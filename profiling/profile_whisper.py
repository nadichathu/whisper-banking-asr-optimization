import time
import json
import pathlib
import platform
import subprocess
import importlib.metadata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import whisper


# ============================================================
# CONFIGURATION
# ============================================================

AUDIO_DIR = pathlib.Path("data/minds14_banking_raw")

MODEL_SIZE = "small"
DEVICE = "cuda:0"

EXPECTED_FILES = 809

WARMUP_RUNS = 3
MEASURED_RUNS = 10

BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 2026


# ============================================================
# HELPERS
# ============================================================

def cuda_sync(device):
    """Synchronize CUDA before/after GPU timing boundaries."""
    torch.cuda.synchronize(device)


def safe_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def git_metadata():
    metadata = {
        "commit": None,
        "branch": None,
        "dirty": None,
    }

    try:
        metadata["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()

        metadata["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True
        ).strip()

        metadata["dirty"] = bool(status)

    except Exception:
        pass

    return metadata


def bootstrap_median_ci(
    values,
    rng,
    n_resamples=10_000
):
    """
    Bootstrap 95% CI for the median.

    Input:
        one median latency value per utterance.

    For this experiment:
        809 utterance-level observations.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    samples = rng.choice(
        values,
        size=(
            n_resamples,
            len(values)
        ),
        replace=True
    )

    medians = np.median(
        samples,
        axis=1
    )

    return (
        float(
            np.percentile(
                medians,
                2.5
            )
        ),
        float(
            np.percentile(
                medians,
                97.5
            )
        ),
    )


# ============================================================
# GPU PROFILE
# ============================================================

def profile_once(
    model,
    audio_path,
    device,
    decode_options
):
    """
    Profile the Whisper Small GPU inference pathway.

    Pipeline
    --------
    1. WAV decoding using Whisper/FFmpeg on CPU
       - measured separately
       - NOT counted as GPU inference latency

    2. Waveform CPU -> GPU transfer

    3. Pad / trim on GPU

    4. Log-Mel spectrogram on GPU

    5. Whisper encoder on GPU

    6. Whisper autoregressive decoding on GPU

    Two aggregate measurements are reported:

    gpu_compute_wall_ms
        Pad/trim + log-Mel + encoder + decoding.
        This is the principal GPU inference metric.

    gpu_pipeline_wall_ms
        CPU->GPU waveform transfer + all GPU computation.

    CPU WAV loading is deliberately reported separately.
    """

    # --------------------------------------------------------
    # Stage 0 — WAV loading / decoding
    #
    # whisper.load_audio() uses FFmpeg and CPU.
    # This cannot be considered GPU model inference.
    # --------------------------------------------------------

    cpu_start = time.perf_counter()

    audio_np = whisper.load_audio(
        str(audio_path)
    )

    cpu_end = time.perf_counter()

    load_audio_cpu_ms = (
        cpu_end - cpu_start
    ) * 1000.0


    # --------------------------------------------------------
    # Convert NumPy waveform to Torch tensor.
    #
    # torch.from_numpy() itself does not copy the waveform.
    # --------------------------------------------------------

    audio_cpu = torch.from_numpy(
        audio_np
    )


    # --------------------------------------------------------
    # Stage 1 — Waveform CPU -> GPU
    # --------------------------------------------------------

    cuda_sync(device)

    pipeline_start = time.perf_counter()

    t0 = time.perf_counter()

    audio_gpu = audio_cpu.to(
        device=device,
        dtype=torch.float32,
        non_blocking=False
    )

    cuda_sync(device)

    t1 = time.perf_counter()

    waveform_h2d_ms = (
        t1 - t0
    ) * 1000.0


    # --------------------------------------------------------
    # Principal GPU computation begins here.
    # --------------------------------------------------------

    cuda_sync(device)

    gpu_compute_start = time.perf_counter()


    # --------------------------------------------------------
    # Stage 2 — Pad / trim ON GPU
    # --------------------------------------------------------

    t0 = time.perf_counter()

    audio_gpu = whisper.pad_or_trim(
        audio_gpu
    )

    cuda_sync(device)

    t1 = time.perf_counter()

    pad_trim_gpu_ms = (
        t1 - t0
    ) * 1000.0


    # --------------------------------------------------------
    # Stage 3 — Log-Mel spectrogram ON GPU
    #
    # Passing a CUDA tensor ensures the STFT / Mel operations
    # execute on the selected GPU.
    # --------------------------------------------------------

    cuda_sync(device)

    t0 = time.perf_counter()

    mel_gpu = whisper.log_mel_spectrogram(
        audio_gpu,
        n_mels=model.dims.n_mels
    )

    cuda_sync(device)

    t1 = time.perf_counter()

    mel_gpu_ms = (
        t1 - t0
    ) * 1000.0


    # --------------------------------------------------------
    # Validate Mel tensor placement
    # --------------------------------------------------------

    if not mel_gpu.is_cuda:
        raise RuntimeError(
            "Log-Mel spectrogram unexpectedly "
            "returned a CPU tensor."
        )

    if mel_gpu.device != device:
        raise RuntimeError(
            "Mel tensor is on the wrong CUDA device. "
            f"Expected {device}, got {mel_gpu.device}."
        )


    # --------------------------------------------------------
    # Stage 4 — Whisper encoder ON GPU
    # --------------------------------------------------------

    cuda_sync(device)

    t0 = time.perf_counter()

    with torch.inference_mode():

        encoded_audio = model.encoder(
            mel_gpu.unsqueeze(0)
        )

    cuda_sync(device)

    t1 = time.perf_counter()

    encoder_gpu_ms = (
        t1 - t0
    ) * 1000.0


    # --------------------------------------------------------
    # Encoder-output validation
    # --------------------------------------------------------

    expected_feature_shape = (
        model.dims.n_audio_ctx,
        model.dims.n_audio_state,
    )

    if (
        tuple(encoded_audio.shape[-2:])
        != expected_feature_shape
    ):
        raise RuntimeError(
            "Unexpected encoder output shape. "
            f"Expected {expected_feature_shape}, "
            f"got "
            f"{tuple(encoded_audio.shape[-2:])}."
        )

    if not encoded_audio.is_cuda:
        raise RuntimeError(
            "Encoder output unexpectedly "
            "resides on CPU."
        )


    # --------------------------------------------------------
    # Stage 5 — Whisper decoding ON GPU
    #
    # encoded_audio is already encoded.
    # whisper.decode() therefore does NOT run
    # the encoder again.
    #
    # This stage includes autoregressive decoding,
    # token selection/filtering, ranking and text
    # construction. It is not pure decoder-kernel time.
    # --------------------------------------------------------

    cuda_sync(device)

    t0 = time.perf_counter()

    with torch.inference_mode():

        decoded_results = whisper.decode(
            model,
            encoded_audio,
            decode_options
        )

    cuda_sync(device)

    t1 = time.perf_counter()

    decode_gpu_ms = (
        t1 - t0
    ) * 1000.0


    # --------------------------------------------------------
    # End principal GPU compute timing
    # --------------------------------------------------------

    cuda_sync(device)

    gpu_compute_end = time.perf_counter()

    gpu_compute_wall_ms = (
        gpu_compute_end
        - gpu_compute_start
    ) * 1000.0


    # --------------------------------------------------------
    # End GPU pipeline:
    # transfer + GPU computation
    # --------------------------------------------------------

    pipeline_end = time.perf_counter()

    gpu_pipeline_wall_ms = (
        pipeline_end
        - pipeline_start
    ) * 1000.0


    # --------------------------------------------------------
    # Decode result validation
    # --------------------------------------------------------

    if not isinstance(
        decoded_results,
        list
    ):
        raise RuntimeError(
            "Expected whisper.decode() "
            "to return a list."
        )

    if len(decoded_results) != 1:
        raise RuntimeError(
            "Expected exactly one decoding "
            f"result, received "
            f"{len(decoded_results)}."
        )

    transcript = (
        decoded_results[0]
        .text
        .strip()
    )


    # --------------------------------------------------------
    # Individually timed GPU stage sum
    # --------------------------------------------------------

    gpu_stage_sum_ms = (
        pad_trim_gpu_ms
        + mel_gpu_ms
        + encoder_gpu_ms
        + decode_gpu_ms
    )

    gpu_pipeline_stage_sum_ms = (
        waveform_h2d_ms
        + gpu_stage_sum_ms
    )


    return {

        # CPU diagnostic only
        "load_audio_cpu_ms":
            load_audio_cpu_ms,

        # Transfer
        "waveform_h2d_ms":
            waveform_h2d_ms,

        # GPU stages
        "pad_trim_gpu_ms":
            pad_trim_gpu_ms,

        "mel_gpu_ms":
            mel_gpu_ms,

        "encoder_gpu_ms":
            encoder_gpu_ms,

        "decode_gpu_ms":
            decode_gpu_ms,

        # Aggregates
        "gpu_stage_sum_ms":
            gpu_stage_sum_ms,

        "gpu_compute_wall_ms":
            gpu_compute_wall_ms,

        "gpu_pipeline_stage_sum_ms":
            gpu_pipeline_stage_sum_ms,

        "gpu_pipeline_wall_ms":
            gpu_pipeline_wall_ms,

        "transcript":
            transcript,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CUDA validation
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. "
            "GPU profiling cannot proceed."
        )

    device = torch.device(
        DEVICE
    )

    if device.type != "cuda":
        raise RuntimeError(
            "This profiler requires CUDA. "
            f"Received device: {DEVICE}"
        )

    torch.cuda.set_device(
        device
    )

    gpu_name = (
        torch.cuda.get_device_name(
            device
        )
    )

    gpu_properties = (
        torch.cuda.get_device_properties(
            device
        )
    )


    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    audio_files = sorted(
        AUDIO_DIR.glob("*.wav")
    )

    if not audio_files:
        raise RuntimeError(
            "No WAV files found in "
            f"{AUDIO_DIR.resolve()}"
        )

    if (
        len(audio_files)
        != EXPECTED_FILES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_FILES} "
            f"WAV files, but found "
            f"{len(audio_files)}."
        )


    # --------------------------------------------------------
    # Unique result directory
    # --------------------------------------------------------

    run_id = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = (
        pathlib.Path("results")
        / f"whisper_gpu_profile_{run_id}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    raw_path = (
        output_dir
        / "raw_runs.csv"
    )

    per_file_path = (
        output_dir
        / "per_file.csv"
    )

    stage_summary_path = (
        output_dir
        / "stage_summary.csv"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )


    # --------------------------------------------------------
    # Environment display
    # --------------------------------------------------------

    print("=" * 80)
    print(
        "WHISPER SMALL GPU INFERENCE PROFILING"
    )
    print("=" * 80)

    print(
        f"GPU                : "
        f"{gpu_name}"
    )

    print(
        f"CUDA device        : "
        f"{device}"
    )

    print(
        f"GPU memory         : "
        f"{gpu_properties.total_memory / 1024**3:.2f} GB"
    )

    print(
        f"PyTorch            : "
        f"{torch.__version__}"
    )

    print(
        f"PyTorch CUDA       : "
        f"{torch.version.cuda}"
    )

    print(
        f"OpenAI Whisper     : "
        f"{safe_package_version('openai-whisper')}"
    )

    print(
        f"Model              : "
        f"{MODEL_SIZE}"
    )

    print(
        f"Audio files        : "
        f"{len(audio_files)}"
    )

    print(
        f"Warm-up runs       : "
        f"{WARMUP_RUNS}"
    )

    print(
        f"Measured runs/file : "
        f"{MEASURED_RUNS}"
    )

    print(
        f"Output directory   : "
        f"{output_dir}"
    )


    # --------------------------------------------------------
    # Model loading
    #
    # Explicitly excluded from latency measurements.
    # --------------------------------------------------------

    print(
        "\nLoading Whisper model..."
    )

    model = whisper.load_model(
        MODEL_SIZE,
        device=device
    )

    model.eval()


    # --------------------------------------------------------
    # Verify model is on CUDA
    # --------------------------------------------------------

    model_parameter = next(
        model.parameters()
    )

    if not model_parameter.is_cuda:
        raise RuntimeError(
            "Whisper model is not on GPU."
        )

    if (
        model_parameter.device
        != device
    ):
        raise RuntimeError(
            "Whisper model loaded on "
            "unexpected device. "
            f"Expected {device}, "
            f"got {model_parameter.device}."
        )


    # --------------------------------------------------------
    # FP32 baseline validation
    # --------------------------------------------------------

    model_dtype = (
        model_parameter.dtype
    )

    print(
        f"Model parameter dtype: "
        f"{model_dtype}"
    )

    if (
        model_dtype
        != torch.float32
    ):
        raise RuntimeError(
            "Expected Whisper Small "
            "FP32 parameters, but found "
            f"{model_dtype}."
        )


    # --------------------------------------------------------
    # Explicit decoding configuration
    # --------------------------------------------------------

    decode_options = (
        whisper.DecodingOptions(
            task="transcribe",
            language="en",
            temperature=0.0,
            fp16=False,
            without_timestamps=False,
        )
    )


    # --------------------------------------------------------
    # Provenance metadata
    # --------------------------------------------------------

    metadata = {

        "created_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "model":
            MODEL_SIZE,

        "device":
            str(device),

        "gpu_name":
            gpu_name,

        "gpu_total_memory_bytes":
            gpu_properties.total_memory,

        "python":
            platform.python_version(),

        "platform":
            platform.platform(),

        "torch_version":
            torch.__version__,

        "torch_cuda_version":
            torch.version.cuda,

        "openai_whisper_version":
            safe_package_version(
                "openai-whisper"
            ),

        "model_dtype":
            str(model_dtype),

        "dataset_directory":
            str(
                AUDIO_DIR.resolve()
            ),

        "number_of_files":
            len(audio_files),

        "warmup_runs":
            WARMUP_RUNS,

        "measured_runs_per_file":
            MEASURED_RUNS,

        "bootstrap_resamples":
            BOOTSTRAP_RESAMPLES,

        "bootstrap_seed":
            RANDOM_SEED,

        "decode_options": {
            "task": "transcribe",
            "language": "en",
            "temperature": 0.0,
            "fp16": False,
            "without_timestamps": False,
        },

        "profiling_design": {
            "wav_loading":
                "CPU/FFmpeg; measured separately "
                "and excluded from GPU inference latency",

            "waveform_transfer":
                "CPU-to-GPU transfer measured separately",

            "pad_trim":
                "GPU",

            "log_mel":
                "GPU",

            "encoder":
                "GPU",

            "decode":
                "GPU",

            "primary_metric":
                "gpu_compute_wall_ms",
        },

        "torch_backend": {

            "cudnn_benchmark":
                torch.backends.cudnn.benchmark,

            "cudnn_deterministic":
                torch.backends.cudnn.deterministic,

            "cuda_matmul_allow_tf32":
                torch.backends.cuda.matmul.allow_tf32,

            "cudnn_allow_tf32":
                torch.backends.cudnn.allow_tf32,
        },

        "git":
            git_metadata(),

        "profiling_scope": (
            "Stage-level profiling of the "
            "Whisper Small FP32 GPU inference path. "
            "CPU WAV decoding is measured separately "
            "and excluded from the principal GPU "
            "inference latency metric."
        ),
    }


    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )


    # --------------------------------------------------------
    # Global GPU warm-up
    # --------------------------------------------------------

    print(
        "\nPerforming GPU warm-up..."
    )

    warmup_file = (
        audio_files[0]
    )

    for i in range(
        WARMUP_RUNS
    ):

        _ = profile_once(
            model,
            warmup_file,
            device,
            decode_options
        )

        print(
            f"Warm-up "
            f"{i + 1}/"
            f"{WARMUP_RUNS} complete"
        )

    cuda_sync(
        device
    )

    print(
        "GPU warm-up complete."
    )


    # --------------------------------------------------------
    # Full profiling experiment
    # --------------------------------------------------------

    rows = []

    for (
        file_index,
        audio_path
    ) in enumerate(
        audio_files,
        start=1
    ):

        print(
            f"\n[{file_index:03d}/"
            f"{len(audio_files)}] "
            f"{audio_path.name}"
        )

        for run in range(
            1,
            MEASURED_RUNS + 1
        ):

            profile = profile_once(
                model,
                audio_path,
                device,
                decode_options
            )

            row = {
                "file":
                    audio_path.name,

                "run":
                    run,

                **profile,
            }

            rows.append(
                row
            )

            print(
                f"Run {run:02d} | "
                f"H2D="
                f"{profile['waveform_h2d_ms']:.2f} | "
                f"pad="
                f"{profile['pad_trim_gpu_ms']:.2f} | "
                f"mel="
                f"{profile['mel_gpu_ms']:.2f} | "
                f"encoder="
                f"{profile['encoder_gpu_ms']:.2f} | "
                f"decode="
                f"{profile['decode_gpu_ms']:.2f} | "
                f"GPU compute="
                f"{profile['gpu_compute_wall_ms']:.2f} ms"
            )


    # --------------------------------------------------------
    # Raw results
    # --------------------------------------------------------

    df = pd.DataFrame(
        rows
    )


    gpu_stage_columns = [
        "pad_trim_gpu_ms",
        "mel_gpu_ms",
        "encoder_gpu_ms",
        "decode_gpu_ms",
    ]


    # --------------------------------------------------------
    # Per-run GPU stage shares
    # --------------------------------------------------------

    measured_gpu_stage_sum = (
        df[gpu_stage_columns]
        .sum(axis=1)
    )

    for stage in (
        gpu_stage_columns
    ):

        df[
            f"{stage}_share_pct"
        ] = (
            df[stage]
            / measured_gpu_stage_sum
            * 100.0
        )


    df.to_csv(
        raw_path,
        index=False
    )


    # --------------------------------------------------------
    # Per-file medians
    #
    # 809 utterances = 809 analytical observations.
    # Each is the median of 10 measured runs.
    # --------------------------------------------------------

    latency_columns = [

        "load_audio_cpu_ms",

        "waveform_h2d_ms",

        "pad_trim_gpu_ms",

        "mel_gpu_ms",

        "encoder_gpu_ms",

        "decode_gpu_ms",

        "gpu_stage_sum_ms",

        "gpu_compute_wall_ms",

        "gpu_pipeline_stage_sum_ms",

        "gpu_pipeline_wall_ms",
    ]


    per_file = (
        df.groupby(
            "file"
        )[latency_columns]
        .median()
        .reset_index()
    )


    per_file.to_csv(
        per_file_path,
        index=False
    )


    # --------------------------------------------------------
    # Stage contribution percentages
    # --------------------------------------------------------

    share_columns = [
        f"{stage}_share_pct"
        for stage
        in gpu_stage_columns
    ]

    per_file_shares = (
        df.groupby(
            "file"
        )[share_columns]
        .mean()
    )

    overall_stage_shares = (
        per_file_shares
        .mean()
    )


    # --------------------------------------------------------
    # GPU stage statistics
    # --------------------------------------------------------

    rng = (
        np.random.default_rng(
            RANDOM_SEED
        )
    )

    summary_rows = []

    for stage in (
        gpu_stage_columns
    ):

        values = (
            per_file[stage]
            .to_numpy(
                dtype=float
            )
        )

        ci_low, ci_high = (
            bootstrap_median_ci(
                values,
                rng,
                BOOTSTRAP_RESAMPLES
            )
        )

        summary_rows.append({

            "stage":
                stage,

            "median_ms":
                float(
                    np.median(
                        values
                    )
                ),

            "ci95_low_ms":
                ci_low,

            "ci95_high_ms":
                ci_high,

            "p95_ms":
                float(
                    np.percentile(
                        values,
                        95
                    )
                ),

            "mean_stage_share_pct":
                float(
                    overall_stage_shares[
                        f"{stage}_share_pct"
                    ]
                ),
        })


    stage_summary = (
        pd.DataFrame(
            summary_rows
        )
    )

    stage_summary.to_csv(
        stage_summary_path,
        index=False
    )


    # --------------------------------------------------------
    # Overall GPU compute latency
    # --------------------------------------------------------

    gpu_values = (
        per_file[
            "gpu_compute_wall_ms"
        ]
        .to_numpy(
            dtype=float
        )
    )

    (
        gpu_ci_low,
        gpu_ci_high
    ) = bootstrap_median_ci(
        gpu_values,
        rng,
        BOOTSTRAP_RESAMPLES
    )


    # --------------------------------------------------------
    # GPU pipeline latency including waveform H2D
    # --------------------------------------------------------

    gpu_pipeline_values = (
        per_file[
            "gpu_pipeline_wall_ms"
        ]
        .to_numpy(
            dtype=float
        )
    )


    # --------------------------------------------------------
    # Final console report
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print(
        "FINAL WHISPER GPU PROFILING SUMMARY"
    )
    print("=" * 80)

    print(
        stage_summary.to_string(
            index=False,
            formatters={

                "median_ms":
                    lambda x:
                    f"{x:.3f}",

                "ci95_low_ms":
                    lambda x:
                    f"{x:.3f}",

                "ci95_high_ms":
                    lambda x:
                    f"{x:.3f}",

                "p95_ms":
                    lambda x:
                    f"{x:.3f}",

                "mean_stage_share_pct":
                    lambda x:
                    f"{x:.2f}%",
            }
        )
    )


    print(
        "\nGPU compute latency "
        "(primary profiling metric)"
    )

    print(
        f"Median: "
        f"{np.median(gpu_values):.3f} ms"
    )

    print(
        f"95% bootstrap CI: "
        f"{gpu_ci_low:.3f}–"
        f"{gpu_ci_high:.3f} ms"
    )


    print(
        "\nGPU pipeline latency "
        "(waveform H2D + GPU compute)"
    )

    print(
        f"Median: "
        f"{np.median(gpu_pipeline_values):.3f} ms"
    )


    print(
        "\nCPU audio loading "
        "(reported separately)"
    )

    print(
        f"Median: "
        f"{per_file['load_audio_cpu_ms'].median():.3f} ms"
    )


    print(
        "\nMean GPU stage shares sum to: "
        f"{stage_summary['mean_stage_share_pct'].sum():.2f}%"
    )


    print("\nSaved:")
    print(
        f"  {raw_path}"
    )
    print(
        f"  {per_file_path}"
    )
    print(
        f"  {stage_summary_path}"
    )
    print(
        f"  {metadata_path}"
    )

    print(
        "\nGPU profiling completed successfully."
    )


if __name__ == "__main__":
    main()