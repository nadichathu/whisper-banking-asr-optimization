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

EXPECTED_FILES = 41

WARMUP_RUNS = 3
MEASURED_RUNS = 10

BOOTSTRAP_RESAMPLES = 10_000
RANDOM_SEED = 2026


# ============================================================
# HELPERS
# ============================================================

def cuda_sync(device):
    """Wait until all queued CUDA work on the selected device is complete."""
    torch.cuda.synchronize(device)


def safe_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return None


def git_metadata():
    """Collect lightweight Git provenance without failing the benchmark."""
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


def bootstrap_median_ci(values, rng, n_resamples=10_000):
    """
    Bootstrap 95% confidence interval for the median.

    Input should be the 41 utterance-level values,
    not the 410 repeated measurements.
    """
    values = np.asarray(values, dtype=float)

    samples = rng.choice(
        values,
        size=(n_resamples, len(values)),
        replace=True
    )

    medians = np.median(samples, axis=1)

    return (
        float(np.percentile(medians, 2.5)),
        float(np.percentile(medians, 97.5)),
    )


# ============================================================
# ONE PROFILED INFERENCE
# ============================================================

def profile_once(model, audio_path, device, decode_options):
    """
    Profile one Whisper Small inference path.

    Important:
    - Encoder is executed exactly once.
    - whisper.decode() receives already-encoded audio features.
    - decode_stage_ms therefore excludes the explicit encoder stage.
    - decode_stage_ms still includes Whisper's autoregressive decoding
      machinery, token selection/filtering, ranking and text construction.
      It is NOT a pure decoder-kernel measurement.
    """

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Stage 1 — Audio loading
    # CPU / FFmpeg path used by OpenAI Whisper
    # --------------------------------------------------------

    t0 = time.perf_counter()

    audio = whisper.load_audio(str(audio_path))

    t1 = time.perf_counter()

    load_audio_ms = (t1 - t0) * 1000.0


    # --------------------------------------------------------
    # Stage 2 — Pad / trim
    # Pads each short command to Whisper's 30-second input
    # --------------------------------------------------------

    t0 = time.perf_counter()

    audio = whisper.pad_or_trim(audio)

    t1 = time.perf_counter()

    pad_trim_ms = (t1 - t0) * 1000.0


    # --------------------------------------------------------
    # Stage 3 — Log-Mel spectrogram
    #
    # Kept on CPU deliberately so that preprocessing and
    # host-to-device transfer remain separate measurements.
    # --------------------------------------------------------

    t0 = time.perf_counter()

    mel_cpu = whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels
    )

    t1 = time.perf_counter()

    mel_cpu_ms = (t1 - t0) * 1000.0


    # --------------------------------------------------------
    # Stage 4 — CPU -> GPU transfer
    # --------------------------------------------------------

    cuda_sync(device)

    t0 = time.perf_counter()

    mel_gpu = mel_cpu.to(
        device=device,
        non_blocking=False
    )

    cuda_sync(device)

    t1 = time.perf_counter()

    h2d_transfer_ms = (t1 - t0) * 1000.0


    # --------------------------------------------------------
    # Stage 5 — Whisper audio encoder
    # --------------------------------------------------------

    cuda_sync(device)

    t0 = time.perf_counter()

    with torch.inference_mode():
        encoded_audio = model.encoder(
            mel_gpu.unsqueeze(0)
        )

    cuda_sync(device)

    t1 = time.perf_counter()

    encoder_ms = (t1 - t0) * 1000.0


    # --------------------------------------------------------
    # Sanity check:
    # encoded features must have Whisper's expected shape
    # --------------------------------------------------------

    expected_feature_shape = (
        model.dims.n_audio_ctx,
        model.dims.n_audio_state,
    )

    if tuple(encoded_audio.shape[-2:]) != expected_feature_shape:
        raise RuntimeError(
            "Unexpected encoder output shape. "
            f"Expected final dimensions {expected_feature_shape}, "
            f"got {tuple(encoded_audio.shape[-2:])}."
        )


    # --------------------------------------------------------
    # Stage 6 — Whisper decoding stage
    #
    # encoded_audio is 3-D:
    # [batch, n_audio_ctx, n_audio_state]
    #
    # whisper.decode() therefore returns List[DecodingResult].
    #
    # The internal _get_audio_features() detects that this is
    # already encoded audio and DOES NOT run the encoder again.
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

    decode_stage_ms = (t1 - t0) * 1000.0


    # --------------------------------------------------------
    # Correct handling of the batched return value
    # --------------------------------------------------------

    if not isinstance(decoded_results, list):
        raise RuntimeError(
            "Expected whisper.decode() to return a list because "
            "encoded_audio is a 3-D batched tensor."
        )

    if len(decoded_results) != 1:
        raise RuntimeError(
            "Expected exactly one decoding result, "
            f"but received {len(decoded_results)}."
        )

    result = decoded_results[0]

    transcript = result.text.strip()


    # --------------------------------------------------------
    # Actual wall-clock time for the complete profiled path
    # --------------------------------------------------------

    total_end = time.perf_counter()

    total_profile_wall_ms = (
        total_end - total_start
    ) * 1000.0


    # --------------------------------------------------------
    # Sum of individually timed stages
    #
    # Kept separately because median(stage sums) and
    # sum(stage medians) are mathematically different.
    # --------------------------------------------------------

    stage_sum_ms = (
        load_audio_ms
        + pad_trim_ms
        + mel_cpu_ms
        + h2d_transfer_ms
        + encoder_ms
        + decode_stage_ms
    )


    return {
        "load_audio_ms": load_audio_ms,
        "pad_trim_ms": pad_trim_ms,
        "mel_cpu_ms": mel_cpu_ms,
        "h2d_transfer_ms": h2d_transfer_ms,
        "encoder_ms": encoder_ms,
        "decode_stage_ms": decode_stage_ms,
        "stage_sum_ms": stage_sum_ms,
        "total_profile_wall_ms": total_profile_wall_ms,
        "transcript": transcript,
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
            "This experiment must be executed on the GPU."
        )

    device = torch.device(DEVICE)

    if device.type != "cuda":
        raise RuntimeError(
            f"GPU profiling requires CUDA; received {DEVICE}."
        )

    torch.cuda.set_device(device)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_properties = torch.cuda.get_device_properties(device)


    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not audio_files:
        raise RuntimeError(
            f"No WAV files found in {AUDIO_DIR.resolve()}"
        )

    if len(audio_files) != EXPECTED_FILES:
        raise RuntimeError(
            f"Expected {EXPECTED_FILES} WAV files, "
            f"but found {len(audio_files)}."
        )


    # --------------------------------------------------------
    # Unique result directory
    # --------------------------------------------------------

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    output_dir = pathlib.Path(
        "results"
    ) / f"whisper_gpu_profile_{run_id}"

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    raw_path = output_dir / "raw_runs.csv"
    per_file_path = output_dir / "per_file.csv"
    stage_summary_path = output_dir / "stage_summary.csv"
    metadata_path = output_dir / "metadata.json"


    # --------------------------------------------------------
    # Display environment
    # --------------------------------------------------------

    print("=" * 80)
    print("WHISPER SMALL GPU STAGE PROFILING")
    print("=" * 80)

    print(f"GPU                : {gpu_name}")
    print(f"CUDA device        : {device}")
    print(f"GPU memory         : {gpu_properties.total_memory / 1024**3:.2f} GB")
    print(f"PyTorch            : {torch.__version__}")
    print(f"PyTorch CUDA       : {torch.version.cuda}")
    print(f"OpenAI Whisper     : {safe_package_version('openai-whisper')}")
    print(f"Model              : {MODEL_SIZE}")
    print(f"Audio files        : {len(audio_files)}")
    print(f"Warm-up runs       : {WARMUP_RUNS}")
    print(f"Measured runs/file : {MEASURED_RUNS}")
    print(f"Output directory   : {output_dir}")


    # --------------------------------------------------------
    # Load Whisper
    # Model loading is deliberately excluded from profiling.
    # --------------------------------------------------------

    print("\nLoading Whisper model...")

    model = whisper.load_model(
        MODEL_SIZE,
        device=device
    )

    model.eval()


    # --------------------------------------------------------
    # Verify FP32 baseline
    # --------------------------------------------------------

    model_dtype = next(
        model.parameters()
    ).dtype

    print(f"Model parameter dtype: {model_dtype}")

    if model_dtype != torch.float32:
        raise RuntimeError(
            "Expected Whisper Small FP32 model parameters, "
            f"but found {model_dtype}."
        )


    # --------------------------------------------------------
    # Explicit decoding configuration
    #
    # This profiles the FP32, timestamps-enabled,
    # temperature-zero decoding path.
    # --------------------------------------------------------

    decode_options = whisper.DecodingOptions(
        task="transcribe",
        language="en",
        temperature=0.0,
        fp16=False,
        without_timestamps=False,
    )


    # --------------------------------------------------------
    # Save provenance metadata
    # --------------------------------------------------------

    metadata = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": MODEL_SIZE,
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_properties.total_memory,

        "python": platform.python_version(),
        "platform": platform.platform(),

        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "openai_whisper_version":
            safe_package_version("openai-whisper"),

        "model_dtype": str(model_dtype),

        "dataset_directory": str(
            AUDIO_DIR.resolve()
        ),
        "number_of_files": len(audio_files),

        "warmup_runs": WARMUP_RUNS,
        "measured_runs_per_file": MEASURED_RUNS,

        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": RANDOM_SEED,

        "decode_options": {
            "task": "transcribe",
            "language": "en",
            "temperature": 0.0,
            "fp16": False,
            "without_timestamps": False,
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

        "git": git_metadata(),

        "profiling_scope": (
            "Exploratory stage-level profiling of one "
            "Whisper Small FP32 single-window decoding path. "
            "Not an exact decomposition of the high-level "
            "model.transcribe() pipeline."
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
    # Global warm-up
    # --------------------------------------------------------

    print("\nPerforming GPU warm-up...")

    warmup_file = audio_files[0]

    for i in range(WARMUP_RUNS):

        _ = profile_once(
            model,
            warmup_file,
            device,
            decode_options
        )

        print(
            f"Warm-up {i + 1}/{WARMUP_RUNS} complete"
        )

    cuda_sync(device)

    print("Warm-up complete.")


    # --------------------------------------------------------
    # Full profiling experiment
    # --------------------------------------------------------

    rows = []

    for file_index, audio_path in enumerate(
        audio_files,
        start=1
    ):

        print(
            f"\n[{file_index:02d}/{len(audio_files)}] "
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
                "file": audio_path.name,
                "run": run,
                **profile,
            }

            rows.append(row)

            print(
                f"Run {run:02d} | "
                f"load={profile['load_audio_ms']:.2f} | "
                f"pad={profile['pad_trim_ms']:.2f} | "
                f"mel={profile['mel_cpu_ms']:.2f} | "
                f"H2D={profile['h2d_transfer_ms']:.2f} | "
                f"encoder={profile['encoder_ms']:.2f} | "
                f"decode={profile['decode_stage_ms']:.2f} | "
                f"total={profile['total_profile_wall_ms']:.2f} ms"
            )


    # --------------------------------------------------------
    # Raw results
    # --------------------------------------------------------

    df = pd.DataFrame(rows)


    stage_columns = [
        "load_audio_ms",
        "pad_trim_ms",
        "mel_cpu_ms",
        "h2d_transfer_ms",
        "encoder_ms",
        "decode_stage_ms",
    ]


    # --------------------------------------------------------
    # Per-run stage percentages
    # --------------------------------------------------------

    measured_stage_sum = df[
        stage_columns
    ].sum(axis=1)

    for stage in stage_columns:

        df[f"{stage}_share_pct"] = (
            df[stage]
            / measured_stage_sum
            * 100.0
        )


    df.to_csv(
        raw_path,
        index=False
    )


    # --------------------------------------------------------
    # Per-file latency medians
    #
    # One file = one analytical latency observation,
    # consistent with the main dissertation methodology.
    # --------------------------------------------------------

    latency_columns = stage_columns + [
        "stage_sum_ms",
        "total_profile_wall_ms",
    ]

    per_file = (
        df.groupby("file")[latency_columns]
        .median()
        .reset_index()
    )

    per_file.to_csv(
        per_file_path,
        index=False
    )


    # --------------------------------------------------------
    # Stage contribution percentages
    #
    # Percentages are calculated at run level first.
    # Then:
    #   10 runs -> mean percentage for each file
    #   41 files -> mean percentage across files
    #
    # This gives each utterance equal weight and causes the
    # final stage percentages to sum to 100%.
    # --------------------------------------------------------

    share_columns = [
        f"{stage}_share_pct"
        for stage in stage_columns
    ]

    per_file_shares = (
        df.groupby("file")[share_columns]
        .mean()
    )

    overall_stage_shares = (
        per_file_shares.mean()
    )


    # --------------------------------------------------------
    # Stage statistics
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    summary_rows = []

    for stage in stage_columns:

        values = per_file[
            stage
        ].to_numpy(dtype=float)

        ci_low, ci_high = (
            bootstrap_median_ci(
                values,
                rng,
                BOOTSTRAP_RESAMPLES
            )
        )

        summary_rows.append({
            "stage": stage,

            "median_ms":
                float(np.median(values)),

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


    stage_summary = pd.DataFrame(
        summary_rows
    )

    stage_summary.to_csv(
        stage_summary_path,
        index=False
    )


    # --------------------------------------------------------
    # Overall profiled wall latency
    # --------------------------------------------------------

    total_values = per_file[
        "total_profile_wall_ms"
    ].to_numpy(dtype=float)

    total_ci_low, total_ci_high = (
        bootstrap_median_ci(
            total_values,
            rng,
            BOOTSTRAP_RESAMPLES
        )
    )


    # --------------------------------------------------------
    # Final console report
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("FINAL GPU STAGE-PROFILING SUMMARY")
    print("=" * 80)

    print(
        stage_summary.to_string(
            index=False,
            formatters={
                "median_ms":
                    lambda x: f"{x:.3f}",

                "ci95_low_ms":
                    lambda x: f"{x:.3f}",

                "ci95_high_ms":
                    lambda x: f"{x:.3f}",

                "p95_ms":
                    lambda x: f"{x:.3f}",

                "mean_stage_share_pct":
                    lambda x: f"{x:.2f}%",
            }
        )
    )

    print("\nOverall profiled wall latency")
    print(
        f"Median: "
        f"{np.median(total_values):.3f} ms"
    )

    print(
        f"95% bootstrap CI: "
        f"{total_ci_low:.3f}–"
        f"{total_ci_high:.3f} ms"
    )

    print(
        "\nMean stage shares sum to: "
        f"{stage_summary['mean_stage_share_pct'].sum():.2f}%"
    )

    print("\nSaved:")
    print(f"  {raw_path}")
    print(f"  {per_file_path}")
    print(f"  {stage_summary_path}")
    print(f"  {metadata_path}")

    print("\nProfiling completed successfully.")


if __name__ == "__main__":
    main()