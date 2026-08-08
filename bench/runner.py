import argparse
import importlib
import math
import pathlib
import statistics as stats
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import torch

from bench.config import (
    AUDIO_DIR,
    METADATA_FILE,
    N_RUNS,
    N_WARMUP,
    RESULTS_DIR,
)
from bench.utils import (
    compute_cer,
    compute_wer,
    ensure_dir,
    env_metadata,
    json_dumps_safe,
    normalize_transcript,
    percentile,
)


# ============================================================
# Adapter registry
# ============================================================

ADAPTER_REGISTRY = {
    "whisper": (
        "bench.adapters.whisper_baseline_adapter",
        "WhisperAdapter",
    ),
    "whisper_greedy": (
        "bench.adapters.whisper_greedy_adapter",
        "WhisperGreedyAdapter",
    ),
    "whisper_no_context": (
        "bench.adapters.whisper_no_context_adapter",
        "WhisperNoContextAdapter",
    ),
    "whisper_vad": (
        "bench.adapters.whisper_vad_adapter",
        "WhisperVadAdapter",
    ),
    "whisper_fp16": (
        "bench.adapters.whisper_fp16_adapter",
        "WhisperFP16Adapter",
    ),
    "whisper_limited_tokens": (
        "bench.adapters.whisper_limited_tokens_adapter",
        "WhisperLimitedTokensAdapter",
    ),
    "whisper_no_timestamps": (
        "bench.adapters.whisper_no_timestamps_adapter",
        "WhisperNoTimestampsAdapter",
    ),
    "whisper_no_fallback": (
        "bench.adapters.whisper_no_fallback_adapter",
        "WhisperNoFallbackAdapter",
    ),
    "whisper_direct_decode": (
        "bench.adapters.whisper_direct_decode_adapter",
        "WhisperDirectDecodeAdapter",
    ),
    "whisper_int8": (
        "bench.adapters.whisper_int8_adapter",
        "WhisperINT8Adapter",
    ),
    "faster_whisper": (
        "bench.adapters.faster_whisper_adapter",
        "FasterWhisperAdapter",
    ),
    "faster_whisper_small": (
        "bench.adapters.faster_whisper_small_adapter",
        "FasterWhisperSmallAdapter",
    ),
    "faster_whisper_medium": (
        "bench.adapters.faster_whisper_medium_adapter",
        "FasterWhisperMediumAdapter",
    ),
    "wav2vec2": (
        "bench.adapters.wav2vec2_adapter",
        "Wav2Vec2Adapter",
    ),
    "parakeet": (
        "bench.adapters.parakeet_adapter",
        "ParakeetAdapter",
    ),
    "nemo_fastconformer": (
        "bench.adapters.nemo_fastconformer_adapter",
        "NeMoFastConformerAdapter",
    ),

    # --------------------------------------------------------
    # New external / precision comparison conditions
    # --------------------------------------------------------
    "qwen3_asr": (
        "bench.adapters.qwen3_asr_adapter",
        "Qwen3ASRAdapter",
    ),
    "distil_whisper": (
        "bench.adapters.distil_whisper_adapter",
        "DistilWhisperAdapter",
    ),
    "faster_whisper_small_bf16": (
        "bench.adapters.faster_whisper_small_bf16_adapter",
        "FasterWhisperSmallBF16Adapter",
    ),
}


# ============================================================
# Adapter loading
# ============================================================

def load_adapter_class(model_key: str) -> Type:
    """Dynamically import the selected adapter only."""

    if model_key not in ADAPTER_REGISTRY:
        available = ", ".join(
            sorted(ADAPTER_REGISTRY.keys())
        )

        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Available models: {available}"
        )

    module_name, class_name = ADAPTER_REGISTRY[
        model_key
    ]

    try:
        module = importlib.import_module(
            module_name
        )
    except Exception as exc:
        raise ImportError(
            f"Could not import module '{module_name}' "
            f"for model '{model_key}'. "
            f"Original error: {exc}"
        ) from exc

    try:
        adapter_class = getattr(
            module,
            class_name,
        )
    except AttributeError as exc:
        available_adapter_classes = [
            name
            for name in dir(module)
            if name.endswith("Adapter")
        ]

        raise ImportError(
            f"Module '{module_name}' does not contain "
            f"'{class_name}'. Adapter classes found: "
            f"{available_adapter_classes or 'none'}"
        ) from exc

    return adapter_class


def resolve_model_name(
    adapter: Any,
    fallback_name: str,
) -> str:
    """Return a stable filename-safe model name."""

    model_name = (
        getattr(adapter, "name", None)
        or getattr(adapter, "model_id", None)
        or fallback_name
    )

    return (
        str(model_name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


# ============================================================
# GPU validation
# ============================================================

def validate_device(
    requested_device: Optional[str],
) -> str:
    """Validate a CUDA device for this GPU-only suite."""

    device_text = (
        requested_device or "cuda"
    ).lower().strip()

    try:
        device = torch.device(device_text)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(
            f"Invalid device value: '{device_text}'."
        ) from exc

    if device.type != "cuda":
        raise SystemExit(
            "This benchmark suite supports GPU execution "
            "only. Use '--device cuda' or '--device cuda:0'."
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but no CUDA-compatible "
            "GPU is available."
        )

    device_index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )

    if (
        device_index < 0
        or device_index >= torch.cuda.device_count()
    ):
        raise SystemExit(
            f"CUDA device index {device_index} is invalid. "
            f"Detected CUDA devices: "
            f"{torch.cuda.device_count()}."
        )

    return f"cuda:{device_index}"


def normalize_cuda_device(
    device_value: str,
) -> str:
    """Normalize CUDA and CUDA:index into CUDA:index."""

    device = torch.device(
        str(device_value).lower().strip()
    )

    if device.type != "cuda":
        raise ValueError(
            f"Expected a CUDA device, received '{device_value}'."
        )

    device_index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )

    return f"cuda:{device_index}"


def get_effective_adapter_device_string(
    adapter: Any,
    fallback: str,
) -> str:
    """Return the device string that actually reflects where the adapter
    will run, preferring an explicit ``device_index`` attribute when the
    adapter exposes one.

    CTranslate2-based adapters (FasterWhisperAdapter, WhisperINT8Adapter,
    and their Small/Medium subclasses) store ``self.device`` as the bare
    string "cuda" and track the GPU index separately in
    ``self.device_index``, per CTranslate2's own API convention
    (WhisperModel(..., device="cuda", device_index=N)). Reading only
    ``adapter.device`` for those adapters would lose the index entirely
    and fall back to whatever torch.cuda.current_device() happens to be,
    which can silently diverge from the actually-requested device on any
    multi-GPU setup or non-default --device index, even though the
    adapter itself is correctly configured via device_index.
    """

    device_index = getattr(adapter, "device_index", None)
    if device_index is not None:
        return f"cuda:{device_index}"

    return str(getattr(adapter, "device", fallback))


def synchronize_cuda(
    device: str,
) -> None:
    """Synchronize the selected CUDA device."""

    torch.cuda.synchronize(
        device=torch.device(device)
    )


# ============================================================
# Metadata and reproducibility helpers
# ============================================================

def load_references(
    metadata_path: pathlib.Path,
) -> Dict[str, str]:
    """Load filename-to-transcript mappings."""

    if not metadata_path.exists():
        raise SystemExit(
            f"Metadata file not found: {metadata_path}"
        )

    metadata_df = pd.read_csv(
        metadata_path
    )

    required_columns = {
        "file_name",
        "transcript",
    }

    missing_columns = (
        required_columns.difference(
            metadata_df.columns
        )
    )

    if missing_columns:
        raise SystemExit(
            "Metadata file is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    metadata_df["file_name"] = (
        metadata_df["file_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    metadata_df["transcript"] = (
        metadata_df["transcript"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    duplicated = metadata_df[
        metadata_df["file_name"].duplicated(
            keep=False
        )
    ]["file_name"].tolist()

    if duplicated:
        print(
            "WARNING: Duplicate filenames found "
            "in metadata:"
        )

        for file_name in sorted(
            set(duplicated)
        ):
            print(f"  - {file_name}")

        print(
            "The final occurrence of each duplicated "
            "filename will be used."
        )

    return dict(
        zip(
            metadata_df["file_name"],
            metadata_df["transcript"],
        )
    )


def run_git_command(
    arguments: List[str],
) -> Optional[str]:
    """Execute a Git command without failing the benchmark."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=True,
        )

        return completed.stdout.strip()

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None


def git_metadata() -> Dict[str, Any]:
    """Return Git commit and working-tree state."""

    commit = run_git_command(
        ["rev-parse", "HEAD"]
    )

    branch = run_git_command(
        ["branch", "--show-current"]
    )

    status = run_git_command(
        ["status", "--porcelain"]
    )

    return {
        "commit": commit,
        "branch": branch,
        "working_tree_clean": (
            status == ""
            if status is not None
            else None
        ),
    }


def create_run_directory(
    results_root: pathlib.Path,
    model_name: str,
) -> tuple[str, pathlib.Path]:
    """Create a unique result folder for this experiment."""

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")

    commit = (
        run_git_command(
            ["rev-parse", "--short", "HEAD"]
        )
        or "no_git"
    )

    run_id = (
        f"{timestamp}_{commit}_{model_name}"
    )

    run_directory = (
        results_root
        / run_id
    )

    ensure_dir(
        run_directory
    )

    return run_id, run_directory


# ============================================================
# Timing and statistics
# ============================================================

def calculate_difference_percent(
    adapter_latency_ms: float,
    runner_wall_latency_ms: float,
) -> float:
    """Calculate wall-clock minus adapter latency percentage."""

    if adapter_latency_ms <= 0:
        return 0.0

    difference = (
        runner_wall_latency_ms
        - adapter_latency_ms
    )

    return (
        difference / adapter_latency_ms
    ) * 100.0


def should_warn_about_timing(
    adapter_latency_ms: float,
    runner_wall_latency_ms: float,
    minimum_difference_ms: float = 20.0,
    minimum_difference_percent: float = 10.0,
) -> bool:
    """Identify a likely timing-boundary mismatch."""

    difference_ms = abs(
        runner_wall_latency_ms
        - adapter_latency_ms
    )

    difference_percent = abs(
        calculate_difference_percent(
            adapter_latency_ms,
            runner_wall_latency_ms,
        )
    )

    return (
        difference_ms >= minimum_difference_ms
        and difference_percent
        >= minimum_difference_percent
    )


def latency_stats(
    values: List[float],
    include_percentiles: bool = True,
) -> Dict[str, Optional[float]]:
    """Calculate descriptive latency statistics."""

    if not values:
        result: Dict[
            str,
            Optional[float],
        ] = {
            "mean_ms": None,
            "median_ms": None,
            "stdev_ms": None,
            "minimum_ms": None,
            "maximum_ms": None,
        }

        if include_percentiles:
            result["p95_ms"] = None
            result["p99_ms"] = None

        return result

    result = {
        "mean_ms": stats.mean(values),
        "median_ms": stats.median(values),
        "stdev_ms": (
            stats.stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }

    if include_percentiles:
        result["p95_ms"] = percentile(
            values,
            95,
        )

        result["p99_ms"] = percentile(
            values,
            99,
        )

    return result


# ============================================================
# Output validation
# ============================================================

def validate_adapter_output(
    output: Any,
    adapter_class: Type,
) -> tuple[str, float, Dict[str, Any]]:
    """Validate the common adapter return contract."""

    if not isinstance(output, dict):
        raise TypeError(
            f"{adapter_class.__name__}.transcribe() "
            "must return a dictionary."
        )

    if "text" not in output:
        raise KeyError(
            f"{adapter_class.__name__}.transcribe() "
            "did not return 'text'."
        )

    if "latency_ms" not in output:
        raise KeyError(
            f"{adapter_class.__name__}.transcribe() "
            "did not return 'latency_ms'."
        )

    if "meta" not in output:
        raise KeyError(
            f"{adapter_class.__name__}.transcribe() "
            "did not return 'meta'."
        )

    text = output["text"]
    metadata = output["meta"]

    if not isinstance(text, str):
        raise TypeError(
            "Adapter output 'text' must be a string."
        )

    if not isinstance(metadata, dict):
        raise TypeError(
            "Adapter output 'meta' must be a dictionary."
        )

    try:
        latency_ms = float(
            output["latency_ms"]
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise TypeError(
            "Adapter output 'latency_ms' must be numeric."
        ) from exc

    if not math.isfinite(
        latency_ms
    ):
        raise ValueError(
            "Adapter latency must be finite."
        )

    if latency_ms < 0:
        raise ValueError(
            "Adapter latency cannot be negative."
        )

    return (
        text.strip(),
        latency_ms,
        metadata,
    )


# ============================================================
# Warm-up
# ============================================================

def run_global_warmup(
    adapter: Any,
    warmup_audio: pathlib.Path,
    n_warmup: int,
    device: str,
) -> None:
    """Perform model warm-up before measured runs."""

    if n_warmup <= 0:
        print(
            "Global warm-up disabled."
        )
        return

    print(
        f"\nRunning {n_warmup} global warm-up "
        f"run(s) using {warmup_audio.name}..."
    )

    for warmup_index in range(
        n_warmup
    ):
        synchronize_cuda(
            device
        )

        adapter.transcribe(
            warmup_audio
        )

        synchronize_cuda(
            device
        )

        print(
            f"  Warm-up "
            f"{warmup_index + 1}/{n_warmup} complete"
        )

    print(
        "Global warm-up complete.\n"
    )


# ============================================================
# Main benchmark
# ============================================================

def run_benchmark(
    adapter_class: Type,
    adapter_config: Optional[
        Dict[str, Any]
    ] = None,
    max_files: Optional[int] = None,
    result_name: Optional[str] = None,
    n_warmup: int = N_WARMUP,
    n_runs: int = N_RUNS,
) -> None:
    """Run one complete GPU benchmark."""

    effective_config = dict(
        adapter_config or {}
    )

    adapter = adapter_class(
        effective_config
    )

    try:
        model_load_start = (
            time.perf_counter()
        )

        adapter.load()

        synchronize_cuda(
            effective_config["device"]
        )

        model_load_ms = (
            time.perf_counter()
            - model_load_start
        ) * 1000.0

        model_name = resolve_model_name(
            adapter,
            result_name
            or adapter_class.__name__.lower(),
        )

        requested_device = normalize_cuda_device(
            effective_config["device"]
        )

        # FIX (device-consistency check): use
        # get_effective_adapter_device_string() rather than reading
        # adapter.device directly, so CTranslate2-based adapters (which
        # store the GPU index separately in self.device_index, not
        # embedded in self.device) are compared correctly instead of
        # silently falling back to torch.cuda.current_device().
        adapter_device = normalize_cuda_device(
            get_effective_adapter_device_string(
                adapter,
                requested_device,
            )
        )

        if (
            adapter_device
            != requested_device
        ):
            raise RuntimeError(
                "Adapter device does not match the "
                "requested CUDA device: "
                f"requested={requested_device}, "
                f"adapter={adapter_device}"
            )

        audio_dir = pathlib.Path(
            AUDIO_DIR
        )

        metadata_path = pathlib.Path(
            METADATA_FILE
        )

        results_root = pathlib.Path(
            RESULTS_DIR
        )

        ensure_dir(
            results_root
        )

        references = load_references(
            metadata_path
        )

        all_audio_files = sorted(
            audio_dir.glob("*.wav")
        )

        if not all_audio_files:
            raise SystemExit(
                f"No WAV files found in: {audio_dir}"
            )

        complete_audio_names = {
            path.name
            for path in all_audio_files
        }

        metadata_names = {
            file_name
            for file_name in references
            if file_name
        }

        metadata_without_audio = sorted(
            metadata_names.difference(
                complete_audio_names
            )
        )

        audio_without_metadata = sorted(
            complete_audio_names.difference(
                metadata_names
            )
        )

        if metadata_without_audio:
            print(
                "\nWARNING: Metadata rows without "
                "matching WAV files:"
            )

            for file_name in (
                metadata_without_audio
            ):
                print(
                    f"  - {file_name}"
                )

        if audio_without_metadata:
            print(
                "\nWARNING: WAV files without "
                "matching metadata rows:"
            )

            for file_name in (
                audio_without_metadata
            ):
                print(
                    f"  - {file_name}"
                )

        selected_audio_files = (
            all_audio_files
        )

        if max_files is not None:
            if max_files <= 0:
                raise SystemExit(
                    "--max-files must be greater than zero."
                )

            selected_audio_files = (
                all_audio_files[:max_files]
            )

        run_id, run_directory = (
            create_run_directory(
                results_root,
                model_name,
            )
        )

        print(
            "\nBenchmark configuration:"
        )
        print(
            f"  Run ID: {run_id}"
        )
        print(
            f"  Adapter: {adapter_class.__name__}"
        )
        print(
            f"  Model result name: {model_name}"
        )
        print(
            f"  Device: {adapter_device}"
        )
        print(
            f"  Audio directory: {audio_dir}"
        )
        print(
            f"  Files selected: "
            f"{len(selected_audio_files)}"
        )
        print(
            f"  Warm-up runs: {n_warmup}"
        )
        print(
            f"  Measured runs per file: {n_runs}"
        )
        print(
            f"  Model loading time: "
            f"{model_load_ms:.2f} ms"
        )
        print(
            f"  Results directory: "
            f"{run_directory}"
        )

        run_global_warmup(
            adapter=adapter,
            warmup_audio=selected_audio_files[0],
            n_warmup=n_warmup,
            device=adapter_device,
        )

        per_run_rows: List[
            Dict[str, Any]
        ] = []

        per_file_rows: List[
            Dict[str, Any]
        ] = []

        files_without_reference: List[
            str
        ] = []

        timing_warning_count = 0
        representative_adapter_meta: Dict[
            str,
            Any,
        ] = {}

        # --------------------------------------------------------
        # Optional adapter diagnostics aggregated across measured
        # runs. These remain generic so adapters that do not expose
        # such metadata are unaffected.
        # --------------------------------------------------------
        fallback_metadata_observed = False
        fallback_detected_run_count = 0
        fallback_detected_files = set()
        accepted_fallback_temperatures = set()

        total_files = len(
            selected_audio_files
        )

        for file_index, audio_path in enumerate(
            selected_audio_files,
            start=1,
        ):
            file_name = audio_path.name

            reference = str(
                references.get(
                    file_name,
                    "",
                )
                or ""
            ).strip()

            print(
                f"[{file_index}/{total_files}] "
                f"Benchmarking {file_name}"
            )

            if not reference:
                files_without_reference.append(
                    file_name
                )

                print(
                    "  WARNING: No reference transcript. "
                    "Accuracy metrics will be omitted."
                )

            adapter_latencies: List[
                float
            ] = []

            runner_wall_latencies: List[
                float
            ] = []

            transcripts: List[
                str
            ] = []

            run_wer_values: List[
                float
            ] = []

            run_cer_values: List[
                float
            ] = []

            run_exact_matches: List[
                bool
            ] = []

            for run_index in range(
                n_runs
            ):
                synchronize_cuda(
                    adapter_device
                )

                wall_start = (
                    time.perf_counter()
                )

                output = adapter.transcribe(
                    audio_path
                )

                synchronize_cuda(
                    adapter_device
                )

                runner_wall_latency_ms = (
                    time.perf_counter()
                    - wall_start
                ) * 1000.0

                if not math.isfinite(
                    runner_wall_latency_ms
                ):
                    raise ValueError(
                        "Runner wall latency is non-finite."
                    )

                if runner_wall_latency_ms < 0:
                    raise ValueError(
                        "Runner wall latency cannot be negative."
                    )

                (
                    transcript,
                    adapter_latency_ms,
                    raw_meta,
                ) = validate_adapter_output(
                    output,
                    adapter_class,
                )

                if not representative_adapter_meta:
                    representative_adapter_meta = dict(
                        raw_meta
                    )

                # ------------------------------------------------
                # Optional fallback diagnostics.
                #
                # New Distil-Whisper metadata uses
                # "fallback_detected_from_emitted_segments".
                # "fallback_used" is also accepted defensively for
                # compatibility with alternative Whisper adapters.
                # ------------------------------------------------
                fallback_flag = raw_meta.get(
                    "fallback_detected_from_emitted_segments",
                    raw_meta.get(
                        "fallback_used",
                        None,
                    ),
                )

                if fallback_flag is not None:
                    fallback_metadata_observed = True

                    if bool(fallback_flag):
                        fallback_detected_run_count += 1
                        fallback_detected_files.add(
                            file_name
                        )

                    temperatures = raw_meta.get(
                        "accepted_segment_temperatures",
                        [],
                    )

                    if isinstance(
                        temperatures,
                        (list, tuple, set),
                    ):
                        for temperature in temperatures:
                            try:
                                temperature_value = float(
                                    temperature
                                )
                            except (
                                TypeError,
                                ValueError,
                            ):
                                continue

                            if math.isfinite(
                                temperature_value
                            ):
                                accepted_fallback_temperatures.add(
                                    temperature_value
                                )

                adapter_latencies.append(
                    adapter_latency_ms
                )

                runner_wall_latencies.append(
                    runner_wall_latency_ms
                )

                transcripts.append(
                    transcript
                )

                timing_difference_ms = (
                    runner_wall_latency_ms
                    - adapter_latency_ms
                )

                timing_difference_percent = (
                    calculate_difference_percent(
                        adapter_latency_ms,
                        runner_wall_latency_ms,
                    )
                )

                timing_warning = (
                    should_warn_about_timing(
                        adapter_latency_ms,
                        runner_wall_latency_ms,
                    )
                )

                if timing_warning:
                    timing_warning_count += 1

                    print(
                        "  WARNING: Adapter latency "
                        "and runner wall latency differ "
                        "significantly."
                    )

                run_wer = (
                    compute_wer(
                        reference,
                        transcript,
                    )
                    if reference
                    else None
                )

                run_cer = (
                    compute_cer(
                        reference,
                        transcript,
                    )
                    if reference
                    else None
                )

                exact_match = (
                    normalize_transcript(
                        reference
                    )
                    == normalize_transcript(
                        transcript
                    )
                    if reference
                    else None
                )

                if run_wer is not None:
                    run_wer_values.append(
                        run_wer
                    )

                if run_cer is not None:
                    run_cer_values.append(
                        run_cer
                    )

                if exact_match is not None:
                    run_exact_matches.append(
                        exact_match
                    )

                per_run_rows.append(
                    {
                        "file": file_name,
                        "run": run_index + 1,
                        "adapter_latency_ms": (
                            adapter_latency_ms
                        ),
                        "runner_wall_latency_ms": (
                            runner_wall_latency_ms
                        ),
                        "timing_difference_ms": (
                            timing_difference_ms
                        ),
                        "timing_difference_percent": (
                            timing_difference_percent
                        ),
                        "timing_warning": timing_warning,
                        "transcript": transcript,
                        "reference": reference,
                        "wer": run_wer,
                        "cer": run_cer,
                        "exact_match": exact_match,
                        "device": adapter_device,
                        "meta": json_dumps_safe(
                            raw_meta
                        ),
                    }
                )

                print(
                    f"  Run {run_index + 1}/{n_runs}: "
                    f"adapter={adapter_latency_ms:.2f} ms, "
                    f"wall={runner_wall_latency_ms:.2f} ms"
                )

            transcript_counts = Counter(
                transcripts
            )

            highest_count = max(
                transcript_counts.values()
            )

            modal_transcripts = sorted(
                transcript
                for transcript, count
                in transcript_counts.items()
                if count == highest_count
            )

            canonical_transcript = (
                modal_transcripts[0]
            )

            transcript_mode_tied = (
                len(modal_transcripts) > 1
            )

            unique_transcript_count = len(
                transcript_counts
            )

            transcript_consistent = (
                unique_transcript_count == 1
            )

            if not transcript_consistent:
                print(
                    "  WARNING: Transcription output "
                    "changed across measured runs."
                )

            file_wer = (
                compute_wer(
                    reference,
                    canonical_transcript,
                )
                if reference
                else None
            )

            file_cer = (
                compute_cer(
                    reference,
                    canonical_transcript,
                )
                if reference
                else None
            )

            file_exact_match = (
                normalize_transcript(
                    reference
                )
                == normalize_transcript(
                    canonical_transcript
                )
                if reference
                else None
            )

            adapter_statistics = latency_stats(
                adapter_latencies,
                include_percentiles=False,
            )

            wall_statistics = latency_stats(
                runner_wall_latencies,
                include_percentiles=False,
            )

            per_file_rows.append(
                {
                    "file": file_name,
                    "n_runs": n_runs,
                    "mean_adapter_latency_ms": (
                        adapter_statistics[
                            "mean_ms"
                        ]
                    ),
                    "median_adapter_latency_ms": (
                        adapter_statistics[
                            "median_ms"
                        ]
                    ),
                    "stdev_adapter_latency_ms": (
                        adapter_statistics[
                            "stdev_ms"
                        ]
                    ),
                    "minimum_adapter_latency_ms": (
                        adapter_statistics[
                            "minimum_ms"
                        ]
                    ),
                    "maximum_adapter_latency_ms": (
                        adapter_statistics[
                            "maximum_ms"
                        ]
                    ),
                    "mean_runner_wall_latency_ms": (
                        wall_statistics[
                            "mean_ms"
                        ]
                    ),
                    "median_runner_wall_latency_ms": (
                        wall_statistics[
                            "median_ms"
                        ]
                    ),
                    "stdev_runner_wall_latency_ms": (
                        wall_statistics[
                            "stdev_ms"
                        ]
                    ),
                    "minimum_runner_wall_latency_ms": (
                        wall_statistics[
                            "minimum_ms"
                        ]
                    ),
                    "maximum_runner_wall_latency_ms": (
                        wall_statistics[
                            "maximum_ms"
                        ]
                    ),
                    "mean_run_wer": (
                        stats.mean(
                            run_wer_values
                        )
                        if run_wer_values
                        else None
                    ),
                    "median_run_wer": (
                        stats.median(
                            run_wer_values
                        )
                        if run_wer_values
                        else None
                    ),
                    "mean_run_cer": (
                        stats.mean(
                            run_cer_values
                        )
                        if run_cer_values
                        else None
                    ),
                    "median_run_cer": (
                        stats.median(
                            run_cer_values
                        )
                        if run_cer_values
                        else None
                    ),
                    "run_exact_match_rate": (
                        sum(run_exact_matches)
                        / len(run_exact_matches)
                        if run_exact_matches
                        else None
                    ),
                    "canonical_transcript": (
                        canonical_transcript
                    ),
                    "unique_transcript_count": (
                        unique_transcript_count
                    ),
                    "transcript_consistent": (
                        transcript_consistent
                    ),
                    "transcript_mode_tied": (
                        transcript_mode_tied
                    ),
                    "reference": reference,
                    "wer": file_wer,
                    "cer": file_cer,
                    "exact_match": (
                        file_exact_match
                    ),
                    "device": adapter_device,
                }
            )

            print()

        runs_df = pd.DataFrame(
            per_run_rows
        )

        per_file_df = pd.DataFrame(
            per_file_rows
        )

        runs_path = (
            run_directory
            / "runs.csv"
        )

        per_file_path = (
            run_directory
            / "per_file.csv"
        )

        summary_path = (
            run_directory
            / "summary.json"
        )

        runs_df.to_csv(
            runs_path,
            index=False,
        )

        per_file_df.to_csv(
            per_file_path,
            index=False,
        )

        valid_reference_rows = [
            row
            for row in per_file_rows
            if row["reference"]
        ]

        corpus_wer = None
        corpus_cer = None
        exact_match_rate = None

        if valid_reference_rows:
            combined_references = " ".join(
                row["reference"]
                for row in valid_reference_rows
            )

            combined_hypotheses = " ".join(
                row["canonical_transcript"]
                for row in valid_reference_rows
            )

            corpus_wer = compute_wer(
                combined_references,
                combined_hypotheses,
            )

            corpus_cer = compute_cer(
                combined_references,
                combined_hypotheses,
            )

            exact_match_rate = (
                sum(
                    bool(row["exact_match"])
                    for row
                    in valid_reference_rows
                )
                / len(valid_reference_rows)
            )

        inconsistent_files = [
            row["file"]
            for row in per_file_rows
            if not row[
                "transcript_consistent"
            ]
        ]

        tied_mode_files = [
            row["file"]
            for row in per_file_rows
            if row[
                "transcript_mode_tied"
            ]
        ]

        all_adapter_latencies = [
            row["adapter_latency_ms"]
            for row in per_run_rows
        ]

        all_wall_latencies = [
            row["runner_wall_latency_ms"]
            for row in per_run_rows
        ]

        adapter_diagnostics: Dict[
            str,
            Any,
        ] = {}

        if fallback_metadata_observed:
            adapter_diagnostics[
                "temperature_fallback"
            ] = {
                "metadata_observed": True,
                "fallback_detected_run_count": (
                    fallback_detected_run_count
                ),
                "fallback_detected_file_count": (
                    len(fallback_detected_files)
                ),
                "fallback_detected_files": sorted(
                    fallback_detected_files
                ),
                "accepted_segment_temperatures_observed": sorted(
                    accepted_fallback_temperatures
                ),
                "interpretation": (
                    "A positive accepted segment temperature "
                    "shows that Whisper fallback advanced beyond "
                    "the initial temperature-zero attempt for that "
                    "emitted segment. Rejected attempts are not "
                    "reconstructed by this runner."
                ),
            }

        summary = {
            "run_id": run_id,
            "model": model_name,
            "adapter_class": (
                adapter_class.__name__
            ),
            "adapter_config": (
                effective_config
            ),
            "adapter_metadata": (
                representative_adapter_meta
            ),
            "adapter_metadata_scope": (
                "First measured run only; complete run-specific "
                "metadata is stored in runs.csv."
            ),
            "adapter_diagnostics": (
                adapter_diagnostics
            ),
            "device": adapter_device,
            "primary_latency_metric": (
                "runner_wall_latency_ms"
            ),
            "diagnostic_latency_metric": (
                "adapter_latency_ms"
            ),
            "model_loading_ms": (
                model_load_ms
            ),
            "n_files": len(
                per_file_rows
            ),
            "n_files_with_reference": len(
                valid_reference_rows
            ),
            "n_files_without_reference": len(
                files_without_reference
            ),
            "files_without_reference": (
                files_without_reference
            ),
            "metadata_entries_without_audio": (
                metadata_without_audio
            ),
            "audio_files_without_metadata": (
                audio_without_metadata
            ),
            "n_warmup_global": n_warmup,
            "n_runs_per_file": n_runs,
            "total_measured_runs": len(
                per_run_rows
            ),
            "timing_warning_count": (
                timing_warning_count
            ),
            "n_files_with_inconsistent_transcripts": (
                len(inconsistent_files)
            ),
            "files_with_inconsistent_transcripts": (
                inconsistent_files
            ),
            "n_files_with_tied_transcript_modes": (
                len(tied_mode_files)
            ),
            "files_with_tied_transcript_modes": (
                tied_mode_files
            ),
            "adapter_latency": latency_stats(
                all_adapter_latencies,
                include_percentiles=True,
            ),
            "runner_wall_latency": latency_stats(
                all_wall_latencies,
                include_percentiles=True,
            ),
            "corpus_wer": corpus_wer,
            "corpus_cer": corpus_cer,
            "exact_match_rate": exact_match_rate,
            "runs_csv": str(
                runs_path
            ),
            "per_file_csv": str(
                per_file_path
            ),
            "command": " ".join(
                sys.argv
            ),
            "git": git_metadata(),
            "environment": env_metadata(),
            "created_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as summary_file:
            summary_file.write(
                json_dumps_safe(
                    summary
                )
            )

        print(
            "=" * 60
        )
        print(
            "Benchmark complete"
        )
        print(
            "=" * 60
        )
        print(
            f"Run ID: {run_id}"
        )
        print(
            f"Model: {model_name}"
        )
        print(
            f"Device: {adapter_device}"
        )
        print(
            f"Files tested: {len(per_file_rows)}"
        )
        print(
            f"Measured runs: {len(per_run_rows)}"
        )
        print(
            f"Timing warnings: "
            f"{timing_warning_count}"
        )
        print(
            f"Inconsistent transcript files: "
            f"{len(inconsistent_files)}"
        )
        print(
            f"Runs CSV: {runs_path}"
        )
        print(
            f"Per-file CSV: {per_file_path}"
        )
        print(
            f"Summary JSON: {summary_path}"
        )

    finally:
        adapter.close()


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GPU-only ASR benchmark for short "
            "banking voice commands."
        )
    )

    parser.add_argument(
        "--model",
        choices=sorted(
            ADAPTER_REGISTRY.keys()
        ),
        required=True,
        help="Adapter to benchmark.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=(
            "CUDA device, for example 'cuda' or "
            "'cuda:0'. CPU execution is not supported."
        ),
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help=(
            "Optional model-size override for adapters that support it. "
            "Fixed experimental adapters (for example Qwen3-ASR, "
            "Distil-Whisper, and Faster-Whisper Small BF16) intentionally "
            "reject incompatible overrides."
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "Optional number of WAV files to benchmark."
        ),
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=N_WARMUP,
        help=(
            f"Global warm-up runs. Default: {N_WARMUP}."
        ),
    )

    parser.add_argument(
        "--measured-runs",
        type=int,
        default=N_RUNS,
        help=(
            f"Measured runs per file. Default: {N_RUNS}."
        ),
    )

    args = parser.parse_args()

    if args.warmup_runs < 0:
        raise SystemExit(
            "--warmup-runs cannot be negative."
        )

    if args.measured_runs <= 0:
        raise SystemExit(
            "--measured-runs must be greater than zero."
        )

    validated_device = validate_device(
        args.device
    )

    adapter_config: Dict[
        str,
        Any,
    ] = {
        "device": validated_device,
    }

    if args.model_size is not None:
        adapter_config["model_size"] = (
            args.model_size
        )

    adapter_class = load_adapter_class(
        args.model
    )

    run_benchmark(
        adapter_class=adapter_class,
        adapter_config=adapter_config,
        max_files=args.max_files,
        result_name=args.model,
        n_warmup=args.warmup_runs,
        n_runs=args.measured_runs,
    )


if __name__ == "__main__":
    main()
