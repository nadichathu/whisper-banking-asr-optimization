import argparse
import importlib
import pathlib
import statistics as stats
import time
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
    compute_wer,
    ensure_dir,
    env_metadata,
    json_dumps_safe,
    percentile,
)


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
    "vosk": (
        "bench.adapters.vosk_adapter",
        "VoskAdapter",
    ),
}


def load_adapter_class(model_key: str) -> Type:
    """Dynamically import only the selected adapter."""

    if model_key not in ADAPTER_REGISTRY:
        available_models = ", ".join(
            sorted(ADAPTER_REGISTRY.keys())
        )

        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Available models: {available_models}"
        )

    module_name, class_name = ADAPTER_REGISTRY[model_key]

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"Could not import module '{module_name}' "
            f"for model '{model_key}'. "
            f"Original error: {exc}"
        ) from exc

    try:
        adapter_class = getattr(module, class_name)
    except AttributeError as exc:
        available_adapter_classes = [
            name
            for name in dir(module)
            if name.endswith("Adapter")
        ]

        raise ImportError(
            f"Module '{module_name}' does not contain "
            f"the required class '{class_name}'. "
            f"Adapter classes found: "
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
    )


def synchronize_if_cuda(device: Optional[str]) -> None:
    """Synchronize CUDA when the selected device is CUDA."""

    if (
        device
        and str(device).lower().startswith("cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()


def validate_device(requested_device: Optional[str]) -> Optional[str]:
    """Validate the user-selected device."""

    if requested_device is None:
        return None

    requested_device = requested_device.lower().strip()

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit(
                f"Device '{requested_device}' was requested, "
                "but CUDA is not available."
            )

    return requested_device


def calculate_difference_percent(
    adapter_latency_ms: float,
    runner_wall_latency_ms: float,
) -> float:
    """Calculate timing difference as a percentage."""

    if adapter_latency_ms <= 0:
        return 0.0

    difference_ms = (
        runner_wall_latency_ms - adapter_latency_ms
    )

    return (
        difference_ms / adapter_latency_ms
    ) * 100.0


def should_warn_about_timing(
    adapter_latency_ms: float,
    runner_wall_latency_ms: float,
    minimum_difference_ms: float = 20.0,
    minimum_difference_percent: float = 10.0,
) -> bool:
    """Determine whether adapter and runner timing differ unusually."""

    difference_ms = abs(
        runner_wall_latency_ms - adapter_latency_ms
    )

    difference_percent = abs(
        calculate_difference_percent(
            adapter_latency_ms,
            runner_wall_latency_ms,
        )
    )

    return (
        difference_ms >= minimum_difference_ms
        and difference_percent >= minimum_difference_percent
    )


def load_references(
    metadata_path: pathlib.Path,
) -> Dict[str, str]:
    """Load filename-to-transcript mappings from the metadata CSV."""

    if not metadata_path.exists():
        raise SystemExit(
            f"Metadata file not found: {metadata_path}"
        )

    metadata_df = pd.read_csv(metadata_path)

    required_columns = {
        "file_name",
        "transcript",
    }

    missing_columns = required_columns.difference(
        metadata_df.columns
    )

    if missing_columns:
        raise SystemExit(
            "Metadata file is missing required columns: "
            + ", ".join(sorted(missing_columns))
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

    duplicate_names = metadata_df[
        metadata_df["file_name"].duplicated(
            keep=False
        )
    ]["file_name"].tolist()

    if duplicate_names:
        unique_duplicates = sorted(
            set(duplicate_names)
        )

        print(
            "WARNING: Duplicate filenames were found "
            "in the metadata CSV:"
        )

        for duplicate_name in unique_duplicates:
            print(f"  - {duplicate_name}")

        print(
            "The last occurrence of each duplicated "
            "filename will be used."
        )

    return dict(
        zip(
            metadata_df["file_name"],
            metadata_df["transcript"],
        )
    )


def run_global_warmup(
    adapter: Any,
    warmup_audio: pathlib.Path,
    n_warmup: int,
) -> None:
    """Run warm-up inference once before measured benchmarking."""

    if n_warmup <= 0:
        print("Global warm-up disabled.")
        return

    print()
    print(
        f"Running {n_warmup} global warm-up "
        f"run(s) using {warmup_audio.name}..."
    )

    for warmup_index in range(n_warmup):
        adapter.transcribe(warmup_audio)

        print(
            f"  Warm-up "
            f"{warmup_index + 1}/{n_warmup} complete"
        )

    print("Global warm-up complete.")
    print()


def run_benchmark(
    adapter_class: Type,
    adapter_config: Optional[Dict[str, Any]] = None,
    max_files: Optional[int] = None,
    result_name: Optional[str] = None,
) -> None:
    """Run an adapter benchmark and save detailed results."""

    effective_config = dict(adapter_config or {})
    adapter = adapter_class(effective_config)

    try:
        adapter.load()

        model_name = resolve_model_name(
            adapter,
            result_name or adapter_class.__name__.lower(),
        )

        adapter_device = str(
            getattr(
                adapter,
                "device",
                effective_config.get(
                    "device",
                    "unspecified",
                ),
            )
        )

        audio_dir = pathlib.Path(AUDIO_DIR)
        results_dir = pathlib.Path(RESULTS_DIR)
        metadata_path = pathlib.Path(METADATA_FILE)

        ensure_dir(results_dir)

        references = load_references(metadata_path)

        audio_files = sorted(
            audio_dir.glob("*.wav")
        )

        if max_files is not None:
            if max_files <= 0:
                raise SystemExit(
                    "--max-files must be greater than zero."
                )

            audio_files = audio_files[:max_files]

        if not audio_files:
            raise SystemExit(
                f"No WAV files found in: {audio_dir}"
            )

        audio_filenames = {
            audio_path.name
            for audio_path in audio_files
        }

        metadata_filenames = {
            file_name
            for file_name in references
            if file_name
        }

        metadata_without_audio = sorted(
            metadata_filenames.difference(
                audio_filenames
            )
        )

        if metadata_without_audio:
            print()
            print(
                "WARNING: Metadata entries without a matching "
                "WAV file were found:"
            )

            for file_name in metadata_without_audio:
                print(f"  - {file_name}")

            print()

        print("Benchmark configuration:")
        print(f"  Adapter: {adapter_class.__name__}")
        print(f"  Result name: {model_name}")
        print(f"  Device: {adapter_device}")
        print(f"  Audio directory: {audio_dir}")
        print(f"  Number of files: {len(audio_files)}")
        print(f"  Warm-up runs: {N_WARMUP}")
        print(f"  Measured runs per file: {N_RUNS}")

        run_global_warmup(
            adapter=adapter,
            warmup_audio=audio_files[0],
            n_warmup=N_WARMUP,
        )

        per_run_rows: List[Dict[str, Any]] = []
        per_file_rows: List[Dict[str, Any]] = []
        files_without_reference: List[str] = []
        timing_warning_count = 0

        total_files = len(audio_files)

        for file_index, audio_path in enumerate(
            audio_files,
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
                files_without_reference.append(file_name)

                print(
                    "  WARNING: No reference transcript "
                    "was found. WER will not be calculated "
                    "for this file."
                )

            adapter_latencies = []
            runner_wall_latencies = []
            transcripts = []

            for run_index in range(N_RUNS):
                synchronize_if_cuda(adapter_device)

                wall_start = time.perf_counter()

                output = adapter.transcribe(
                    audio_path
                )

                synchronize_if_cuda(adapter_device)

                runner_wall_latency_ms = (
                    time.perf_counter() - wall_start
                ) * 1000.0

                if not isinstance(output, dict):
                    raise TypeError(
                        f"{adapter_class.__name__}."
                        "transcribe() must return a dictionary."
                    )

                if "latency_ms" not in output:
                    raise KeyError(
                        f"{adapter_class.__name__}."
                        "transcribe() did not return "
                        "'latency_ms'."
                    )

                adapter_latency_ms = float(
                    output["latency_ms"]
                )

                transcript = str(
                    output.get(
                        "text",
                        "",
                    )
                    or ""
                ).strip()

                adapter_latencies.append(
                    adapter_latency_ms
                )

                runner_wall_latencies.append(
                    runner_wall_latency_ms
                )

                transcripts.append(transcript)

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
                        "  WARNING: Adapter latency and "
                        "runner wall-clock latency differ "
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
                        "device": adapter_device,
                        "meta": json_dumps_safe(
                            output.get(
                                "meta",
                                {},
                            )
                        ),
                    }
                )

                print(
                    f"  Run {run_index + 1}/{N_RUNS}: "
                    f"adapter={adapter_latency_ms:.2f} ms, "
                    f"wall={runner_wall_latency_ms:.2f} ms"
                )

            median_run_index = sorted(
                range(len(adapter_latencies)),
                key=lambda index: (
                    adapter_latencies[index]
                ),
            )[len(adapter_latencies) // 2]

            canonical_transcript = transcripts[
                median_run_index
            ]

            file_wer = (
                compute_wer(
                    reference,
                    canonical_transcript,
                )
                if reference
                else None
            )

            per_file_rows.append(
                {
                    "file": file_name,
                    "n_runs": len(adapter_latencies),
                    "mean_adapter_latency_ms": stats.mean(
                        adapter_latencies
                    ),
                    "median_adapter_latency_ms": percentile(
                        adapter_latencies,
                        50,
                    ),
                    "p95_adapter_latency_ms": percentile(
                        adapter_latencies,
                        95,
                    ),
                    "p99_adapter_latency_ms": percentile(
                        adapter_latencies,
                        99,
                    ),
                    "best_adapter_latency_ms": min(
                        adapter_latencies
                    ),
                    "worst_adapter_latency_ms": max(
                        adapter_latencies
                    ),
                    "mean_runner_wall_latency_ms": stats.mean(
                        runner_wall_latencies
                    ),
                    "median_runner_wall_latency_ms": percentile(
                        runner_wall_latencies,
                        50,
                    ),
                    "p95_runner_wall_latency_ms": percentile(
                        runner_wall_latencies,
                        95,
                    ),
                    "p99_runner_wall_latency_ms": percentile(
                        runner_wall_latencies,
                        99,
                    ),
                    "canonical_transcript": (
                        canonical_transcript
                    ),
                    "reference": reference,
                    "wer": file_wer,
                    "device": adapter_device,
                }
            )

            print()

        runs_df = pd.DataFrame(per_run_rows)
        per_file_df = pd.DataFrame(per_file_rows)

        runs_path = (
            results_dir
            / f"{model_name}_runs.csv"
        )

        per_file_path = (
            results_dir
            / f"{model_name}_per_file.csv"
        )

        summary_path = (
            results_dir
            / f"{model_name}_summary.json"
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

        all_adapter_latencies = [
            row["adapter_latency_ms"]
            for row in per_run_rows
        ]

        all_runner_wall_latencies = [
            row["runner_wall_latency_ms"]
            for row in per_run_rows
        ]

        environment = env_metadata()

        summary = {
            "model": model_name,
            "adapter_class": adapter_class.__name__,
            "adapter_config": effective_config,
            "device": adapter_device,
            "n_files": len(per_file_rows),
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
            "n_warmup_global": N_WARMUP,
            "n_runs_per_file": N_RUNS,
            "total_measured_runs": len(
                per_run_rows
            ),
            "timing_warning_count": (
                timing_warning_count
            ),
            "adapter_latency": {
                "mean_ms": (
                    stats.mean(
                        all_adapter_latencies
                    )
                    if all_adapter_latencies
                    else None
                ),
                "median_ms": (
                    percentile(
                        all_adapter_latencies,
                        50,
                    )
                    if all_adapter_latencies
                    else None
                ),
                "p95_ms": (
                    percentile(
                        all_adapter_latencies,
                        95,
                    )
                    if all_adapter_latencies
                    else None
                ),
                "p99_ms": (
                    percentile(
                        all_adapter_latencies,
                        99,
                    )
                    if all_adapter_latencies
                    else None
                ),
                "best_ms": (
                    min(all_adapter_latencies)
                    if all_adapter_latencies
                    else None
                ),
                "worst_ms": (
                    max(all_adapter_latencies)
                    if all_adapter_latencies
                    else None
                ),
            },
            "runner_wall_latency": {
                "mean_ms": (
                    stats.mean(
                        all_runner_wall_latencies
                    )
                    if all_runner_wall_latencies
                    else None
                ),
                "median_ms": (
                    percentile(
                        all_runner_wall_latencies,
                        50,
                    )
                    if all_runner_wall_latencies
                    else None
                ),
                "p95_ms": (
                    percentile(
                        all_runner_wall_latencies,
                        95,
                    )
                    if all_runner_wall_latencies
                    else None
                ),
                "p99_ms": (
                    percentile(
                        all_runner_wall_latencies,
                        99,
                    )
                    if all_runner_wall_latencies
                    else None
                ),
                "best_ms": (
                    min(all_runner_wall_latencies)
                    if all_runner_wall_latencies
                    else None
                ),
                "worst_ms": (
                    max(all_runner_wall_latencies)
                    if all_runner_wall_latencies
                    else None
                ),
            },
            "corpus_wer": corpus_wer,
            "runs_csv": str(runs_path),
            "per_file_csv": str(
                per_file_path
            ),
            "environment": environment,
        }

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as summary_file:
            summary_file.write(
                json_dumps_safe(summary)
            )

        print("=" * 60)
        print("Benchmark complete")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Device: {adapter_device}")
        print(f"Files tested: {len(per_file_rows)}")
        print(
            "Files with references: "
            f"{len(valid_reference_rows)}"
        )
        print(
            "Files without references: "
            f"{len(files_without_reference)}"
        )
        print(
            "Timing warnings: "
            f"{timing_warning_count}"
        )
        print(f"Runs CSV: {runs_path}")
        print(
            f"Per-file CSV: {per_file_path}"
        )
        print(f"Summary JSON: {summary_path}")

    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark speech-recognition adapters "
            "for the banking voice-command dataset."
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
        default=None,
        help=(
            "Device passed to the adapter, for example "
            "'cpu', 'cuda', or 'cuda:0'. "
            "When omitted, the adapter or bench.config "
            "selects the device."
        ),
    )

    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help=(
            "Optional model size passed to the adapter, "
            "for example 'tiny', 'base', 'small', "
            "'medium', or 'large'."
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "Optional maximum number of WAV files "
            "to benchmark."
        ),
    )

    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help=(
            "Number of PyTorch CPU threads. "
            "Default: 4."
        ),
    )

    args = parser.parse_args()

    if args.cpu_threads <= 0:
        raise SystemExit(
            "--cpu-threads must be greater than zero."
        )

    torch.set_num_threads(
        args.cpu_threads
    )

    requested_device = validate_device(
        args.device
    )

    adapter_config: Dict[str, Any] = {}

    if requested_device is not None:
        adapter_config["device"] = (
            requested_device
        )

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
    )


if __name__ == "__main__":
    main()