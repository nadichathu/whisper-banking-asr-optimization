import argparse
import importlib
import pathlib
import statistics as stats
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import torch

from bench.config import AUDIO_DIR, METADATA_FILE, N_RUNS, N_WARMUP, RESULTS_DIR
from bench.utils import compute_cer, compute_wer, ensure_dir, env_metadata, json_dumps_safe, percentile


ADAPTER_REGISTRY = {
    "whisper": ("bench.adapters.whisper_baseline_adapter", "WhisperAdapter"),
    "whisper_greedy": ("bench.adapters.whisper_greedy_adapter", "WhisperGreedyAdapter"),
    "whisper_no_context": ("bench.adapters.whisper_no_context_adapter", "WhisperNoContextAdapter"),
    "whisper_vad": ("bench.adapters.whisper_vad_adapter", "WhisperVadAdapter"),
    "whisper_fp16": ("bench.adapters.whisper_fp16_adapter", "WhisperFP16Adapter"),
    "whisper_limited_tokens": ("bench.adapters.whisper_limited_tokens_adapter", "WhisperLimitedTokensAdapter"),
    "whisper_no_timestamps": ("bench.adapters.whisper_no_timestamps_adapter", "WhisperNoTimestampsAdapter"),
    "whisper_no_fallback": ("bench.adapters.whisper_no_fallback_adapter", "WhisperNoFallbackAdapter"),
    "whisper_direct_decode": ("bench.adapters.whisper_direct_decode_adapter", "WhisperDirectDecodeAdapter"),
    "whisper_int8": ("bench.adapters.whisper_int8_adapter", "WhisperINT8Adapter"),
    "faster_whisper": ("bench.adapters.faster_whisper_adapter", "FasterWhisperAdapter"),
    "faster_whisper_small": ("bench.adapters.faster_whisper_small_adapter", "FasterWhisperSmallAdapter"),
    "faster_whisper_medium": ("bench.adapters.faster_whisper_medium_adapter", "FasterWhisperMediumAdapter"),
    "wav2vec2": ("bench.adapters.wav2vec2_adapter", "Wav2Vec2Adapter"),
    "parakeet": ("bench.adapters.parakeet_adapter", "ParakeetAdapter"),
    "nemo_fastconformer": ("bench.adapters.nemo_fastconformer_adapter", "NeMoFastConformerAdapter"),
    # "vosk" removed from the comparator set: Vosk is CPU-only (Kaldi-based)
    # and this benchmark suite is GPU-only throughout.
}


def load_adapter_class(model_key: str) -> Type:
    """Dynamically import only the selected adapter."""
    if model_key not in ADAPTER_REGISTRY:
        available_models = ", ".join(sorted(ADAPTER_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{model_key}'. Available models: {available_models}")

    module_name, class_name = ADAPTER_REGISTRY[model_key]

    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise ImportError(
            f"Could not import module '{module_name}' for model '{model_key}'. Original error: {exc}"
        ) from exc

    try:
        adapter_class = getattr(module, class_name)
    except AttributeError as exc:
        available_adapter_classes = [name for name in dir(module) if name.endswith("Adapter")]
        raise ImportError(
            f"Module '{module_name}' does not contain the required class '{class_name}'. "
            f"Adapter classes found: {available_adapter_classes or 'none'}"
        ) from exc

    return adapter_class


def resolve_model_name(adapter: Any, fallback_name: str) -> str:
    """Return a stable, filename-safe model name."""
    model_name = getattr(adapter, "name", None) or getattr(adapter, "model_id", None) or fallback_name
    return str(model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")


def validate_device(requested_device: Optional[str]) -> str:
    """Validate the selected device. This benchmark suite is GPU-only:
    defaults to 'cuda' if not specified, and fails fast (rather than
    falling back to CPU) if CUDA is unavailable or a non-CUDA device
    string is passed. Also validates a specific CUDA index if given
    (e.g. 'cuda:1'), since torch.cuda.is_available() alone does not
    confirm that a particular index actually exists."""

    device_str = (requested_device or "cuda").lower().strip()

    if not device_str.startswith("cuda"):
        raise SystemExit(
            f"This benchmark suite is GPU-only. Received device='{device_str}'. "
            "Use '--device cuda' or '--device cuda:0'."
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            f"Device '{device_str}' was requested, but CUDA is not available."
        )

    device = torch.device(device_str)
    if device.index is not None:
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise SystemExit(
                f"Invalid CUDA device index: {device.index}. "
                f"Detected {torch.cuda.device_count()} CUDA device(s)."
            )

    return device_str


def calculate_difference_percent(adapter_latency_ms: float, runner_wall_latency_ms: float) -> float:
    """Calculate the timing difference between adapter-reported and wall-clock latency, as a percentage."""
    if adapter_latency_ms <= 0:
        return 0.0
    difference_ms = runner_wall_latency_ms - adapter_latency_ms
    return (difference_ms / adapter_latency_ms) * 100.0


def should_warn_about_timing(
    adapter_latency_ms: float,
    runner_wall_latency_ms: float,
    minimum_difference_ms: float = 20.0,
    minimum_difference_percent: float = 10.0,
) -> bool:
    """Flag runs where adapter-reported and wall-clock timing diverge unusually,
    which typically indicates the adapter's internal timer starts at a different
    pipeline stage than the baseline (e.g. after audio loading instead of before it)."""
    difference_ms = abs(runner_wall_latency_ms - adapter_latency_ms)
    difference_percent = abs(calculate_difference_percent(adapter_latency_ms, runner_wall_latency_ms))
    return difference_ms >= minimum_difference_ms and difference_percent >= minimum_difference_percent


def load_references(metadata_path: pathlib.Path) -> Dict[str, str]:
    """Load filename-to-transcript mappings from the metadata CSV."""
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    metadata_df = pd.read_csv(metadata_path)

    required_columns = {"file_name", "transcript"}
    missing_columns = required_columns.difference(metadata_df.columns)
    if missing_columns:
        raise SystemExit("Metadata file is missing required columns: " + ", ".join(sorted(missing_columns)))

    metadata_df["file_name"] = metadata_df["file_name"].fillna("").astype(str).str.strip()
    metadata_df["transcript"] = metadata_df["transcript"].fillna("").astype(str).str.strip()

    duplicate_names = metadata_df[metadata_df["file_name"].duplicated(keep=False)]["file_name"].tolist()
    if duplicate_names:
        print("WARNING: Duplicate filenames were found in the metadata CSV:")
        for duplicate_name in sorted(set(duplicate_names)):
            print(f"  - {duplicate_name}")
        print("The last occurrence of each duplicated filename will be used.")

    return dict(zip(metadata_df["file_name"], metadata_df["transcript"]))


def run_global_warmup(adapter: Any, warmup_audio: pathlib.Path, n_warmup: int) -> None:
    """Run warm-up inference once before measured benchmarking."""
    if n_warmup <= 0:
        print("Global warm-up disabled.")
        return

    print(f"\nRunning {n_warmup} global warm-up run(s) using {warmup_audio.name}...")
    for warmup_index in range(n_warmup):
        adapter.transcribe(warmup_audio)
        print(f"  Warm-up {warmup_index + 1}/{n_warmup} complete")
    print("Global warm-up complete.\n")


def _latency_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Compute the standard set of descriptive latency statistics for a list of
    measurements. Shared by both the adapter-reported and runner-wall-clock
    latency summaries so the two block are guaranteed to stay consistent."""
    if not values:
        return {
            "mean_ms": None, "median_ms": None, "stdev_ms": None,
            "p95_ms": None, "p99_ms": None, "best_ms": None, "worst_ms": None,
        }
    return {
        "mean_ms": stats.mean(values),
        "median_ms": percentile(values, 50),
        "stdev_ms": stats.stdev(values) if len(values) > 1 else 0.0,
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "best_ms": min(values),
        "worst_ms": max(values),
    }


def run_benchmark(
    adapter_class: Type,
    adapter_config: Optional[Dict[str, Any]] = None,
    max_files: Optional[int] = None,
    result_name: Optional[str] = None,
    n_warmup: int = N_WARMUP,
    n_runs: int = N_RUNS,
) -> None:
    """Run an adapter benchmark and save detailed per-run, per-file, and summary results."""

    effective_config = dict(adapter_config or {})
    adapter = adapter_class(effective_config)

    try:
        adapter.load()

        model_name = resolve_model_name(adapter, result_name or adapter_class.__name__.lower())
        adapter_device = str(getattr(adapter, "device", effective_config.get("device", "cuda")))

        audio_dir = pathlib.Path(AUDIO_DIR)
        results_dir = pathlib.Path(RESULTS_DIR)
        metadata_path = pathlib.Path(METADATA_FILE)
        ensure_dir(results_dir)

        references = load_references(metadata_path)
        audio_files = sorted(audio_dir.glob("*.wav"))

        if max_files is not None:
            if max_files <= 0:
                raise SystemExit("--max-files must be greater than zero.")
            audio_files = audio_files[:max_files]

        if not audio_files:
            raise SystemExit(f"No WAV files found in: {audio_dir}")

        audio_filenames = {audio_path.name for audio_path in audio_files}
        metadata_filenames = {file_name for file_name in references if file_name}
        metadata_without_audio = sorted(metadata_filenames.difference(audio_filenames))

        if metadata_without_audio:
            print("\nWARNING: Metadata entries without a matching WAV file were found:")
            for file_name in metadata_without_audio:
                print(f"  - {file_name}")
            print()

        print("Benchmark configuration:")
        print(f"  Adapter: {adapter_class.__name__}")
        print(f"  Result name: {model_name}")
        print(f"  Device: {adapter_device}")
        print(f"  Audio directory: {audio_dir}")
        print(f"  Number of files: {len(audio_files)}")
        print(f"  Warm-up runs: {n_warmup}")
        print(f"  Measured runs per file: {n_runs}")

        run_global_warmup(adapter=adapter, warmup_audio=audio_files[0], n_warmup=n_warmup)

        per_run_rows: List[Dict[str, Any]] = []
        per_file_rows: List[Dict[str, Any]] = []
        files_without_reference: List[str] = []
        timing_warning_count = 0

        total_files = len(audio_files)

        for file_index, audio_path in enumerate(audio_files, start=1):
            file_name = audio_path.name
            reference = str(references.get(file_name, "") or "").strip()

            print(f"[{file_index}/{total_files}] Benchmarking {file_name}")
            if not reference:
                files_without_reference.append(file_name)
                print("  WARNING: No reference transcript was found. WER will not be calculated for this file.")

            adapter_latencies = []
            runner_wall_latencies = []
            transcripts = []

            for run_index in range(n_runs):
                torch.cuda.synchronize(device=adapter_device)
                wall_start = time.perf_counter()

                output = adapter.transcribe(audio_path)

                torch.cuda.synchronize(device=adapter_device)
                runner_wall_latency_ms = (time.perf_counter() - wall_start) * 1000.0

                if not isinstance(output, dict):
                    raise TypeError(f"{adapter_class.__name__}.transcribe() must return a dictionary.")
                if "latency_ms" not in output:
                    raise KeyError(f"{adapter_class.__name__}.transcribe() did not return 'latency_ms'.")

                adapter_latency_ms = float(output["latency_ms"])
                transcript = str(output.get("text", "") or "").strip()

                adapter_latencies.append(adapter_latency_ms)
                runner_wall_latencies.append(runner_wall_latency_ms)
                transcripts.append(transcript)

                timing_warning = should_warn_about_timing(adapter_latency_ms, runner_wall_latency_ms)
                if timing_warning:
                    timing_warning_count += 1
                    print("  WARNING: Adapter latency and runner wall-clock latency differ significantly.")

                run_wer = compute_wer(reference, transcript) if reference else None
                run_cer = compute_cer(reference, transcript) if reference else None

                per_run_rows.append({
                    "file": file_name,
                    "run": run_index + 1,
                    "adapter_latency_ms": adapter_latency_ms,
                    "runner_wall_latency_ms": runner_wall_latency_ms,
                    "timing_difference_ms": runner_wall_latency_ms - adapter_latency_ms,
                    "timing_difference_percent": calculate_difference_percent(adapter_latency_ms, runner_wall_latency_ms),
                    "timing_warning": timing_warning,
                    "transcript": transcript,
                    "reference": reference,
                    "wer": run_wer,
                    "cer": run_cer,
                    "device": adapter_device,
                    "meta": json_dumps_safe(output.get("meta", {})),
                })

                print(f"  Run {run_index + 1}/{n_runs}: adapter={adapter_latency_ms:.2f} ms, wall={runner_wall_latency_ms:.2f} ms")

            # Canonical transcript is the MODAL (most common) transcript
            # across measured runs, not the transcript from whichever run
            # happened to have median latency -- accuracy should not
            # depend on runtime performance. For a fully deterministic
            # adapter, all runs should produce an identical transcript;
            # transcript_consistent records whether that held here.
            transcript_consistent = len(set(transcripts)) == 1
            if not transcript_consistent:
                print(f"  WARNING: Non-deterministic transcripts across runs for {file_name}")

            canonical_transcript = Counter(transcripts).most_common(1)[0][0]
            unique_transcript_count = len(set(transcripts))

            file_wer = compute_wer(reference, canonical_transcript) if reference else None
            file_cer = compute_cer(reference, canonical_transcript) if reference else None

            adapter_stats = _latency_stats(adapter_latencies)
            wall_stats = _latency_stats(runner_wall_latencies)

            per_file_rows.append({
                "file": file_name,
                "n_runs": len(adapter_latencies),
                "mean_adapter_latency_ms": adapter_stats["mean_ms"],
                "median_adapter_latency_ms": adapter_stats["median_ms"],
                "stdev_adapter_latency_ms": adapter_stats["stdev_ms"],
                "p95_adapter_latency_ms": adapter_stats["p95_ms"],
                "p99_adapter_latency_ms": adapter_stats["p99_ms"],
                "best_adapter_latency_ms": adapter_stats["best_ms"],
                "worst_adapter_latency_ms": adapter_stats["worst_ms"],
                "mean_runner_wall_latency_ms": wall_stats["mean_ms"],
                "median_runner_wall_latency_ms": wall_stats["median_ms"],
                "stdev_runner_wall_latency_ms": wall_stats["stdev_ms"],
                "p95_runner_wall_latency_ms": wall_stats["p95_ms"],
                "p99_runner_wall_latency_ms": wall_stats["p99_ms"],
                "canonical_transcript": canonical_transcript,
                "unique_transcript_count": unique_transcript_count,
                "transcript_consistent": transcript_consistent,
                "reference": reference,
                "wer": file_wer,
                "cer": file_cer,
                "device": adapter_device,
            })

            print()

        runs_df = pd.DataFrame(per_run_rows)
        per_file_df = pd.DataFrame(per_file_rows)

        runs_path = results_dir / f"{model_name}_runs.csv"
        per_file_path = results_dir / f"{model_name}_per_file.csv"
        summary_path = results_dir / f"{model_name}_summary.json"

        runs_df.to_csv(runs_path, index=False)
        per_file_df.to_csv(per_file_path, index=False)

        valid_reference_rows = [row for row in per_file_rows if row["reference"]]
        corpus_wer = None
        corpus_cer = None
        if valid_reference_rows:
            combined_references = " ".join(row["reference"] for row in valid_reference_rows)
            combined_hypotheses = " ".join(row["canonical_transcript"] for row in valid_reference_rows)
            corpus_wer = compute_wer(combined_references, combined_hypotheses)
            corpus_cer = compute_cer(combined_references, combined_hypotheses)

        inconsistent_transcript_files = [
            row["file"] for row in per_file_rows if not row["transcript_consistent"]
        ]

        all_adapter_latencies = [row["adapter_latency_ms"] for row in per_run_rows]
        all_runner_wall_latencies = [row["runner_wall_latency_ms"] for row in per_run_rows]

        summary = {
            "model": model_name,
            "adapter_class": adapter_class.__name__,
            "adapter_config": effective_config,
            "device": adapter_device,
            "n_files": len(per_file_rows),
            "n_files_with_reference": len(valid_reference_rows),
            "n_files_without_reference": len(files_without_reference),
            "files_without_reference": files_without_reference,
            "metadata_entries_without_audio": metadata_without_audio,
            "n_warmup_global": n_warmup,
            "n_runs_per_file": n_runs,
            "total_measured_runs": len(per_run_rows),
            "timing_warning_count": timing_warning_count,
            "n_files_with_inconsistent_transcripts": len(inconsistent_transcript_files),
            "files_with_inconsistent_transcripts": inconsistent_transcript_files,
            "adapter_latency": _latency_stats(all_adapter_latencies),
            "runner_wall_latency": _latency_stats(all_runner_wall_latencies),
            "corpus_wer": corpus_wer,
            "corpus_cer": corpus_cer,
            "runs_csv": str(runs_path),
            "per_file_csv": str(per_file_path),
            "environment": env_metadata(),
        }

        with open(summary_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(json_dumps_safe(summary))

        print("=" * 60)
        print("Benchmark complete")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Device: {adapter_device}")
        print(f"Files tested: {len(per_file_rows)}")
        print(f"Files with references: {len(valid_reference_rows)}")
        print(f"Files without references: {len(files_without_reference)}")
        print(f"Timing warnings: {timing_warning_count}")
        print(f"Runs CSV: {runs_path}")
        print(f"Per-file CSV: {per_file_path}")
        print(f"Summary JSON: {summary_path}")

    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark speech-recognition adapters for the banking voice-command dataset. GPU-only."
    )

    parser.add_argument(
        "--model",
        choices=sorted(ADAPTER_REGISTRY.keys()),
        required=True,
        help="Adapter to benchmark.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device, e.g. 'cuda' or 'cuda:0'. Defaults to 'cuda'. "
             "This benchmark suite is GPU-only; non-CUDA values are rejected.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        help="Optional model size passed to the adapter, e.g. 'tiny', 'base', 'small', 'medium', or 'large'.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of WAV files to benchmark.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=N_WARMUP,
        help=f"Global warm-up runs before measurement. Default: {N_WARMUP} (from config).",
    )
    parser.add_argument(
        "--measured-runs",
        type=int,
        default=N_RUNS,
        help=f"Measured runs per file. Default: {N_RUNS} (from config). "
             "Note: per-file p95/p99 are unreliable with very few runs; "
             "increase this for final results if you want stable percentile estimates.",
    )

    args = parser.parse_args()

    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs cannot be negative.")
    if args.measured_runs <= 0:
        raise SystemExit("--measured-runs must be greater than zero.")

    validated_device = validate_device(args.device)

    adapter_config: Dict[str, Any] = {"device": validated_device}
    if args.model_size is not None:
        adapter_config["model_size"] = args.model_size

    adapter_class = load_adapter_class(args.model)

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
