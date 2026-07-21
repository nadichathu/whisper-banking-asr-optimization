import time
import pathlib
import statistics as stats
from typing import Type, Dict

import pandas as pd
import torch

torch.set_num_threads(4)

from bench.config import AUDIO_DIR, RESULTS_DIR, METADATA_FILE, N_WARMUP, N_RUNS
from bench.utils import compute_wer, percentile, ensure_dir, json_dumps_safe, env_metadata


def run_benchmark(adapter_class: Type, adapter_config: Dict = None, max_files: int = None):
    adapter = adapter_class(adapter_config)
    adapter.load()

    model_name = getattr(adapter, "model_id", None) or adapter.__class__.__name__.lower()
    model_name = model_name.replace("/", "_")

    audio_dir = pathlib.Path(AUDIO_DIR)
    results_dir = pathlib.Path(RESULTS_DIR)
    ensure_dir(results_dir)

    # Load metadata references
    refs = {}
    metadata_path = pathlib.Path(METADATA_FILE)
    if metadata_path.exists():
        meta_df = pd.read_csv(metadata_path)
        refs = dict(zip(meta_df["file_name"], meta_df["transcript"]))
    else:
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    audio_files = sorted(audio_dir.glob("*.wav"))
    if max_files:
        audio_files = audio_files[:max_files]

    if not audio_files:
        raise SystemExit(f"No WAV files found in {audio_dir}")

    per_run_rows = []
    per_file_rows = []

    for audio in audio_files:
        file_name = audio.name
        reference = refs.get(file_name, "")

        # warmup
        for _ in range(N_WARMUP):
            adapter.transcribe(str(audio))

        latencies = []
        transcripts = []

        for run in range(N_RUNS):
            t0 = time.perf_counter()
            out = adapter.transcribe(str(audio))
            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000
            text = out.get("text", "").strip()

            latencies.append(latency)
            transcripts.append(text)

            per_run_rows.append({
                "file": file_name,
                "run": run + 1,
                "latency_ms": latency,
                "transcript": text,
                "meta": json_dumps_safe(out.get("meta", {})),
                "reference": reference,
                "wer": compute_wer(reference, text) if reference else None,
            })

        # canonical transcript = transcript from median-latency run
        sorted_pairs = sorted(zip(latencies, transcripts), key=lambda x: x[0])
        canonical_transcript = sorted_pairs[len(sorted_pairs) // 2][1]

        per_file_rows.append({
            "file": file_name,
            "n_runs": len(latencies),
            "mean_latency_ms": stats.mean(latencies),
            "median_latency_ms": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "best_ms": min(latencies),
            "worst_ms": max(latencies),
            "canonical_transcript": canonical_transcript,
            "reference": reference,
            "wer": compute_wer(reference, canonical_transcript) if reference else None,
        })

    runs_df = pd.DataFrame(per_run_rows)
    per_file_df = pd.DataFrame(per_file_rows)

    runs_path = results_dir / f"{model_name}_runs.csv"
    per_file_path = results_dir / f"{model_name}_per_file.csv"

    runs_df.to_csv(runs_path, index=False)
    per_file_df.to_csv(per_file_path, index=False)

    corpus_wer = None
    if per_file_rows:
        refs_all = [row["reference"] for row in per_file_rows if row["reference"]]
        hyps_all = [row["canonical_transcript"] for row in per_file_rows if row["reference"]]
        if refs_all:
            corpus_wer = compute_wer(" ".join(refs_all), " ".join(hyps_all))

    meta = env_metadata()
    summary = {
        "model": model_name,
        "n_files": len(per_file_rows),
        "runs_csv": str(runs_path),
        "per_file_csv": str(per_file_path),
        "env": json_dumps_safe(meta),
        "corpus_wer": corpus_wer,
    }

    summary_path = results_dir / f"{model_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(json_dumps_safe(summary))

    print("Benchmark complete. Results:")
    print(runs_path)
    print(per_file_path)

    adapter.close()


if __name__ == "__main__":
    import argparse

    from bench.adapters.whisper_baseline_adapter import WhisperAdapter
    from bench.adapters.whisper_greedy_adapter import WhisperGreedyAdapter
    from bench.adapters.whisper_no_context_adapter import WhisperNoContextAdapter
    from bench.adapters.whisper_vad_adapter import WhisperVadAdapter
    from bench.adapters.whisper_fp16_adapter import WhisperFP16Adapter
    from bench.adapters.whisper_limited_tokens_adapter import (
        WhisperLimitedTokensAdapter
    )
    from bench.adapters.whisper_no_timestamps_adapter import (
        WhisperNoTimestampsAdapter
    )
    from bench.adapters.whisper_no_fallback_adapter import (
        WhisperNoFallbackAdapter
    )
    from bench.adapters.whisper_direct_decode_adapter import (
        WhisperDirectDecodeAdapter
    )
    from bench.adapters.whisper_int8_adapter import (
        WhisperINT8Adapter,
    )
    from bench.adapters.faster_whisper_adapter import (
        FasterWhisperAdapter,
    )
    from bench.adapters.faster_whisper_small_adapter import (
    FasterWhisperSmallAdapter,
    )

    from bench.adapters.faster_whisper_medium_adapter import (
        FasterWhisperMediumAdapter,
    )
    from bench.adapters.wav2vec2_adapter import Wav2Vec2Adapter
    from bench.adapters.vosk_adapter import VoskAdapter

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=[
            "whisper",
            "whisper_greedy",
            "whisper_no_context",
            "whisper_vad",
            "whisper_fp16",
            "whisper_limited_tokens",
            "whisper_no_timestamps",
            "whisper_no_fallback",
            "whisper_direct_decode",
            "whisper_int8",
            "faster_whisper",
            "faster_whisper_small",
            "faster_whisper_medium",
            "wav2vec2",
            "vosk",
        ],
        required=True
    )
    args = parser.parse_args()

    model_map = {
        "whisper": WhisperAdapter,
        "whisper_greedy": WhisperGreedyAdapter,
        "whisper_no_context": WhisperNoContextAdapter,
        "whisper_vad": WhisperVadAdapter,
        "whisper_fp16": WhisperFP16Adapter,
        "whisper_limited_tokens": WhisperLimitedTokensAdapter,
        "whisper_no_timestamps": WhisperNoTimestampsAdapter,
        "whisper_no_fallback": WhisperNoFallbackAdapter,
        "whisper_direct_decode": WhisperDirectDecodeAdapter,
        "whisper_int8": WhisperINT8Adapter,
        "faster_whisper": FasterWhisperAdapter, 
        "faster_whisper_small": FasterWhisperSmallAdapter,
        "faster_whisper_medium": FasterWhisperMediumAdapter,
        "wav2vec2": Wav2Vec2Adapter,
        "vosk": VoskAdapter,
    }

    run_benchmark(model_map[args.model])