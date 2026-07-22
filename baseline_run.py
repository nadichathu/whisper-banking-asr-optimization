import pathlib
import statistics as stats
import time

import pandas as pd
import torch
import whisper


# =========================
# CONFIGURATION
# =========================

AUDIO_DIR = pathlib.Path("audio_samples")

RUN_RESULTS_PATH = pathlib.Path(
    "results/baseline_runs.csv"
)

SUMMARY_RESULTS_PATH = pathlib.Path(
    "results/baseline_results.csv"
)

MODEL_SIZE = "small"

REQUESTED_DEVICE = "cpu"

N_WARMUP = 2
N_RUNS = 5

LANGUAGE = "en"

# =========================


def resolve_device(requested_device: str) -> str:
    requested_device = requested_device.lower()

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            print(
                "CUDA was requested but is unavailable. "
                "Falling back to CPU."
            )
            return "cpu"

        return requested_device

    return "cpu"


def synchronize_cuda(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    device = resolve_device(REQUESTED_DEVICE)
    use_fp16 = device.startswith("cuda")

    print(
        f"Loading Whisper {MODEL_SIZE} "
        f"on {device} using "
        f"{'FP16' if use_fp16 else 'FP32'}..."
    )

    model = whisper.load_model(
        MODEL_SIZE,
        device=device,
    )

    model.eval()

    audio_files = sorted(
        AUDIO_DIR.glob("*.wav")
    )

    if not audio_files:
        raise SystemExit(
            f"No WAV files found in: {AUDIO_DIR.resolve()}"
        )

    run_rows = []
    summary_rows = []

    for audio_path in audio_files:
        print(f"\nProcessing: {audio_path.name}")

        # Load audio once only to obtain its duration.
        audio_data = whisper.load_audio(
            str(audio_path)
        )

        audio_duration_s = (
            len(audio_data) / whisper.audio.SAMPLE_RATE
        )

        print(
            f"Audio duration: {audio_duration_s:.3f} seconds"
        )

        print(
            f"Running {N_WARMUP} warm-up transcription(s)..."
        )

        for warmup_number in range(
            1,
            N_WARMUP + 1,
        ):
            model.transcribe(
                str(audio_path),
                task="transcribe",
                language=LANGUAGE,
                fp16=use_fp16,
            )

            synchronize_cuda(device)

            print(
                f"Warm-up {warmup_number}/{N_WARMUP} complete"
            )

        latencies = []
        transcriptions = []

        print(
            f"Running {N_RUNS} measured transcription(s)..."
        )

        for run_number in range(
            1,
            N_RUNS + 1,
        ):
            synchronize_cuda(device)

            start = time.perf_counter()

            result = model.transcribe(
                str(audio_path),
                task="transcribe",
                language=LANGUAGE,
                fp16=use_fp16,
            )

            synchronize_cuda(device)

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            text = result.get(
                "text",
                "",
            ).strip()

            real_time_factor = (
                latency_ms / 1000
            ) / audio_duration_s

            latencies.append(latency_ms)
            transcriptions.append(text)

            run_rows.append(
                {
                    "file": audio_path.name,
                    "run": run_number,
                    "model_size": MODEL_SIZE,
                    "device": device,
                    "precision": (
                        "fp16"
                        if use_fp16
                        else "fp32"
                    ),
                    "language": LANGUAGE,
                    "audio_duration_s": audio_duration_s,
                    "latency_ms": latency_ms,
                    "real_time_factor": real_time_factor,
                    "text": text,
                }
            )

            print(
                f"Run {run_number}: "
                f"{latency_ms:.2f} ms | "
                f"RTF: {real_time_factor:.4f}"
            )

        latency_series = pd.Series(
            latencies,
            dtype="float64",
        )

        representative_text = (
            stats.mode(transcriptions)
            if transcriptions
            else ""
        )

        summary_rows.append(
            {
                "file": audio_path.name,
                "model_size": MODEL_SIZE,
                "device": device,
                "precision": (
                    "fp16"
                    if use_fp16
                    else "fp32"
                ),
                "language": LANGUAGE,
                "audio_duration_s": audio_duration_s,
                "warmup_runs": N_WARMUP,
                "measured_runs": N_RUNS,
                "mean_ms": stats.mean(latencies),
                "median_ms": stats.median(latencies),
                "p50_ms": latency_series.quantile(0.50),
                "p95_ms": latency_series.quantile(0.95),
                "p99_ms": latency_series.quantile(0.99),
                "std_ms": (
                    stats.stdev(latencies)
                    if len(latencies) > 1
                    else 0.0
                ),
                "best_ms": min(latencies),
                "worst_ms": max(latencies),
                "mean_rtf": (
                    stats.mean(latencies) / 1000
                ) / audio_duration_s,
                "text": representative_text,
            }
        )

    run_df = pd.DataFrame(run_rows)
    summary_df = pd.DataFrame(summary_rows)

    RUN_RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_df.to_csv(
        RUN_RESULTS_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_RESULTS_PATH,
        index=False,
    )

    print(
        "\nIndividual runs saved to:",
        RUN_RESULTS_PATH,
    )

    print(
        "Summary results saved to:",
        SUMMARY_RESULTS_PATH,
    )

    print("\nSummary:")
    print(
        summary_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()