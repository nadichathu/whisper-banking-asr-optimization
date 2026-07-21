import time
import pathlib
import statistics as stats
import pandas as pd
import whisper

# ===== CONFIG =====

AUDIO_DIR = pathlib.Path("audio_samples")
RESULT_PATH = pathlib.Path("results/baseline_results.csv")

MODEL_SIZE = "small"
DEVICE = "cpu"

N_WARMUP = 2
N_RUNS = 5

FP16 = False

# ===================


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def main():

    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)

    audio_files = sorted(AUDIO_DIR.glob("*.wav"))

    if not audio_files:
        raise SystemExit("No WAV files found in audio_samples")

    rows = []

    for audio in audio_files:

        print(f"\nProcessing {audio.name}")

        latencies = []

        # warmup
        for _ in range(N_WARMUP):
            model.transcribe(
                str(audio),
                fp16=(FP16 and DEVICE == "cuda"),
                task="transcribe"
            )

        # measured runs
        for run in range(N_RUNS):

            t0 = time.perf_counter()

            result = model.transcribe(
                str(audio),
                fp16=(FP16 and DEVICE == "cuda"),
                task="transcribe"
            )

            t1 = time.perf_counter()

            latency = (t1 - t0) * 1000
            latencies.append(latency)

            print(f"run {run+1}: {latency:.2f} ms")

        row = {
            "file": audio.name,
            "mean": stats.mean(latencies),
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "best": min(latencies),
            "worst": max(latencies)
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    RESULT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RESULT_PATH, index=False)

    print("\nBaseline results saved to:", RESULT_PATH)
    print(df)


if __name__ == "__main__":
    main()