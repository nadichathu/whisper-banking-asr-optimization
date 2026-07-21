import time
import pathlib
from .config import RECORD_SECONDS
from .record import record_wav
from .stt import transcribe
from .logger import log_run

AUDIO_DIR = pathlib.Path("audio_samples")
AUDIO_DIR.mkdir(exist_ok=True)

def run_record_mode():
    out_path = AUDIO_DIR / "latest.wav"
    print(f"\nRecording {RECORD_SECONDS}s... speak your banking command now.")
    record_wav(str(out_path), RECORD_SECONDS)
    print("Recording saved:", out_path)

    t0 = time.perf_counter()
    out = transcribe(str(out_path))
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000
    text = out.get("text", "").strip()

    print(f"\nTranscription: {text}")
    print(f"Latency: {latency_ms:.2f} ms")

    log_run("record", out_path.name, latency_ms, text)

def run_file_mode(file_path: str):
    p = pathlib.Path(file_path)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    t0 = time.perf_counter()
    out = transcribe(str(p))
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000
    text = out.get("text", "").strip()

    print(f"\nFile: {p.name}")
    print(f"Transcription: {text}")
    print(f"Latency: {latency_ms:.2f} ms")

    log_run("file", p.name, latency_ms, text)

def main():
    print("Banking Voice Command Shell (minimal)")
    print("1) Record a command")
    print("2) Transcribe an existing WAV file")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        run_record_mode()
    elif choice == "2":
        file_path = input("Enter path to .wav: ").strip().strip('"')
        run_file_mode(file_path)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
