import csv
import os
from datetime import datetime

LOG_PATH = os.path.join("logs", "runs.csv")
FIELDS = ["ts", "mode", "audio_file", "latency_ms", "text"]

def log_run(mode: str, audio_file: str, latency_ms: float, text: str):
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "audio_file": audio_file,
            "latency_ms": f"{latency_ms:.2f}",
            "text": text.strip(),
        })
