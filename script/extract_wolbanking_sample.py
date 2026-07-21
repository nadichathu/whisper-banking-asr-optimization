import pandas as pd
from pathlib import Path

parquet_path = Path("data/raw/wolbanking77/train.parquet")
out_dir = Path("data/extracted")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(parquet_path)

print("Columns:", df.columns.tolist())
print("\nFirst row:\n", df.iloc[0])

audio_obj = df.iloc[0]["audio"]   # get only the audio field
print("\nAudio keys:", audio_obj.keys())

wav_bytes = audio_obj["bytes"]

out_path = out_dir / "sample0.wav"
with open(out_path, "wb") as f:
    f.write(wav_bytes)

print("Saved:", out_path)
