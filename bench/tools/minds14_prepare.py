from datasets import load_dataset, Audio
import os
import io
import soundfile as sf
import pandas as pd

OUTPUT_DIR = "data/minds14_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset = load_dataset(
    "PolyAI/minds14",
    "en-US",
    split="train",
    streaming=True
)

# Disable automatic audio decoding
dataset = dataset.cast_column("audio", Audio(decode=False))

rows = []

for i, item in enumerate(dataset):
    audio_info = item["audio"]

    if audio_info["bytes"] is not None:
        audio_array, sr = sf.read(io.BytesIO(audio_info["bytes"]))
    else:
        audio_array, sr = sf.read(audio_info["path"])

    transcript = item.get("english_transcription", item.get("transcription", ""))
    intent = item["intent_class"]
    duration = len(audio_array) / sr

    filename = f"minds14_{i}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)

    sf.write(filepath, audio_array, sr)

    rows.append({
        "file_name": filename,
        "transcript": transcript,
        "intent": intent,
        "duration": duration,
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)

print(f"Done. Saved {len(df)} files to {OUTPUT_DIR}")