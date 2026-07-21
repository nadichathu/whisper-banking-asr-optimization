import pandas as pd
import soundfile as sf
import librosa
from pathlib import Path

DATA_DIR = Path("data/minds14_banking")
META_PATH = DATA_DIR / "metadata.csv"

TARGET_SR = 16000

df = pd.read_csv(META_PATH)

for _, row in df.iterrows():
    audio_path = DATA_DIR / row["file_name"]

    audio, sr = sf.read(audio_path)

    # convert to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # resample
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    # overwrite
    sf.write(audio_path, audio, TARGET_SR)

print("Done. All files converted to 16kHz mono.")