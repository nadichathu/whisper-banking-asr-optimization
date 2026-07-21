import sounddevice as sd
import soundfile as sf
import numpy as np
from .config import SAMPLE_RATE

def record_wav(out_path: str, seconds: int) -> str:
    frames = int(seconds * SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype="float32")    
    sd.wait()
    audio = np.squeeze(audio)
    sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return out_path