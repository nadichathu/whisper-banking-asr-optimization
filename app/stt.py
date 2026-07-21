import whisper
from .config import MODEL_SIZE, DEVICE, LANGUAGE, FP16

_model = None

def get_model():
    global _model
    if _model is None:
        _model = whisper.load_model(MODEL_SIZE, device=DEVICE)
    return _model

def transcribe(audio_path: str) -> dict:
    model = get_model()
    # task="transcribe" keeps it in transcription mode (not translate)
    return model.transcribe(
        audio_path,
        language=LANGUAGE,
        fp16=(FP16 and DEVICE == "cuda"),
        task="transcribe",
    )
