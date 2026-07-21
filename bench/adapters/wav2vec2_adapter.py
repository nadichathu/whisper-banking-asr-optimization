import time
from typing import Dict, Any

try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import torch
    import soundfile as sf
except Exception:
    Wav2Vec2ForCTC = None
    Wav2Vec2Processor = None
    torch = None

from .base_adapter import BaseAdapter
from bench.config import WAV2VEC_MODEL_ID, DEVICE, SAMPLE_RATE


class Wav2Vec2Adapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_id = (config or {}).get("model_id", WAV2VEC_MODEL_ID)
        self.device = (config or {}).get("device", DEVICE)
        self._model = None
        self._processor = None

    def load(self):
        if Wav2Vec2ForCTC is None:
            raise RuntimeError("transformers/torch not installed. Install transformers and torch to use Wav2Vec2Adapter.")
        if self._model is None:
            self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
            if self.device == "cuda" and torch.cuda.is_available():
                self._model.to("cuda")
        return self._model

    def _load_audio(self, path: str):
        data, sr = sf.read(path)
        if sr != SAMPLE_RATE:
            # resample using torchaudio if available
            try:
                import torchaudio
                waveform = torch.tensor(data).unsqueeze(0)
                waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
                return waveform.squeeze(0).numpy()
            except Exception:
                # fallback: naive resample (not ideal)
                return data
        return data

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        model = self.load()
        processor = self._processor
        t0 = time.perf_counter()
        audio = self._load_audio(audio_path)
        inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            model.to("cuda")
        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        meta = {"model_id": self.model_id, "device": self.device, "latency_ms": latency_ms}
        return {"text": text, "meta": meta}

    def close(self):
        self._model = None
        self._processor = None
