import time
from typing import Dict, Any
import whisper
from .base_adapter import BaseAdapter
from bench.config import MODEL_SIZE, DEVICE


class WhisperAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # allow override from config
        self.model_size = (config or {}).get("model_size", MODEL_SIZE)
        self.device = (config or {}).get("device", DEVICE)
        self._model = None

    def load(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        model = self.load()
        t0 = time.perf_counter()
        result = model.transcribe(
            audio_path,
            task="transcribe",
            language="en",
            #fp16=False,
            fp16=self.device=="cuda"
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        text = result.get("text", "").strip()
        meta = {
            "model_size": self.model_size,
            "device": self.device,
            "latency_ms": latency_ms,
        }
        return {"text": text, "meta": meta}

    def profile(self, audio_path: str) -> Dict[str, float]:
        # mimic profiling from profiling/profile_whisper.py
        model = self.load()
        t0 = time.perf_counter()
        audio = whisper.load_audio(str(audio_path))
        t1 = time.perf_counter()
        audio = whisper.pad_or_trim(audio)
        t2 = time.perf_counter()
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        t3 = time.perf_counter()
        enc_out = model.encoder(mel.unsqueeze(0))
        t4 = time.perf_counter()
        options = whisper.DecodingOptions()
        whisper.decode(model, mel.unsqueeze(0), options)
        t5 = time.perf_counter()
        return {
            "load_audio_ms": (t1 - t0) * 1000,
            "pad_trim_ms": (t2 - t1) * 1000,
            "mel_ms": (t3 - t2) * 1000,
            "encoder_ms": (t4 - t3) * 1000,
            "decoder_ms": (t5 - t4) * 1000,
            "total_ms": (t5 - t0) * 1000,
        }

    def close(self):
        # Whisper model does not expose explicit close. Free reference.
        self._model = None
