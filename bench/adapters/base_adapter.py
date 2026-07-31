import time
from typing import Dict, Any

import whisper

from .base_adapter import BaseAdapter
from bench.config import MODEL_SIZE, DEVICE


class WhisperAdapter(BaseAdapter):
    """OpenAI Whisper FP32 baseline adapter.

    This baseline deliberately forces FP32 (fp16=False) unconditionally,
    regardless of device, so that whisper_fp16_adapter.py represents a
    genuine, isolated experimental condition on CUDA hardware. Do not
    make fp16 device-conditional here -- that reintroduces the baseline/
    FP16 precision confound documented in Chapter 3, Section 3.2.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        config = config or {}
        self.name = "whisper"
        self.model_size = config.get("model_size", MODEL_SIZE)

        requested_device = str(config.get("device", DEVICE)).lower()
        self.device = requested_device

        self._model = None

    def load(self):
        if self._model is None:
            self._model = whisper.load_model(self.model_size, device=self.device)
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        model = self.load()

        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        result = model.transcribe(
            audio_path,
            task="transcribe",
            language="en",
            fp16=False,  # unconditional FP32 -- do not make this device-conditional
        )

        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000

        text = result.get("text", "").strip()

        # latency_ms MUST be top-level per BaseAdapter's documented contract.
        # Do not move this inside meta -- the runner reads output["latency_ms"]
        # directly and raises KeyError if it is only nested inside meta.
        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp32",
            },
        }

    def profile(self, audio_path: str) -> Dict[str, Any]:
        # mimic profiling from profiling/profile_whisper.py
        import torch
        is_cuda = self.device.startswith("cuda")

        model = self.load()

        if is_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        audio = whisper.load_audio(str(audio_path))
        t1 = time.perf_counter()

        audio = whisper.pad_or_trim(audio)
        t2 = time.perf_counter()

        mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        if is_cuda:
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        with torch.inference_mode():
            model.encoder(mel.unsqueeze(0))
        if is_cuda:
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        options = whisper.DecodingOptions(fp16=False)
        with torch.inference_mode():
            whisper.decode(model, mel, options)
        if is_cuda:
            torch.cuda.synchronize()
        t5 = time.perf_counter()

        return {
            "load_audio_ms": (t1 - t0) * 1000,
            "pad_trim_ms": (t2 - t1) * 1000,
            "mel_ms": (t3 - t2) * 1000,
            "encoder_ms": (t4 - t3) * 1000,
            "decoder_ms": (t5 - t4) * 1000,
            "total_ms": (t5 - t0) * 1000,
            "note": (
                "decoder_ms includes a redundant second encoder pass, since "
                "whisper.decode() re-encodes the mel spectrogram internally. "
                "Not a clean decoder-only measurement."
            ),
        }

    def close(self):
        # Whisper model does not expose explicit close. Free reference.
        self._model = None
