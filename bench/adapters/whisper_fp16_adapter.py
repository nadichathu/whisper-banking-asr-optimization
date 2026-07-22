import time

import torch
import whisper


class WhisperFP16Adapter:
    def __init__(self, config=None):
        self.config = config or {}
        self.device = "cuda"
        self.model = None
        self.name = "whisper_fp16"

    def load(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for the Whisper FP16 experiment."
            )

        model_size = self.config.get("model_size", "small")

        self.model = whisper.load_model(
            model_size,
            device=self.device,
        )

    def transcribe(self, audio_path):
        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Call load() before transcribing."
            )

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        result = self.model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            fp16=True,
            temperature=0.0,
            condition_on_previous_text=False,
        )

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "text": result["text"].strip(),
            "latency_ms": latency_ms,
            "meta": {
                "model": self.config.get("model_size", "small"),
                "device": self.device,
                "precision": "fp16",
            },
        }

    def close(self):
        self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()