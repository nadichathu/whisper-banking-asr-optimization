import time
import whisper

from bench.config import MODEL_SIZE, DEVICE


class WhisperNoContextAdapter:
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.model_size = self.config.get("model_size", MODEL_SIZE)
        self.device = self.config.get("device", DEVICE)
        self.model_id = "whisper_no_context"

    def load(self):
        if self.model is None:
            self.model = whisper.load_model(self.model_size, device=self.device)

    def transcribe(self, audio_path: str):
        self.load()
        start = time.perf_counter()

        result = self.model.transcribe(
            audio_path,
            task="transcribe",
            language="en",
            fp16=False,
            condition_on_previous_text=False,
        )

        end = time.perf_counter()

        return {
            "text": result.get("text", "").strip(),
            "meta": {
                "optimization": "no_previous_text_conditioning",
                "condition_on_previous_text": False,
                "latency_ms": (end - start) * 1000,
            },
        }

    def close(self):
        self.model = None