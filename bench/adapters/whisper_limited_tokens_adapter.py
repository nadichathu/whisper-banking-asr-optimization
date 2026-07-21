import time
import torch
import whisper


class WhisperLimitedTokensAdapter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = whisper.load_model(
            "small",
            device=self.device
        )

        self.name = "whisper_limited_tokens"

    def transcribe(self, audio_path):
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        result = self.model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            fp16=self.device == "cuda",
            temperature=0.0,
            condition_on_previous_text=False,
            sample_len=32,
            verbose=None,
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "text": result["text"].strip(),
            "latency_ms": latency_ms,
            "meta": {
                "device": self.device,
                "sample_len": 32,
            },
        }