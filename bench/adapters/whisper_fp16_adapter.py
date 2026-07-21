import time
import torch
import whisper


class WhisperFP16Adapter:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU is required for the Whisper FP16 experiment."
            )

        self.device = "cuda"
        self.model = whisper.load_model("small", device=self.device)
        self.name = "whisper_fp16"

    def transcribe(self, audio_path):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = self.model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            fp16=True,
            temperature=0.0,
            condition_on_previous_text=False,
        )

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "text": result["text"].strip(),
            "latency_ms": latency_ms,
            "meta": {
                "device": self.device,
                "precision": "fp16",
            },
        }