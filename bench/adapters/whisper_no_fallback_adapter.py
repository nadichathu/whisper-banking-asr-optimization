import time
import torch
import whisper


class WhisperNoFallbackAdapter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = whisper.load_model(
            "small",
            device=self.device
        )

        self.name = "whisper_no_fallback"

    def transcribe(self, audio_path):
        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        result = self.model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            fp16=self.device == "cuda",

            # One deterministic decoding attempt only
            temperature=0.0,
            beam_size=None,
            best_of=None,

            # Disable fallback-related checks
            compression_ratio_threshold=None,
            logprob_threshold=None,

            condition_on_previous_text=False,
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
                "fallback_disabled": True,
            },
        }