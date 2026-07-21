import time

import torch
import whisper


class WhisperDirectDecodeAdapter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = whisper.load_model(
            "small",
            device=self.device
        )

        self.name = "whisper_direct_decode"

    def transcribe(self, audio_path):
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=self.model.dims.n_mels
        ).to(self.device)

        options = whisper.DecodingOptions(
            task="transcribe",
            language="en",
            temperature=0.0,
            without_timestamps=True,
            fp16=self.device == "cuda",
            sample_len=32,
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        result = whisper.decode(
            self.model,
            mel,
            options
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "text": result.text.strip(),
            "latency_ms": latency_ms,
            "meta": {
                "device": self.device,
                "direct_decode": True,
                "sample_len": 32,
            },
        }