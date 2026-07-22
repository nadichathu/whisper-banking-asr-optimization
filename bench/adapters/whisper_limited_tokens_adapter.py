import time

import torch
import whisper


class WhisperLimitedTokensAdapter:
    """
    OpenAI Whisper adapter that limits the maximum number of tokens generated
    during decoding.

    This adapter is intended for short banking voice commands, where long
    transcription outputs are unnecessary.
    """

    def __init__(self, config=None):
        self.config = config or {}

        self.device = self.config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )

        self.model_size = self.config.get("model_size", "small")
        self.max_tokens = int(self.config.get("max_tokens", 32))

        self.model = None
        self.name = "whisper_limited_tokens"

    def load(self):
        """
        Load the Whisper model onto the configured device.
        """

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible GPU is available."
            )

        self.model = whisper.load_model(
            self.model_size,
            device=self.device,
        )

    def transcribe(self, audio_path):
        """
        Transcribe one audio file using limited-token decoding.
        """

        if self.model is None:
            raise RuntimeError(
                "Model is not loaded. Call load() before transcribing."
            )

        # Load and prepare audio using Whisper's standard preprocessing.
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=self.model.dims.n_mels,
        ).to(self.device)

        use_fp16 = self.device == "cuda"

        decoding_options = whisper.DecodingOptions(
            task="transcribe",
            language="en",
            fp16=use_fp16,
            temperature=0.0,
            without_timestamps=True,
            sample_len=self.max_tokens,
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            result = whisper.decode(
                self.model,
                mel,
                decoding_options,
            )

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "text": result.text.strip(),
            "latency_ms": latency_ms,
            "meta": {
                "model": self.model_size,
                "device": self.device,
                "precision": "fp16" if use_fp16 else "fp32",
                "max_tokens": self.max_tokens,
                "decoding": "limited_tokens",
            },
        }

    def close(self):
        """
        Release the model and clear unused GPU memory.
        """

        self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()