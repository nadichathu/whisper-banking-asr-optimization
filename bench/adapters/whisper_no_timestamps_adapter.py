import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperNoTimestampsAdapter(BaseAdapter):
    """OpenAI Whisper with timestamp-token generation disabled.

    Experimental change relative to the FP32 Whisper baseline:

        without_timestamps=True

    All other relevant settings remain aligned with the baseline:

        - FP32 precision
        - Default temperature fallback
        - Previous-text conditioning enabled
        - High-level transcribe() pipeline
        - End-to-end file-to-text timing
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.config = config or {}
        self.name = "whisper_no_timestamps"

        self.model_size = self.config.get(
            "model_size",
            MODEL_SIZE,
        )

        requested_device = str(
            self.config.get(
                "device",
                DEVICE,
            )
        ).lower()

        if requested_device.startswith("cuda"):
            if torch.cuda.is_available():
                self.device = requested_device
            else:
                print(
                    "CUDA was requested but is unavailable. "
                    "Falling back to CPU."
                )
                self.device = "cpu"
        else:
            self.device = "cpu"

        self._model = None

    def load(self):
        """Load the Whisper model."""

        if self._model is None:
            print(
                f"Loading Whisper {self.model_size} "
                f"no-timestamps adapter on {self.device}"
            )

            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
            )

            self._model.eval()

        return self._model

    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Transcribe audio without generating timestamp tokens."""

        if self._model is None:
            raise RuntimeError(
                "Whisper model is not loaded. "
                "Call load() before transcribe()."
            )

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file was not found: {audio_path}"
            )

        if not audio_path.is_file():
            raise ValueError(
                f"Audio path is not a file: {audio_path}"
            )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        result = self._model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",

            # Match the corrected FP32 baseline.
            fp16=False,

            # The only experimental variable.
            without_timestamps=True,

            # Prevent console output during benchmarking.
            verbose=None,

            # Do not explicitly set:
            # - temperature
            # - condition_on_previous_text
            # - sample_len
            #
            # This preserves the baseline defaults.
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        text = str(
            result.get(
                "text",
                "",
            )
        ).strip()

        segments = result.get(
            "segments",
            [],
        )

        generated_token_count = 0

        for segment in segments:
            tokens = segment.get(
                "tokens",
                [],
            )

            if tokens is not None:
                generated_token_count += len(tokens)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "openai-whisper",
                "backend": "whisper",
                "optimization": (
                    "disabled_timestamp_generation"
                ),
                "experimental_change": (
                    "without_timestamps=True"
                ),
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp32",
                "language": "en",
                "decoding": (
                    "default_transcribe_without_timestamps"
                ),
                "without_timestamps": True,
                "timestamp_tokens_enabled": False,
                "word_timestamps": False,
                "condition_on_previous_text": True,
                "temperature_fallback_enabled": True,
                "token_limit_applied": False,
                "segment_count": len(segments),
                "generated_token_count": (
                    generated_token_count
                ),
                "latency_type": "end_to_end",
                "latency_includes": [
                    "audio_loading",
                    "audio_decoding",
                    "audio_resampling",
                    "padding",
                    "mel_spectrogram",
                    "segmentation",
                    "encoder",
                    "decoder_without_timestamp_tokens",
                    "text_generation",
                ],
            },
        }

    def close(self) -> None:
        """Release Whisper model resources."""

        used_cuda = self.device.startswith("cuda")

        self._model = None

        gc.collect()

        if used_cuda:
            torch.cuda.empty_cache()