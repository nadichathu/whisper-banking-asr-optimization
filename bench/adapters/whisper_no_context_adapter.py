import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperNoContextAdapter(BaseAdapter):
    """OpenAI Whisper adapter with previous-text conditioning disabled.

    Experimental change relative to the FP32 Whisper baseline:

        condition_on_previous_text=False

    All other relevant transcription settings remain aligned with the
    baseline, including FP32 precision and Whisper's default temperature
    fallback behaviour.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.config = config or {}
        self.name = "whisper_no_context"

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
                f"no-context adapter on {self.device}"
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
        """Transcribe an audio file without previous-text conditioning."""

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

            # Keep precision identical to the FP32 baseline.
            fp16=False,

            # Experimental variable.
            condition_on_previous_text=False,
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        text = result.get(
            "text",
            "",
        ).strip()

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "openai-whisper",
                "backend": "whisper",
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp32",
                "language": "en",
                "decoding": "default_transcribe",
                "condition_on_previous_text": False,
                "temperature_fallback_enabled": True,
                "optimization": "no_previous_text_conditioning",
                "experimental_change": (
                    "condition_on_previous_text=False"
                ),
                "latency_type": "end_to_end",
                "latency_includes": [
                    "audio_loading",
                    "audio_resampling",
                    "padding_or_trimming",
                    "mel_spectrogram",
                    "encoder",
                    "decoder",
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