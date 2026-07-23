import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperFP16Adapter(BaseAdapter):
    """OpenAI Whisper FP16 adapter for CUDA inference.

    This adapter is intended to isolate the effect of FP16 precision
    relative to the FP32 Whisper baseline.

    A methodologically valid comparison requires:

    - the same CUDA device
    - the same Whisper model size
    - the same audio files
    - the same transcription settings
    - the same latency boundary

    FP16 inference is not supported as a valid CPU benchmark condition.
    Therefore, this adapter rejects CPU instead of silently falling back
    to FP32 or changing hardware.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.config = config or {}
        self.name = "whisper_fp16"

        self.model_size = self.config.get(
            "model_size",
            MODEL_SIZE,
        )

        requested_device = str(
            self.config.get(
                "device",
                DEVICE,
            )
        ).lower().strip()

        if not requested_device.startswith("cuda"):
            raise ValueError(
                "WhisperFP16Adapter requires a CUDA device. "
                f"Received device='{requested_device}'. "
                "Run with '--device cuda' or '--device cuda:0'."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "WhisperFP16Adapter requires CUDA, but CUDA is not "
                "available in the current environment."
            )

        try:
            torch.device(requested_device)
        except RuntimeError as exc:
            raise ValueError(
                f"Invalid CUDA device: '{requested_device}'."
            ) from exc

        if ":" in requested_device:
            try:
                device_index = int(
                    requested_device.split(":", maxsplit=1)[1]
                )
            except ValueError as exc:
                raise ValueError(
                    f"Invalid CUDA device index in "
                    f"'{requested_device}'."
                ) from exc

            if device_index < 0:
                raise ValueError(
                    "CUDA device index cannot be negative."
                )

            if device_index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device '{requested_device}' does not exist. "
                    f"Available CUDA device count: "
                    f"{torch.cuda.device_count()}."
                )

        self.device = requested_device
        self._model = None

    def load(self):
        """Load the Whisper model on CUDA."""

        if self._model is None:
            print(
                f"Loading Whisper {self.model_size} "
                f"FP16 on {self.device}"
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
        """Transcribe an audio file using FP16 inference."""

        if self._model is None:
            raise RuntimeError(
                "Whisper FP16 model is not loaded. "
                "Call load() before transcribe()."
            )

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file was not found: {audio_path}"
            )

        torch.cuda.synchronize(
            device=self.device
        )

        start_time = time.perf_counter()

        result = self._model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            fp16=True,
        )

        torch.cuda.synchronize(
            device=self.device
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        text = str(
            result.get(
                "text",
                "",
            )
            or ""
        ).strip()

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "openai-whisper",
                "backend": "whisper",
                "adapter": self.name,
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp16",
                "decoding": "default_transcribe",
                "task": "transcribe",
                "language": "en",
                "latency_type": "end_to_end",
                "model_loading_included": False,
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
        """Release model and CUDA resources."""

        self._model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()