import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperLimitedTokensAdapter(BaseAdapter):
    """OpenAI Whisper with a limited decoder output length. GPU-only.

    Experimental change relative to the FP32 Whisper baseline:

        sample_len=max_tokens

    The adapter deliberately uses Whisper's high-level ``transcribe()``
    method so that the timing boundary, audio loading, preprocessing,
    segmentation, timestamp behaviour, temperature fallback, and context
    propagation remain aligned with the baseline.

    Only the maximum number of tokens generated during each decoding
    window is changed.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.name = "whisper_limited_tokens"

        self.model_size = self.config.get(
            "model_size",
            MODEL_SIZE,
        )

        self.max_tokens = int(
            self.config.get(
                "max_tokens",
                32,
            )
        )

        if self.max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        requested_device = str(
            self.config.get(
                "device",
                DEVICE,
            )
        ).lower()

        if not requested_device.startswith("cuda"):
            raise ValueError(
                "This benchmark suite is GPU-only. "
                f"Received device='{requested_device}'. "
                "Run with '--device cuda' or '--device cuda:0'."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible GPU "
                "is available in the current environment."
            )

        self.device = requested_device

        self._model = None

    def load(self):
        """Load the Whisper model."""

        if self._model is None:
            print(
                f"Loading Whisper {self.model_size} "
                f"limited-token adapter on {self.device}"
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
        """Transcribe an audio file with limited decoder tokens."""

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

        torch.cuda.synchronize(device=self.device)

        start_time = time.perf_counter()

        result = self._model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",

            # Keep precision identical to the corrected
            # FP32 Whisper baseline.
            fp16=False,

            # Experimental variable:
            # limit generated tokens per decoding window.
            sample_len=self.max_tokens,

            # Do not specify:
            # - temperature
            # - condition_on_previous_text
            # - without_timestamps
            #
            # This preserves Whisper's baseline defaults.
        )

        torch.cuda.synchronize(device=self.device)

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
                "optimization": "limited_decoder_tokens",
                "experimental_change": (
                    f"sample_len={self.max_tokens}"
                ),
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp32",
                "language": "en",
                "decoding": "default_transcribe_limited_tokens",
                "max_tokens_per_decode_window": self.max_tokens,
                "generated_token_count": generated_token_count,
                "segment_count": len(segments),

                # These remain at Whisper baseline behaviour.
                "timestamps_disabled": False,
                "condition_on_previous_text": True,
                "temperature_fallback_enabled": True,

                "latency_type": "end_to_end",
                "model_loading_included": False,
                "latency_includes": [
                    "audio_loading",
                    "audio_decoding",
                    "audio_resampling",
                    "padding",
                    "mel_spectrogram",
                    "segmentation",
                    "encoder",
                    "limited_token_decoder",
                    "timestamp_processing",
                    "text_generation",
                ],
            },
        }

    def close(self) -> None:
        """Release Whisper model resources."""

        self._model = None

        gc.collect()

        torch.cuda.empty_cache()
