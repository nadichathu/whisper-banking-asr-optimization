import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperNoFallbackAdapter(BaseAdapter):
    """OpenAI Whisper with quality-triggered fallback disabled. GPU-only.

    Experimental changes relative to the FP32 Whisper baseline:

        compression_ratio_threshold=None
        logprob_threshold=None
        no_speech_threshold=None

    Whisper's normal temperature sequence is retained, but because all
    fallback-triggering quality checks are disabled, transcription accepts
    the first decoding result and does not escalate to another temperature.

    Important research limitation:
    Whisper's first default temperature is 0.0. Therefore, this experiment
    normally produces one greedy decoding attempt and can be operationally
    equivalent to a greedy-only adapter using ``temperature=0.0``.

    Do not present greedy-only decoding and disabled fallback as fully
    independent optimization techniques without acknowledging this overlap.
    """

    DEFAULT_TEMPERATURES = (
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
    )

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.name = "whisper_no_fallback"

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
                f"no-fallback adapter on {self.device}"
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
        """Transcribe audio without quality-triggered fallback attempts."""

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

            # Match the corrected FP32 baseline.
            fp16=False,

            # Retain Whisper's baseline temperature sequence.
            # With fallback thresholds disabled, the first result
            # will be accepted and later temperatures will not run.
            temperature=self.DEFAULT_TEMPERATURES,

            # Experimental variables: disable all quality checks
            # capable of requesting another decoding attempt.
            compression_ratio_threshold=None,
            logprob_threshold=None,
            no_speech_threshold=None,

            # Preserve the baseline context behaviour.
            condition_on_previous_text=True,

            # Suppress console output during benchmarking.
            verbose=None,

            # Do not set:
            # - without_timestamps
            # - sample_len
            # - beam_size
            # - best_of
            #
            # Their baseline defaults are preserved.
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

        segment_temperatures = []

        for segment in segments:
            temperature = segment.get("temperature")

            if temperature is not None:
                segment_temperatures.append(
                    float(temperature)
                )

        temperatures_used = sorted(
            set(segment_temperatures)
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
                    "disabled_quality_triggered_fallback"
                ),
                "experimental_change": (
                    "compression_ratio_threshold=None, "
                    "logprob_threshold=None, "
                    "no_speech_threshold=None"
                ),
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp32",
                "language": "en",
                "decoding": (
                    "default_transcribe_without_quality_fallback"
                ),
                "configured_temperature_sequence": list(
                    self.DEFAULT_TEMPERATURES
                ),
                "temperatures_used": temperatures_used,
                "fallback_disabled": True,
                "fallback_trigger_checks_enabled": False,
                "compression_ratio_threshold": None,
                "logprob_threshold": None,
                "no_speech_threshold": None,
                "condition_on_previous_text": True,
                "without_timestamps": False,
                "token_limit_applied": False,
                "segment_count": len(segments),
                "generated_token_count": (
                    generated_token_count
                ),
                "overlaps_with_greedy_only": True,
                "overlap_explanation": (
                    "Whisper starts its default temperature "
                    "sequence at 0.0. With all fallback triggers "
                    "disabled, the first greedy result is accepted."
                ),
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
                    "single_accepted_decode_attempt",
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
