import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperDirectDecodeAdapter(BaseAdapter):
    """OpenAI Whisper using the lower-level ``whisper.decode()`` API.

    This adapter measures a direct single-window decoding pipeline:

        audio file
        -> whisper.load_audio()
        -> whisper.pad_or_trim()
        -> whisper.log_mel_spectrogram()
        -> whisper.decode()
        -> final text extraction

    The complete file-to-text pipeline is included in the reported latency.

    Temperature fallback follows OpenAI Whisper's high-level transcription
    policy as closely as possible:

    - excessive compression ratio can trigger another temperature attempt;
    - low average log probability can trigger another temperature attempt;
    - likely silence cancels fallback only when both:
        1. no-speech probability exceeds the configured threshold; and
        2. average log probability is below the configured threshold.

    Each call to ``whisper.decode()`` receives a mel spectrogram and therefore
    performs its own encoder and decoder execution. OpenAI Whisper's
    high-level ``transcribe()`` fallback loop similarly calls the model's
    decode operation again for each attempted temperature.

    Research limitation:
    This adapter processes only one padded or trimmed 30-second Whisper
    window. It does not reproduce the complete long-audio segmentation,
    timestamp handling, and context propagation performed by
    ``model.transcribe()``. This is acceptable for the short banking-command
    recordings used in this study but must be documented.
    """

    DEFAULT_TEMPERATURES: Tuple[float, ...] = (
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
    ) -> None:
        super().__init__(config)

        self.name = "whisper_direct_decode"

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

        self.device = self._validate_cuda_device(
            requested_device
        )

        configured_temperatures = self.config.get(
            "temperatures",
            self.DEFAULT_TEMPERATURES,
        )

        if isinstance(
            configured_temperatures,
            (int, float),
        ):
            configured_temperatures = (
                float(configured_temperatures),
            )

        self.temperatures = tuple(
            float(value)
            for value in configured_temperatures
        )

        if not self.temperatures:
            raise ValueError(
                "At least one decoding temperature is required."
            )

        if any(
            value < 0.0
            for value in self.temperatures
        ):
            raise ValueError(
                "Decoding temperatures cannot be negative."
            )

        self.compression_ratio_threshold = (
            self._optional_float(
                self.config.get(
                    "compression_ratio_threshold",
                    2.4,
                )
            )
        )

        self.logprob_threshold = self._optional_float(
            self.config.get(
                "logprob_threshold",
                -1.0,
            )
        )

        self.no_speech_threshold = self._optional_float(
            self.config.get(
                "no_speech_threshold",
                0.6,
            )
        )

        self._model = None

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> Optional[float]:
        """Convert a configurable value to an optional float."""

        if value is None:
            return None

        return float(value)

    @staticmethod
    def _validate_cuda_device(
        requested_device: str,
    ) -> str:
        """Validate and normalize the requested CUDA device."""

        try:
            device = torch.device(requested_device)
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid device value: '{requested_device}'."
            ) from exc

        if device.type != "cuda":
            raise ValueError(
                "WhisperDirectDecodeAdapter requires CUDA. "
                f"Received device='{requested_device}'. "
                "Use '--device cuda' or '--device cuda:0'."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible GPU "
                "is available in the current environment."
            )

        device_index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )

        if (
            device_index < 0
            or device_index >= torch.cuda.device_count()
        ):
            raise ValueError(
                f"CUDA device index {device_index} is unavailable. "
                f"Available CUDA devices: "
                f"{torch.cuda.device_count()}."
            )

        return (
            f"cuda:{device_index}"
            if device.index is not None
            else "cuda"
        )

    def load(self):
        """Load the Whisper model."""

        if self._model is None:
            print(
                f"Loading Whisper {self.model_size} "
                f"direct-decode adapter on {self.device}"
            )

            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
            )

            self._model.eval()

        return self._model

    def _decode_with_fallback(
        self,
        mel: torch.Tensor,
    ):
        """Decode one mel window using Whisper-style fallback logic.

        Returns:
            A tuple containing:

            - selected decoding result;
            - temperatures attempted;
            - whether multiple temperatures were attempted;
            - final fallback reason;
            - whether the silence override stopped fallback.
        """

        if self._model is None:
            raise RuntimeError(
                "Whisper model is not loaded."
            )

        selected_result = None
        temperatures_attempted = []
        final_fallback_reason = None
        no_speech_override_used = False

        for temperature in self.temperatures:
            temperatures_attempted.append(
                temperature
            )

            options = whisper.DecodingOptions(
                task="transcribe",
                language="en",
                fp16=False,
                without_timestamps=False,
                temperature=temperature,
            )

            with torch.inference_mode():
                result = whisper.decode(
                    self._model,
                    mel,
                    options,
                )

            selected_result = result

            needs_fallback = False
            fallback_reasons = []

            if (
                self.compression_ratio_threshold
                is not None
                and result.compression_ratio
                > self.compression_ratio_threshold
            ):
                needs_fallback = True
                fallback_reasons.append(
                    "compression_ratio"
                )

            if (
                self.logprob_threshold is not None
                and result.avg_logprob
                < self.logprob_threshold
            ):
                needs_fallback = True
                fallback_reasons.append(
                    "average_log_probability"
                )

            this_attempt_no_speech_override = False

            # Match OpenAI Whisper's current silence override:
            # fallback is cancelled only when the result has both
            # high no-speech probability and low average log probability.
            if (
                self.no_speech_threshold is not None
                and result.no_speech_prob
                > self.no_speech_threshold
                and self.logprob_threshold is not None
                and result.avg_logprob
                < self.logprob_threshold
            ):
                needs_fallback = False
                this_attempt_no_speech_override = True

            if not needs_fallback:
                no_speech_override_used = (
                    this_attempt_no_speech_override
                )

                final_fallback_reason = (
                    None
                    if (
                        not fallback_reasons
                        or this_attempt_no_speech_override
                    )
                    else ",".join(fallback_reasons)
                )

                break

            final_fallback_reason = ",".join(
                fallback_reasons
            )

        if selected_result is None:
            raise RuntimeError(
                "Whisper decoding did not produce a result."
            )

        fallback_used = (
            len(temperatures_attempted) > 1
        )

        return (
            selected_result,
            temperatures_attempted,
            fallback_used,
            final_fallback_reason,
            no_speech_override_used,
        )

    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Transcribe one short audio file using direct decoding."""

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

        torch.cuda.synchronize(
            device=self.device
        )

        total_start = time.perf_counter()

        # -------------------------------------------------
        # Stage 1: Decode and resample the source file
        # -------------------------------------------------
        audio_load_start = time.perf_counter()

        try:
            audio = whisper.load_audio(
                str(audio_path)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio file "
                f"'{audio_path}': {exc}"
            ) from exc

        audio_load_ms = (
            time.perf_counter()
            - audio_load_start
        ) * 1000.0

        if audio.size == 0:
            raise ValueError(
                f"Audio file contains no samples: "
                f"{audio_path}"
            )

        original_sample_count = int(
            audio.shape[-1]
        )

        original_duration_s = (
            original_sample_count
            / float(whisper.audio.SAMPLE_RATE)
        )

        # -------------------------------------------------
        # Stage 2: Pad or trim to one Whisper window
        # -------------------------------------------------
        pad_trim_start = time.perf_counter()

        audio = whisper.pad_or_trim(
            audio
        )

        pad_trim_ms = (
            time.perf_counter()
            - pad_trim_start
        ) * 1000.0

        # -------------------------------------------------
        # Stage 3: Create the log-Mel spectrogram
        # -------------------------------------------------
        mel_start = time.perf_counter()

        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=self._model.dims.n_mels,
        ).to(self.device)

        torch.cuda.synchronize(
            device=self.device
        )

        mel_spectrogram_ms = (
            time.perf_counter()
            - mel_start
        ) * 1000.0

        # -------------------------------------------------
        # Stage 4: Decode with temperature fallback
        # -------------------------------------------------
        torch.cuda.synchronize(
            device=self.device
        )

        decode_start = time.perf_counter()

        (
            result,
            temperatures_attempted,
            fallback_used,
            final_fallback_reason,
            no_speech_override_used,
        ) = self._decode_with_fallback(
            mel
        )

        torch.cuda.synchronize(
            device=self.device
        )

        decode_latency_ms = (
            time.perf_counter()
            - decode_start
        ) * 1000.0

        # -------------------------------------------------
        # Stage 5: Apply Whisper-style silence handling
        # -------------------------------------------------
        no_speech_detected = False

        if (
            self.no_speech_threshold is not None
            and result.no_speech_prob
            > self.no_speech_threshold
        ):
            no_speech_detected = True

            if (
                self.logprob_threshold is not None
                and result.avg_logprob
                > self.logprob_threshold
            ):
                no_speech_detected = False

        text = (
            ""
            if no_speech_detected
            else str(result.text or "").strip()
        )

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        real_time_factor = (
            (total_latency_ms / 1000.0)
            / original_duration_s
            if original_duration_s > 0
            else None
        )

        generated_token_count = len(
            result.tokens
        )

        return {
            "text": text,
            "latency_ms": total_latency_ms,
            "meta": {
                "implementation": "openai-whisper",
                "backend": "whisper",
                "optimization": "direct_decode_api",
                "experimental_change": (
                    "Use whisper.decode() instead of "
                    "model.transcribe()"
                ),
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp32",
                "language": "en",
                "task": "transcribe",
                "direct_decode": True,
                "single_window_decode": True,
                "long_audio_chunking_enabled": False,
                "condition_on_previous_text_applicable": False,
                "without_timestamps": False,
                "timestamp_tokens_enabled": True,
                "token_limit_applied": False,
                "sample_len": None,
                "configured_temperature_sequence": list(
                    self.temperatures
                ),
                "temperatures_attempted": (
                    temperatures_attempted
                ),
                "fallback_enabled": (
                    len(self.temperatures) > 1
                ),
                "fallback_used": fallback_used,
                "final_fallback_reason": (
                    final_fallback_reason
                ),
                "no_speech_override_used": (
                    no_speech_override_used
                ),
                "compression_ratio_threshold": (
                    self.compression_ratio_threshold
                ),
                "logprob_threshold": (
                    self.logprob_threshold
                ),
                "no_speech_threshold": (
                    self.no_speech_threshold
                ),
                "no_speech_probability": float(
                    result.no_speech_prob
                ),
                "no_speech_detected": (
                    no_speech_detected
                ),
                "average_log_probability": float(
                    result.avg_logprob
                ),
                "compression_ratio": float(
                    result.compression_ratio
                ),
                "generated_token_count": (
                    generated_token_count
                ),
                "original_sample_count": (
                    original_sample_count
                ),
                "original_sample_rate": (
                    whisper.audio.SAMPLE_RATE
                ),
                "original_duration_s": (
                    original_duration_s
                ),
                "audio_load_ms": audio_load_ms,
                "pad_trim_ms": pad_trim_ms,
                "mel_spectrogram_ms": (
                    mel_spectrogram_ms
                ),
                "decode_latency_ms": (
                    decode_latency_ms
                ),
                "real_time_factor": (
                    real_time_factor
                ),
                "latency_type": "end_to_end",
                "model_loading_included": False,
                "latency_includes": [
                    "audio_loading",
                    "audio_decoding",
                    "audio_resampling",
                    "padding_or_trimming",
                    "mel_spectrogram",
                    "direct_encoder_execution",
                    "direct_decoder_execution",
                    "temperature_fallback_attempts",
                    "silence_decision",
                    "text_extraction",
                ],
                "methodological_limitation": (
                    "Direct decode processes one padded or "
                    "trimmed 30-second window and does not "
                    "replicate model.transcribe() long-audio "
                    "segmentation or context propagation. "
                    "Each fallback attempt invokes another "
                    "decode operation and encoder pass, as in "
                    "OpenAI Whisper's high-level fallback loop."
                ),
            },
        }

    def close(self) -> None:
        """Release Whisper model and CUDA resources."""

        self._model = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
