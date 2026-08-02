import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperDirectDecodeAdapter(BaseAdapter):
    """OpenAI Whisper using the lower-level ``whisper.decode()`` API.
    GPU-only.

    Experimental approach:

        audio path
        -> whisper.load_audio()
        -> whisper.pad_or_trim()
        -> whisper.log_mel_spectrogram()
        -> whisper.decode()

    The complete file-to-text pipeline is included in the reported latency.

    Unlike the previous implementation, this adapter does not additionally:

        - limit decoder tokens
        - disable timestamp tokens
        - disable previous-text conditioning
        - change precision to FP16
        - permanently force a single greedy attempt

    The standard Whisper temperature-fallback sequence and quality
    thresholds are reproduced manually because ``whisper.decode()`` performs
    only one decoding attempt per call.

    Research limitation:
    Direct decoding operates on one padded or trimmed 30-second window. It
    does not reproduce the complete long-audio segmentation and context
    management performed by ``model.transcribe()``. This is acceptable for
    short banking commands but must be documented in the methodology.

    Research limitation (encoder cost on fallback):
    ``whisper.decode()`` recomputes the encoder on every call, since the
    public API provides no way to reuse a precomputed encoder output across
    calls. Whisper's internal ``transcribe()`` pipeline, by contrast,
    computes the encoder once and reuses it across every temperature
    attempt in its fallback loop. Consequently, on any file where fallback
    actually triggers (see ``fallback_used`` in the returned metadata),
    this adapter pays for one additional full encoder pass per extra
    temperature attempt -- a cost that is specific to this adapter's API
    choice, not to the direct-decode technique in general, and that
    inflates latency only on fallback-triggering files.
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

        configured_temperatures = self.config.get(
            "temperatures",
            self.DEFAULT_TEMPERATURES,
        )

        if isinstance(configured_temperatures, (int, float)):
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

        if any(value < 0.0 for value in self.temperatures):
            raise ValueError(
                "Decoding temperatures cannot be negative."
            )

        self.compression_ratio_threshold = self._optional_float(
            self.config.get(
                "compression_ratio_threshold",
                2.4,
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

            - selected DecodingResult
            - temperatures attempted
            - whether fallback occurred
            - reason that requested the final fallback
        """

        selected_result = None
        temperatures_attempted = []
        final_fallback_reason = None

        for temperature in self.temperatures:
            temperatures_attempted.append(
                temperature
            )

            options = whisper.DecodingOptions(
                task="transcribe",
                language="en",

                # Keep precision aligned with the corrected
                # FP32 Whisper baseline.
                fp16=False,

                # Preserve timestamp-token generation.
                without_timestamps=False,

                # Each call performs one decoding attempt.
                temperature=temperature,
            )

            with torch.inference_mode():
                result = whisper.decode(
                    self._model,
                    mel,
                    options,
                )

            selected_result = result

            fallback_reasons = []

            if (
                self.compression_ratio_threshold is not None
                and result.compression_ratio
                > self.compression_ratio_threshold
            ):
                fallback_reasons.append(
                    "compression_ratio"
                )

            if (
                self.logprob_threshold is not None
                and result.avg_logprob
                < self.logprob_threshold
            ):
                fallback_reasons.append(
                    "average_log_probability"
                )

            if not fallback_reasons:
                final_fallback_reason = None
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

        torch.cuda.synchronize(device=self.device)

        # Start before audio loading so the latency boundary
        # matches the file-to-text baseline measurement.
        total_start = time.perf_counter()

        # -------------------------------------------------
        # Stage 1: Load and resample audio
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
            time.perf_counter() - audio_load_start
        ) * 1000

        if audio.size == 0:
            raise ValueError(
                f"Audio file contains no samples: {audio_path}"
            )

        original_sample_count = int(
            audio.shape[-1]
        )

        original_duration_s = (
            original_sample_count
            / whisper.audio.SAMPLE_RATE
        )

        # -------------------------------------------------
        # Stage 2: Pad or trim to one Whisper window
        # -------------------------------------------------
        pad_trim_start = time.perf_counter()

        audio = whisper.pad_or_trim(
            audio
        )

        pad_trim_ms = (
            time.perf_counter() - pad_trim_start
        ) * 1000

        # -------------------------------------------------
        # Stage 3: Create the log-Mel spectrogram
        # -------------------------------------------------
        mel_start = time.perf_counter()

        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=self._model.dims.n_mels,
        ).to(self.device)

        torch.cuda.synchronize(device=self.device)

        mel_spectrogram_ms = (
            time.perf_counter() - mel_start
        ) * 1000

        # -------------------------------------------------
        # Stage 4: Direct decoding with fallback
        # -------------------------------------------------
        torch.cuda.synchronize(device=self.device)

        decode_start = time.perf_counter()

        (
            result,
            temperatures_attempted,
            fallback_used,
            final_fallback_reason,
        ) = self._decode_with_fallback(
            mel
        )

        torch.cuda.synchronize(device=self.device)

        decode_latency_ms = (
            time.perf_counter() - decode_start
        ) * 1000

        # Match Whisper's normal no-speech handling as closely
        # as possible for this single-window direct pipeline.
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
            else result.text.strip()
        )

        total_latency_ms = (
            time.perf_counter() - total_start
        ) * 1000

        real_time_factor = (
            (total_latency_ms / 1000)
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
                "fallback_enabled": True,
                "fallback_used": fallback_used,
                "final_fallback_reason": (
                    final_fallback_reason
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
                    "text_generation",
                ],
                "methodological_limitation": (
                    "Direct decode processes one padded or "
                    "trimmed 30-second window and does not "
                    "replicate transcribe() long-audio "
                    "segmentation or context propagation."
                ),
                "fallback_encoder_recompute_note": (
                    "whisper.decode() recomputes the encoder on "
                    "every call, unlike transcribe()'s internal "
                    "pipeline which reuses one encoder pass across "
                    "all fallback attempts. On files where "
                    "fallback_used is True, this adapter's latency "
                    "includes one additional full encoder pass per "
                    "extra temperature attempt, a cost specific to "
                    "this adapter's use of the low-level decode() "
                    "API rather than to the direct-decode technique "
                    "in general."
                ),
            },
        }

    def close(self) -> None:
        """Release Whisper model resources."""

        self._model = None

        gc.collect()

        torch.cuda.empty_cache()
