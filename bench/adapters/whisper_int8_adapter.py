import gc
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from faster_whisper import WhisperModel

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperINT8Adapter(BaseAdapter):
    """Faster-Whisper using CTranslate2 INT8 quantisation.

    This adapter does not represent INT8 quantisation within the original
    OpenAI Whisper/PyTorch implementation. It evaluates INT8 quantisation
    within the Faster-Whisper/CTranslate2 backend.

    The appropriate isolated comparison is therefore:

        Faster-Whisper FP32
        versus
        Faster-Whisper INT8

    The adapter intentionally preserves the Faster-Whisper baseline
    transcription settings. The only intended experimental difference
    between those two Faster-Whisper configurations is ``compute_type``.

    Default compute types:

        CPU:
            int8

        CUDA:
            int8_float16

    Reported latency is end-to-end and includes:

        - audio loading and decoding
        - resampling
        - feature extraction
        - encoder inference
        - decoder inference
        - temperature fallback attempts
        - segment generation
        - final text assembly

    Model-loading time is excluded.
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

        self.config = config or {}
        self.name = "whisper_int8"

        self.model_size = str(
            self.config.get(
                "model_size",
                MODEL_SIZE,
            )
        )

        requested_device = str(
            self.config.get(
                "device",
                DEVICE,
            )
        ).lower()

        (
            self.device,
            self.device_index,
        ) = self._resolve_device(requested_device)

        if self.device == "cuda":
            default_compute_type = "int8_float16"
            default_cpu_threads = 0
        else:
            default_compute_type = "int8"
            default_cpu_threads = max(
                1,
                os.cpu_count() or 1,
            )

        self.compute_type = str(
            self.config.get(
                "compute_type",
                default_compute_type,
            )
        ).lower()

        self._validate_compute_type()

        self.cpu_threads = int(
            self.config.get(
                "cpu_threads",
                default_cpu_threads,
            )
        )

        self.num_workers = int(
            self.config.get(
                "num_workers",
                1,
            )
        )

        if self.cpu_threads < 0:
            raise ValueError(
                "cpu_threads cannot be negative."
            )

        if self.num_workers <= 0:
            raise ValueError(
                "num_workers must be greater than zero."
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

        self.beam_size = int(
            self.config.get(
                "beam_size",
                5,
            )
        )

        self.best_of = int(
            self.config.get(
                "best_of",
                5,
            )
        )

        if self.beam_size <= 0:
            raise ValueError(
                "beam_size must be greater than zero."
            )

        if self.best_of <= 0:
            raise ValueError(
                "best_of must be greater than zero."
            )

        self.compression_ratio_threshold = (
            self._optional_float(
                self.config.get(
                    "compression_ratio_threshold",
                    2.4,
                )
            )
        )

        self.log_prob_threshold = (
            self._optional_float(
                self.config.get(
                    "log_prob_threshold",
                    -1.0,
                )
            )
        )

        self.no_speech_threshold = (
            self._optional_float(
                self.config.get(
                    "no_speech_threshold",
                    0.6,
                )
            )
        )

        self.condition_on_previous_text = bool(
            self.config.get(
                "condition_on_previous_text",
                True,
            )
        )

        self.without_timestamps = bool(
            self.config.get(
                "without_timestamps",
                False,
            )
        )

        self.word_timestamps = bool(
            self.config.get(
                "word_timestamps",
                False,
            )
        )

        self.vad_filter = bool(
            self.config.get(
                "vad_filter",
                False,
            )
        )

        self.download_root = self.config.get(
            "download_root"
        )

        self.local_files_only = bool(
            self.config.get(
                "local_files_only",
                False,
            )
        )

        self._model: Optional[WhisperModel] = None

    @staticmethod
    def _optional_float(
        value: Any,
    ) -> Optional[float]:
        """Convert a configuration value to an optional float."""

        if value is None:
            return None

        return float(value)

    @staticmethod
    def _resolve_device(
        requested_device: str,
    ) -> Tuple[str, Union[int, list[int]]]:
        """Resolve CPU/CUDA device and optional CUDA index.

        Faster-Whisper expects ``device="cuda"`` together with a separate
        ``device_index`` rather than a PyTorch-style value such as
        ``device="cuda:0"``.
        """

        if requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(
                    "CUDA was requested but is unavailable. "
                    "Falling back to CPU."
                )
                return "cpu", 0

            if ":" in requested_device:
                index_text = requested_device.split(
                    ":",
                    maxsplit=1,
                )[1]

                try:
                    device_index = int(index_text)
                except ValueError as exc:
                    raise ValueError(
                        "Invalid CUDA device value: "
                        f"{requested_device}. Expected a value "
                        "such as 'cuda' or 'cuda:0'."
                    ) from exc
            else:
                device_index = 0

            if device_index < 0:
                raise ValueError(
                    "CUDA device index cannot be negative."
                )

            if device_index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {device_index} is unavailable. "
                    f"Detected {torch.cuda.device_count()} CUDA device(s)."
                )

            return "cuda", device_index

        if requested_device == "cpu":
            return "cpu", 0

        raise ValueError(
            f"Unsupported device: {requested_device}. "
            "Use 'cpu', 'cuda', or an indexed CUDA device "
            "such as 'cuda:0'."
        )

    def _validate_compute_type(self) -> None:
        """Ensure that this adapter remains an INT8 experiment."""

        valid_int8_compute_types = {
            "int8",
            "int8_float32",
            "int8_float16",
            "int8_bfloat16",
        }

        if self.compute_type not in valid_int8_compute_types:
            raise ValueError(
                "WhisperINT8Adapter requires an INT8 CTranslate2 "
                "compute type. Supported values are: "
                f"{sorted(valid_int8_compute_types)}. "
                f"Received: {self.compute_type}"
            )

        if (
            self.device == "cpu"
            and self.compute_type
            in {
                "int8_float16",
                "int8_bfloat16",
            }
        ):
            raise ValueError(
                f"compute_type='{self.compute_type}' is not an "
                "appropriate explicit CPU configuration for this "
                "benchmark. Use 'int8' or 'int8_float32' on CPU."
            )

    def load(self) -> WhisperModel:
        """Load the Faster-Whisper CTranslate2 model."""

        if self._model is None:
            device_description = self.device

            if self.device == "cuda":
                device_description = (
                    f"{self.device}:{self.device_index}"
                )

            print(
                f"Loading Faster-Whisper {self.model_size} "
                f"INT8 adapter on {device_description} "
                f"with compute_type={self.compute_type}"
            )

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
                download_root=self.download_root,
                local_files_only=self.local_files_only,
            )

        return self._model

    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Transcribe one file using Faster-Whisper INT8 inference."""

        if self._model is None:
            raise RuntimeError(
                "Faster-Whisper model is not loaded. "
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

        if self.device == "cuda":
            torch.cuda.synchronize(
                self.device_index
            )

        start_time = time.perf_counter()

        segments_generator, info = self._model.transcribe(
            str(audio_path),
            language="en",
            task="transcribe",

            # Preserve Faster-Whisper baseline decoding.
            beam_size=self.beam_size,
            best_of=self.best_of,
            temperature=self.temperatures,
            compression_ratio_threshold=(
                self.compression_ratio_threshold
            ),
            log_prob_threshold=(
                self.log_prob_threshold
            ),
            no_speech_threshold=(
                self.no_speech_threshold
            ),
            condition_on_previous_text=(
                self.condition_on_previous_text
            ),

            # Keep timestamp behaviour aligned with the
            # Faster-Whisper baseline.
            without_timestamps=(
                self.without_timestamps
            ),
            word_timestamps=(
                self.word_timestamps
            ),

            # VAD is not part of the INT8 experiment.
            vad_filter=self.vad_filter,
        )

        # Faster-Whisper performs transcription lazily.
        # Materialising the generator is required before stopping
        # the timer; otherwise inference latency is under-measured.
        segments = list(
            segments_generator
        )

        if self.device == "cuda":
            torch.cuda.synchronize(
                self.device_index
            )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text
            and segment.text.strip()
        ).strip()

        generated_token_count = sum(
            len(segment.tokens)
            for segment in segments
            if segment.tokens is not None
        )

        temperatures_used = sorted(
            {
                float(segment.temperature)
                for segment in segments
                if segment.temperature is not None
            }
        )

        fallback_used = any(
            temperature
            != self.temperatures[0]
            for temperature in temperatures_used
        )

        detected_language = getattr(
            info,
            "language",
            None,
        )

        language_probability = getattr(
            info,
            "language_probability",
            None,
        )

        audio_duration = getattr(
            info,
            "duration",
            None,
        )

        duration_after_vad = getattr(
            info,
            "duration_after_vad",
            None,
        )

        real_time_factor = None

        if (
            audio_duration is not None
            and float(audio_duration) > 0
        ):
            real_time_factor = (
                latency_ms / 1000
            ) / float(audio_duration)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "faster-whisper",
                "backend": "CTranslate2",
                "optimization": (
                    "faster_whisper_int8_quantization"
                ),
                "experimental_change": (
                    f"compute_type={self.compute_type} "
                    "relative to a matching Faster-Whisper "
                    "non-quantized baseline"
                ),
                "model_size": self.model_size,
                "device": self.device,
                "device_index": (
                    self.device_index
                    if self.device == "cuda"
                    else None
                ),
                "compute_type": self.compute_type,
                "quantization": "int8",
                "quantization_backend": "CTranslate2",
                "comparison_baseline": (
                    "matching_faster_whisper_fp32_or_fp16"
                ),
                "not_directly_isolated_against": (
                    "openai_whisper_pytorch"
                ),
                "language": "en",
                "beam_size": self.beam_size,
                "best_of": self.best_of,
                "configured_temperature_sequence": list(
                    self.temperatures
                ),
                "temperatures_used": temperatures_used,
                "fallback_enabled": (
                    len(self.temperatures) > 1
                ),
                "fallback_used": fallback_used,
                "compression_ratio_threshold": (
                    self.compression_ratio_threshold
                ),
                "log_prob_threshold": (
                    self.log_prob_threshold
                ),
                "no_speech_threshold": (
                    self.no_speech_threshold
                ),
                "condition_on_previous_text": (
                    self.condition_on_previous_text
                ),
                "without_timestamps": (
                    self.without_timestamps
                ),
                "word_timestamps": (
                    self.word_timestamps
                ),
                "vad_filter": self.vad_filter,
                "greedy_decoding": (
                    self.beam_size == 1
                    and self.best_of == 1
                    and len(self.temperatures) == 1
                    and self.temperatures[0] == 0.0
                ),
                "segment_count": len(segments),
                "generated_token_count": (
                    generated_token_count
                ),
                "detected_language": (
                    detected_language
                ),
                "language_probability": (
                    float(language_probability)
                    if language_probability is not None
                    else None
                ),
                "audio_duration_s": (
                    float(audio_duration)
                    if audio_duration is not None
                    else None
                ),
                "duration_after_vad_s": (
                    float(duration_after_vad)
                    if duration_after_vad is not None
                    else None
                ),
                "real_time_factor": real_time_factor,
                "cpu_threads": self.cpu_threads,
                "num_workers": self.num_workers,
                "latency_type": "end_to_end",
                "latency_includes": [
                    "audio_loading",
                    "audio_decoding",
                    "audio_resampling",
                    "feature_extraction",
                    "ctranslate2_encoder",
                    "ctranslate2_int8_decoder",
                    "temperature_fallback_attempts",
                    "segment_generation",
                    "text_assembly",
                ],
                "methodological_interpretation": (
                    "This result isolates INT8 only when compared "
                    "with a Faster-Whisper adapter using identical "
                    "decoding settings and a non-INT8 compute type."
                ),
            },
        }

    def close(self) -> None:
        """Release Faster-Whisper and CTranslate2 resources."""

        used_cuda = self.device == "cuda"

        self._model = None

        gc.collect()

        if used_cuda:
            torch.cuda.empty_cache()