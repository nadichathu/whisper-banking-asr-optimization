import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import whisper
from silero_vad import (
    collect_chunks,
    get_speech_timestamps,
    load_silero_vad,
)

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperVadAdapter(BaseAdapter):
    """OpenAI Whisper with Silero VAD silence trimming. GPU-only.

    Experimental change relative to the FP32 Whisper baseline:

        Silero VAD removes detected non-speech regions before Whisper
        transcription.

    The end-to-end latency measurement includes:

        - Whisper-compatible audio loading and resampling
        - Silero VAD speech detection
        - Speech-chunk collection
        - Waveform conversion
        - Whisper transcription
        - Final text extraction

    Model-loading time is intentionally excluded from benchmark latency.
    The Silero VAD model itself runs on CPU by design (it is lightweight
    and this avoids competing with Whisper for GPU memory); Whisper
    inference within this adapter still requires and runs on CUDA.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.name = "whisper_vad"

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

        self.sample_rate = int(
            self.config.get(
                "sample_rate",
                16000,
            )
        )

        self.vad_threshold = float(
            self.config.get(
                "vad_threshold",
                0.5,
            )
        )

        self.min_speech_duration_ms = int(
            self.config.get(
                "min_speech_duration_ms",
                100,
            )
        )

        self.min_silence_duration_ms = int(
            self.config.get(
                "min_silence_duration_ms",
                100,
            )
        )

        self.speech_pad_ms = int(
            self.config.get(
                "speech_pad_ms",
                100,
            )
        )

        if self.sample_rate not in {8000, 16000}:
            raise ValueError(
                "Silero VAD supports sample rates of "
                "8000 Hz or 16000 Hz."
            )

        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError(
                "vad_threshold must be between 0.0 and 1.0."
            )

        if self.min_speech_duration_ms < 0:
            raise ValueError(
                "min_speech_duration_ms cannot be negative."
            )

        if self.min_silence_duration_ms < 0:
            raise ValueError(
                "min_silence_duration_ms cannot be negative."
            )

        if self.speech_pad_ms < 0:
            raise ValueError(
                "speech_pad_ms cannot be negative."
            )

        self._model = None
        self._vad_model = None

    def load(self):
        """Load Whisper and Silero VAD models."""

        if self._model is None:
            print(
                f"Loading Whisper {self.model_size} "
                f"VAD adapter on {self.device}"
            )

            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
            )

            self._model.eval()

        if self._vad_model is None:
            print(
                "Loading Silero VAD on CPU"
            )

            # Keep VAD on CPU so that it does not consume
            # Whisper's GPU memory.
            self._vad_model = load_silero_vad()

            if hasattr(self._vad_model, "eval"):
                self._vad_model.eval()

        return self._model

    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Transcribe an audio file after Silero VAD trimming."""

        if self._model is None or self._vad_model is None:
            raise RuntimeError(
                "Models are not loaded. "
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

        total_start = time.perf_counter()

        # -------------------------------------------------
        # Stage 1: Whisper-compatible audio loading
        # -------------------------------------------------
        audio_load_start = time.perf_counter()

        try:
            # whisper.load_audio() uses Whisper's standard
            # file-decoding, mono-conversion and 16 kHz
            # resampling path, matching the baseline.
            audio_data = whisper.load_audio(
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

        if audio_data.size == 0:
            raise ValueError(
                f"Audio file contains no samples: {audio_path}"
            )

        if not np.isfinite(audio_data).all():
            raise ValueError(
                f"Audio file contains invalid numeric values: "
                f"{audio_path}"
            )

        audio_data = np.asarray(
            audio_data,
            dtype=np.float32,
        )

        # VAD (Silero) runs on CPU by design; this tensor
        # intentionally stays on CPU for the VAD stage.
        waveform = torch.from_numpy(
            np.ascontiguousarray(audio_data)
        ).to(
            dtype=torch.float32,
            device="cpu",
        )

        original_samples = int(
            waveform.numel()
        )

        original_duration_s = (
            original_samples / self.sample_rate
        )

        # -------------------------------------------------
        # Stage 2: Silero VAD inference (CPU)
        # -------------------------------------------------
        vad_inference_start = time.perf_counter()

        with torch.inference_mode():
            speech_timestamps = get_speech_timestamps(
                waveform,
                self._vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.vad_threshold,
                min_speech_duration_ms=(
                    self.min_speech_duration_ms
                ),
                min_silence_duration_ms=(
                    self.min_silence_duration_ms
                ),
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=False,
            )

        vad_inference_ms = (
            time.perf_counter() - vad_inference_start
        ) * 1000

        # -------------------------------------------------
        # Stage 3: Collect detected speech chunks
        # -------------------------------------------------
        chunk_collection_start = time.perf_counter()

        if speech_timestamps:
            trimmed_waveform = collect_chunks(
                speech_timestamps,
                waveform,
            )

            speech_detected = True
            used_original_audio_fallback = False
        else:
            # Safe fallback: do not pass empty audio to
            # Whisper when Silero detects no speech.
            trimmed_waveform = waveform

            speech_detected = False
            used_original_audio_fallback = True

        if not isinstance(
            trimmed_waveform,
            torch.Tensor,
        ):
            trimmed_waveform = torch.as_tensor(
                trimmed_waveform,
                dtype=torch.float32,
            )

        trimmed_waveform = (
            trimmed_waveform
            .detach()
            .to(
                dtype=torch.float32,
                device="cpu",
            )
            .contiguous()
            .flatten()
        )

        chunk_collection_ms = (
            time.perf_counter() - chunk_collection_start
        ) * 1000

        trimmed_samples = int(
            trimmed_waveform.numel()
        )

        if trimmed_samples == 0:
            # Additional safety fallback in case chunk
            # collection unexpectedly returns an empty tensor.
            trimmed_waveform = waveform
            trimmed_samples = original_samples
            speech_detected = False
            used_original_audio_fallback = True

        trimmed_duration_s = (
            trimmed_samples / self.sample_rate
        )

        removed_duration_s = max(
            0.0,
            original_duration_s - trimmed_duration_s,
        )

        retained_audio_ratio = (
            trimmed_duration_s / original_duration_s
            if original_duration_s > 0
            else 0.0
        )

        removed_audio_ratio = (
            removed_duration_s / original_duration_s
            if original_duration_s > 0
            else 0.0
        )

        # Whisper accepts a NumPy float32 waveform.
        audio_array = (
            trimmed_waveform
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        vad_preprocessing_ms = (
            vad_inference_ms + chunk_collection_ms
        )

        # -------------------------------------------------
        # Stage 4: Whisper transcription (GPU)
        # -------------------------------------------------
        torch.cuda.synchronize(device=self.device)

        whisper_start = time.perf_counter()

        result = self._model.transcribe(
            audio_array,
            task="transcribe",
            language="en",

            # Keep precision aligned with the corrected
            # Whisper FP32 baseline.
            fp16=False,

            # Temperature and condition_on_previous_text
            # are intentionally left unspecified so Whisper
            # retains the baseline defaults.
        )

        torch.cuda.synchronize(device=self.device)

        whisper_latency_ms = (
            time.perf_counter() - whisper_start
        ) * 1000

        text = result.get(
            "text",
            "",
        ).strip()

        total_latency_ms = (
            time.perf_counter() - total_start
        ) * 1000

        real_time_factor = (
            total_latency_ms / 1000
        ) / original_duration_s if original_duration_s > 0 else None

        whisper_real_time_factor = (
            whisper_latency_ms / 1000
        ) / trimmed_duration_s if trimmed_duration_s > 0 else None

        return {
            "text": text,
            "latency_ms": total_latency_ms,
            "meta": {
                "implementation": "openai-whisper",
                "backend": "whisper",
                "optimization": (
                    "silero_vad_silence_trimming"
                ),
                "experimental_change": (
                    "Silero VAD preprocessing before "
                    "Whisper transcription"
                ),
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp32",
                "language": "en",
                "decoding": "default_transcribe",
                "condition_on_previous_text": True,
                "temperature_fallback_enabled": True,
                "vad_backend": "silero_vad",
                "vad_device": "cpu",
                "sample_rate": self.sample_rate,
                "vad_threshold": self.vad_threshold,
                "min_speech_duration_ms": (
                    self.min_speech_duration_ms
                ),
                "min_silence_duration_ms": (
                    self.min_silence_duration_ms
                ),
                "speech_pad_ms": self.speech_pad_ms,
                "speech_detected": speech_detected,
                "used_original_audio_fallback": (
                    used_original_audio_fallback
                ),
                "speech_segment_count": len(
                    speech_timestamps
                ),
                "original_samples": original_samples,
                "trimmed_samples": trimmed_samples,
                "original_duration_s": original_duration_s,
                "trimmed_duration_s": trimmed_duration_s,
                "removed_duration_s": removed_duration_s,
                "retained_audio_ratio": retained_audio_ratio,
                "removed_audio_ratio": removed_audio_ratio,
                "audio_load_ms": audio_load_ms,
                "vad_inference_ms": vad_inference_ms,
                "chunk_collection_ms": chunk_collection_ms,
                "vad_preprocessing_ms": vad_preprocessing_ms,
                "whisper_latency_ms": whisper_latency_ms,
                "real_time_factor": real_time_factor,
                "whisper_real_time_factor": (
                    whisper_real_time_factor
                ),
                "latency_type": "end_to_end",
                "model_loading_included": False,
                "latency_includes": [
                    "whisper_compatible_audio_loading",
                    "audio_decoding",
                    "mono_conversion",
                    "audio_resampling",
                    "silero_vad_inference",
                    "speech_chunk_collection",
                    "waveform_conversion",
                    "whisper_encoder",
                    "whisper_decoder",
                    "text_generation",
                ],
            },
        }

    def close(self) -> None:
        """Release Whisper and Silero VAD resources."""

        self._model = None
        self._vad_model = None

        gc.collect()

        torch.cuda.empty_cache()
