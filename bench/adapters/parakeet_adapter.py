from __future__ import annotations

import gc
import pathlib
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio

from .base_adapter import BaseAdapter
from bench.config import PARAKEET_MODEL_ID


class ParakeetAdapter(BaseAdapter):
    """NVIDIA Parakeet (FastConformer-TDT) ASR adapter using NVIDIA NeMo.
    GPU-only.

    Used as an external comparator model during the model-selection phase
    (alongside Wav2Vec2 and NeMoFastConformerAdapter), not as a subject of
    the Whisper inference-time optimisation experiments.

    NVIDIA's documentation states that NeMo transcription inputs must be
    mono and 16 kHz, and that preparing NumPy/tensor inputs into this
    format is the caller's responsibility. This adapter performs explicit
    mono-conversion and resampling via torchaudio before calling
    transcribe(), and cross-checks the loaded checkpoint's own configured
    preprocessor sample rate against the target rate in load().
    """

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)

        self.model_id = self.config.get(
            "model_id",
            PARAKEET_MODEL_ID,
        )

        requested_device_str = str(
            self.config.get("device", "cuda")
        ).lower().strip()

        self.device = self._validate_device(requested_device_str)

        self.batch_size = int(
            self.config.get("batch_size", 1)
        )

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.model = None
        self._parameter_count: Optional[int] = None

        # Used by the runner for result filenames.
        self.name = (
            self.model_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _validate_device(requested_device: str) -> str:
        """Validate the requested device. GPU-only: no CPU fallback."""

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

        device = torch.device(requested_device)

        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable. "
                f"Detected {torch.cuda.device_count()} CUDA device(s)."
            )

        return str(device)

    def load(self) -> None:
        """Load the pretrained Parakeet model."""

        if self.model is not None:
            return

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ImportError(
                "NVIDIA NeMo is not installed. Install it with:\n"
                'pip install "nemo_toolkit[asr]"'
            ) from exc

        print(f"Loading Parakeet model: {self.model_id}")

        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_id
        )

        self.model.eval()

        # Use .to(device) rather than .cuda()/.cpu() so a specific CUDA
        # index (e.g. "cuda:1") is respected, matching the device-selection
        # contract used by the rest of the adapter suite.
        self.model = self.model.to(self.device)

        # Cross-check the loaded checkpoint's own configured preprocessor
        # sample rate against TARGET_SAMPLE_RATE. Wrapped defensively:
        # the exact config path can vary slightly across NeMo model
        # classes/versions, and a missing attribute here should not crash
        # load() outright -- it should be reported, not silently ignored.
        try:
            configured_sample_rate = int(self.model.cfg.preprocessor.sample_rate)
            if configured_sample_rate != self.TARGET_SAMPLE_RATE:
                raise RuntimeError(
                    "Model preprocessing sample rate mismatch: "
                    f"adapter targets {self.TARGET_SAMPLE_RATE} Hz, "
                    f"but the loaded checkpoint's preprocessor is "
                    f"configured for {configured_sample_rate} Hz. "
                    "Update TARGET_SAMPLE_RATE or use a different checkpoint."
                )
        except AttributeError:
            print(
                "WARNING: could not read model.cfg.preprocessor.sample_rate "
                "to verify the checkpoint's expected sample rate; proceeding "
                f"with the configured TARGET_SAMPLE_RATE={self.TARGET_SAMPLE_RATE}."
            )

        # Parameter count computed once here, not per-transcribe() call,
        # so it never adds overhead inside the timed region.
        try:
            self._parameter_count = sum(p.numel() for p in self.model.parameters())
        except Exception:
            self._parameter_count = None

        print(f"Parakeet loaded on device: {self.device}")

    def _load_and_prepare_audio(
        self,
        audio_path: pathlib.Path,
    ):
        """Load audio, convert to mono, and resample to 16 kHz.

        Returns (waveform_1d_float32_numpy, original_sample_rate,
        original_channels, duration_s), where duration_s is computed from
        the ORIGINAL (pre-resample) sample count, since resampling can
        introduce small rounding differences in sample count.
        """

        try:
            waveform, original_sample_rate = torchaudio.load(str(audio_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio file '{audio_path}': {exc}"
            ) from exc

        if waveform.ndim != 2:
            raise ValueError(
                "Expected waveform with shape [channels, samples], "
                f"received {tuple(waveform.shape)}."
            )

        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

        if original_sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {original_sample_rate}")

        if not torch.isfinite(waveform).all():
            raise ValueError(
                f"Audio contains NaN or infinite values: {audio_path}"
            )

        original_channels = int(waveform.shape[0])
        original_num_samples = int(waveform.shape[-1])
        duration_s = original_num_samples / float(original_sample_rate)

        # Convert multi-channel audio to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to the rate NeMo expects.
        if original_sample_rate != self.TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=self.TARGET_SAMPLE_RATE,
            )

        waveform = waveform.squeeze(0).to(torch.float32)
        waveform_np = np.ascontiguousarray(
            waveform.detach().cpu().numpy(),
            dtype=np.float32,
        )

        return waveform_np, original_sample_rate, original_channels, duration_s

    def transcribe(
        self,
        audio_path: pathlib.Path,
    ) -> Dict[str, Any]:
        """Transcribe one audio file and measure end-to-end latency."""

        if self.model is None:
            raise RuntimeError(
                "Parakeet model has not been loaded. "
                "Call load() before transcribe()."
            )

        audio_path = pathlib.Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        if not audio_path.is_file():
            raise ValueError(f"Audio path is not a file: {audio_path}")

        torch.cuda.synchronize(device=self.device)

        start_time = time.perf_counter()

        waveform, original_sample_rate, original_channels, duration_s = (
            self._load_and_prepare_audio(audio_path)
        )

        with torch.inference_mode():
            outputs = self.model.transcribe(
                audio=[waveform],
                batch_size=self.batch_size,
                verbose=False,
            )

        torch.cuda.synchronize(device=self.device)

        # Text extraction included inside the timed region, so latency
        # reflects file-to-final-text, not file-to-raw-model-output.
        text = self._extract_text(outputs)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        real_time_factor = (
            (latency_ms / 1000) / duration_s if duration_s > 0 else None
        )

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "nemo",
                "backend": "pytorch",
                "model_id": self.model_id,
                "model_class": type(self.model).__name__,
                "architecture": "FastConformer-TDT",
                "decoder": "TDT",
                "language": "en",
                "device": self.device,
                "parameter_count": self._parameter_count,
                "batch_size": self.batch_size,
                "latency_type": "end_to_end",
                "model_loading_included": False,
                "original_channels": original_channels,
                "processed_channels": 1,
                "original_sample_rate": original_sample_rate,
                "target_sample_rate": self.TARGET_SAMPLE_RATE,
                "audio_duration_s": duration_s,
                "real_time_factor": real_time_factor,
                "preprocessing_steps": [
                    "torchaudio_load",
                    "mono_conversion",
                    "resampling_to_16khz",
                    "contiguous_numpy_conversion",
                ],
                "latency_includes": [
                    "audio_loading",
                    "mono_conversion",
                    "resampling",
                    "numpy_conversion",
                    "feature_extraction",
                    "encoder",
                    "decoder",
                    "text_extraction",
                ],
            },
        }

    @staticmethod
    def _extract_text(outputs: Any) -> str:
        """Handle different NeMo transcription return formats.

        Some NeMo model/version combinations return a bare list of
        hypotheses; others return a tuple whose first element is that
        list (e.g. (hypotheses, other_metadata)). Unwrap the tuple case
        before proceeding with the usual list handling.
        """

        if outputs is None:
            return ""

        if (
            isinstance(outputs, tuple)
            and len(outputs) >= 1
            and isinstance(outputs[0], (list, tuple))
        ):
            outputs = outputs[0]

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        if not outputs:
            return ""

        first_output = outputs[0]

        # Current NeMo API normally returns a hypothesis object.
        if hasattr(first_output, "text"):
            return str(first_output.text or "").strip()

        # Some NeMo versions may return plain strings.
        if isinstance(first_output, str):
            return first_output.strip()

        # Defensive support for dictionary-like output.
        if isinstance(first_output, dict):
            return str(first_output.get("text", "") or "").strip()

        # Do not silently stringify an unrecognised object -- that could
        # store a Python repr (e.g. "<...Hypothesis object at 0x...>")
        # into the transcript field and corrupt downstream WER scoring
        # without any visible error.
        raise TypeError(
            "Unsupported NeMo transcription output type: "
            f"{type(first_output).__name__}"
        )

    def close(self) -> None:
        """Release model and GPU memory."""

        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()

        torch.cuda.empty_cache()
