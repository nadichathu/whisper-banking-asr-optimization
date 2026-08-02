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
    """NVIDIA Parakeet ASR adapter using NVIDIA NeMo.

    Used as an external comparator model during the model-selection phase
    (alongside Wav2Vec2 and FastConformer), not as a subject of the Whisper
    inference-time optimisation experiments.

    NVIDIA's documentation states that NeMo transcription inputs must be
    mono and 16 kHz, and that preparing NumPy/tensor inputs into this
    format is the caller's responsibility (NeMo accepts file paths, NumPy
    arrays, and tensors). This adapter therefore performs explicit
    mono-conversion and resampling via torchaudio before calling
    transcribe(), rather than relying on undocumented internal behaviour
    when passed a raw file path directly.
    """

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)

        self.config = dict(config or {})

        self.model_id = self.config.get(
            "model_id",
            PARAKEET_MODEL_ID,
        )

        requested_device = str(
            self.config.get(
                "device",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        ).lower().strip()

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print(
                "CUDA was requested but is unavailable. "
                "Falling back to CPU."
            )
            requested_device = "cpu"

        self.device = requested_device

        self.batch_size = int(
            self.config.get("batch_size", 1)
        )

        self.model = None

        # Used by the runner for result filenames.
        self.name = (
            self.model_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

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

        print(f"Parakeet loaded on device: {self.device}")

    def _load_and_prepare_audio(
        self,
        audio_path: pathlib.Path,
    ):
        """Load audio, convert to mono, and resample to 16 kHz.

        Returns (waveform_1d_float32_numpy, original_sample_rate, duration_s).
        """

        try:
            waveform, original_sample_rate = torchaudio.load(str(audio_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio file '{audio_path}': {exc}"
            ) from exc

        if waveform.numel() == 0:
            raise ValueError(f"Audio file is empty: {audio_path}")

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
        duration_s = waveform.shape[-1] / self.TARGET_SAMPLE_RATE

        return waveform.numpy(), original_sample_rate, duration_s

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

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        start_time = time.perf_counter()

        waveform, original_sample_rate, duration_s = self._load_and_prepare_audio(audio_path)

        with torch.inference_mode():
            outputs = self.model.transcribe(
                audio=[waveform],
                batch_size=self.batch_size,
                verbose=False,
            )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        text = self._extract_text(outputs)

        real_time_factor = (
            (latency_ms / 1000) / duration_s if duration_s > 0 else None
        )

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "NVIDIA NeMo",
                "model_id": self.model_id,
                "architecture": "Parakeet TDT",
                "device": self.device,
                "batch_size": self.batch_size,
                "latency_type": "end_to_end",
                "model_loading_included": False,
                "original_sample_rate": original_sample_rate,
                "target_sample_rate": self.TARGET_SAMPLE_RATE,
                "audio_duration_s": duration_s,
                "real_time_factor": real_time_factor,
                "preprocessing_steps": [
                    "torchaudio_load",
                    "mono_conversion",
                    "resampling_to_16khz",
                ],
            },
        }

    @staticmethod
    def _extract_text(outputs: Any) -> str:
        """Handle different NeMo transcription return formats."""

        if outputs is None:
            return ""

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

        return str(first_output).strip()

    def close(self) -> None:
        """Release model and GPU memory."""

        used_cuda = self.device.startswith("cuda")

        if self.model is not None:
            del self.model
            self.model = None

        gc.collect()

        if used_cuda:
            torch.cuda.empty_cache()
