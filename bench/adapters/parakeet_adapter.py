from __future__ import annotations

import gc
import pathlib
import time
from typing import Any, Dict, Optional

import torch

from .base_adapter import BaseAdapter


class ParakeetAdapter(BaseAdapter):
    """NVIDIA Parakeet ASR adapter using NVIDIA NeMo.

    Used as an external comparator model during the model-selection phase
    (alongside Wav2Vec2 and Vosk), not as a subject of the Whisper
    inference-time optimisation experiments.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)

        self.config = dict(config or {})

        self.model_id = self.config.get(
            "model_id",
            "nvidia/parakeet-tdt-0.6b-v2",
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
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.transcribe(
                audio=[str(audio_path)],
                batch_size=self.batch_size,
                verbose=False,
            )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        text = self._extract_text(outputs)

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