import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from .base_adapter import BaseAdapter


class NeMoFastConformerAdapter(BaseAdapter):
    """Adapter for NVIDIA NeMo FastConformer ASR models.

    Used as an external comparator model during the model-selection phase,
    alongside Wav2Vec2 and Parakeet, not as a subject of the Whisper
    inference-time optimisation experiments.

    Unlike an earlier version of this adapter, this one REQUIRES an
    explicit model_name and does not silently fall back across a list of
    candidate architectures (e.g. FastConformer -> QuartzNet). Reporting
    "FastConformer" results that were actually produced by a different
    architecture, chosen implicitly based on what happened to load
    successfully in a given environment, would be a reproducibility
    problem for the dissertation. Pass model_name explicitly, e.g.:

        NeMoFastConformerAdapter({"model_name": "stt_en_fastconformer_base"})
    """

    DEFAULT_MODEL_NAME = "stt_en_fastconformer_base"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.config = config or {}
        self.model_name = self.config.get("model_name", self.DEFAULT_MODEL_NAME)
        self.name = "nemo_fastconformer"

        requested_device = str(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        ).lower().strip()

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA was requested but is unavailable. Falling back to CPU.")
            requested_device = "cpu"

        self.device = requested_device
        self.batch_size = int(self.config.get("batch_size", 1))
        self.model = None

    def load(self) -> None:
        """Load the specified pretrained NeMo FastConformer model.

        Fails loudly (RuntimeError) if the exact requested model_name
        cannot be loaded, rather than silently substituting a different
        architecture.
        """

        if self.model is not None:
            return

        try:
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise ImportError(
                "NVIDIA NeMo is not installed. Install it with:\n"
                'pip install "nemo_toolkit[asr]"'
            ) from exc

        print(f"Loading NeMo model: {self.model_name}")

        try:
            self.model = ASRModel.from_pretrained(model_name=self.model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load NeMo model '{self.model_name}'. "
                "Check that this exact model name is available for your "
                "installed NeMo version, and that model_name is set "
                "explicitly in config rather than relying on a default."
            ) from exc

        self.model.eval()
        self.model = self.model.to(self.device)

        print(f"NeMo FastConformer loaded on device: {self.device}")

    def transcribe(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        """Transcribe one audio file and measure end-to-end latency."""

        if self.model is None:
            raise RuntimeError(
                "NeMo model has not been loaded. Call load() before transcribe()."
            )

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

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

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        text = self._extract_text(outputs)

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "NVIDIA NeMo",
                "model_name": self.model_name,
                "architecture": "FastConformer",
                "device": self.device,
                "batch_size": self.batch_size,
                "latency_type": "end_to_end",
            },
        }

    @staticmethod
    def _extract_text(outputs: Any) -> str:
        """Handle different NeMo transcription return formats.

        NeMo's transcribe() may return plain strings, Hypothesis objects
        with a .text attribute, or dict-like results depending on version
        and model type. Do not assume a bare string -- that would silently
        store a non-string object and corrupt downstream WER computation.
        """

        if outputs is None:
            return ""

        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        if not outputs:
            return ""

        first_output = outputs[0]

        if hasattr(first_output, "text"):
            return str(first_output.text or "").strip()

        if isinstance(first_output, str):
            return first_output.strip()

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
