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
    """NVIDIA Parakeet FastConformer-TDT ASR adapter.

    This is a GPU-only external comparator used during the ASR
    model-selection phase. It is not a Whisper optimisation adapter.

    Audio preprocessing includes:

    - Loading with torchaudio
    - Validation
    - Multichannel-to-mono conversion
    - Resampling to 16 kHz
    - Conversion to a contiguous float32 NumPy array

    CUDA-graph decoding is disabled because the current RunPod
    CUDA-driver environment raises CUDA error 35 while NeMo attempts
    to compile the TDT CUDA graph.
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

        requested_device = str(
            self.config.get(
                "device",
                "cuda",
            )
        ).lower().strip()

        self.device = self._validate_device(
            requested_device
        )

        self.batch_size = int(
            self.config.get(
                "batch_size",
                1,
            )
        )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model = None
        self._parameter_count: Optional[int] = None
        self._cuda_graphs_disabled = False

        self.name = (
            self.model_id
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _validate_device(
        requested_device: str,
    ) -> str:
        """Validate the requested GPU device."""

        try:
            device = torch.device(
                requested_device
            )
        except (RuntimeError, ValueError) as exc:
            raise ValueError(
                f"Invalid device value: "
                f"'{requested_device}'."
            ) from exc

        if device.type != "cuda":
            raise ValueError(
                "ParakeetAdapter is GPU-only. "
                f"Received device='{requested_device}'. "
                "Use '--device cuda' or '--device cuda:0'."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible "
                "GPU is available."
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
                f"CUDA device index {device_index} is "
                f"unavailable. Detected "
                f"{torch.cuda.device_count()} CUDA device(s)."
            )

        return f"cuda:{device_index}"

    def load(self):
        """Load and prepare the pretrained Parakeet model."""

        if self.model is not None:
            return self.model

        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise ImportError(
                "NVIDIA NeMo is not installed. Install it with:\n"
                'pip install "nemo_toolkit[asr]"'
            ) from exc

        print(
            f"Loading Parakeet model: {self.model_id}"
        )

        self.model = (
            nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_id
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        # NeMo TDT decoding can attempt to compile CUDA graphs.
        # The current RunPod driver/runtime combination raises
        # cudaErrorInsufficientDriver during that compilation.
        # Disable CUDA graphs and use standard GPU decoding.
        if not hasattr(
            self.model,
            "disable_cuda_graphs",
        ):
            raise RuntimeError(
                "The installed NeMo model does not expose "
                "disable_cuda_graphs(). Check the installed "
                "nemo_toolkit version before benchmarking."
            )

        try:
            self.model.disable_cuda_graphs()
            self._cuda_graphs_disabled = True
        except Exception as exc:
            raise RuntimeError(
                "Failed to disable CUDA-graph decoding for "
                f"Parakeet model '{self.model_id}': {exc}"
            ) from exc

        # Confirm that the checkpoint expects the same sample
        # rate used by this adapter.
        try:
            configured_sample_rate = int(
                self.model.cfg.preprocessor.sample_rate
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                "Could not verify the Parakeet model's "
                "configured preprocessing sample rate."
            ) from exc

        if (
            configured_sample_rate
            != self.TARGET_SAMPLE_RATE
        ):
            raise RuntimeError(
                "Model preprocessing sample-rate mismatch: "
                f"adapter={self.TARGET_SAMPLE_RATE} Hz, "
                f"checkpoint={configured_sample_rate} Hz."
            )

        try:
            self._parameter_count = sum(
                parameter.numel()
                for parameter
                in self.model.parameters()
            )
        except Exception:
            self._parameter_count = None

        print(
            f"Parakeet loaded on device: {self.device}"
        )

        print(
            "Parakeet CUDA-graph decoding disabled; "
            "using standard GPU decoding."
        )

        return self.model

    def _load_and_prepare_audio(
        self,
        audio_path: pathlib.Path,
    ) -> tuple[np.ndarray, int, int, float]:
        """Load, validate, mono-convert and resample audio.

        Returns:
            waveform:
                Contiguous one-dimensional float32 NumPy array.
            original_sample_rate:
                Sample rate stored in the source audio file.
            original_channels:
                Channel count before mono conversion.
            duration_s:
                Duration calculated from the original audio.
        """

        try:
            (
                waveform,
                original_sample_rate,
            ) = torchaudio.load(
                str(audio_path)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio file "
                f"'{audio_path}': {exc}"
            ) from exc

        if waveform.ndim != 2:
            raise ValueError(
                "Expected waveform shape "
                "[channels, samples], but received "
                f"{tuple(waveform.shape)}."
            )

        if waveform.numel() == 0:
            raise ValueError(
                f"Audio file is empty: {audio_path}"
            )

        if original_sample_rate <= 0:
            raise ValueError(
                "Invalid source-audio sample rate: "
                f"{original_sample_rate}"
            )

        if not torch.isfinite(
            waveform
        ).all():
            raise ValueError(
                "Audio contains NaN or infinite values: "
                f"{audio_path}"
            )

        original_channels = int(
            waveform.shape[0]
        )

        original_num_samples = int(
            waveform.shape[-1]
        )

        duration_s = (
            original_num_samples
            / float(original_sample_rate)
        )

        if original_channels > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        if (
            original_sample_rate
            != self.TARGET_SAMPLE_RATE
        ):
            waveform = (
                torchaudio.functional.resample(
                    waveform,
                    orig_freq=original_sample_rate,
                    new_freq=self.TARGET_SAMPLE_RATE,
                )
            )

        waveform = (
            waveform
            .squeeze(0)
            .to(torch.float32)
        )

        waveform_numpy = np.ascontiguousarray(
            waveform.detach().cpu().numpy(),
            dtype=np.float32,
        )

        return (
            waveform_numpy,
            original_sample_rate,
            original_channels,
            duration_s,
        )

    def transcribe(
        self,
        audio_path: pathlib.Path,
    ) -> Dict[str, Any]:
        """Transcribe one file and measure end-to-end latency."""

        if self.model is None:
            raise RuntimeError(
                "Parakeet model is not loaded. "
                "Call load() before transcribe()."
            )

        audio_path = pathlib.Path(
            audio_path
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        if not audio_path.is_file():
            raise ValueError(
                f"Audio path is not a file: {audio_path}"
            )

        torch.cuda.synchronize(
            device=self.device
        )

        start_time = time.perf_counter()

        (
            waveform,
            original_sample_rate,
            original_channels,
            duration_s,
        ) = self._load_and_prepare_audio(
            audio_path
        )

        with torch.inference_mode():
            outputs = self.model.transcribe(
                audio=[waveform],
                batch_size=self.batch_size,
                verbose=False,
            )

        torch.cuda.synchronize(
            device=self.device
        )

        # Include final text extraction in the timed region.
        text = self._extract_text(
            outputs
        )

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        real_time_factor = (
            (latency_ms / 1000.0) / duration_s
            if duration_s > 0
            else None
        )

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "nemo",
                "backend": "pytorch",
                "model_id": self.model_id,
                "model_class": (
                    type(self.model).__name__
                ),
                "architecture": "FastConformer-TDT",
                "decoder": "TDT",
                "language": "en",
                "device": self.device,
                "parameter_count": (
                    self._parameter_count
                ),
                "batch_size": self.batch_size,
                "cuda_graphs_disabled": (
                    self._cuda_graphs_disabled
                ),
                "decoding_execution": (
                    "standard_gpu_without_cuda_graphs"
                ),
                "latency_type": "end_to_end",
                "model_loading_included": False,
                "original_channels": (
                    original_channels
                ),
                "processed_channels": 1,
                "original_sample_rate": (
                    original_sample_rate
                ),
                "target_sample_rate": (
                    self.TARGET_SAMPLE_RATE
                ),
                "audio_duration_s": duration_s,
                "real_time_factor": (
                    real_time_factor
                ),
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
                    "standard_gpu_tdt_decoding",
                    "text_extraction",
                ],
                "environment_note": (
                    "CUDA-graph decoding was disabled "
                    "because CUDA graph compilation failed "
                    "with cudaErrorInsufficientDriver in "
                    "the RunPod environment."
                ),
            },
        }

    @staticmethod
    def _extract_text(
        outputs: Any,
    ) -> str:
        """Extract text from supported NeMo output formats."""

        if outputs is None:
            return ""

        if (
            isinstance(outputs, tuple)
            and len(outputs) >= 1
            and isinstance(
                outputs[0],
                (list, tuple),
            )
        ):
            outputs = outputs[0]

        if not isinstance(
            outputs,
            (list, tuple),
        ):
            outputs = [outputs]

        if not outputs:
            return ""

        first_output = outputs[0]

        if hasattr(
            first_output,
            "text",
        ):
            return str(
                first_output.text or ""
            ).strip()

        if isinstance(
            first_output,
            str,
        ):
            return first_output.strip()

        if isinstance(
            first_output,
            dict,
        ):
            return str(
                first_output.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

        raise TypeError(
            "Unsupported NeMo transcription output type: "
            f"{type(first_output).__name__}"
        )

    def close(self) -> None:
        """Release the model and GPU memory."""

        if self.model is not None:
            del self.model
            self.model = None

        self._parameter_count = None
        self._cuda_graphs_disabled = False

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
