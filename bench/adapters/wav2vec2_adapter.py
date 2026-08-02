import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .base_adapter import BaseAdapter
from bench.config import WAV2VEC_MODEL_ID


class Wav2Vec2Adapter(BaseAdapter):
    """Wav2Vec2 CTC adapter for end-to-end ASR benchmarking.

    The reported latency includes:

    - Audio-file loading
    - Stereo-to-mono conversion
    - Resampling
    - Processor feature preparation
    - Model inference
    - CTC greedy decoding
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.config = config or {}
        self.name = "wav2vec2"

        self.model_id = self.config.get(
            "model_id",
            WAV2VEC_MODEL_ID,
        )

        requested_device = str(
            self.config.get("device", "cuda")
        ).lower()

        if requested_device.startswith("cuda"):
            if torch.cuda.is_available():
                self.device = requested_device
            else:
                print(
                    "CUDA was requested for Wav2Vec2, "
                    "but CUDA is unavailable. Falling back to CPU."
                )
                self.device = "cpu"
        else:
            self.device = "cpu"

        self._processor = None
        self._model = None
        self.sample_rate = None

    def load(self):
        """Load the Wav2Vec2 processor and model."""

        if self._processor is None:
            print(
                f"Loading Wav2Vec2 processor: {self.model_id}"
            )

            self._processor = Wav2Vec2Processor.from_pretrained(
                self.model_id
            )

        if self._model is None:
            print(
                f"Loading Wav2Vec2 model: {self.model_id} "
                f"on {self.device}"
            )

            self._model = Wav2Vec2ForCTC.from_pretrained(
                self.model_id
            )

            self._model.to(self.device)
            self._model.eval()

        self.sample_rate = int(
            self._processor.feature_extractor.sampling_rate
        )

        return self._model

    def _load_audio(
        self,
        audio_path: Union[str, Path],
    ) -> torch.Tensor:
        """Load, convert to mono and resample an audio file."""

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        try:
            waveform, original_sample_rate = torchaudio.load(
                str(audio_path)
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio file "
                f"'{audio_path}': {exc}"
            ) from exc

        if waveform.numel() == 0:
            raise ValueError(
                f"Audio file is empty: {audio_path}"
            )

        # Convert multi-channel audio to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        # Resample to the rate required by Wav2Vec2.
        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=self.sample_rate,
            )

        # Convert [1, samples] into [samples].
        waveform = waveform.squeeze(0)

        # Ensure floating-point audio.
        waveform = waveform.to(torch.float32)

        return waveform

    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Transcribe an audio file using CTC greedy decoding."""

        if self._model is None or self._processor is None:
            raise RuntimeError(
                "Wav2Vec2 is not loaded. "
                "Call load() before transcribe()."
            )

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        start_time = time.perf_counter()

        # Audio loading, mono conversion and resampling are
        # deliberately included in end-to-end latency.
        waveform = self._load_audio(audio_path)

        audio_duration_s = (
            waveform.shape[-1] / self.sample_rate
        )

        # Hugging Face processors operate on CPU waveform data.
        audio_array = waveform.cpu().numpy()

        processed_inputs = self._processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        model_inputs = {
            key: value.to(self.device)
            for key, value in processed_inputs.items()
            if isinstance(value, torch.Tensor)
        }

        with torch.inference_mode():
            outputs = self._model(**model_inputs)
            logits = outputs.logits

            # CTC greedy decoding.
            predicted_ids = torch.argmax(
                logits,
                dim=-1,
            )

        predicted_ids_cpu = predicted_ids.detach().cpu()

        text = self._processor.batch_decode(
            predicted_ids_cpu
        )[0].strip()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        real_time_factor = (
            latency_ms / 1000
        ) / audio_duration_s if audio_duration_s > 0 else None

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "huggingface-transformers",
                "backend": "wav2vec2",
                "model_id": self.model_id,
                "device": self.device,
                "precision": "fp32",
                "sample_rate": self.sample_rate,
                "audio_duration_s": audio_duration_s,
                "real_time_factor": real_time_factor,
                "decoding": "ctc_greedy",
                "latency_type": "end_to_end",
                "latency_includes": [
                    "audio_loading",
                    "mono_conversion",
                    "resampling",
                    "processor_preparation",
                    "model_inference",
                    "ctc_decoding",
                ],
            },
        }

    def close(self) -> None:
        """Release model and processor resources."""

        used_cuda = self.device.startswith("cuda")

        self._model = None
        self._processor = None

        gc.collect()

        if used_cuda:
            torch.cuda.empty_cache()
