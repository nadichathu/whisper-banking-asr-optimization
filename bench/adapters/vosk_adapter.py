import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torchaudio
from vosk import KaldiRecognizer, Model

from .base_adapter import BaseAdapter


class VoskAdapter(BaseAdapter):
    """Vosk adapter for end-to-end ASR benchmarking.

    Reported latency includes:

    - Audio-file loading
    - Stereo-to-mono conversion
    - Resampling
    - Conversion to 16-bit PCM
    - Chunked Vosk recognition
    - Final JSON decoding
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.config = config or {}
        self.name = "vosk"

        self.model_path = Path(
            self.config.get(
                "model_path",
                "models/vosk-model-small-en-us-0.15",
            )
        )

        self.sample_rate = int(
            self.config.get(
                "sample_rate",
                16000,
            )
        )

        self.frame_chunk_size = int(
            self.config.get(
                "frame_chunk_size",
                4000,
            )
        )

        if self.sample_rate <= 0:
            raise ValueError(
                "sample_rate must be greater than zero."
            )

        if self.frame_chunk_size <= 0:
            raise ValueError(
                "frame_chunk_size must be greater than zero."
            )

        self.device = "cpu"
        self._model = None

    def load(self):
        """Load the Vosk model."""

        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    "Vosk model directory was not found: "
                    f"{self.model_path.resolve()}"
                )

            print(
                f"Loading Vosk model from: "
                f"{self.model_path}"
            )

            self._model = Model(
                str(self.model_path)
            )

        return self._model

    def _load_audio(
        self,
        audio_path: Union[str, Path],
    ) -> tuple[torch.Tensor, int]:
        """Load audio, convert it to mono and resample it."""

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

        if waveform.ndim != 2:
            raise ValueError(
                "Expected waveform shape [channels, samples], "
                f"but received {tuple(waveform.shape)}."
            )

        # Convert multi-channel audio to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(
                dim=0,
                keepdim=True,
            )

        # Resample to the rate expected by the Vosk model.
        if original_sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=original_sample_rate,
                new_freq=self.sample_rate,
            )

        waveform = waveform.squeeze(0)
        waveform = waveform.to(torch.float32)

        return waveform, original_sample_rate

    @staticmethod
    def _waveform_to_pcm16(
        waveform: torch.Tensor,
    ) -> bytes:
        """Convert a float waveform in [-1, 1] to PCM signed 16-bit bytes."""

        waveform = waveform.detach().cpu()

        waveform = torch.clamp(
            waveform,
            min=-1.0,
            max=1.0,
        )

        pcm16 = (
            waveform.numpy() * 32767.0
        ).astype(np.int16)

        return pcm16.tobytes()

    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Transcribe an audio file using Vosk streaming recognition."""

        if self._model is None:
            raise RuntimeError(
                "Vosk model is not loaded. "
                "Call load() before transcribe()."
            )

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # Start timing before audio loading so this measures
        # complete file-to-text end-to-end latency.
        start_time = time.perf_counter()

        waveform, original_sample_rate = self._load_audio(
            audio_path
        )

        audio_duration_s = (
            waveform.shape[-1] / self.sample_rate
        )

        pcm_bytes = self._waveform_to_pcm16(
            waveform
        )

        recognizer = KaldiRecognizer(
            self._model,
            self.sample_rate,
        )

        byte_chunk_size = (
            self.frame_chunk_size * 2
        )

        for offset in range(
            0,
            len(pcm_bytes),
            byte_chunk_size,
        ):
            chunk = pcm_bytes[
                offset : offset + byte_chunk_size
            ]

            if chunk:
                recognizer.AcceptWaveform(
                    chunk
                )

        try:
            final_result = json.loads(
                recognizer.FinalResult()
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Vosk returned invalid JSON."
            ) from exc

        text = str(
            final_result.get(
                "text",
                "",
            )
        ).strip()

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
                "implementation": "vosk",
                "backend": "kaldi",
                "model_path": str(self.model_path),
                "device": self.device,
                "original_sample_rate": original_sample_rate,
                "sample_rate": self.sample_rate,
                "audio_duration_s": audio_duration_s,
                "real_time_factor": real_time_factor,
                "channels": 1,
                "sample_width_bits": 16,
                "frame_chunk_size": self.frame_chunk_size,
                "decoding": "streaming_incremental",
                "latency_type": "end_to_end",
                "latency_includes": [
                    "audio_loading",
                    "mono_conversion",
                    "resampling",
                    "pcm16_conversion",
                    "streaming_recognition",
                    "final_json_decoding",
                ],
            },
        }

    def close(self) -> None:
        """Release Vosk model resources."""

        self._model = None

        gc.collect()