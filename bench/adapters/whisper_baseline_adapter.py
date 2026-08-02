import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperAdapter(BaseAdapter):
    """OpenAI Whisper FP32 baseline adapter.

    The reported latency is end-to-end file-to-text latency and includes
    Whisper's internal audio loading, resampling, preprocessing, inference,
    and decoding.

    This baseline deliberately forces FP32 so that a separate FP16 adapter
    represents a genuine experimental condition on CUDA hardware.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config)

        self.config = config or {}
        self.name = "whisper"

        self.model_size = self.config.get(
            "model_size",
            MODEL_SIZE,
        )

        requested_device = str(
            self.config.get("device", DEVICE)
        ).lower()

        if requested_device.startswith("cuda"):
            if torch.cuda.is_available():
                self.device = requested_device
            else:
                print(
                    "CUDA was requested but is unavailable. "
                    "Falling back to CPU."
                )
                self.device = "cpu"
        else:
            self.device = "cpu"

        self._model = None

    def load(self):
        """Load the Whisper model."""

        if self._model is None:
            print(
                f"Loading Whisper {self.model_size} "
                f"FP32 baseline on {self.device}"
            )

            self._model = whisper.load_model(
                self.model_size,
                device=self.device,
            )

            self._model.eval()

        return self._model

    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Transcribe an audio file using the FP32 baseline."""

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

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        start_time = time.perf_counter()

        result = self._model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            fp16=False,
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        text = result.get(
            "text",
            "",
        ).strip()

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "openai-whisper",
                "backend": "whisper",
                "model_size": self.model_size,
                "device": self.device,
                "precision": "fp32",
                "decoding": "default_transcribe",
                "language": "en",
                "latency_type": "end_to_end",
                "latency_includes": [
                    "audio_loading",
                    "audio_resampling",
                    "padding_or_trimming",
                    "mel_spectrogram",
                    "encoder",
                    "decoder",
                    "text_generation",
                ],
            },
        }

    def profile(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Return diagnostic stage timings.

        Important:
        `decode_pipeline_ms` includes a second encoder pass because
        `whisper.decode()` internally encodes the mel spectrogram again.

        Therefore:
        - `decode_pipeline_ms` is not decoder-only latency.
        - `profiling_wall_time_ms` is not representative of one production
          transcription pipeline.
        - These values should not be added together as a clean stage-level
          decomposition.
        """

        if self._model is None:
            raise RuntimeError(
                "Whisper model is not loaded. "
                "Call load() before profile()."
            )

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file was not found: {audio_path}"
            )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        profiling_start = time.perf_counter()

        stage_start = time.perf_counter()

        audio = whisper.load_audio(
            str(audio_path)
        )

        load_audio_ms = (
            time.perf_counter() - stage_start
        ) * 1000

        stage_start = time.perf_counter()

        audio = whisper.pad_or_trim(
            audio
        )

        pad_trim_ms = (
            time.perf_counter() - stage_start
        ) * 1000

        stage_start = time.perf_counter()

        mel = whisper.log_mel_spectrogram(
            audio,
            n_mels=self._model.dims.n_mels,
        ).to(self.device)

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        mel_ms = (
            time.perf_counter() - stage_start
        ) * 1000

        stage_start = time.perf_counter()

        with torch.inference_mode():
            self._model.encoder(
                mel.unsqueeze(0)
            )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        encoder_ms = (
            time.perf_counter() - stage_start
        ) * 1000

        decoding_options = whisper.DecodingOptions(
            task="transcribe",
            language="en",
            fp16=False,
        )

        stage_start = time.perf_counter()

        with torch.inference_mode():
            whisper.decode(
                self._model,
                mel,
                decoding_options,
            )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize(device=self.device)

        decode_pipeline_ms = (
            time.perf_counter() - stage_start
        ) * 1000

        profiling_wall_time_ms = (
            time.perf_counter() - profiling_start
        ) * 1000

        return {
            "load_audio_ms": load_audio_ms,
            "pad_trim_ms": pad_trim_ms,
            "mel_ms": mel_ms,
            "encoder_ms": encoder_ms,
            "decode_pipeline_ms": decode_pipeline_ms,
            "profiling_wall_time_ms": profiling_wall_time_ms,
            "encoder_recomputed_during_decode": True,
            "precision": "fp32",
            "note": (
                "decode_pipeline_ms includes Whisper's internal "
                "encoder and decoder execution. It is not a "
                "decoder-only measurement, and profiling_wall_time_ms "
                "contains a redundant encoder pass."
            ),
        }

    def close(self) -> None:
        """Release Whisper model resources."""

        used_cuda = self.device.startswith("cuda")

        self._model = None

        gc.collect()

        if used_cuda:
            torch.cuda.empty_cache()
