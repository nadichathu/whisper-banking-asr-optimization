import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import whisper

from .base_adapter import BaseAdapter
from bench.config import DEVICE, MODEL_SIZE


class WhisperAdapter(BaseAdapter):
    """OpenAI Whisper FP32 baseline adapter. GPU-only.

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

        self.name = "whisper"

        self.model_size = self.config.get(
            "model_size",
            MODEL_SIZE,
        )

        requested_device = str(
            self.config.get("device", DEVICE)
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

        torch.cuda.synchronize(device=self.device)

        start_time = time.perf_counter()

        result = self._model.transcribe(
            str(audio_path),
            task="transcribe",
            language="en",
            fp16=False,
        )

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
                "model_loading_included": False,
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

        torch.cuda.synchronize(device=self.device)

        mel_ms = (
            time.perf_counter() - stage_start
        ) * 1000

        stage_start = time.perf_counter()

        with torch.inference_mode():
            self._model.encoder(
                mel.unsqueeze(0)
            )

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

        self._model = None

        gc.collect()

        torch.cuda.empty_cache()
