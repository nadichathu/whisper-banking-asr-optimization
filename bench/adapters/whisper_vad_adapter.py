import time
from pathlib import Path

import numpy as np
import whisper
from silero_vad import (
    collect_chunks,
    get_speech_timestamps,
    load_silero_vad,
)
import soundfile as sf
import torch

from bench.config import DEVICE, MODEL_SIZE


class WhisperVadAdapter:
    """
    Whisper inference with Silero VAD preprocessing.

    VAD removes non-speech regions before transcription.
    The reported latency includes both VAD and Whisper inference.
    """

    def __init__(self, config=None):
        self.config = config or {}

        self.model_size = self.config.get("model_size", MODEL_SIZE)
        self.device = self.config.get("device", DEVICE)

        self.whisper_model = None
        self.vad_model = None

        self.model_id = "whisper_vad"
        self.sample_rate = 16000

    def load(self):
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(
                self.model_size,
                device=self.device,
            )

        if self.vad_model is None:
            # Keep VAD on CPU so that it does not compete with Whisper for GPU memory.
            self.vad_model = load_silero_vad()

    def transcribe(self, audio_path: str):
        self.load()

        total_start = time.perf_counter()

        # -------------------------
        # Stage 1: Read audio
        # -------------------------
        vad_start = time.perf_counter()

        audio_data, sample_rate = sf.read(
            str(Path(audio_path)),
            dtype="float32",
        )

        # Convert stereo to mono if necessary
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz audio, "
                f"but received {sample_rate} Hz: {audio_path}"
            )

        waveform = torch.from_numpy(audio_data)


        original_samples = int(waveform.numel())

        # -------------------------
        # Stage 2: Detect speech
        # -------------------------
        speech_timestamps = get_speech_timestamps(
            waveform,
            self.vad_model,
            sampling_rate=self.sample_rate,
            threshold=0.5,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
            speech_pad_ms=100,
            return_seconds=False,
        )

        # -------------------------
        # Stage 3: Trim silence
        # -------------------------
        if speech_timestamps:
            trimmed_waveform = collect_chunks(
                speech_timestamps,
                waveform,
            )
            speech_detected = True
        else:
            # Safe fallback: use the original audio when VAD finds no speech.
            trimmed_waveform = waveform
            speech_detected = False

        vad_end = time.perf_counter()

        trimmed_samples = int(trimmed_waveform.numel())

        original_duration_s = original_samples / self.sample_rate
        trimmed_duration_s = trimmed_samples / self.sample_rate
        removed_duration_s = max(
            0.0,
            original_duration_s - trimmed_duration_s,
        )

        # Whisper accepts a NumPy float32 waveform.
        audio_array = (
            trimmed_waveform
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        # -------------------------
        # Stage 4: Whisper inference
        # -------------------------
        whisper_start = time.perf_counter()

        result = self.whisper_model.transcribe(
            audio_array,
            task="transcribe",
            language="en",
            fp16=self.device == "cuda",
        )

        whisper_end = time.perf_counter()
        total_end = time.perf_counter()

        return {
            "text": result.get("text", "").strip(),
            "meta": {
                "optimization": "silero_vad_silence_trimming",
                "speech_detected": speech_detected,
                "speech_segment_count": len(speech_timestamps),
                "original_duration_s": original_duration_s,
                "trimmed_duration_s": trimmed_duration_s,
                "removed_duration_s": removed_duration_s,
                "vad_latency_ms": (vad_end - vad_start) * 1000,
                "whisper_latency_ms": (
                    whisper_end - whisper_start
                ) * 1000,
                "latency_ms": (total_end - total_start) * 1000,
            },
        }

    def close(self):
        self.whisper_model = None
        self.vad_model = None