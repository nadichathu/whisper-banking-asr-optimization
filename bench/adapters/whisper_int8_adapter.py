import os
import time
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel


class WhisperINT8Adapter:
    """
    Whisper Small quantized with CTranslate2.

    CPU:
        device="cpu"
        compute_type="int8"

    CUDA:
        device="cuda"
        compute_type="int8_float16"
    """

    def __init__(self) -> None:
        self.model_name = "small"

        if torch.cuda.is_available():
            self.device = "cuda"

            # INT8 weights with FP16 computation where required.
            self.compute_type = "int8_float16"
            self.cpu_threads = 0
        else:
            self.device = "cpu"
            self.compute_type = "int8"

            # Use all available logical CPU cores.
            self.cpu_threads = max(1, os.cpu_count() or 1)

        self.name = "whisper_int8"

        print(
            f"Loading {self.model_name} using "
            f"device={self.device}, "
            f"compute_type={self.compute_type}"
        )

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=1,
        )

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        audio_path = str(audio_path)

        self.device = "cpu"
        self.compute_type = "int8"
        self.cpu_threads = max(1, os.cpu_count() or 1)

        #if self.device == "cuda":
        #    torch.cuda.synchronize()

        start = time.perf_counter()

        segments, info = self.model.transcribe(
            audio_path,
            language="en",
            task="transcribe",

            # Match the original Whisper greedy configuration.
            beam_size=1,
            best_of=1,
            temperature=0.0,

            # Short independent banking commands.
            condition_on_previous_text=False,

            # Avoid timestamp generation.
            without_timestamps=True,
            word_timestamps=False,

            # Do not include VAD in this experiment.
            vad_filter=False,
        )

        # faster-whisper returns a generator.
        # It must be consumed before stopping the timer.
        segment_list = list(segments)

        if self.device == "cuda":
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        text = " ".join(
            segment.text.strip()
            for segment in segment_list
            if segment.text.strip()
        ).strip()

        return {
            "text": text,
            "latency_ms": latency_ms,
            "meta": {
                "implementation": "faster-whisper",
                "backend": "CTranslate2",
                "model_size": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "quantization": "int8",
                "beam_size": 1,
                "segment_count": len(segment_list),
                "detected_language": info.language,
                "language_probability": float(
                    info.language_probability
                ),
                "audio_duration_s": float(info.duration),
            },
        }