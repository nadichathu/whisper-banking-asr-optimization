import time
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


class FasterWhisperMediumAdapter:
    def __init__(self) -> None:
        self.name = "faster_whisper_medium"
        self.model_name = "medium"
        self.device = "cuda"
        self.compute_type = "float16"

        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=1,
        )

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> dict[str, Any]:

        start = time.perf_counter()

        segments, info = self.model.transcribe(
            str(audio_path),
            language="en",
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            word_timestamps=False,
            vad_filter=False,
        )

        segment_list = list(segments)

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

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
                "segment_count": len(segment_list),
                "audio_duration_s": float(info.duration),
            },
        }