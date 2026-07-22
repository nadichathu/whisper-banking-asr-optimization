from typing import Any, Dict, Optional

from .faster_whisper_adapter import FasterWhisperAdapter


class FasterWhisperMediumAdapter(FasterWhisperAdapter):
    """Faster-Whisper adapter fixed to the Whisper Medium model.

    This adapter reuses the complete FasterWhisperAdapter implementation
    rather than duplicating its loading, transcription, timing, metadata,
    validation, and cleanup logic.

    The only model-specific change is:

        model_size="medium"

    This guarantees that the Small vs. Medium comparison isolates only
    the model size while every other benchmark setting remains identical.
    """

    MODEL_SIZE = "medium"

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved_config = dict(config or {})

        requested_model_size = resolved_config.get(
            "model_size",
            self.MODEL_SIZE,
        )

        if str(requested_model_size).lower() != self.MODEL_SIZE:
            raise ValueError(
                "FasterWhisperMediumAdapter only supports "
                "model_size='medium'. "
                f"Received model_size='{requested_model_size}'. "
                "Use FasterWhisperSmallAdapter for the small model "
                "or FasterWhisperAdapter for a configurable model size."
            )

        resolved_config["model_size"] = self.MODEL_SIZE

        super().__init__(resolved_config)

        self.name = "faster_whisper_medium"
        self.model_size = self.MODEL_SIZE