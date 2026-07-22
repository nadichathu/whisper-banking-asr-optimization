from typing import Any, Dict, Optional

from .faster_whisper_adapter import FasterWhisperAdapter


class FasterWhisperSmallAdapter(FasterWhisperAdapter):
    """Faster-Whisper adapter fixed to the Whisper Small model.

    This adapter reuses the complete FasterWhisperAdapter implementation
    rather than duplicating its loading, transcription, timing, metadata,
    validation, and cleanup logic.

    The only model-specific change is:

        model_size="small"

    This design prevents the Small and Medium adapters from silently drifting
    apart when decoding parameters or benchmark methodology are updated.
    """

    MODEL_SIZE = "small"

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
                "FasterWhisperSmallAdapter only supports "
                "model_size='small'. "
                f"Received model_size='{requested_model_size}'. "
                "Use FasterWhisperMediumAdapter for the medium model "
                "or FasterWhisperAdapter for a configurable model size."
            )

        resolved_config["model_size"] = self.MODEL_SIZE

        super().__init__(resolved_config)

        self.name = "faster_whisper_small"
        self.model_size = self.MODEL_SIZE