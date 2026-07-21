from typing import Any, Dict

class BaseAdapter:
    """Base adapter interface for ASR backends.

    Implementations should override load(), transcribe(), and close().
    transcribe must be synchronous and return a dict with at least:
      {"text": str, "meta": { ... }}
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def load(self):
        """Load model resources. May be expensive."""
        raise NotImplementedError()

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """Synchronously transcribe the given audio file.

        Returns: {"text": str, "meta": {...}}
        """
        raise NotImplementedError()

    def profile(self, audio_path: str) -> Dict[str, float]:
        """Optional: return per-stage timings as a dict.

        Raise NotImplementedError if profiling not supported.
        """
        raise NotImplementedError()

    def close(self):
        """Free resources if necessary."""
        return None
