from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseAdapter(ABC):
    """Base interface for all ASR benchmark adapters.

    Every adapter must implement:
        __init__(config=None)
        load()
        transcribe(audio_path)
        close()

    transcribe() must return:
        {
            "text": str,
            "latency_ms": float,
            "meta": dict,
        }
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.name = self.__class__.__name__

    @abstractmethod
    def load(self) -> Any:
        """Load model resources.

        This method may download model files and allocate CPU or GPU memory.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
    ) -> Dict[str, Any]:
        """Synchronously transcribe an audio file.

        Returns:
            {
                "text": str,
                "latency_ms": float,
                "meta": dict,
            }
        """
        raise NotImplementedError

    def profile(
        self,
        audio_path: str | Path,
    ) -> Dict[str, Any]:
        """Return optional stage-level timing information.

        Adapters that do not support profiling may keep this default
        implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support profiling."
        )

    def close(self) -> None:
        """Release model resources.

        Adapters may override this method when explicit cleanup is required.
        """
        return None
