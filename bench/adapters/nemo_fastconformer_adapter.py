from typing import Any, Dict
import time

from bench.adapters.base_adapter import BaseAdapter


class NeMoFastConformerAdapter(BaseAdapter):
    """Adapter for NVIDIA NeMo FastConformer/ASR models.

    This adapter tries to load a pretrained NeMo ASR model via
    ``ASRModel.from_pretrained(...)``. If NeMo or compatible pretrained
    models are not available in the environment this will raise a
    RuntimeError with an actionable message.

    Notes:
    - NeMo typically requires a matching PyTorch/CUDA setup for GPU models.
    - If you only have CPU, some NeMo models may not run or will be extremely slow.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = None
        self.model_id = None

    def load(self):
        try:
            # Imported lazily so environments without NeMo still import the package
            from nemo.collections.asr.models import ASRModel
        except Exception as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "NeMo is not available in this Python environment. "
                "Install nemo-toolkit (and a compatible PyTorch/CUDA) to use the NeMo adapter."
            ) from e

        # model_name can be provided via config; otherwise try a few common names
        requested = self.config.get("model_name") if self.config else None
        candidates = [requested] if requested else []
        candidates += [
            # common NeMo Hub identifiers - availability depends on your nemo version
            "stt_en_fastconformer_base",
            "stt_en_fastconformer_small",
            "stt_en_fastconformer_tiny",
            "stt_en_quartznet15x5",
        ]

        last_exc = None
        for cand in filter(None, candidates):
            try:
                # from_pretrained will download the model if needed
                self.model = ASRModel.from_pretrained(cand)
                self.model_id = cand
                return
            except Exception as e:  # try next candidate
                last_exc = e

        # nothing loaded
        msg = (
            "Failed to load a NeMo ASR pretrained model. "
            "Tried candidates: %s. Ensure nemo-toolkit is installed and you have "
            "the correct CUDA/PyTorch environment for NeMo models. "
            "%s" % (candidates, str(last_exc))
        )
        raise RuntimeError(msg)

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not loaded; call load() before transcribing")

        # NeMo ASR models expose a .transcribe(list_of_files) helper
        t0 = time.perf_counter()
        hyps = self.model.transcribe([audio_path])
        t1 = time.perf_counter()

        text = hyps[0] if hyps else ""
        return {"text": text, "meta": {"transcribe_ms": (t1 - t0) * 1000}}

    def profile(self, audio_path: str) -> Dict[str, float]:
        # Basic profiling wraps the transcribe call. NeMo does not currently
        # expose easy stage-level hooks here without deeper instrumentation.
        t0 = time.perf_counter()
        _ = self.model.transcribe([audio_path])
        t1 = time.perf_counter()
        return {"transcribe_ms": (t1 - t0) * 1000}

    def close(self):
        # Try to free GPU memory if used
        try:
            import gc
            del self.model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass
