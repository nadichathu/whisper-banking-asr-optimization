import time
from typing import Dict, Any

try:
    from vosk import Model, KaldiRecognizer
    import wave
except Exception:
    Model = None
    KaldiRecognizer = None

from .base_adapter import BaseAdapter
from bench.config import VOSK_MODEL_PATH, SAMPLE_RATE


class VoskAdapter(BaseAdapter):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model_path = (config or {}).get("model_path", VOSK_MODEL_PATH)
        self._model = None

    def load(self):
        if Model is None:
            raise RuntimeError("vosk not installed. Install vosk to use VoskAdapter.")
        if not self._model:
            # Model path may point to directory containing model files
            self._model = Model(self.model_path)
        return self._model

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        model = self.load()
        wf = wave.open(audio_path, "rb")
        rec = KaldiRecognizer(model, wf.getframerate())
        t0 = time.perf_counter()
        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                results.append(rec.Result())
        results.append(rec.FinalResult())
        t1 = time.perf_counter()
        # concatenate partial JSON results
        import json
        texts = []
        for r in results:
            try:
                j = json.loads(r)
                if j.get('text'):
                    texts.append(j['text'])
            except Exception:
                continue
        text = " ".join(texts).strip()
        latency_ms = (t1 - t0) * 1000
        meta = {"model_path": self.model_path, "device": "cpu", "latency_ms": latency_ms}
        return {"text": text, "meta": meta}

    def close(self):
        self._model = None
