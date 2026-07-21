"""Small smoke script that runs Whisper adapter on up to 2 audio files."""
from bench.adapters.whisper_baseline_adapter import WhisperAdapter
from bench.adapters.wav2vec2_adapter import Wav2Vec2Adapter
from bench.adapters.vosk_adapter import VoskAdapter
from bench.adapters.nemo_fastconformer_adapter import NeMoFastConformerAdapter
from bench.runner import run_benchmark


def main():
    # run on up to 2 files to smoke-test multiple adapters
    print("Running Whisper smoke test...")
    run_benchmark(WhisperAdapter, adapter_config=None, max_files=2)

    print("Running Wav2Vec2 smoke test (may require transformers/torch)...")
    try:
        run_benchmark(Wav2Vec2Adapter, adapter_config=None, max_files=2)
    except Exception as e:
        print("Wav2Vec2 smoke test skipped/error:", e)

    print("Running Vosk smoke test (may require vosk model)...")
    try:
        run_benchmark(VoskAdapter, adapter_config=None, max_files=2)
    except Exception as e:
        print("Vosk smoke test skipped/error:", e)

    print("Running NeMo FastConformer smoke test (may require nemo-toolkit + GPU)...")
    try:
        run_benchmark(NeMoFastConformerAdapter, adapter_config=None, max_files=2)
    except Exception as e:
        print("NeMo smoke test skipped/error:", e)


if __name__ == '__main__':
    main()
