import pathlib

# Benchmark defaults
N_WARMUP = 1
N_RUNS = 3

# Use MINDS-14 dataset
AUDIO_DIR = pathlib.Path("data/minds14_banking")
METADATA_FILE = pathlib.Path("data/minds14_banking/metadata.csv")

RESULTS_DIR = pathlib.Path("results")

SAMPLE_RATE = 16000

# This benchmark suite is GPU-only. Every adapter requires and validates
# a CUDA device; there is no CPU execution path.
DEVICE = "cuda"

MODEL_SIZE = "small"

# Wav2Vec2 default model id (HF)
WAV2VEC_MODEL_ID = "facebook/wav2vec2-base-960h"

# Parakeet default model id (NVIDIA NeMo)
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"

# FastConformer default model id (NVIDIA NeMo)
FASTCONFORMER_MODEL_ID = "nvidia/stt_en_fastconformer_ctc_large"
