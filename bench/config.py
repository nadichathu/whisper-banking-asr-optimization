import pathlib

# Benchmark defaults
N_WARMUP = 1
N_RUNS = 3

# Use MINDS-14 dataset
AUDIO_DIR = pathlib.Path("data/minds14_banking")
METADATA_FILE = pathlib.Path("data/minds14_banking/metadata.csv")

RESULTS_DIR = pathlib.Path("results")

SAMPLE_RATE = 16000
# DEVICE = "cpu"  # override to "cuda" when available
DEVICE = "cuda"
MODEL_SIZE = "small"

# Whether to include network latency for cloud adapters
# NOTE: not currently read by any adapter or by runner.py -- either wire
# this into a cloud-API adapter if one is planned, or remove it if it is
# dead configuration left over from an earlier design.
INCLUDE_NETWORK_LATENCY = True

# Wav2Vec2 default model id (HF)
WAV2VEC_MODEL_ID = "facebook/wav2vec2-base-960h"

# Parakeet default model id (NVIDIA NeMo)
PARAKEET_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"