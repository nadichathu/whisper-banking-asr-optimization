import pathlib

# Benchmark defaults
N_WARMUP = 1
N_RUNS = 3

# Use MINDS-14 dataset
AUDIO_DIR = pathlib.Path("data/minds14_banking")
METADATA_FILE = pathlib.Path("data/minds14_banking/metadata.csv")

RESULTS_DIR = pathlib.Path("results")

SAMPLE_RATE = 16000
#DEVICE = "cpu"  # override to "cuda" when available
DEVICE = "cuda"  
MODEL_SIZE = "small"

# Whether to include network latency for cloud adapters
INCLUDE_NETWORK_LATENCY = False

# Wav2Vec2 default model id (HF)
WAV2VEC_MODEL_ID = "facebook/wav2vec2-base-960h"

# Vosk model local path
VOSK_MODEL_PATH = "bench/models/vosk/vosk-model-small-en-us-0.15"