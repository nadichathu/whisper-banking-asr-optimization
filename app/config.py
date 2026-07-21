MODEL_SIZE = "small" # tiny/base/small/medium/large
DEVICE = "cpu" # "cuda"
LANGUAGE = "en"
FP16 = False
SAMPLE_RATE = 16000
RECORD_SECONDS = 4

# MODEL_SIZE ========================================
# tiny (~39M params) - Very fast, Lower accuracy, Good for edge devices / quick demos, Weak for serious research (already very optimized + too simple)
# base (~74M) - Slightly better accuracy than tiny, Very fast, Limited research headroom
## small (~244M) - Good balance of speed + accuracy, Realistic deployment size, Enough complexity to optimize, Suitable for real-time systems research
# medium (~769M) - Much better accuracy, Much slower, Needs strong GPU, Overkill for short banking commands
# large (~1.5B) - Best accuracy, Heavy GPU requirement, Not realistic for real-time banking on normal infra

# SAMPLE_RATE =======================================
# 8 kHz → Old telephone quality (very low quality, narrowband)
## 16 kHz → Standard speech processing (Whisper default) - Matches training distribution, Avoids extra resampling overhead, Keeps experiments consistent
# 22.05 kHz → Mid-quality audio
# 44.1 kHz → CD quality (music standard)
# 48 kHz → Professional audio / video production

# RECORD_SECONDS ====================================
# 1–2 sec → very short commands
## 3–5 sec → realistic short commands
# 6–10 sec → longer queries
# 10+ sec → becomes general transcription, not command-style