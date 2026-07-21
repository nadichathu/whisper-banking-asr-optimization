import time
import pathlib
import pandas as pd
import whisper

# AUDIO_DIR = pathlib.Path("audio_samples")
# RESULT_PATH = pathlib.Path("results/profile_results.csv")
AUDIO_DIR = pathlib.Path("data/minds14_banking")
RESULT_PATH = pathlib.Path("results/whisper_profile_baseline.csv")

MODEL_SIZE = "small"
DEVICE = "cpu"


def main():

    print("Loading Whisper model...")
    model = whisper.load_model(MODEL_SIZE, device=DEVICE)

    rows = []

    for audio_path in sorted(AUDIO_DIR.glob("*.wav")):

        print(f"\nProfiling {audio_path.name}")

        t0 = time.perf_counter()

        # Stage 1 — load audio
        audio = whisper.load_audio(str(audio_path))
        t1 = time.perf_counter()

        # Stage 2 — pad / trim
        audio = whisper.pad_or_trim(audio)
        t2 = time.perf_counter()

        # Stage 3 — mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        t3 = time.perf_counter()

        # Stage 4 — encoder
        enc_out = model.encoder(mel.unsqueeze(0))
        t4 = time.perf_counter()

        # Stage 5 — decoder
        # options = whisper.DecodingOptions()
        # whisper.decode(model, mel.unsqueeze(0), options)
        # t5 = time.perf_counter()

        options = whisper.DecodingOptions(
            language = "en",
            fp16 = False
        )

        whisper.decode(model, mel.unsqueeze(0), options)
        t5 = time.perf_counter()

        row = {
            "file": audio_path.name,
            "load_audio_ms": (t1 - t0) * 1000,
            "pad_trim_ms": (t2 - t1) * 1000,
            "mel_ms": (t3 - t2) * 1000,
            "encoder_ms": (t4 - t3) * 1000,
            "decoder_ms": (t5 - t4) * 1000,
            "total_ms": (t5 - t0) * 1000,
        }

        rows.append(row)

        print(row)

    df = pd.DataFrame(rows)

    RESULT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(RESULT_PATH, index=False)

    print("\nSaved profiling results:", RESULT_PATH)


if __name__ == "__main__":
    main()