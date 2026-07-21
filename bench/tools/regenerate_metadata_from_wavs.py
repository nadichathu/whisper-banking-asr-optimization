from pathlib import Path
import argparse
import csv
try:
    import soundfile as sf
except Exception:
    raise SystemExit("Please install soundfile: pip install soundfile")
try:
    import pandas as pd
except Exception:
    raise SystemExit("Please install pandas: pip install pandas")

def load_transcripts(transcript_csv: Path):
    if not transcript_csv.exists():
        return {}
    try:
        df = pd.read_csv(transcript_csv)
    except Exception:
        mapping = {}
        with open(transcript_csv, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) >= 2:
                    mapping[Path(row[0]).name] = row[1]
        return mapping

    cols = list(df.columns)
    if len(cols) >= 2:
        fname_col = None
        for c in cols:
            if "file" in c.lower() or "path" in c.lower() or "name" in c.lower():
                fname_col = c
                break
        if fname_col is None:
            fname_col = cols[0]

        text_col = None
        for c in cols:
            if "trans" in c.lower() or "text" in c.lower() or "utter" in c.lower():
                text_col = c
                break
        if text_col is None:
            text_col = cols[1]

        mapping = {}
        for _, row in df.iterrows():
            key = Path(str(row[fname_col])).name
            mapping[key] = str(row[text_col])
        return mapping
    else:
        return {}

def main():
    parser = argparse.ArgumentParser(description="Regenerate metadata.csv from WAVs and transcripts CSV")
    parser.add_argument("--dir", default="data/extracted", help="Directory with WAV files")
    parser.add_argument("--transcripts", default=None, help="Optional transcripts CSV file")
    parser.add_argument("--out", default=None, help="Output metadata CSV path (defaults to <dir>/metadata.csv)")
    args = parser.parse_args()

    d = Path(args.dir)
    if not d.exists():
        print(f"Directory not found: {d}")
        return

    transcripts = {}
    if args.transcripts:
        transcripts = load_transcripts(Path(args.transcripts))
    else:
        tpath = d / "transcripts.csv"
        if tpath.exists():
            transcripts = load_transcripts(tpath)

    rows = []
    wavs = sorted([p for p in d.glob("**/*.wav")])
    for p in wavs:
        try:
            info = sf.info(str(p))
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            try:
                data, sr = sf.read(str(p))
                duration = float(len(data)) / float(sr)
            except Exception:
                duration = None

        fn = p.name
        transcript = transcripts.get(fn, "")

        rows.append({"file_name": fn, "path": str(p.resolve()), "transcript": transcript, "duration_s": duration})

    out_path = Path(args.out) if args.out else d / "metadata.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote metadata to: {out_path} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
