"""Merge per-model bench results into combined CSVs and compute Pareto frontier.

Usage:
    python bench/merge_results.py
"""
from pathlib import Path
import pandas as pd
import numpy as np


VALID_MODELS = {
    "whisperadapter",
    "facebook_wav2vec2-base-960h",
    "voskadapter",
    "whisper_greedy",
}


def safe_read_csv(p):
    try:
        df = pd.read_csv(p)
        df = df.loc[:, ~df.columns.duplicated()]
        return df
    except Exception:
        return pd.DataFrame()


def normalize_text(text):
    text = str(text).lower()
    import re
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_word_wer(ref, hyp):
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()

    n = len(r)
    if n == 0:
        return 0.0 if len(h) == 0 else 1.0

    dp = [[0] * (len(h) + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1   # substitution
                )

    wer = dp[n][len(h)] / float(n)
    return max(0.0, min(float(wer), 1.0))


def main():
    res = Path("results")
    res.mkdir(exist_ok=True)

    # A: combine runs
    runs = []
    for p in res.glob("*_runs.csv"):
        model = p.stem.replace("_runs", "")
        if model not in VALID_MODELS:
            continue

        df = safe_read_csv(p)
        if df.empty:
            continue

        df["model"] = model
        runs.append(df)

    if runs:
        combined_runs = pd.concat(runs, ignore_index=True)
        combined_runs = combined_runs.loc[:, ~combined_runs.columns.duplicated()]
        combined_runs.to_csv(res / "combined_runs.csv", index=False)

    # B: combine per-file
    dfpf = pd.DataFrame()
    perfs = []

    for p in res.glob("*_per_file.csv"):
        model = p.stem.replace("_per_file", "")
        if model not in VALID_MODELS:
            continue

        df = safe_read_csv(p)
        if df.empty:
            continue

        df["model"] = model
        perfs.append(df)

    if perfs:
        dfpf = pd.concat(perfs, ignore_index=True)
        dfpf = dfpf.loc[:, ~dfpf.columns.duplicated()]

        # normalize possible latency column names
        if "p50" in dfpf.columns and "median_latency_ms" not in dfpf.columns:
            dfpf = dfpf.rename(columns={"p50": "median_latency_ms"})

        # load metadata for references
        metadata = pd.read_csv("data/minds14_banking/metadata.csv")
        metadata = metadata.rename(columns={
            "file_name": "file",
            "transcript": "reference",
        })

        # normalize possible filename column names
        if "file" not in dfpf.columns:
            if "file_name" in dfpf.columns:
                dfpf = dfpf.rename(columns={"file_name": "file"})
            else:
                print("dfpf columns:", dfpf.columns.tolist())
                raise KeyError("No 'file' column found in combined per-file data")

        # remove old broken reference column if present, then merge fresh one
        dfpf = dfpf.drop(columns=["reference"], errors="ignore")
        dfpf = dfpf.merge(metadata[["file", "reference"]], on="file", how="left")

        # recompute per-file WER from merged references
        if "canonical_transcript" in dfpf.columns:
            dfpf["wer"] = dfpf.apply(
                lambda row: compute_word_wer(
                    row.get("reference", ""),
                    row.get("canonical_transcript", "")
                ) if pd.notna(row.get("reference", "")) and str(row.get("reference", "")).strip() else None,
                axis=1
            )

        dfpf.to_csv(res / "combined_per_file.csv", index=False)

    # C: model-level summary
    rows = []

    if not dfpf.empty:
        dfpf = dfpf.loc[:, ~dfpf.columns.duplicated()]

        for model, g in dfpf.groupby("model"):
            if "median_latency_ms" in g.columns:
                col = g["median_latency_ms"]
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                p50s = pd.to_numeric(col, errors="coerce").dropna()
            else:
                p50s = pd.Series(dtype=float)

            if "mean_latency_ms" in g.columns:
                means = pd.to_numeric(g["mean_latency_ms"], errors="coerce").dropna()
            else:
                means = pd.Series(dtype=float)

            if "p95" in g.columns:
                p95s = pd.to_numeric(g["p95"], errors="coerce").dropna()
            elif "p95_latency_ms" in g.columns:
                p95s = pd.to_numeric(g["p95_latency_ms"], errors="coerce").dropna()
            else:
                p95s = pd.Series(dtype=float)

            if "wer" in g.columns:
                wers = pd.to_numeric(g["wer"], errors="coerce").dropna()
            else:
                wers = pd.Series(dtype=float)

            corpus_wer = None
            if "reference" in g.columns and "canonical_transcript" in g.columns:
                refs = " ".join(g["reference"].fillna("").astype(str).tolist())
                hyps = " ".join(g["canonical_transcript"].fillna("").astype(str).tolist())
                if refs.strip():
                    corpus_wer = compute_word_wer(refs, hyps)

            rows.append({
                "model": model,
                "n_files": len(g),
                "median_latency_ms": float(p50s.median()) if len(p50s) else None,
                "mean_latency_ms": float(means.mean()) if len(means) else None,
                "p95_latency_ms": float(np.nanpercentile(p95s, 95)) if len(p95s) else None,
                "std_latency_ms": float(means.std()) if len(means) else None,
                "mean_wer": float(wers.mean()) if len(wers) else None,
                "corpus_wer": corpus_wer,
            })

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(res / "bench_summary.csv", index=False)

        # D: pareto frontier
        pts = summary_df[["median_latency_ms", "corpus_wer"]].dropna()
        pareto_idx = []

        for i, row in pts.iterrows():
            lat, werv = row["median_latency_ms"], row["corpus_wer"]
            dominated = False

            for j, row2 in pts.iterrows():
                if j == i:
                    continue

                if (
                    row2["median_latency_ms"] <= lat and
                    row2["corpus_wer"] <= werv and
                    (
                        row2["median_latency_ms"] < lat or
                        row2["corpus_wer"] < werv
                    )
                ):
                    dominated = True
                    break

            if not dominated:
                pareto_idx.append(i)

        pareto_df = summary_df.loc[pareto_idx]
        pareto_df.to_csv(res / "bench_pareto.csv", index=False)

    print("Merged results written to results/:")
    print("- combined_runs.csv")
    print("- combined_per_file.csv")
    print("- bench_summary.csv")
    print("- bench_pareto.csv")


if __name__ == "__main__":
    main()