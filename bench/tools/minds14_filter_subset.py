from pathlib import Path
import shutil
import pandas as pd

SOURCE_DIR = Path("data/minds14_audio")
SOURCE_CSV = SOURCE_DIR / "metadata.csv"

TARGET_DIR = Path("data/minds14_banking")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
TARGET_CSV = TARGET_DIR / "metadata.csv"

INTENT_MAP = {
    3: "atm_limit",
    4: "balance",
    6: "card_issues",
    7: "cash_deposit",
    12: "latest_transactions",
    13: "pay_bill",
}

SELECTED_INTENTS = [3, 4, 6, 7, 12, 13]

MAX_DURATION = 4.0
SAMPLES_PER_INTENT = 10
RANDOM_SEED = 42


def main() -> None:
    df = pd.read_csv(SOURCE_CSV)

    # Keep only chosen intents and duration <= 4 seconds
    filtered = df[
        df["intent"].isin(SELECTED_INTENTS) &
        (df["duration"] <= MAX_DURATION)
    ].copy()

    print(f"Total rows after intent+duration filtering: {len(filtered)}")

    selected_parts = []

    for intent in SELECTED_INTENTS:
        intent_df = filtered[filtered["intent"] == intent].copy()
        available = len(intent_df)

        if available == 0:
            print(f"[WARN] No samples found for intent: {intent}")
            continue

        n_take = min(SAMPLES_PER_INTENT, available)

        sampled = intent_df.sample(n=n_take, random_state=RANDOM_SEED)
        selected_parts.append(sampled)

        print(f"{intent}: selected {n_take} / available {available}")

    if not selected_parts:
        raise RuntimeError("No samples selected. Check intent names and durations.")

    final_df = pd.concat(selected_parts, ignore_index=True)

    # Copy files and rename them cleanly
    output_rows = []
    intent_counts = {}

    for _, row in final_df.iterrows():
        intent_id = row["intent"]
        intent = INTENT_MAP[intent_id]
        old_name = row["file_name"]
        src_file = SOURCE_DIR / old_name

        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        idx = intent_counts[intent]

        new_name = f"{intent}_{idx:02d}.wav"
        dst_file = TARGET_DIR / new_name

        shutil.copy2(src_file, dst_file)

        output_rows.append({
            "file_name": new_name,
            "transcript": row["transcript"],
            "intent": intent,
            "duration": row["duration"],
        })

    out_df = pd.DataFrame(output_rows).sort_values(["intent", "file_name"])
    out_df.to_csv(TARGET_CSV, index=False)

    print("\nDone.")
    print(f"Saved {len(out_df)} files to: {TARGET_DIR}")
    print(f"Metadata saved to: {TARGET_CSV}")
    print("\nFiles per intent:")
    print(out_df["intent"].value_counts().sort_index())
    print("\nDuration stats:")
    print(out_df["duration"].describe())


if __name__ == "__main__":
    main()