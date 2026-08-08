"""
Curate and stratify the prepared English MINDS-14 dataset.

This stage deliberately DOES NOT sample a fixed number of files per intent.

Default behaviour:
    - retain en-AU, en-GB and en-US;
    - retain all 14 banking intents;
    - retain all valid durations;
    - add duration strata;
    - mark <= 4 s utterances for short-command analysis;
    - preserve language-variety labels;
    - copy the untouched source audio into a curated raw directory.

Input:
    data/minds14_audio/
        metadata.csv
        *.wav

Output:
    data/minds14_banking_raw/
        metadata.csv
        stratification_summary.csv
        intent_language_summary.csv
        *.wav
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path("data/minds14_audio")
SOURCE_CSV = SOURCE_DIR / "metadata.csv"

TARGET_DIR = Path("data/minds14_banking_raw")
TARGET_CSV = TARGET_DIR / "metadata.csv"

STRATIFICATION_CSV = (
    TARGET_DIR / "stratification_summary.csv"
)

INTENT_LANGUAGE_CSV = (
    TARGET_DIR / "intent_language_summary.csv"
)

LANGUAGE_VARIETIES = (
    "en-AU",
    "en-GB",
    "en-US",
)

# None = keep every MINDS-14 banking intent.
#
# If a future experiment deliberately needs a subset,
# replace None with intent names, e.g.:
#
# SELECTED_INTENTS = [
#     "atm_limit",
#     "balance",
#     "card_issues",
# ]
#
SELECTED_INTENTS: list[str] | None = None

# Do NOT discard longer files.
# Instead, mark short commands and create duration strata.
SHORT_COMMAND_MAX_SECONDS = 6.0

DURATION_BINS = [
    0.0,
    2.0,
    4.0,
    8.0,
    np.inf,
]

DURATION_LABELS = [
    "≤2 s",
    ">2–4 s",
    ">4–8 s",
    ">8 s",
]

CLEAN_OUTPUT = True


# ============================================================
# HELPERS
# ============================================================

def clean_target_directory() -> None:
    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CLEAN_OUTPUT:
        return

    for path in TARGET_DIR.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def require_columns(
    dataframe: pd.DataFrame,
    columns: set[str],
) -> None:
    missing = columns.difference(dataframe.columns)

    if missing:
        raise ValueError(
            "Source metadata is missing required columns: "
            f"{sorted(missing)}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(
            f"Source metadata not found: {SOURCE_CSV}\n"
            "Run minds14_prepare.py first."
        )

    source_df = pd.read_csv(SOURCE_CSV)

    required_columns = {
        "file_name",
        "language_variety",
        "intent_id",
        "intent",
        "transcript",
        "duration",
        "sample_rate",
        "channels",
    }

    require_columns(
        source_df,
        required_columns,
    )

    if source_df.empty:
        raise RuntimeError(
            "Source metadata contains no rows."
        )

    # --------------------------------------------------------
    # VALIDATE SOURCE DATA
    # --------------------------------------------------------

    if source_df["file_name"].duplicated().any():
        raise RuntimeError(
            "Duplicate file_name values exist in source metadata."
        )

    if source_df["transcript"].isna().any():
        raise RuntimeError(
            "Missing transcripts detected."
        )

    if (source_df["duration"] <= 0).any():
        raise RuntimeError(
            "Non-positive audio durations detected."
        )

    source_files_missing = [
        file_name
        for file_name in source_df["file_name"]
        if not (SOURCE_DIR / file_name).is_file()
    ]

    if source_files_missing:
        raise FileNotFoundError(
            f"{len(source_files_missing)} source audio files are missing. "
            f"First missing file: {source_files_missing[0]}"
        )

    # --------------------------------------------------------
    # LANGUAGE SELECTION
    # --------------------------------------------------------

    selected = source_df[
        source_df["language_variety"].isin(
            LANGUAGE_VARIETIES
        )
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "No rows remain after language filtering."
        )

    # --------------------------------------------------------
    # OPTIONAL INTENT SELECTION
    #
    # Default = all 14 intents.
    # --------------------------------------------------------

    if SELECTED_INTENTS is not None:
        unknown_intents = (
            set(SELECTED_INTENTS)
            - set(selected["intent"].unique())
        )

        if unknown_intents:
            raise ValueError(
                "Requested intents were not found in the dataset: "
                f"{sorted(unknown_intents)}"
            )

        selected = selected[
            selected["intent"].isin(
                SELECTED_INTENTS
            )
        ].copy()

    if selected.empty:
        raise RuntimeError(
            "No rows remain after intent selection."
        )

    # --------------------------------------------------------
    # DURATION STRATIFICATION
    #
    # No duration filtering occurs here.
    # --------------------------------------------------------

    selected["duration_bucket"] = pd.cut(
        selected["duration"],
        bins=DURATION_BINS,
        labels=DURATION_LABELS,
        include_lowest=True,
        right=True,
    )

    if selected["duration_bucket"].isna().any():
        problem_rows = selected[
            selected["duration_bucket"].isna()
        ]

        raise RuntimeError(
            "Some recordings could not be assigned to a "
            f"duration bucket:\n{problem_rows[['file_name', 'duration']]}"
        )

    selected["is_short_le_4s"] = (
        selected["duration"]
        <= SHORT_COMMAND_MAX_SECONDS
    )

    # --------------------------------------------------------
    # DETERMINISTIC ORDER
    # --------------------------------------------------------

    selected = (
        selected
        .sort_values(
            [
                "language_variety",
                "intent_id",
                "duration",
                "file_name",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # REBUILD TARGET DIRECTORY
    # --------------------------------------------------------

    clean_target_directory()

    for file_name in selected["file_name"]:
        source_file = SOURCE_DIR / file_name
        destination_file = TARGET_DIR / file_name

        shutil.copy2(
            source_file,
            destination_file,
        )

    # --------------------------------------------------------
    # WRITE CURATED METADATA
    # --------------------------------------------------------

    temporary_csv = TARGET_CSV.with_suffix(
        ".csv.tmp"
    )

    selected.to_csv(
        temporary_csv,
        index=False,
    )

    temporary_csv.replace(
        TARGET_CSV
    )

    # --------------------------------------------------------
    # STRATIFICATION SUMMARY
    # --------------------------------------------------------

    stratification_summary = (
        selected
        .groupby(
            [
                "language_variety",
                "intent",
                "duration_bucket",
            ],
            observed=True,
        )
        .agg(
            n_samples=("file_name", "size"),
            mean_duration_s=("duration", "mean"),
            median_duration_s=("duration", "median"),
            min_duration_s=("duration", "min"),
            max_duration_s=("duration", "max"),
        )
        .reset_index()
    )

    stratification_summary[
        [
            "mean_duration_s",
            "median_duration_s",
            "min_duration_s",
            "max_duration_s",
        ]
    ] = stratification_summary[
        [
            "mean_duration_s",
            "median_duration_s",
            "min_duration_s",
            "max_duration_s",
        ]
    ].round(4)

    stratification_summary.to_csv(
        STRATIFICATION_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # INTENT x LANGUAGE SUMMARY
    # --------------------------------------------------------

    intent_language_summary = (
        selected
        .groupby(
            [
                "language_variety",
                "intent",
            ],
            observed=True,
        )
        .agg(
            n_samples=("file_name", "size"),
            short_le_4s=("is_short_le_4s", "sum"),
            median_duration_s=("duration", "median"),
        )
        .reset_index()
    )

    intent_language_summary[
        "median_duration_s"
    ] = (
        intent_language_summary[
            "median_duration_s"
        ]
        .round(4)
    )

    intent_language_summary.to_csv(
        INTENT_LANGUAGE_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    copied_files = list(
        TARGET_DIR.glob("*.wav")
    )

    if len(copied_files) != len(selected):
        raise RuntimeError(
            f"Expected {len(selected)} copied WAV files but found "
            f"{len(copied_files)}."
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("=" * 72)
    print("MINDS-14 CURATION COMPLETE")
    print("=" * 72)

    print(f"\nTotal retained utterances: {len(selected)}")

    print(
        f"Short utterances (<= {SHORT_COMMAND_MAX_SECONDS:.1f} s): "
        f"{int(selected['is_short_le_4s'].sum())}"
    )

    print(
        f"Longer utterances (> {SHORT_COMMAND_MAX_SECONDS:.1f} s): "
        f"{int((~selected['is_short_le_4s']).sum())}"
    )

    print("\nSamples by language:")
    print(
        selected["language_variety"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSamples by intent:")
    print(
        selected["intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSamples by duration bucket:")
    print(
        selected["duration_bucket"]
        .value_counts(sort=False)
        .to_string()
    )

    print("\nLanguage × short-command status:")
    print(
        pd.crosstab(
            selected["language_variety"],
            selected["is_short_le_4s"],
        ).to_string()
    )

    print(f"\nCurated audio : {TARGET_DIR.resolve()}")
    print(f"Metadata      : {TARGET_CSV.resolve()}")
    print(
        f"Stratification: "
        f"{STRATIFICATION_CSV.resolve()}"
    )
    print(
        f"Intent/language: "
        f"{INTENT_LANGUAGE_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()