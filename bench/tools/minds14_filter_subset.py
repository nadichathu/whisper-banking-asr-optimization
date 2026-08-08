"""
Curate the prepared English MINDS-14 dataset for short banking commands.

Default behaviour:
    - retain en-AU, en-GB and en-US;
    - retain all 14 banking intents;
    - retain only utterances <= 6.0 seconds;
    - create duration strata;
    - preserve language-variety labels;
    - copy only the selected source audio into the curated raw directory.

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

STRATIFICATION_CSV = TARGET_DIR / "stratification_summary.csv"
INTENT_LANGUAGE_CSV = TARGET_DIR / "intent_language_summary.csv"

LANGUAGE_VARIETIES = (
    "en-AU",
    "en-GB",
    "en-US",
)

# None = retain all MINDS-14 intents.
SELECTED_INTENTS: list[str] | None = None

# Primary short-command benchmark threshold.
SHORT_COMMAND_MAX_SECONDS = 6.0

# Expected count for the pinned three-English-variety MINDS-14 dataset.
EXPECTED_SELECTED_FILES = 809

DURATION_BINS = [
    0.0,
    2.0,
    4.0,
    6.0,
]

DURATION_LABELS = [
    "≤2 s",
    ">2–4 s",
    ">4–6 s",
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

    # --------------------------------------------------------
    # LOAD SOURCE METADATA
    # --------------------------------------------------------

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

    eligible = source_df[
        source_df["language_variety"].isin(
            LANGUAGE_VARIETIES
        )
    ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No rows remain after language filtering."
        )

    # --------------------------------------------------------
    # OPTIONAL INTENT SELECTION
    # --------------------------------------------------------

    if SELECTED_INTENTS is not None:

        unknown_intents = (
            set(SELECTED_INTENTS)
            - set(eligible["intent"].unique())
        )

        if unknown_intents:
            raise ValueError(
                "Requested intents were not found in the dataset: "
                f"{sorted(unknown_intents)}"
            )

        eligible = eligible[
            eligible["intent"].isin(
                SELECTED_INTENTS
            )
        ].copy()

    if eligible.empty:
        raise RuntimeError(
            "No rows remain after intent selection."
        )

    # --------------------------------------------------------
    # SHORT-COMMAND DURATION FILTER
    #
    # IMPORTANT:
    # Files longer than 6 seconds are removed here.
    # --------------------------------------------------------

    total_eligible = len(eligible)

    selected = eligible[
        eligible["duration"]
        <= SHORT_COMMAND_MAX_SECONDS
    ].copy()

    if selected.empty:
        raise RuntimeError(
            "No rows remain after duration filtering."
        )

    excluded_by_duration = (
        total_eligible - len(selected)
    )

    selected_share_pct = (
        len(selected)
        / total_eligible
        * 100.0
    )

    # --------------------------------------------------------
    # EXPECTED DATASET SIZE VALIDATION
    # --------------------------------------------------------

    if (
        EXPECTED_SELECTED_FILES is not None
        and len(selected) != EXPECTED_SELECTED_FILES
    ):
        raise RuntimeError(
            "Unexpected short-command dataset size.\n"
            f"Expected: {EXPECTED_SELECTED_FILES}\n"
            f"Found   : {len(selected)}\n"
            "Check the dataset revision and duration filtering."
        )

    # --------------------------------------------------------
    # DURATION STRATIFICATION
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
            f"duration bucket:\n"
            f"{problem_rows[['file_name', 'duration']]}"
        )

    # Explicit provenance flag.
    selected["is_short_le_6s"] = True

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

    observed_stratification = (
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

    observed_stratification[
        "duration_bucket"
    ] = (
        observed_stratification[
            "duration_bucket"
        ]
        .astype(str)
    )

    # Explicit complete grid so zero-count strata are not hidden.
    intent_names = sorted(
        selected["intent"].unique()
    )

    complete_stratification_grid = (
        pd.MultiIndex.from_product(
            [
                LANGUAGE_VARIETIES,
                intent_names,
                DURATION_LABELS,
            ],
            names=[
                "language_variety",
                "intent",
                "duration_bucket",
            ],
        )
        .to_frame(index=False)
    )

    stratification_summary = (
        complete_stratification_grid
        .merge(
            observed_stratification,
            on=[
                "language_variety",
                "intent",
                "duration_bucket",
            ],
            how="left",
        )
    )

    stratification_summary[
        "n_samples"
    ] = (
        stratification_summary[
            "n_samples"
        ]
        .fillna(0)
        .astype(int)
    )

    duration_stat_columns = [
        "mean_duration_s",
        "median_duration_s",
        "min_duration_s",
        "max_duration_s",
    ]

    stratification_summary[
        duration_stat_columns
    ] = (
        stratification_summary[
            duration_stat_columns
        ]
        .round(4)
    )

    stratification_summary.to_csv(
        STRATIFICATION_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # INTENT × LANGUAGE SUMMARY
    # --------------------------------------------------------

    observed_intent_language = (
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
            median_duration_s=("duration", "median"),
            mean_duration_s=("duration", "mean"),
        )
        .reset_index()
    )

    complete_intent_language_grid = (
        pd.MultiIndex.from_product(
            [
                LANGUAGE_VARIETIES,
                intent_names,
            ],
            names=[
                "language_variety",
                "intent",
            ],
        )
        .to_frame(index=False)
    )

    intent_language_summary = (
        complete_intent_language_grid
        .merge(
            observed_intent_language,
            on=[
                "language_variety",
                "intent",
            ],
            how="left",
        )
    )

    intent_language_summary[
        "n_samples"
    ] = (
        intent_language_summary[
            "n_samples"
        ]
        .fillna(0)
        .astype(int)
    )

    intent_language_summary[
        [
            "median_duration_s",
            "mean_duration_s",
        ]
    ] = (
        intent_language_summary[
            [
                "median_duration_s",
                "mean_duration_s",
            ]
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

    copied_names = {
        path.name
        for path in copied_files
    }

    metadata_names = set(
        selected["file_name"]
    )

    if copied_names != metadata_names:
        raise RuntimeError(
            "Copied WAV filenames do not exactly match metadata."
        )

    if (
        selected["duration"]
        > SHORT_COMMAND_MAX_SECONDS
    ).any():
        raise RuntimeError(
            "Duration filtering validation failed: "
            "a file longer than 6 seconds remains."
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("=" * 72)
    print("MINDS-14 SHORT-COMMAND CURATION COMPLETE")
    print("=" * 72)

    print(
        f"\nEligible English files before duration filter: "
        f"{total_eligible}"
    )

    print(
        f"Selected files (<= {SHORT_COMMAND_MAX_SECONDS:.1f} s): "
        f"{len(selected)}"
    )

    print(
        f"Not selected (> {SHORT_COMMAND_MAX_SECONDS:.1f} s): "
        f"{excluded_by_duration}"
    )

    print(
        f"Selected share: "
        f"{selected_share_pct:.2f}%"
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

    print("\nLanguage × intent counts:")
    print(
        pd.crosstab(
            selected["language_variety"],
            selected["intent"],
        ).to_string()
    )

    print(
        "\nMinimum samples in any language × intent cell: "
        f"{intent_language_summary['n_samples'].min()}"
    )

    print(
        "Zero-count cells: "
        f"{int((intent_language_summary['n_samples'] == 0).sum())}"
    )

    print(
        "Cells with <10 samples: "
        f"{int((intent_language_summary['n_samples'] < 10).sum())}"
    )

    print(
        "Cells with <15 samples: "
        f"{int((intent_language_summary['n_samples'] < 15).sum())}"
    )

    print(
        "Cells with <20 samples: "
        f"{int((intent_language_summary['n_samples'] < 20).sum())}"
    )

    print(
        f"\nCurated audio   : "
        f"{TARGET_DIR.resolve()}"
    )

    print(
        f"Metadata        : "
        f"{TARGET_CSV.resolve()}"
    )

    print(
        f"Stratification  : "
        f"{STRATIFICATION_CSV.resolve()}"
    )

    print(
        f"Intent/language : "
        f"{INTENT_LANGUAGE_CSV.resolve()}"
    )


if __name__ == "__main__":
    main()