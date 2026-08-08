"""
Prepare all English MINDS-14 audio for ASR benchmarking.

Downloads the pinned PolyAI/MINDS-14 revision for:
    - en-AU
    - en-GB
    - en-US

The source audio is preserved byte-for-byte. Files are written with a .wav
extension only after the encoded container has been verified as WAV-compatible.

Output:
    data/minds14_audio/
        *.wav
        metadata.csv
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "PolyAI/minds14"

# Pinned Hugging Face revision for reproducibility.
# This revision is the Parquet conversion of MINDS-14.
DATASET_REVISION = "40ce77cb32a384e4d50a568e1ec39ac804019d33"

LANGUAGE_VARIETIES = (
    "en-AU",
    "en-GB",
    "en-US",
)

# Expected counts at the pinned revision.
EXPECTED_COUNTS = {
    "en-AU": 654,
    "en-GB": 592,
    "en-US": 563,
}

OUTPUT_DIR = Path("data/minds14_audio")
METADATA_PATH = OUTPUT_DIR / "metadata.csv"

# WAV-family containers that are valid with a .wav filename.
VALID_WAV_FORMATS = {
    "WAV",
    "WAVEX",
    "RF64",
}

# Rebuild the dedicated generated directory on every run
# so stale files cannot remain from an older preparation run.
CLEAN_OUTPUT = True

# Copy/hash chunk size when a local source path is used.
COPY_CHUNK_SIZE = 1024 * 1024


# ============================================================
# HELPERS
# ============================================================

def safe_text(value: Any) -> str:
    """Return a stripped string, or an empty string for None."""
    return "" if value is None else str(value).strip()


def clean_output_directory() -> None:
    """
    Recreate only the dedicated generated output directory.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CLEAN_OUTPUT:
        return

    for path in OUTPUT_DIR.iterdir():

        if path.is_file() or path.is_symlink():
            path.unlink()

        elif path.is_dir():
            shutil.rmtree(path)


def iter_existing_source_paths(
    audio_info: dict[str, Any],
    item: dict[str, Any],
):
    """
    Yield distinct existing local audio paths in preference order.
    """

    seen: set[Path] = set()

    for raw_path in (
        audio_info.get("path"),
        item.get("path"),
    ):
        if not raw_path:
            continue

        candidate = Path(str(raw_path))

        if candidate in seen:
            continue

        seen.add(candidate)

        if candidate.is_file():
            yield candidate


def copy_source_and_hash(
    source: Path,
    destination: Path,
) -> str:
    """
    Copy a source file while calculating SHA-256 in one read pass.
    """

    digest = hashlib.sha256()

    with source.open("rb") as src, destination.open("wb") as dst:

        while True:

            chunk = src.read(COPY_CHUNK_SIZE)

            if not chunk:
                break

            digest.update(chunk)
            dst.write(chunk)

    # Preserve source timestamps/permissions where possible.
    try:
        shutil.copystat(
            source,
            destination,
        )
    except OSError:
        pass

    return digest.hexdigest()


def persist_verified_wav(
    *,
    audio_info: dict[str, Any],
    item: dict[str, Any],
    destination: Path,
):
    """
    Preserve the encoded source audio byte-for-byte, verify that
    the real container is WAV-compatible, then atomically publish
    it with a .wav filename.

    Returns
    -------
    info:
        soundfile.SoundFileInfo describing the verified audio.

    checksum:
        SHA-256 of the exact stored source audio.

    source_reference:
        Path/reference actually associated with the source audio.
    """

    temporary_path = destination.with_name(
        destination.name + ".part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    audio_bytes = audio_info.get("bytes")

    source_reference = ""

    try:

        # ----------------------------------------------------
        # CASE 1:
        # Hugging Face provides embedded encoded bytes.
        # ----------------------------------------------------

        if audio_bytes is not None:

            if not isinstance(
                audio_bytes,
                (
                    bytes,
                    bytearray,
                    memoryview,
                ),
            ):
                raise TypeError(
                    "audio['bytes'] must be bytes-like when present; "
                    f"received {type(audio_bytes)!r}."
                )

            raw_bytes = bytes(audio_bytes)

            temporary_path.write_bytes(
                raw_bytes
            )

            # No second disk read required.
            checksum = hashlib.sha256(
                raw_bytes
            ).hexdigest()

            source_reference = safe_text(
                item.get("path")
                or audio_info.get("path")
            )

        # ----------------------------------------------------
        # CASE 2:
        # Hugging Face provides a local cached source path.
        # ----------------------------------------------------

        else:

            source_path = next(
                iter_existing_source_paths(
                    audio_info,
                    item,
                ),
                None,
            )

            if source_path is None:
                raise FileNotFoundError(
                    "Audio contained neither embedded bytes nor "
                    "an existing local source path."
                )

            checksum = copy_source_and_hash(
                source=source_path,
                destination=temporary_path,
            )

            source_reference = str(
                source_path
            )

        # ----------------------------------------------------
        # VERIFY ACTUAL AUDIO CONTAINER
        #
        # Do not trust a filename extension. libsndfile reads
        # the encoded file header.
        # ----------------------------------------------------

        info = sf.info(
            temporary_path
        )

        detected_format = safe_text(
            info.format
        ).upper()

        if detected_format not in VALID_WAV_FORMATS:

            raise ValueError(
                "MINDS-14 source audio was expected to be "
                "WAV-compatible, but "
                f"{destination.name} was detected as "
                f"format={info.format!r}. "
                "The file has not been published with a "
                "misleading .wav extension."
            )

        if info.samplerate <= 0:
            raise ValueError(
                f"Invalid sample rate for {destination.name}: "
                f"{info.samplerate}"
            )

        if info.frames <= 0:
            raise ValueError(
                f"Audio contains no frames: "
                f"{destination.name}"
            )

        if info.channels <= 0:
            raise ValueError(
                f"Invalid channel count for "
                f"{destination.name}: "
                f"{info.channels}"
            )

        # Atomic move on the same filesystem.
        os.replace(
            temporary_path,
            destination,
        )

        return (
            info,
            checksum,
            source_reference,
        )

    except Exception:

        # Never leave a partial audio artefact behind.
        if temporary_path.exists():
            temporary_path.unlink()

        raise


def validate_class_label(
    feature: Any,
    field_name: str,
) -> list[str]:
    """
    Return Hugging Face ClassLabel names or fail clearly.
    """

    names = getattr(
        feature,
        "names",
        None,
    )

    if not names:
        raise TypeError(
            f"Dataset feature {field_name!r} is expected "
            "to expose ClassLabel names, but none were found."
        )

    return list(names)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    clean_output_directory()

    rows: list[dict[str, Any]] = []

    observed_counts: dict[str, int] = {}

    print("=" * 72)
    print("Preparing MINDS-14 English ASR dataset")
    print("=" * 72)

    print(
        f"Dataset       : {DATASET_NAME}"
    )

    print(
        f"Revision      : {DATASET_REVISION}"
    )

    print(
        f"Languages     : "
        f"{', '.join(LANGUAGE_VARIETIES)}"
    )

    print(
        f"Output        : "
        f"{OUTPUT_DIR.resolve()}"
    )

    print()

    # ========================================================
    # PROCESS EACH ENGLISH VARIETY
    # ========================================================

    for language in LANGUAGE_VARIETIES:

        print(
            f"[LOAD] {language}"
        )

        # The pinned revision is Parquet-based.
        # No trust_remote_code=True is required.
        dataset = load_dataset(
            DATASET_NAME,
            language,
            split="train",
            revision=DATASET_REVISION,
        )

        # Disable automatic decoding.
        #
        # This exposes the encoded bytes/path instead of
        # automatically converting audio to an array.
        dataset = dataset.cast_column(
            "audio",
            Audio(decode=False),
        )

        # ----------------------------------------------------
        # SCHEMA VALIDATION
        # ----------------------------------------------------

        required_columns = {
            "path",
            "audio",
            "transcription",
            "english_transcription",
            "intent_class",
            "lang_id",
        }

        missing_columns = (
            required_columns
            .difference(
                dataset.column_names
            )
        )

        if missing_columns:

            raise RuntimeError(
                f"Unexpected MINDS-14 schema for "
                f"{language}; missing columns: "
                f"{sorted(missing_columns)}"
            )

        # ----------------------------------------------------
        # CLASS-LABEL DEFINITIONS
        # ----------------------------------------------------

        intent_names = validate_class_label(
            dataset.features[
                "intent_class"
            ],
            "intent_class",
        )

        language_names = validate_class_label(
            dataset.features[
                "lang_id"
            ],
            "lang_id",
        )

        # ----------------------------------------------------
        # SAMPLE COUNT VALIDATION
        # ----------------------------------------------------

        observed_count = len(
            dataset
        )

        observed_counts[
            language
        ] = observed_count

        expected_count = (
            EXPECTED_COUNTS[
                language
            ]
        )

        if observed_count != expected_count:

            raise RuntimeError(
                f"Unexpected sample count for "
                f"{language}: expected "
                f"{expected_count}, found "
                f"{observed_count}. "
                "Because the dataset revision is pinned, "
                "this indicates a loading or schema problem "
                "that should be investigated before "
                "benchmarking."
            )

        # Independent counter for every intent
        # inside each language variety.
        intent_counters: Counter[str] = (
            Counter()
        )

        # ====================================================
        # PROCESS RECORDS
        # ====================================================

        for source_index, item in enumerate(
            dataset
        ):

            audio_info = item.get(
                "audio"
            )

            if not isinstance(
                audio_info,
                dict,
            ):
                raise TypeError(
                    f"Unexpected audio object for "
                    f"{language}, row "
                    f"{source_index}: "
                    f"{type(audio_info)!r}"
                )

            # -----------------------------------------------
            # INTENT
            # -----------------------------------------------

            intent_id = int(
                item["intent_class"]
            )

            if not (
                0
                <= intent_id
                < len(intent_names)
            ):
                raise ValueError(
                    f"Invalid intent_class="
                    f"{intent_id} for "
                    f"{language}, row "
                    f"{source_index}."
                )

            intent_name = (
                intent_names[
                    intent_id
                ]
            )

            # -----------------------------------------------
            # LANGUAGE ID
            # -----------------------------------------------

            lang_id = int(
                item["lang_id"]
            )

            if not (
                0
                <= lang_id
                < len(language_names)
            ):
                raise ValueError(
                    f"Invalid lang_id="
                    f"{lang_id} for "
                    f"{language}, row "
                    f"{source_index}."
                )

            resolved_language = (
                language_names[
                    lang_id
                ]
            )

            if (
                resolved_language
                != language
            ):
                raise RuntimeError(
                    f"Language mismatch at "
                    f"{language}, row "
                    f"{source_index}: "
                    f"configuration={language}, "
                    f"lang_id resolves to "
                    f"{resolved_language}."
                )

            # -----------------------------------------------
            # TRANSCRIPTS
            # -----------------------------------------------

            transcript = safe_text(
                item.get(
                    "transcription"
                )
            )

            english_transcription = (
                safe_text(
                    item.get(
                        "english_transcription"
                    )
                )
            )

            # All selected configurations are English
            # varieties. The native transcript is therefore
            # the correct ASR reference.
            if not transcript:

                raise ValueError(
                    f"Empty transcription for "
                    f"{language}, row "
                    f"{source_index}."
                )

            # -----------------------------------------------
            # DETERMINISTIC OUTPUT NAME
            # -----------------------------------------------

            intent_counters[
                intent_name
            ] += 1

            intent_number = (
                intent_counters[
                    intent_name
                ]
            )

            filename = (
                f"{language}_"
                f"{intent_name}_"
                f"{intent_number:04d}.wav"
            )

            destination = (
                OUTPUT_DIR
                / filename
            )

            # -----------------------------------------------
            # WRITE + VERIFY SOURCE AUDIO
            # -----------------------------------------------

            (
                info,
                checksum,
                actual_source_reference,
            ) = persist_verified_wav(
                audio_info=audio_info,
                item=item,
                destination=destination,
            )

            duration_seconds = (
                info.frames
                / float(
                    info.samplerate
                )
            )

            # -----------------------------------------------
            # METADATA
            # -----------------------------------------------

            rows.append(
                {
                    "file_name": filename,

                    "language_variety": (
                        language
                    ),

                    "lang_id": lang_id,

                    "intent_id": intent_id,

                    "intent": intent_name,

                    "transcript": (
                        transcript
                    ),

                    "english_transcription": (
                        english_transcription
                    ),

                    "duration": (
                        duration_seconds
                    ),

                    "sample_rate": int(
                        info.samplerate
                    ),

                    "channels": int(
                        info.channels
                    ),

                    "frames": int(
                        info.frames
                    ),

                    "audio_container": (
                        safe_text(
                            info.format
                        )
                    ),

                    "audio_subtype": (
                        safe_text(
                            info.subtype
                        )
                    ),

                    "source_index": int(
                        source_index
                    ),

                    # Logical path supplied by MINDS-14.
                    "dataset_path": (
                        safe_text(
                            item.get(
                                "path"
                            )
                        )
                    ),

                    # Actual path/reference used/found
                    # by this preparation run.
                    "source_path": (
                        safe_text(
                            actual_source_reference
                            or item.get(
                                "path"
                            )
                            or audio_info.get(
                                "path"
                            )
                        )
                    ),

                    "sha256": checksum,

                    "dataset_revision": (
                        DATASET_REVISION
                    ),
                }
            )

        print(
            f"       Saved "
            f"{observed_count} "
            f"utterances."
        )

    # ========================================================
    # BUILD METADATA
    # ========================================================

    metadata = pd.DataFrame(
        rows
    )

    if metadata.empty:

        raise RuntimeError(
            "No MINDS-14 records "
            "were prepared."
        )

    # --------------------------------------------------------
    # TOTAL COUNT
    # --------------------------------------------------------

    expected_total = sum(
        EXPECTED_COUNTS.values()
    )

    if len(metadata) != expected_total:

        raise RuntimeError(
            f"Expected {expected_total} "
            "total English utterances "
            f"but produced "
            f"{len(metadata)} "
            "metadata rows."
        )

    # --------------------------------------------------------
    # UNIQUE FILENAMES
    # --------------------------------------------------------

    if metadata[
        "file_name"
    ].duplicated().any():

        duplicates = (
            metadata.loc[
                metadata[
                    "file_name"
                ].duplicated(
                    keep=False
                ),
                "file_name",
            ]
            .tolist()
        )

        raise RuntimeError(
            "Duplicate output filenames "
            "detected. First duplicates: "
            f"{duplicates[:10]}"
        )

    # --------------------------------------------------------
    # DUPLICATE AUDIO CONTENT
    # --------------------------------------------------------

    if metadata[
        "sha256"
    ].duplicated().any():

        duplicate_hash_rows = (
            metadata[
                metadata[
                    "sha256"
                ].duplicated(
                    keep=False
                )
            ]
            .sort_values(
                "sha256"
            )
        )

        print(
            "\n[WARN] Duplicate audio "
            "content detected by SHA-256: "
            f"{len(duplicate_hash_rows)} rows. "
            "Review metadata before deciding "
            "whether duplicates should be "
            "retained."
        )

    # --------------------------------------------------------
    # CHECKSUM VALIDATION
    # --------------------------------------------------------

    if (
        metadata[
            "sha256"
        ].isna().any()
        or (
            metadata[
                "sha256"
            ] == ""
        ).any()
    ):
        raise RuntimeError(
            "Missing SHA-256 values "
            "detected."
        )

    # --------------------------------------------------------
    # DURATION VALIDATION
    # --------------------------------------------------------

    if (
        metadata[
            "duration"
        ] <= 0
    ).any():

        raise RuntimeError(
            "Non-positive audio "
            "durations detected."
        )

    # --------------------------------------------------------
    # CONTAINER VALIDATION
    # --------------------------------------------------------

    if not (
        metadata[
            "audio_container"
        ]
        .str.upper()
        .isin(
            VALID_WAV_FORMATS
        )
        .all()
    ):
        raise RuntimeError(
            "A non-WAV-compatible "
            "audio container reached metadata."
        )

    # --------------------------------------------------------
    # DETERMINISTIC ORDER
    # --------------------------------------------------------

    metadata = (
        metadata
        .sort_values(
            [
                "language_variety",
                "intent_id",
                "file_name",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # AUDIO <-> METADATA CONSISTENCY
    # ========================================================

    output_audio_files = sorted(
        OUTPUT_DIR.glob(
            "*.wav"
        )
    )

    if (
        len(output_audio_files)
        != len(metadata)
    ):
        raise RuntimeError(
            f"Expected "
            f"{len(metadata)} WAV files "
            f"but found "
            f"{len(output_audio_files)} "
            f"in {OUTPUT_DIR}."
        )

    metadata_names = set(
        metadata[
            "file_name"
        ].astype(str)
    )

    output_names = {
        path.name
        for path
        in output_audio_files
    }

    missing_audio = sorted(
        metadata_names
        - output_names
    )

    untracked_audio = sorted(
        output_names
        - metadata_names
    )

    if (
        missing_audio
        or untracked_audio
    ):

        raise RuntimeError(
            "Audio/metadata mismatch "
            "detected. "
            f"Missing audio="
            f"{missing_audio[:5]}, "
            f"untracked audio="
            f"{untracked_audio[:5]}"
        )

    # ========================================================
    # ATOMIC METADATA WRITE
    # ========================================================

    temporary_metadata = (
        METADATA_PATH.with_name(
            METADATA_PATH.name
            + ".tmp"
        )
    )

    metadata.to_csv(
        temporary_metadata,
        index=False,
    )

    os.replace(
        temporary_metadata,
        METADATA_PATH,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()

    print("=" * 72)
    print("PREPARATION COMPLETE")
    print("=" * 72)

    print(
        f"Total utterances: "
        f"{len(metadata)}"
    )

    print(
        "\nUtterances by "
        "language variety:"
    )

    print(
        metadata[
            "language_variety"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nUtterances by intent:"
    )

    print(
        metadata[
            "intent"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nLanguage × intent counts:"
    )

    print(
        pd.crosstab(
            metadata[
                "language_variety"
            ],
            metadata[
                "intent"
            ],
        ).to_string()
    )

    print(
        "\nDuration statistics "
        "(seconds):"
    )

    print(
        metadata[
            "duration"
        ]
        .describe()
        .round(3)
        .to_string()
    )

    print(
        "\nSource sample rates:"
    )

    print(
        metadata[
            "sample_rate"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nAudio directory : "
        f"{OUTPUT_DIR.resolve()}"
    )

    print(
        f"Metadata        : "
        f"{METADATA_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()