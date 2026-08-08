"""
Convert the curated MINDS-14 English banking corpus to
16-kHz mono WAV for ASR benchmarking.

Non-destructive design:
    input  -> data/minds14_banking_raw/
    output -> data/minds14_banking/

The original downloaded and curated audio remains untouched.

Resampling uses scipy.signal.resample_poly with an exact
integer up/down ratio derived using gcd().

The final metadata preserves the original source SHA-256
checksum and records a new checksum for the generated
16-kHz mono PCM16 WAV file.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path("data/minds14_banking_raw")
SOURCE_METADATA = SOURCE_DIR / "metadata.csv"

TARGET_DIR = Path("data/minds14_banking")
TARGET_METADATA = TARGET_DIR / "metadata.csv"

TARGET_SAMPLE_RATE = 16_000

OUTPUT_FORMAT = "WAV"
OUTPUT_SUBTYPE = "PCM_16"

CLEAN_OUTPUT = True

# Maximum allowed absolute duration difference between
# source and output audio after resampling.
DURATION_TOLERANCE_SECONDS = 0.002

HASH_CHUNK_SIZE = 1024 * 1024


# ============================================================
# HASHING
# ============================================================

def sha256_file(path: Path) -> str:
    """
    Calculate the SHA-256 checksum of a file.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file_obj:

        while True:

            chunk = file_obj.read(
                HASH_CHUNK_SIZE
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# TARGET DIRECTORY
# ============================================================

def clean_target_directory() -> None:
    """
    Safely recreate the generated output directory.
    """

    source_resolved = SOURCE_DIR.resolve()
    target_resolved = TARGET_DIR.resolve()

    if source_resolved == target_resolved:

        raise RuntimeError(
            "SOURCE_DIR and TARGET_DIR must be different. "
            "Refusing destructive in-place resampling."
        )

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not CLEAN_OUTPUT:
        return

    for path in TARGET_DIR.iterdir():

        if path.is_file() or path.is_symlink():

            path.unlink()

        elif path.is_dir():

            shutil.rmtree(path)


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio_mono(
    audio_path: Path,
) -> tuple[np.ndarray, int, Any]:
    """
    Load source audio as float32 and convert it to mono.

    Returns
    -------
    mono_audio
        One-dimensional float32 waveform.

    sample_rate
        Original source sample rate.

    source_info
        soundfile metadata for the original file.
    """

    source_info = sf.info(
        audio_path
    )

    if source_info.samplerate <= 0:

        raise ValueError(
            f"Invalid sample rate for {audio_path}: "
            f"{source_info.samplerate}"
        )

    if source_info.frames <= 0:

        raise ValueError(
            f"Empty audio file: {audio_path}"
        )

    if source_info.channels <= 0:

        raise ValueError(
            f"Invalid channel count for {audio_path}: "
            f"{source_info.channels}"
        )

    audio, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )

    if sample_rate != source_info.samplerate:

        raise RuntimeError(
            f"Inconsistent sample rate reported for {audio_path}: "
            f"sf.info={source_info.samplerate}, "
            f"sf.read={sample_rate}"
        )

    if audio.shape[0] != source_info.frames:

        raise RuntimeError(
            f"Inconsistent frame count reported for {audio_path}: "
            f"sf.info={source_info.frames}, "
            f"sf.read={audio.shape[0]}"
        )

    if not np.isfinite(audio).all():

        raise ValueError(
            f"NaN or infinite samples detected in {audio_path}"
        )

    # soundfile returns:
    # [samples, channels]
    #
    # Averaging across channels creates a mono waveform.
    mono = audio.mean(
        axis=1,
        dtype=np.float32,
    )

    if not np.isfinite(mono).all():

        raise ValueError(
            "NaN or infinite samples detected after "
            f"mono conversion: {audio_path}"
        )

    return (
        mono,
        int(sample_rate),
        source_info,
    )


# ============================================================
# RESAMPLING
# ============================================================

def resample_audio(
    audio: np.ndarray,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    """
    Resample using polyphase filtering.

    The up/down ratio is reduced to its smallest integer
    representation using gcd().
    """

    if source_rate <= 0 or target_rate <= 0:

        raise ValueError(
            "Sample rates must be positive: "
            f"source={source_rate}, "
            f"target={target_rate}"
        )

    if source_rate == target_rate:

        output = np.asarray(
            audio,
            dtype=np.float32,
        )

    else:

        divisor = math.gcd(
            source_rate,
            target_rate,
        )

        up = (
            target_rate
            // divisor
        )

        down = (
            source_rate
            // divisor
        )

        output = resample_poly(
            audio,
            up=up,
            down=down,
        )

        output = np.asarray(
            output,
            dtype=np.float32,
        )

    if output.size == 0:

        raise ValueError(
            "Resampling produced an empty waveform."
        )

    if not np.isfinite(output).all():

        raise ValueError(
            "Resampling produced NaN or infinite values."
        )

    return output


# ============================================================
# PCM16 PREPARATION
# ============================================================

def prepare_pcm16_audio(
    audio: np.ndarray,
) -> np.ndarray:
    """
    Prepare floating-point audio for safe PCM16 conversion.

    This clipping is applied regardless of whether resampling
    was required, preventing out-of-range float samples from
    being passed to the PCM16 encoder.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if not np.isfinite(audio).all():

        raise ValueError(
            "Audio contains NaN or infinite values "
            "before PCM16 writing."
        )

    return np.clip(
        audio,
        -1.0,
        1.0,
    )


# ============================================================
# ATOMIC CSV WRITING
# ============================================================

def atomic_write_csv(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    """
    Write a CSV atomically.
    """

    temporary_path = destination.with_name(
        destination.name + ".tmp"
    )

    if temporary_path.exists():

        temporary_path.unlink()

    try:

        dataframe.to_csv(
            temporary_path,
            index=False,
        )

        os.replace(
            temporary_path,
            destination,
        )

    except Exception:

        if temporary_path.exists():

            temporary_path.unlink()

        raise


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    # ========================================================
    # 1. LOAD SOURCE METADATA
    # ========================================================

    if not SOURCE_METADATA.is_file():

        raise FileNotFoundError(
            f"Metadata not found: {SOURCE_METADATA}\n"
            "Run minds14_filter_subset.py first."
        )

    metadata = pd.read_csv(
        SOURCE_METADATA
    )

    if metadata.empty:

        raise RuntimeError(
            "Source metadata contains no rows."
        )


    # ========================================================
    # 2. REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "file_name",
        "transcript",
        "duration",
        "language_variety",
        "intent",
        "sha256",
    }

    missing_columns = (
        required_columns
        - set(metadata.columns)
    )

    if missing_columns:

        raise ValueError(
            "Source metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )


    # ========================================================
    # 3. SOURCE METADATA VALIDATION
    # ========================================================

    if metadata[
        "file_name"
    ].isna().any():

        raise RuntimeError(
            "Missing file names detected in source metadata."
        )

    if metadata[
        "file_name"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate file names detected in source metadata."
        )

    if metadata[
        "sha256"
    ].isna().any():

        raise RuntimeError(
            "Missing source SHA-256 values detected."
        )


    # ========================================================
    # 4. VERIFY SOURCE FILES EXIST
    # ========================================================

    missing_files = [

        str(file_name)

        for file_name
        in metadata[
            "file_name"
        ]

        if not (
            SOURCE_DIR
            / str(file_name)
        ).is_file()
    ]

    if missing_files:

        raise FileNotFoundError(
            f"{len(missing_files)} source audio files "
            "are missing. "
            f"First missing file: {missing_files[0]}"
        )


    # ========================================================
    # 5. PREPARE OUTPUT DIRECTORY
    # ========================================================

    clean_target_directory()

    output_rows: list[
        dict[str, Any]
    ] = []

    total = len(
        metadata
    )

    print("=" * 72)
    print("RESAMPLING MINDS-14 TO 16-kHz MONO")
    print("=" * 72)

    print(
        f"Files: {total}"
    )

    print()


    # ========================================================
    # 6. PROCESS EACH AUDIO FILE
    # ========================================================

    for position, (_, row) in enumerate(
        metadata.iterrows(),
        start=1,
    ):

        file_name = str(
            row[
                "file_name"
            ]
        )

        source_path = (
            SOURCE_DIR
            / file_name
        )

        target_path = (
            TARGET_DIR
            / file_name
        )


        # ----------------------------------------------------
        # VERIFY SOURCE SHA-256
        # ----------------------------------------------------

        expected_source_sha256 = (
            str(
                row[
                    "sha256"
                ]
            )
            .strip()
            .lower()
        )

        if len(
            expected_source_sha256
        ) != 64:

            raise ValueError(
                "Invalid SHA-256 value in metadata for "
                f"{file_name}: "
                f"{expected_source_sha256!r}"
            )

        observed_source_sha256 = (
            sha256_file(
                source_path
            )
        )

        if (
            observed_source_sha256.lower()
            != expected_source_sha256
        ):

            raise RuntimeError(
                f"Source SHA-256 mismatch for {file_name}.\n"
                f"Expected: {expected_source_sha256}\n"
                f"Observed: {observed_source_sha256}"
            )


        # ----------------------------------------------------
        # LOAD + MONO
        # ----------------------------------------------------

        (
            audio,
            source_rate,
            source_info,
        ) = load_audio_mono(
            source_path
        )

        source_duration = (
            source_info.frames
            / float(
                source_info.samplerate
            )
        )


        # ----------------------------------------------------
        # VERIFY SOURCE DURATION AGAINST METADATA
        # ----------------------------------------------------

        metadata_source_duration = float(
            row[
                "duration"
            ]
        )

        if (
            abs(
                metadata_source_duration
                - source_duration
            )
            > DURATION_TOLERANCE_SECONDS
        ):

            raise RuntimeError(
                "Source duration does not match metadata "
                f"for {file_name}: "
                f"metadata={metadata_source_duration:.6f}s, "
                f"audio={source_duration:.6f}s"
            )


        # ----------------------------------------------------
        # RESAMPLE
        # ----------------------------------------------------

        resampled = resample_audio(
            audio=audio,
            source_rate=source_rate,
            target_rate=TARGET_SAMPLE_RATE,
        )


        # ----------------------------------------------------
        # CLIP SAFELY FOR PCM16
        #
        # This happens for BOTH:
        #   - resampled audio;
        #   - audio already at 16 kHz.
        # ----------------------------------------------------

        output_audio = prepare_pcm16_audio(
            resampled
        )


        # ----------------------------------------------------
        # ATOMIC AUDIO WRITE
        # ----------------------------------------------------

        temporary_path = target_path.with_name(
            target_path.stem
            + ".tmp.wav"
        )

        if temporary_path.exists():

            temporary_path.unlink()

        try:

            sf.write(
                temporary_path,
                output_audio,
                TARGET_SAMPLE_RATE,
                subtype=OUTPUT_SUBTYPE,
                format=OUTPUT_FORMAT,
            )


            # ------------------------------------------------
            # VERIFY WRITTEN TEMPORARY FILE
            # ------------------------------------------------

            output_info = sf.info(
                temporary_path
            )

            if (
                output_info.format
                != OUTPUT_FORMAT
            ):

                raise RuntimeError(
                    f"Incorrect output format for {file_name}: "
                    f"{output_info.format}"
                )

            if (
                output_info.subtype
                != OUTPUT_SUBTYPE
            ):

                raise RuntimeError(
                    f"Incorrect output subtype for {file_name}: "
                    f"{output_info.subtype}"
                )

            if (
                output_info.samplerate
                != TARGET_SAMPLE_RATE
            ):

                raise RuntimeError(
                    "Incorrect output sample rate for "
                    f"{file_name}: "
                    f"{output_info.samplerate}"
                )

            if output_info.channels != 1:

                raise RuntimeError(
                    f"Output is not mono for {file_name}: "
                    f"{output_info.channels} channels"
                )

            if output_info.frames <= 0:

                raise RuntimeError(
                    "Output contains no audio frames: "
                    f"{file_name}"
                )


            # ------------------------------------------------
            # VERIFY DURATION
            # ------------------------------------------------

            output_duration = (
                output_info.frames
                / float(
                    output_info.samplerate
                )
            )

            if (
                abs(
                    output_duration
                    - source_duration
                )
                > DURATION_TOLERANCE_SECONDS
            ):

                raise RuntimeError(
                    "Unexpected duration change after "
                    f"resampling {file_name}: "
                    f"source={source_duration:.6f}s, "
                    f"output={output_duration:.6f}s"
                )


            # ------------------------------------------------
            # PUBLISH FINAL FILE ATOMICALLY
            # ------------------------------------------------

            os.replace(
                temporary_path,
                target_path,
            )


        except Exception:

            if temporary_path.exists():

                temporary_path.unlink()

            raise


        # ----------------------------------------------------
        # COMPUTE FINAL OUTPUT SHA-256
        # ----------------------------------------------------

        output_sha256 = sha256_file(
            target_path
        )


        # ----------------------------------------------------
        # UPDATED METADATA
        # ----------------------------------------------------

        output_row = (
            row.to_dict()
        )


        # ----------------------------------------------------
        # PRESERVE ORIGINAL SOURCE INFORMATION
        # ----------------------------------------------------

        output_row[
            "source_sha256"
        ] = (
            expected_source_sha256
        )

        output_row[
            "source_sample_rate"
        ] = int(
            source_info.samplerate
        )

        output_row[
            "source_channels"
        ] = int(
            source_info.channels
        )

        output_row[
            "source_frames"
        ] = int(
            source_info.frames
        )

        output_row[
            "source_duration"
        ] = (
            source_duration
        )

        output_row[
            "source_audio_format"
        ] = str(
            source_info.format
        )

        output_row[
            "source_audio_subtype"
        ] = str(
            source_info.subtype
        )


        # ----------------------------------------------------
        # FINAL BENCHMARK FILE INFORMATION
        # ----------------------------------------------------

        output_row[
            "sha256"
        ] = (
            output_sha256
        )

        output_row[
            "sample_rate"
        ] = (
            TARGET_SAMPLE_RATE
        )

        output_row[
            "channels"
        ] = 1

        output_row[
            "frames"
        ] = int(
            output_info.frames
        )

        output_row[
            "duration"
        ] = (
            output_duration
        )

        output_row[
            "audio_format"
        ] = str(
            output_info.format
        )

        output_row[
            "audio_subtype"
        ] = str(
            output_info.subtype
        )

        output_row[
            "resampled"
        ] = (
            source_rate
            != TARGET_SAMPLE_RATE
        )

        output_row[
            "downmixed_to_mono"
        ] = (
            source_info.channels
            != 1
        )


        output_rows.append(
            output_row
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            position == 1
            or position % 100 == 0
            or position == total
        ):

            print(
                f"[{position:4d}/{total}] "
                f"{file_name}"
            )


    # ========================================================
    # 7. BUILD FINAL METADATA
    # ========================================================

    output_metadata = pd.DataFrame(
        output_rows
    )

    output_metadata = (
        output_metadata
        .sort_values(
            [
                "language_variety",
                "intent",
                "file_name",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # 8. FINAL FILE COUNT VALIDATION
    # ========================================================

    output_files = sorted(
        TARGET_DIR.glob(
            "*.wav"
        )
    )

    if (
        len(output_files)
        != len(output_metadata)
    ):

        raise RuntimeError(
            f"Expected {len(output_metadata)} "
            "output WAV files but found "
            f"{len(output_files)}."
        )


    # ========================================================
    # 9. FILE/METADATA CONSISTENCY
    # ========================================================

    metadata_names = set(
        output_metadata[
            "file_name"
        ].astype(str)
    )

    output_names = {
        path.name
        for path
        in output_files
    }

    missing_outputs = sorted(
        metadata_names
        - output_names
    )

    unexpected_outputs = sorted(
        output_names
        - metadata_names
    )

    if (
        missing_outputs
        or unexpected_outputs
    ):

        raise RuntimeError(
            "Output audio/metadata mismatch. "
            f"Missing={missing_outputs[:5]}, "
            f"unexpected={unexpected_outputs[:5]}"
        )


    # ========================================================
    # 10. FINAL AUDIO METADATA VALIDATION
    # ========================================================

    if not (
        output_metadata[
            "sample_rate"
        ]
        == TARGET_SAMPLE_RATE
    ).all():

        raise RuntimeError(
            "Metadata contains an unexpected "
            "output sample rate."
        )

    if not (
        output_metadata[
            "channels"
        ]
        == 1
    ).all():

        raise RuntimeError(
            "Metadata contains non-mono output audio."
        )

    if output_metadata[
        "sha256"
    ].isna().any():

        raise RuntimeError(
            "Final metadata contains missing "
            "output SHA-256 values."
        )


    # ========================================================
    # 11. WRITE FINAL METADATA
    # ========================================================

    atomic_write_csv(
        output_metadata,
        TARGET_METADATA,
    )


    # ========================================================
    # 12. REPORT
    # ========================================================

    print()

    print("=" * 72)
    print("RESAMPLING COMPLETE")
    print("=" * 72)

    print(
        f"Processed files : "
        f"{len(output_metadata)}"
    )

    print(
        f"Sample rate     : "
        f"{TARGET_SAMPLE_RATE} Hz"
    )

    print(
        "Channels        : 1 (mono)"
    )

    print(
        f"Format          : "
        f"{OUTPUT_FORMAT} / "
        f"{OUTPUT_SUBTYPE}"
    )

    print(
        f"Audio directory : "
        f"{TARGET_DIR.resolve()}"
    )

    print(
        f"Metadata        : "
        f"{TARGET_METADATA.resolve()}"
    )

    print(
        "\nDuration statistics "
        "(seconds):"
    )

    print(
        output_metadata[
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
        output_metadata[
            "source_sample_rate"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nResampled:"
    )

    print(
        output_metadata[
            "resampled"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()