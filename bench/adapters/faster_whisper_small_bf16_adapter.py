"""
Faster-Whisper Small BF16 adapter.

Experimental condition
----------------------
Model:
    Whisper Small

Implementation:
    Faster-Whisper

Backend:
    CTranslate2

Compute type:
    BF16 / bfloat16

Device:
    CUDA only

Research purpose
----------------
This adapter defines a dedicated BF16 precision experiment.

Its primary matched comparison is:

    FasterWhisperSmallAdapter
        compute_type="float16"

versus

    FasterWhisperSmallBF16Adapter
        compute_type="bfloat16"

When every other configuration and benchmark setting is identical,
the intended experimental factor is therefore the CTranslate2
floating-point compute type:

    FP16 -> BF16

The full transcription, timing, decoding, metadata and cleanup
implementation is inherited from FasterWhisperAdapter to prevent
experimental drift caused by duplicated code.

IMPORTANT
---------
CTranslate2 officially supports bfloat16 as a compute_type.
This adapter verifies both:

    1. that BF16 is supported on the selected CUDA device; and
    2. that the loaded CTranslate2 Whisper model actually resolved
       to bfloat16.

It therefore fails rather than silently accepting a different
runtime compute type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import ctranslate2

from .faster_whisper_adapter import FasterWhisperAdapter


class FasterWhisperSmallBF16Adapter(
    FasterWhisperAdapter
):
    """
    Dedicated Faster-Whisper Small BF16 experimental condition.
    """

    # ================================================================
    # FIXED EXPERIMENTAL IDENTITY
    # ================================================================

    NAME = "faster_whisper_small_bf16"

    MODEL_SIZE = "small"

    COMPUTE_TYPE = "bfloat16"


    # ================================================================
    # INITIALISATION
    # ================================================================

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        resolved_config = dict(
            config or {}
        )


        # ------------------------------------------------------------
        # Lock model size
        # ------------------------------------------------------------

        requested_model_size = str(
            resolved_config.get(
                "model_size",
                self.MODEL_SIZE,
            )
        ).strip().lower()


        if requested_model_size != self.MODEL_SIZE:

            raise ValueError(
                f"{self.__class__.__name__} is fixed to "
                f"model_size='{self.MODEL_SIZE}'. "
                f"Received model_size="
                f"'{requested_model_size}'."
            )


        # ------------------------------------------------------------
        # Lock BF16 compute type
        #
        # This adapter must never silently become FP16, FP32,
        # INT8 or another CTranslate2 condition.
        # ------------------------------------------------------------

        requested_compute_type = str(
            resolved_config.get(
                "compute_type",
                self.COMPUTE_TYPE,
            )
        ).strip().lower()


        if requested_compute_type != self.COMPUTE_TYPE:

            raise ValueError(
                f"{self.__class__.__name__} is fixed to "
                f"compute_type='{self.COMPUTE_TYPE}'. "
                f"Received compute_type="
                f"'{requested_compute_type}'."
            )


        # ------------------------------------------------------------
        # Force the two defining experimental values
        # ------------------------------------------------------------

        resolved_config[
            "model_size"
        ] = self.MODEL_SIZE

        resolved_config[
            "compute_type"
        ] = self.COMPUTE_TYPE


        # ------------------------------------------------------------
        # Parent owns:
        #
        #   - CUDA-device resolution
        #   - num_workers
        #   - decoding temperatures
        #   - beam_size
        #   - best_of
        #   - fallback thresholds
        #   - context propagation
        #   - timestamps
        #   - VAD
        #   - model download options
        #   - transcription
        #   - timing
        #   - cleanup
        #
        # Reusing this implementation is essential to ensuring
        # FP16-vs-BF16 differs only in compute_type.
        # ------------------------------------------------------------

        super().__init__(
            resolved_config
        )


        # ------------------------------------------------------------
        # Stable experiment identity
        # ------------------------------------------------------------

        self.name = self.NAME

        self.model_size = (
            self.MODEL_SIZE
        )

        self.compute_type = (
            self.COMPUTE_TYPE
        )


        # ------------------------------------------------------------
        # Runtime validation/provenance
        # ------------------------------------------------------------

        self._supported_compute_types = None

        self._resolved_compute_type = None

        self._resolved_ct2_device = None

        self._resolved_ct2_device_indices = None


    # ================================================================
    # MODEL LOADING
    # ================================================================

    def load(self):
        """
        Validate native BF16 support and load Faster-Whisper Small.

        The underlying FasterWhisperAdapter performs the actual model
        loading with:

            compute_type="bfloat16"

        Before loading, the selected CUDA device is queried through
        CTranslate2's official runtime capability API.

        After loading, the actual CTranslate2 Whisper model is inspected
        to verify that the resolved runtime compute type is genuinely
        bfloat16.
        """

        # ------------------------------------------------------------
        # Official CTranslate2 capability query
        #
        # This checks the exact GPU selected by the benchmark,
        # rather than assuming cuda:0 or relying only on PyTorch.
        # ------------------------------------------------------------

        supported_compute_types = set(
            ctranslate2.get_supported_compute_types(
                self.device,
                self.device_index,
            )
        )


        self._supported_compute_types = sorted(
            supported_compute_types
        )


        if (
            self.COMPUTE_TYPE
            not in supported_compute_types
        ):

            raise RuntimeError(
                "CTranslate2 BF16 is not supported on the "
                "selected benchmark device.\n"
                f"Device: "
                f"{self.device}:{self.device_index}\n"
                f"Requested compute type: "
                f"{self.COMPUTE_TYPE}\n"
                f"Supported compute types: "
                f"{sorted(supported_compute_types)}\n\n"
                "The adapter will not silently fall back to "
                "FP16 or FP32 because that would invalidate "
                "the BF16 experimental condition."
            )


        # ------------------------------------------------------------
        # Load through the SAME FasterWhisperAdapter implementation
        # used by your FP16 condition.
        # ------------------------------------------------------------

        model = super().load()


        # ============================================================
        # VERIFY ACTUAL CTRANSLATE2 MODEL
        # ============================================================

        # Faster-Whisper exposes the underlying
        # ctranslate2.models.Whisper instance as `.model`.
        ct2_model = getattr(
            model,
            "model",
            None,
        )


        if ct2_model is None:

            raise RuntimeError(
                "Faster-Whisper did not expose the underlying "
                "CTranslate2 Whisper model as '.model'. "
                "Cannot verify the actual BF16 runtime condition."
            )


        # ------------------------------------------------------------
        # Actual compute type
        # ------------------------------------------------------------

        actual_compute_type = getattr(
            ct2_model,
            "compute_type",
            None,
        )


        if actual_compute_type is None:

            raise RuntimeError(
                "The underlying CTranslate2 Whisper model "
                "does not expose its compute_type. "
                "Cannot verify BF16 execution."
            )


        actual_compute_type = str(
            actual_compute_type
        ).strip().lower()


        self._resolved_compute_type = (
            actual_compute_type
        )


        if (
            actual_compute_type
            != self.COMPUTE_TYPE
        ):

            raise RuntimeError(
                "CTranslate2 compute-type validation failed.\n"
                f"Requested: "
                f"{self.COMPUTE_TYPE}\n"
                f"Resolved: "
                f"{actual_compute_type}\n\n"
                "The run is rejected because it is not a "
                "verified BF16 condition."
            )


        # ------------------------------------------------------------
        # Actual CTranslate2 device
        # ------------------------------------------------------------

        actual_device = str(
            getattr(
                ct2_model,
                "device",
                "",
            )
        ).strip().lower()


        self._resolved_ct2_device = (
            actual_device
        )


        if actual_device != self.device:

            raise RuntimeError(
                "Unexpected CTranslate2 execution device.\n"
                f"Expected: {self.device}\n"
                f"Observed: {actual_device}"
            )


        # ------------------------------------------------------------
        # Actual CTranslate2 GPU index
        #
        # CTranslate2 exposes device_index as a list for model
        # instances, but handling an int defensively makes this robust
        # across compatible runtime versions.
        # ------------------------------------------------------------

        raw_device_indices = getattr(
            ct2_model,
            "device_index",
            None,
        )


        if raw_device_indices is None:

            raise RuntimeError(
                "CTranslate2 did not expose its device_index. "
                "Cannot verify GPU placement."
            )


        if isinstance(
            raw_device_indices,
            int,
        ):

            actual_device_indices = [
                raw_device_indices
            ]

        else:

            actual_device_indices = list(
                raw_device_indices
            )


        self._resolved_ct2_device_indices = (
            actual_device_indices
        )


        expected_device_indices = [
            self.device_index
        ]


        if (
            actual_device_indices
            != expected_device_indices
        ):

            raise RuntimeError(
                "Unexpected CTranslate2 CUDA placement.\n"
                f"Expected: "
                f"{expected_device_indices}\n"
                f"Observed: "
                f"{actual_device_indices}"
            )


        # ------------------------------------------------------------
        # Successful validation
        # ------------------------------------------------------------

        print(
            "Faster-Whisper Small BF16 "
            "loaded and validated successfully."
        )

        print(
            "Requested compute type : "
            f"{self.COMPUTE_TYPE}"
        )

        print(
            "Resolved compute type  : "
            f"{self._resolved_compute_type}"
        )

        print(
            "CTranslate2 device     : "
            f"{self._resolved_ct2_device}"
        )

        print(
            "CTranslate2 GPU index  : "
            f"{self._resolved_ct2_device_indices}"
        )

        print(
            "Supported compute types: "
            f"{self._supported_compute_types}"
        )


        return model


    # ================================================================
    # TRANSCRIPTION
    # ================================================================

    def transcribe(
        self,
        audio_path: Union[
            str,
            Path,
        ],
    ) -> Dict[str, Any]:
        """
        Run the inherited Faster-Whisper transcription pathway.

        The complete inference implementation remains identical to the
        Faster-Whisper FP16 condition. Only CTranslate2 compute_type is
        changed to bfloat16.
        """

        result = super().transcribe(
            audio_path
        )


        # ------------------------------------------------------------
        # Defensive result-contract check
        # ------------------------------------------------------------

        if not isinstance(
            result,
            dict,
        ):

            raise TypeError(
                "FasterWhisperAdapter.transcribe() returned "
                f"{type(result).__name__}; expected dict."
            )


        if "text" not in result:

            raise KeyError(
                "Faster-Whisper result is missing 'text'."
            )


        if "latency_ms" not in result:

            raise KeyError(
                "Faster-Whisper result is missing "
                "'latency_ms'."
            )


        meta = result.setdefault(
            "meta",
            {},
        )


        if not isinstance(
            meta,
            dict,
        ):

            raise TypeError(
                "Faster-Whisper result['meta'] "
                "must be a dictionary."
            )


        # ============================================================
        # BF16-SPECIFIC METADATA
        # ============================================================

        meta[
            "adapter_name"
        ] = self.name


        meta[
            "model_size"
        ] = self.MODEL_SIZE


        meta[
            "compute_type"
        ] = self.COMPUTE_TYPE


        meta[
            "precision"
        ] = self.COMPUTE_TYPE


        # BF16 is a floating-point precision condition,
        # not integer quantisation.
        meta[
            "quantization"
        ] = "none"


        meta[
            "optimization"
        ] = "bfloat16_precision"


        meta[
            "experimental_change"
        ] = (
            "CTranslate2/Faster-Whisper Small "
            "compute_type changed from float16 "
            "to bfloat16."
        )


        # ------------------------------------------------------------
        # Runtime verification provenance
        # ------------------------------------------------------------

        meta[
            "requested_ctranslate2_compute_type"
        ] = self.COMPUTE_TYPE


        meta[
            "resolved_ctranslate2_compute_type"
        ] = self._resolved_compute_type


        meta[
            "supported_compute_types_on_device"
        ] = list(
            self._supported_compute_types
            or []
        )


        meta[
            "resolved_ctranslate2_device"
        ] = self._resolved_ct2_device


        meta[
            "resolved_ctranslate2_device_indices"
        ] = (
            list(
                self._resolved_ct2_device_indices
            )

            if self._resolved_ct2_device_indices
            is not None

            else None
        )


        # ------------------------------------------------------------
        # Remove inherited metadata that describes the INT8 comparison.
        #
        # That statement is correct for the generic FP16 adapter but
        # would be misleading for this dedicated BF16 condition.
        # ------------------------------------------------------------

        meta.pop(
            "valid_int8_comparison",
            None,
        )


        # ------------------------------------------------------------
        # Correct matched parent
        # ------------------------------------------------------------

        meta[
            "matched_parent_condition"
        ] = "faster_whisper_small_fp16"


        meta[
            "valid_fp16_comparison"
        ] = (
            "Compare with Faster-Whisper Small "
            "compute_type='float16' using identical "
            "model, decoding parameters, hardware, "
            "dataset and benchmark protocol."
        )


        meta[
            "methodological_interpretation"
        ] = (
            "The matched Faster-Whisper Small FP16 "
            "versus BF16 comparison changes only the "
            "CTranslate2 floating-point compute type, "
            "provided every runner-supplied decoding and "
            "benchmark setting remains identical. "
            "Comparison against OpenAI Whisper is instead "
            "a broader system-level comparison because "
            "the inference backend also differs."
        )


        return result