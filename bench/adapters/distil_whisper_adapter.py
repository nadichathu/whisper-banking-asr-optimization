"""
Distil-Whisper Small.en FP32 external-comparator adapter.

Model:
    distil-whisper/distil-small.en

Checkpoint:
    original-model.fp32.bin

Pinned Hugging Face revision:
    f31b126aca9d63fde470e45e9e744fe97fc4a917

Backend:
    OpenAI Whisper

Runtime model dtype:
    FP32

Research role:
    External distilled-model comparator.

The transcription call intentionally matches the existing OpenAI Whisper
Small FP32 baseline:

    model.transcribe(
        audio_path,
        task="transcribe",
        language="en",
        fp16=False,
    )

All remaining decoding behaviour therefore comes from the same installed
OpenAI Whisper Python API defaults used by the baseline.

This is still a model-level comparison rather than a pure decoder-depth
ablation because Distil-Whisper Small.en is English-only and uses a
distilled 4-layer decoder.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch

from .base_adapter import BaseAdapter


class DistilWhisperAdapter(BaseAdapter):
    """Distil-Whisper Small.en, OpenAI Whisper backend, FP32."""

    # =================================================================
    # FIXED EXPERIMENT IDENTITY
    # =================================================================

    NAME = "distil_whisper_small_en_fp32"

    MODEL_ID = "distil-whisper/distil-small.en"

    CHECKPOINT_FILENAME = "original-model.fp32.bin"

    HF_REVISION = (
        "f31b126aca9d63fde470e45e9e744fe97fc4a917"
    )

    CHECKPOINT_SHA256 = (
        "15ef9cdbfb4ab85c93c7620a71b869127cfc6feacbd13551c3afe7df3e3e7eea"
    )

    CHECKPOINT_SIZE_BYTES = 664_621_144

    BACKEND = "openai-whisper"

    PRECISION = "fp32"

    TASK = "transcribe"

    LANGUAGE = "en"

    BATCH_SIZE = 1


    # =================================================================
    # EXPECTED ARCHITECTURE
    # =================================================================

    EXPECTED_DIMS = {
        "n_mels": 80,
        "n_audio_ctx": 1500,
        "n_audio_state": 768,
        "n_audio_head": 12,
        "n_audio_layer": 12,
        "n_text_ctx": 448,
        "n_text_state": 768,
        "n_text_head": 12,
        "n_text_layer": 4,
        "n_vocab": 51864,
    }


    # =================================================================
    # INITIALISATION
    # =================================================================

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(config)

        self.name = self.NAME


        # -------------------------------------------------------------
        # Lock model identity
        # -------------------------------------------------------------

        configured_model_id = str(
            self.config.get(
                "model_id",
                self.MODEL_ID,
            )
        ).strip()

        if configured_model_id != self.MODEL_ID:
            raise ValueError(
                f"{self.__class__.__name__} is fixed to "
                f"'{self.MODEL_ID}'. "
                f"Received model_id='{configured_model_id}'."
            )


        # -------------------------------------------------------------
        # Lock immutable Hugging Face revision
        # -------------------------------------------------------------

        configured_revision = str(
            self.config.get(
                "revision",
                self.HF_REVISION,
            )
        ).strip()

        if configured_revision != self.HF_REVISION:
            raise ValueError(
                f"{self.__class__.__name__} is pinned to "
                f"revision '{self.HF_REVISION}'. "
                f"Received revision='{configured_revision}'."
            )


        # -------------------------------------------------------------
        # Lock checkpoint filename
        # -------------------------------------------------------------

        configured_checkpoint = str(
            self.config.get(
                "checkpoint_filename",
                self.CHECKPOINT_FILENAME,
            )
        ).strip()

        if configured_checkpoint != self.CHECKPOINT_FILENAME:
            raise ValueError(
                f"{self.__class__.__name__} is fixed to "
                f"'{self.CHECKPOINT_FILENAME}'. "
                f"Received checkpoint_filename="
                f"'{configured_checkpoint}'."
            )


        # -------------------------------------------------------------
        # Lock precision
        # -------------------------------------------------------------

        configured_precision = str(
            self.config.get(
                "precision",
                self.PRECISION,
            )
        ).strip().lower()

        if configured_precision not in {
            "fp32",
            "float32",
        }:
            raise ValueError(
                "This adapter defines the FP32 Distil-Whisper "
                "condition. "
                f"Received precision='{configured_precision}'."
            )


        # -------------------------------------------------------------
        # Lock batch size
        # -------------------------------------------------------------

        configured_batch_size = int(
            self.config.get(
                "batch_size",
                self.BATCH_SIZE,
            )
        )

        if configured_batch_size != self.BATCH_SIZE:
            raise ValueError(
                "DistilWhisperAdapter is fixed to batch_size=1. "
                f"Received batch_size={configured_batch_size}."
            )

        self.batch_size = self.BATCH_SIZE


        # -------------------------------------------------------------
        # Resolve GPU
        # -------------------------------------------------------------

        requested_device = str(
            self.config.get(
                "device",
                "cuda",
            )
        ).strip().lower()

        self.device = self._resolve_cuda_device(
            requested_device
        )

        self._torch_device = torch.device(
            self.device
        )


        # -------------------------------------------------------------
        # Runtime state
        # -------------------------------------------------------------

        self.model = None

        self._whisper = None

        self._checkpoint_path: Optional[
            Path
        ] = None

        self._resolved_snapshot_revision: Optional[
            str
        ] = None

        self._parameter_count: Optional[
            int
        ] = None

        self._floating_parameter_dtypes: Set[
            torch.dtype
        ] = set()

        self._floating_buffer_dtypes: Set[
            torch.dtype
        ] = set()

        self._parameter_devices: Set[
            torch.device
        ] = set()


        # -------------------------------------------------------------
        # Software provenance
        # -------------------------------------------------------------

        self._openai_whisper_version: Optional[
            str
        ] = None

        self._huggingface_hub_version: Optional[
            str
        ] = None


        # -------------------------------------------------------------
        # Actual defaults from installed OpenAI Whisper
        # -------------------------------------------------------------

        self._python_api_defaults: Dict[
            str,
            Any,
        ] = {}

        self._decoding_defaults: Dict[
            str,
            Any,
        ] = {}


        # -------------------------------------------------------------
        # Backend numerical state
        # -------------------------------------------------------------

        self._backend_state: Dict[
            str,
            Any,
        ] = {}


    # =================================================================
    # CUDA DEVICE
    # =================================================================

    @staticmethod
    def _resolve_cuda_device(
        requested_device: str,
    ) -> str:

        try:
            parsed = torch.device(
                requested_device
            )

        except (
            RuntimeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Invalid PyTorch device specification: "
                f"'{requested_device}'."
            ) from exc


        if parsed.type != "cuda":
            raise ValueError(
                "DistilWhisperAdapter is GPU-only. "
                f"Received device='{requested_device}'."
            )


        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but "
                "torch.cuda.is_available() returned False."
            )


        device_count = torch.cuda.device_count()

        if device_count <= 0:
            raise RuntimeError(
                "CUDA is reported as available, "
                "but no CUDA devices are visible."
            )


        if parsed.index is None:
            device_index = torch.cuda.current_device()
        else:
            device_index = parsed.index


        if (
            device_index < 0
            or device_index >= device_count
        ):
            raise ValueError(
                f"Requested cuda:{device_index}, "
                f"but only {device_count} CUDA device(s) "
                "are visible."
            )


        return f"cuda:{device_index}"


    # =================================================================
    # CUDA SYNCHRONISATION
    # =================================================================

    def _cuda_sync(self) -> None:

        torch.cuda.synchronize(
            device=self._torch_device
        )


    # =================================================================
    # PACKAGE VERSION
    # =================================================================

    @staticmethod
    def _package_version(
        package_name: str,
    ) -> Optional[str]:

        try:
            return importlib.metadata.version(
                package_name
            )

        except importlib.metadata.PackageNotFoundError:
            return None


    # =================================================================
    # CHECKPOINT SHA256
    # =================================================================

    @staticmethod
    def _sha256_file(
        path: Path,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> str:

        digest = hashlib.sha256()

        with path.open(
            "rb"
        ) as handle:

            for block in iter(
                lambda: handle.read(
                    chunk_size
                ),
                b"",
            ):
                digest.update(
                    block
                )

        return digest.hexdigest()


    # =================================================================
    # HF SNAPSHOT REVISION
    # =================================================================

    @staticmethod
    def _extract_snapshot_revision(
        path: Path,
    ) -> Optional[str]:

        parts = path.parts

        try:
            snapshot_index = parts.index(
                "snapshots"
            )

        except ValueError:
            return None


        revision_index = (
            snapshot_index + 1
        )

        if revision_index >= len(parts):
            return None


        return parts[
            revision_index
        ]


    # =================================================================
    # PYTORCH BACKEND STATE
    # =================================================================

    @staticmethod
    def _capture_backend_state(
    ) -> Dict[str, Any]:
        """
        Record numerical/determinism state.

        Important:
        This adapter does NOT change TF32 policy.

        TF32 must be controlled benchmark-wide by the runner if a strict
        IEEE-FP32 experiment is required. Changing it only for this adapter
        would invalidate the matched comparison with Whisper Small.
        """

        return {
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),

            "cudnn_allow_tf32": bool(
                torch.backends.cudnn.allow_tf32
            ),

            "float32_matmul_precision":
                torch.get_float32_matmul_precision(),

            "cudnn_benchmark": bool(
                torch.backends.cudnn.benchmark
            ),

            "cudnn_deterministic": bool(
                torch.backends.cudnn.deterministic
            ),
        }


    def _assert_backend_state_unchanged(
        self,
    ) -> None:

        current_state = (
            self._capture_backend_state()
        )

        if current_state != self._backend_state:
            raise RuntimeError(
                "PyTorch backend state changed after the "
                "Distil-Whisper model was loaded.\n"
                f"Load-time state: {self._backend_state}\n"
                f"Current state:   {current_state}\n"
                "Backend numerical policy must be controlled "
                "at benchmark/runner level rather than being "
                "changed by individual adapters."
            )


    # =================================================================
    # RECORD ACTUAL OPENAI WHISPER DEFAULTS
    # =================================================================

    def _record_openai_defaults(
        self,
    ) -> None:

        if self._whisper is None:
            raise RuntimeError(
                "OpenAI Whisper has not been imported."
            )


        # -------------------------------------------------------------
        # High-level transcribe() defaults
        # -------------------------------------------------------------

        signature = inspect.signature(
            self._whisper.transcribe
        )

        transcribe_default_names = (
            "verbose",
            "temperature",
            "compression_ratio_threshold",
            "logprob_threshold",
            "no_speech_threshold",
            "condition_on_previous_text",
            "initial_prompt",
            "carry_initial_prompt",
            "word_timestamps",
            "clip_timestamps",
            "hallucination_silence_threshold",
        )


        self._python_api_defaults = {
            name: signature.parameters[
                name
            ].default

            for name
            in transcribe_default_names

            if name in signature.parameters
        }


        # -------------------------------------------------------------
        # Low-level DecodingOptions defaults
        #
        # This distinguishes Python API defaults from CLI defaults.
        # -------------------------------------------------------------

        options = self._whisper.DecodingOptions()


        self._decoding_defaults = {
            "best_of": options.best_of,
            "beam_size": options.beam_size,
            "patience": options.patience,
            "length_penalty": options.length_penalty,
            "without_timestamps":
                options.without_timestamps,
            "suppress_tokens":
                options.suppress_tokens,
            "suppress_blank":
                options.suppress_blank,
            "max_initial_timestamp":
                options.max_initial_timestamp,
        }


    # =================================================================
    # LOAD
    # =================================================================

    def load(self) -> Any:
        """
        Download, verify and load the official pinned FP32 checkpoint.

        Downloading, hashing and model loading are outside measured
        per-utterance latency.
        """

        if self.model is not None:
            return self.model


        # -------------------------------------------------------------
        # Lazy imports
        # -------------------------------------------------------------

        try:
            import whisper

        except ImportError as exc:
            raise ImportError(
                "The 'openai-whisper' package is required."
            ) from exc


        try:
            from huggingface_hub import (
                hf_hub_download,
            )

        except ImportError as exc:
            raise ImportError(
                "The 'huggingface_hub' package is required."
            ) from exc


        self._whisper = whisper


        # -------------------------------------------------------------
        # Software provenance
        # -------------------------------------------------------------

        self._openai_whisper_version = (
            self._package_version(
                "openai-whisper"
            )
        )

        self._huggingface_hub_version = (
            self._package_version(
                "huggingface-hub"
            )
        )


        # -------------------------------------------------------------
        # Record actual runtime defaults and backend state
        # -------------------------------------------------------------

        self._record_openai_defaults()

        self._backend_state = (
            self._capture_backend_state()
        )


        print(
            f"Loading {self.MODEL_ID}"
        )

        print(
            f"Revision   : {self.HF_REVISION}"
        )

        print(
            f"Checkpoint : {self.CHECKPOINT_FILENAME}"
        )

        print(
            f"Backend    : {self.BACKEND}"
        )

        print(
            f"Device     : {self.device}"
        )

        print(
            "Precision  : FP32"
        )


        # =============================================================
        # DOWNLOAD EXACT PINNED CHECKPOINT
        # =============================================================

        checkpoint_path = hf_hub_download(
            repo_id=self.MODEL_ID,
            filename=self.CHECKPOINT_FILENAME,
            revision=self.HF_REVISION,
            repo_type="model",
        )


        self._checkpoint_path = Path(
            checkpoint_path
        )


        if not self._checkpoint_path.is_file():
            raise FileNotFoundError(
                "Pinned Distil-Whisper checkpoint "
                f"was not found: {self._checkpoint_path}"
            )


        # =============================================================
        # VERIFY SIZE
        # =============================================================

        actual_size = (
            self._checkpoint_path.stat().st_size
        )


        if actual_size != self.CHECKPOINT_SIZE_BYTES:
            raise RuntimeError(
                "Distil-Whisper checkpoint size mismatch. "
                f"Expected {self.CHECKPOINT_SIZE_BYTES} bytes, "
                f"observed {actual_size} bytes."
            )


        # =============================================================
        # VERIFY SHA256
        # =============================================================

        actual_sha256 = self._sha256_file(
            self._checkpoint_path
        )


        if actual_sha256 != self.CHECKPOINT_SHA256:
            raise RuntimeError(
                "Distil-Whisper checkpoint SHA256 mismatch. "
                f"Expected {self.CHECKPOINT_SHA256}, "
                f"observed {actual_sha256}."
            )


        # =============================================================
        # VERIFY RESOLVED REVISION WHEN CACHE PATH EXPOSES IT
        # =============================================================

        self._resolved_snapshot_revision = (
            self._extract_snapshot_revision(
                self._checkpoint_path
            )
        )


        if (
            self._resolved_snapshot_revision
            is not None
            and self._resolved_snapshot_revision
            != self.HF_REVISION
        ):
            raise RuntimeError(
                "Unexpected Hugging Face snapshot revision. "
                f"Expected {self.HF_REVISION}, "
                f"observed "
                f"{self._resolved_snapshot_revision}."
            )


        # =============================================================
        # LOAD THROUGH OPENAI WHISPER
        # =============================================================

        with torch.cuda.device(
            self._torch_device
        ):

            self.model = whisper.load_model(
                str(
                    self._checkpoint_path
                ),
                device=self.device,
            )


            # ---------------------------------------------------------
            # The pinned checkpoint is already the official FP32 file.
            #
            # This is intentionally redundant defence: it guarantees
            # runtime floating parameters/buffers are FP32 even if
            # loading behaviour changes in a future Whisper version.
            # ---------------------------------------------------------

            self.model.float()

            self.model.eval()

            self._cuda_sync()


        # =============================================================
        # ENGLISH-ONLY VALIDATION
        # =============================================================

        if self.model.is_multilingual:
            raise RuntimeError(
                "Expected Distil-Whisper Small.en to be "
                "English-only, but is_multilingual=True."
            )


        # =============================================================
        # ARCHITECTURE VALIDATION
        # =============================================================

        for (
            field_name,
            expected_value,
        ) in self.EXPECTED_DIMS.items():

            actual_value = getattr(
                self.model.dims,
                field_name,
                None,
            )

            if actual_value != expected_value:
                raise RuntimeError(
                    "Unexpected Distil-Whisper architecture. "
                    f"{field_name}: expected {expected_value}, "
                    f"observed {actual_value}."
                )


        # =============================================================
        # PARAMETER DTYPE VALIDATION
        # =============================================================

        self._floating_parameter_dtypes = {
            parameter.dtype

            for parameter
            in self.model.parameters()

            if parameter.is_floating_point()
        }


        if self._floating_parameter_dtypes != {
            torch.float32
        }:
            raise RuntimeError(
                "Distil-Whisper FP32 parameter validation failed. "
                f"Observed dtypes: "
                f"{sorted(str(x) for x in self._floating_parameter_dtypes)}"
            )


        # =============================================================
        # BUFFER DTYPE VALIDATION
        # =============================================================

        self._floating_buffer_dtypes = {
            buffer.dtype

            for buffer
            in self.model.buffers()

            if buffer.is_floating_point()
        }


        if (
            self._floating_buffer_dtypes
            and self._floating_buffer_dtypes
            != {torch.float32}
        ):
            raise RuntimeError(
                "Distil-Whisper FP32 floating-buffer "
                "validation failed. "
                f"Observed dtypes: "
                f"{sorted(str(x) for x in self._floating_buffer_dtypes)}"
            )


        # =============================================================
        # DEVICE VALIDATION
        # =============================================================

        self._parameter_devices = {
            parameter.device

            for parameter
            in self.model.parameters()
        }


        if len(
            self._parameter_devices
        ) != 1:
            raise RuntimeError(
                "Distil-Whisper parameters span multiple devices. "
                f"Observed: "
                f"{sorted(str(x) for x in self._parameter_devices)}"
            )


        parameter_device = next(
            iter(
                self._parameter_devices
            )
        )


        if parameter_device.type != "cuda":
            raise RuntimeError(
                "Distil-Whisper parameters are not on CUDA. "
                f"Observed device: {parameter_device}."
            )


        expected_index = (
            self._torch_device.index
        )

        actual_index = (
            parameter_device.index
        )


        if actual_index is None:
            actual_index = (
                torch.cuda.current_device()
            )


        if actual_index != expected_index:
            raise RuntimeError(
                "Distil-Whisper is on the wrong CUDA device. "
                f"Expected cuda:{expected_index}, "
                f"observed cuda:{actual_index}."
            )


        # =============================================================
        # PARAMETER COUNT
        # =============================================================

        self._parameter_count = sum(
            parameter.numel()

            for parameter
            in self.model.parameters()
        )


        # =============================================================
        # ENSURE BACKEND STATE DID NOT CHANGE DURING LOAD
        # =============================================================

        self._assert_backend_state_unchanged()


        print(
            "Distil-Whisper Small.en FP32 loaded successfully."
        )

        print(
            f"Parameters : {self._parameter_count:,}"
        )

        print(
            f"SHA256     : {actual_sha256}"
        )

        print(
            "Dtypes     : "
            f"{sorted(str(x) for x in self._floating_parameter_dtypes)}"
        )

        print(
            "Devices    : "
            f"{sorted(str(x) for x in self._parameter_devices)}"
        )

        print(
            f"Backend state: {self._backend_state}"
        )


        return self.model


    # =================================================================
    # FALLBACK DIAGNOSTICS
    # =================================================================

    @staticmethod
    def _extract_fallback_diagnostics(
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Inspect temperatures recorded on emitted Whisper segments.

        Because the standard fallback sequence starts at temperature 0,
        an emitted segment with temperature > 0 proves that fallback
        advanced beyond the initial greedy attempt for that segment.

        This diagnostic does NOT reconstruct rejected decoding attempts
        and cannot observe a decoded window that produced no emitted
        segment.
        """

        segments = result.get(
            "segments",
            [],
        )


        if not isinstance(
            segments,
            list,
        ):
            raise TypeError(
                "OpenAI Whisper result['segments'] "
                "must be a list."
            )


        temperatures: List[
            float
        ] = []


        for (
            index,
            segment,
        ) in enumerate(
            segments
        ):

            if not isinstance(
                segment,
                dict,
            ):
                raise TypeError(
                    f"Whisper segment {index} is not a dictionary."
                )


            value = segment.get(
                "temperature",
                None,
            )


            if value is None:
                continue


            temperature = float(
                value
            )


            if math.isfinite(
                temperature
            ):
                temperatures.append(
                    temperature
                )


        # -------------------------------------------------------------
        # Preserve order while removing duplicates
        # -------------------------------------------------------------

        unique_temperatures: List[
            float
        ] = []


        for temperature in temperatures:

            already_present = any(
                math.isclose(
                    temperature,
                    existing,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )

                for existing
                in unique_temperatures
            )


            if not already_present:
                unique_temperatures.append(
                    temperature
                )


        fallback_detected = any(
            temperature > 0.0

            for temperature
            in temperatures
        )


        return {
            "fallback_detected_from_emitted_segments":
                fallback_detected,

            "accepted_segment_temperatures":
                unique_temperatures,

            "max_accepted_temperature":
                max(temperatures)
                if temperatures
                else None,

            "segment_count":
                len(segments),

            "temperature_observation_available":
                bool(temperatures),
        }


    # =================================================================
    # TRANSCRIPTION
    # =================================================================

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Run the same OpenAI Whisper Python transcription call used by
        the primary Whisper Small FP32 baseline.

        Only the checkpoint changes.
        """

        if (
            self.model is None
            or self._whisper is None
        ):
            raise RuntimeError(
                "Distil-Whisper has not been loaded. "
                "Call load() before transcribe()."
            )


        # -------------------------------------------------------------
        # Validate input
        # -------------------------------------------------------------

        path = Path(
            audio_path
        ).expanduser()


        if not path.is_file():
            raise FileNotFoundError(
                f"Audio file was not found: {path}"
            )


        # -------------------------------------------------------------
        # Detect accidental global backend-state drift
        # -------------------------------------------------------------

        self._assert_backend_state_unchanged()


        # -------------------------------------------------------------
        # Ensure prior GPU work does not contaminate timing
        # -------------------------------------------------------------

        self._cuda_sync()


        start_time = time.perf_counter()


        # =============================================================
        # CRITICAL MATCHED-COMPARISON CALL
        # =============================================================
        #
        # These are intentionally the SAME kwargs as the existing
        # Whisper Small FP32 baseline adapter.
        #
        # We do NOT manually set:
        #
        #     temperature
        #     best_of
        #     beam_size
        #     patience
        #     compression_ratio_threshold
        #     logprob_threshold
        #     no_speech_threshold
        #     condition_on_previous_text
        #     without_timestamps
        #
        # because the baseline does not set them either.
        #
        # Both models therefore inherit the SAME defaults from the SAME
        # installed OpenAI Whisper package.
        # =============================================================

        with torch.cuda.device(
            self._torch_device
        ):

            result = self.model.transcribe(
                str(path),
                task=self.TASK,
                language=self.LANGUAGE,
                fp16=False,
            )


        # -------------------------------------------------------------
        # Complete asynchronous GPU work
        # -------------------------------------------------------------

        self._cuda_sync()


        # -------------------------------------------------------------
        # Validate result
        # -------------------------------------------------------------

        if not isinstance(
            result,
            dict,
        ):
            raise TypeError(
                "OpenAI Whisper transcribe() returned "
                f"{type(result).__name__}; expected dict."
            )


        raw_text = result.get(
            "text",
            "",
        )


        if not isinstance(
            raw_text,
            str,
        ):
            raise TypeError(
                "OpenAI Whisper result['text'] "
                "must be a string."
            )


        # -------------------------------------------------------------
        # Final usable text
        # -------------------------------------------------------------

        text = raw_text.strip()


        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0


        if (
            not math.isfinite(
                latency_ms
            )
            or latency_ms < 0.0
        ):
            raise RuntimeError(
                "Invalid Distil-Whisper latency: "
                f"{latency_ms}"
            )


        # -------------------------------------------------------------
        # Diagnostic metadata
        # -------------------------------------------------------------

        fallback = (
            self._extract_fallback_diagnostics(
                result
            )
        )


        returned_language = result.get(
            "language",
            None,
        )


        if returned_language is not None:
            returned_language = str(
                returned_language
            )


        strict_ieee_fp32_math = (
            not self._backend_state[
                "cuda_matmul_allow_tf32"
            ]

            and not self._backend_state[
                "cudnn_allow_tf32"
            ]

            and self._backend_state[
                "float32_matmul_precision"
            ] == "highest"
        )


        # =================================================================
        # COMMON ADAPTER CONTRACT
        # =================================================================

        return {
            "text": text,

            "latency_ms": float(
                latency_ms
            ),

            "meta": {

                # =====================================================
                # MODEL
                # =====================================================

                "implementation":
                    "distil-whisper",

                "backend":
                    self.BACKEND,

                "adapter_name":
                    self.name,

                "model_id":
                    self.MODEL_ID,

                "model_variant":
                    "distil-small.en",

                "architecture_family":
                    "Whisper",

                "teacher_family":
                    "Whisper Small.en",

                "english_only":
                    True,


                # =====================================================
                # CHECKPOINT PROVENANCE
                # =====================================================

                "checkpoint_filename":
                    self.CHECKPOINT_FILENAME,

                "pinned_hf_revision":
                    self.HF_REVISION,

                "resolved_snapshot_revision":
                    self._resolved_snapshot_revision,

                "checkpoint_sha256":
                    self.CHECKPOINT_SHA256,

                "checkpoint_size_bytes":
                    self.CHECKPOINT_SIZE_BYTES,


                # =====================================================
                # ARCHITECTURE
                # =====================================================

                "encoder_layers":
                    self.model.dims.n_audio_layer,

                "decoder_layers":
                    self.model.dims.n_text_layer,

                "encoder_width":
                    self.model.dims.n_audio_state,

                "decoder_width":
                    self.model.dims.n_text_state,

                "parameter_count":
                    self._parameter_count,


                # =====================================================
                # PRECISION / DEVICE
                # =====================================================

                "device":
                    self.device,

                "precision":
                    self.PRECISION,

                "torch_dtype":
                    "float32",

                "floating_parameter_dtypes":
                    sorted(
                        str(dtype).replace(
                            "torch.",
                            "",
                        )

                        for dtype
                        in self._floating_parameter_dtypes
                    ),

                "floating_buffer_dtypes":
                    sorted(
                        str(dtype).replace(
                            "torch.",
                            "",
                        )

                        for dtype
                        in self._floating_buffer_dtypes
                    ),

                "parameter_devices":
                    sorted(
                        str(device)

                        for device
                        in self._parameter_devices
                    ),


                # =====================================================
                # NUMERICAL BACKEND STATE
                # =====================================================

                "backend_state":
                    dict(
                        self._backend_state
                    ),

                "strict_ieee_fp32_math":
                    strict_ieee_fp32_math,

                "tf32_policy_note": (
                    "The adapter records and locks the "
                    "benchmark process's existing TF32 policy "
                    "but does not alter it. A strict IEEE-FP32 "
                    "policy, if required, must be applied "
                    "benchmark-wide to both Whisper and "
                    "Distil-Whisper."
                ),


                # =====================================================
                # INFERENCE
                # =====================================================

                "batch_size":
                    self.batch_size,

                "task":
                    self.TASK,

                "language_forced":
                    self.LANGUAGE,

                "returned_language":
                    returned_language,

                "fp16":
                    False,

                "decoding_policy": (
                    "openai_whisper_python_api_defaults"
                ),


                # =====================================================
                # ACTUAL INSTALLED LIBRARY DEFAULTS
                # =====================================================

                "python_api_transcribe_defaults":
                    dict(
                        self._python_api_defaults
                    ),

                "decoding_options_defaults":
                    dict(
                        self._decoding_defaults
                    ),


                # =====================================================
                # FALLBACK DIAGNOSTICS
                # =====================================================

                "fallback_detected_from_emitted_segments":
                    fallback[
                        "fallback_detected_from_emitted_segments"
                    ],

                "accepted_segment_temperatures":
                    fallback[
                        "accepted_segment_temperatures"
                    ],

                "max_accepted_temperature":
                    fallback[
                        "max_accepted_temperature"
                    ],

                "segment_count":
                    fallback[
                        "segment_count"
                    ],

                "temperature_observation_available":
                    fallback[
                        "temperature_observation_available"
                    ],

                "fallback_note": (
                    "An emitted segment with temperature "
                    "greater than zero proves that standard "
                    "Whisper fallback advanced beyond the "
                    "initial temperature-zero attempt for "
                    "that segment. Rejected attempts and "
                    "windows producing no emitted segment "
                    "are not reconstructed."
                ),


                # =====================================================
                # TIMING
                # =====================================================

                "latency_type": (
                    "adapter_file_to_final_text_wall_clock"
                ),

                "model_loading_included":
                    False,

                "checkpoint_download_included":
                    False,

                "checkpoint_hashing_included":
                    False,

                "latency_includes": [
                    "audio_loading",
                    "ffmpeg_decode_and_resampling",
                    "log_mel_preprocessing",
                    "host_to_device_transfer",
                    "encoder_inference",
                    "autoregressive_decoding",
                    "temperature_fallback_if_triggered",
                    "segment_processing",
                    "text_assembly",
                    "final_text_strip",
                ],


                # =====================================================
                # SOFTWARE
                # =====================================================

                "openai_whisper_version":
                    self._openai_whisper_version,

                "huggingface_hub_version":
                    self._huggingface_hub_version,

                "torch_version":
                    torch.__version__,

                "torch_cuda_version":
                    torch.version.cuda,


                # =====================================================
                # RESEARCH INTERPRETATION
                # =====================================================

                "comparison_scope": (
                    "matched_backend_fp32_"
                    "distilled_model_comparator"
                ),

                "comparison_note": (
                    "This condition uses the same "
                    "OpenAI Whisper Python transcription "
                    "call and FP32 model dtype as the "
                    "Whisper Small FP32 baseline. "
                    "The comparison remains model-level "
                    "because Distil-Whisper Small.en is "
                    "English-only and has a distilled "
                    "4-layer decoder."
                ),

                "rtf_note": (
                    "Audio duration and RTF should be "
                    "derived by the benchmark runner "
                    "outside the timed adapter call."
                ),
            },
        }


    # =================================================================
    # CLEANUP
    # =================================================================

    def close(self) -> None:

        if self.model is not None:

            self._cuda_sync()

            self.model = None


        self._whisper = None

        self._checkpoint_path = None

        self._resolved_snapshot_revision = None

        self._parameter_count = None

        self._floating_parameter_dtypes.clear()

        self._floating_buffer_dtypes.clear()

        self._parameter_devices.clear()

        self._python_api_defaults.clear()

        self._decoding_defaults.clear()

        self._backend_state.clear()


        gc.collect()


        if torch.cuda.is_available():

            with torch.cuda.device(
                self._torch_device
            ):

                torch.cuda.empty_cache()