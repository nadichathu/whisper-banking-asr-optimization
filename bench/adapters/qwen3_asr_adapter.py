"""
Qwen3-ASR 0.6B BF16 external-comparator adapter.

Official model:
    Qwen/Qwen3-ASR-0.6B

Backend:
    Official qwen-asr Transformers backend

Precision:
    BF16 (torch.bfloat16)

Experimental configuration:
    - CUDA only
    - Batch size 1
    - English forced
    - Empty context / no banking prompt
    - No timestamps
    - No forced aligner
    - Official greedy generation configuration
    - max_new_tokens = 256
    - FlashAttention 2 NOT explicitly enabled

Research role:
    External ASR system-level comparator.

Timing:
    latency_ms measures the adapter-side file-to-final-text wall-clock
    duration of the official Qwen transcribe() path, excluding model
    loading.

    The benchmark runner's outer CUDA-synchronised latency should remain
    the authoritative latency for cross-model analysis.
"""

from __future__ import annotations

import gc
import importlib.metadata
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch

from .base_adapter import BaseAdapter


class Qwen3ASRAdapter(BaseAdapter):
    """
    Official Qwen3-ASR 0.6B BF16 adapter using the qwen-asr
    Transformers backend.
    """

    # ================================================================
    # FIXED EXPERIMENTAL CONDITION
    # ================================================================

    MODEL_ID = "Qwen/Qwen3-ASR-0.6B"

    BACKEND = "transformers"

    PRECISION = "bf16"

    TORCH_DTYPE = torch.bfloat16

    LANGUAGE = "English"

    CONTEXT = ""

    BATCH_SIZE = 1

    MAX_INFERENCE_BATCH_SIZE = 1

    MAX_NEW_TOKENS = 256

    RETURN_TIME_STAMPS = False

    NAME = "qwen3_asr_0_6b_bf16"


    # ================================================================
    # INITIALISATION
    # ================================================================

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(config)

        self.name = self.NAME

        # ------------------------------------------------------------
        # Lock checkpoint identity
        # ------------------------------------------------------------

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
                f"Received model_id='{configured_model_id}'. "
                "Create a separate adapter for a different checkpoint."
            )

        self.model_id = self.MODEL_ID


        # ------------------------------------------------------------
        # Resolve CUDA device
        # ------------------------------------------------------------

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


        # ------------------------------------------------------------
        # Validate BF16 support on the ACTUAL target GPU.
        #
        # Important for multi-GPU correctness.
        # ------------------------------------------------------------

        with torch.cuda.device(
            self._torch_device
        ):

            if not torch.cuda.is_bf16_supported():

                raise RuntimeError(
                    f"{self.device} does not report CUDA BF16 support. "
                    "This adapter defines an official BF16 experimental "
                    "condition and must not silently fall back to FP16 "
                    "or FP32."
                )


        # ------------------------------------------------------------
        # Lock batch-one condition
        # ------------------------------------------------------------

        configured_batch_size = int(
            self.config.get(
                "batch_size",
                self.BATCH_SIZE,
            )
        )

        if configured_batch_size != self.BATCH_SIZE:

            raise ValueError(
                "Qwen3ASRAdapter is fixed to batch_size=1. "
                f"Received batch_size={configured_batch_size}."
            )

        self.batch_size = self.BATCH_SIZE

        self.max_inference_batch_size = (
            self.MAX_INFERENCE_BATCH_SIZE
        )

        self.max_new_tokens = (
            self.MAX_NEW_TOKENS
        )

        self.language = self.LANGUAGE

        self.context = self.CONTEXT

        self.return_time_stamps = (
            self.RETURN_TIME_STAMPS
        )


        # ------------------------------------------------------------
        # Runtime resources / provenance
        # ------------------------------------------------------------

        self.model = None

        self._parameter_count: Optional[int] = None

        self._floating_parameter_dtypes: Set[
            torch.dtype
        ] = set()

        self._floating_buffer_dtypes: Set[
            torch.dtype
        ] = set()

        self._parameter_devices: Set[
            torch.device
        ] = set()

        self._attention_implementation: Optional[
            str
        ] = None

        self._generation_do_sample: Optional[
            bool
        ] = None

        self._generation_num_beams: Optional[
            int
        ] = None

        self._generation_temperature: Optional[
            float
        ] = None

        self._qwen_asr_version: Optional[
            str
        ] = None

        self._transformers_version: Optional[
            str
        ] = None

        self._accelerate_version: Optional[
            str
        ] = None


    # ================================================================
    # CUDA DEVICE VALIDATION
    # ================================================================

    @staticmethod
    def _resolve_cuda_device(
        requested_device: str,
    ) -> str:
        """
        Validate CUDA and return an explicit cuda:<index> device.

        Examples
        --------
        cuda
            -> cuda:<current device>

        cuda:0
            -> cuda:0
        """

        try:

            parsed = torch.device(
                requested_device
            )

        except (
            RuntimeError,
            ValueError,
        ) as exc:

            raise ValueError(
                "Invalid device specification: "
                f"'{requested_device}'."
            ) from exc


        if parsed.type != "cuda":

            raise ValueError(
                "Qwen3ASRAdapter is GPU-only. "
                f"Received device='{requested_device}'."
            )


        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but "
                "torch.cuda.is_available() returned False."
            )


        device_count = torch.cuda.device_count()

        if device_count <= 0:

            raise RuntimeError(
                "CUDA is reported as available, but no CUDA "
                "devices are visible to PyTorch."
            )


        if parsed.index is None:

            device_index = (
                torch.cuda.current_device()
            )

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


    # ================================================================
    # CUDA SYNCHRONISATION
    # ================================================================

    def _cuda_sync(self) -> None:
        """
        Synchronise the exact CUDA device used by this adapter.
        """

        torch.cuda.synchronize(
            self._torch_device
        )


    # ================================================================
    # PACKAGE VERSION HELPER
    # ================================================================

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


    # ================================================================
    # MODEL LOADING
    # ================================================================

    def load(self) -> Any:
        """
        Load official Qwen3-ASR 0.6B in BF16.

        Model-loading time is intentionally excluded from
        per-utterance benchmark latency.
        """

        if self.model is not None:

            return self.model


        # ------------------------------------------------------------
        # Lazy import keeps Qwen dependencies isolated from adapters
        # that do not require qwen-asr.
        # ------------------------------------------------------------

        try:

            from qwen_asr import Qwen3ASRModel

        except ImportError as exc:

            raise ImportError(
                "The official 'qwen-asr' package is required.\n\n"
                "Install it in the Qwen environment using:\n"
                "    pip install -U qwen-asr\n\n"
                "Do not substitute another Qwen implementation, "
                "because doing so would change the experimental "
                "condition."
            ) from exc


        # ------------------------------------------------------------
        # Dependency provenance
        # ------------------------------------------------------------

        self._qwen_asr_version = (
            self._package_version(
                "qwen-asr"
            )
        )

        self._transformers_version = (
            self._package_version(
                "transformers"
            )
        )

        self._accelerate_version = (
            self._package_version(
                "accelerate"
            )
        )


        print(
            f"Loading {self.model_id}"
        )

        print(
            f"Backend   : {self.BACKEND}"
        )

        print(
            f"Device    : {self.device}"
        )

        print(
            "Precision : BF16 "
            "(torch.bfloat16)"
        )

        print(
            f"Batch size: {self.batch_size}"
        )


        # ------------------------------------------------------------
        # Official Transformers-backend Qwen loading.
        #
        # IMPORTANT:
        #
        # BF16 is explicit.
        #
        # We intentionally DO NOT explicitly enable:
        #   - FlashAttention 2
        #   - vLLM
        #   - timestamps
        #   - forced alignment
        #   - contextual/hotword prompting
        # ------------------------------------------------------------

        self.model = Qwen3ASRModel.from_pretrained(
            self.model_id,

            dtype=self.TORCH_DTYPE,

            device_map=self.device,

            max_inference_batch_size=(
                self.max_inference_batch_size
            ),

            max_new_tokens=(
                self.max_new_tokens
            ),

            forced_aligner=None,
        )


        # ============================================================
        # WRAPPER VALIDATION
        # ============================================================

        # ------------------------------------------------------------
        # Backend
        # ------------------------------------------------------------

        actual_backend = getattr(
            self.model,
            "backend",
            None,
        )

        if actual_backend != self.BACKEND:

            raise RuntimeError(
                "Unexpected Qwen backend. "
                f"Expected '{self.BACKEND}', "
                f"received '{actual_backend}'."
            )


        # ------------------------------------------------------------
        # max_inference_batch_size
        # ------------------------------------------------------------

        actual_batch_limit = getattr(
            self.model,
            "max_inference_batch_size",
            None,
        )

        if (
            actual_batch_limit
            != self.max_inference_batch_size
        ):

            raise RuntimeError(
                "Qwen3-ASR did not retain the requested "
                "max_inference_batch_size. "
                f"Expected {self.max_inference_batch_size}, "
                f"received {actual_batch_limit}."
            )


        # ------------------------------------------------------------
        # max_new_tokens
        #
        # The official Qwen wrapper stores this attribute and passes
        # it directly to model.generate().
        # ------------------------------------------------------------

        actual_max_new_tokens = getattr(
            self.model,
            "max_new_tokens",
            None,
        )

        if (
            actual_max_new_tokens
            != self.max_new_tokens
        ):

            raise RuntimeError(
                "Qwen3-ASR did not retain the requested "
                "max_new_tokens setting. "
                f"Expected {self.max_new_tokens}, "
                f"received {actual_max_new_tokens}."
            )


        # ------------------------------------------------------------
        # No forced aligner
        # ------------------------------------------------------------

        if getattr(
            self.model,
            "forced_aligner",
            None,
        ) is not None:

            raise RuntimeError(
                "A Qwen forced aligner was unexpectedly loaded. "
                "This condition requires timestamps and forced "
                "alignment to remain disabled."
            )


        # ------------------------------------------------------------
        # Wrapper device
        # ------------------------------------------------------------

        wrapper_device = getattr(
            self.model,
            "device",
            None,
        )

        if wrapper_device is None:

            raise RuntimeError(
                "Qwen3ASRModel did not expose its resolved device."
            )


        wrapper_device = torch.device(
            wrapper_device
        )

        expected_index = (
            self._torch_device.index
        )

        actual_index = (
            wrapper_device.index
        )


        if wrapper_device.type != "cuda":

            raise RuntimeError(
                "Qwen3-ASR was expected on CUDA but "
                f"resolved to '{wrapper_device}'."
            )


        if actual_index is None:

            actual_index = (
                torch.cuda.current_device()
            )


        if actual_index != expected_index:

            raise RuntimeError(
                "Qwen3-ASR loaded on the wrong CUDA device. "
                f"Expected cuda:{expected_index}, "
                f"received cuda:{actual_index}."
            )


        # ------------------------------------------------------------
        # Wrapper dtype
        # ------------------------------------------------------------

        wrapper_dtype = getattr(
            self.model,
            "dtype",
            None,
        )

        if wrapper_dtype != self.TORCH_DTYPE:

            raise RuntimeError(
                "Qwen3-ASR BF16 condition requires "
                "torch.bfloat16. "
                f"Wrapper reports dtype={wrapper_dtype}."
            )


        # ============================================================
        # UNDERLYING TRANSFORMERS MODEL VALIDATION
        # ============================================================

        underlying_model = getattr(
            self.model,
            "model",
            None,
        )

        if underlying_model is None:

            raise RuntimeError(
                "Qwen3ASRModel did not expose its underlying "
                "Transformers model."
            )


        underlying_model.eval()


        # ------------------------------------------------------------
        # Validate floating-point parameter precision.
        #
        # This prevents a supposed BF16 benchmark silently becoming
        # FP32 or mixed parameter precision.
        # ------------------------------------------------------------

        floating_parameter_dtypes = {
            parameter.dtype

            for parameter
            in underlying_model.parameters()

            if parameter.is_floating_point()
        }


        if not floating_parameter_dtypes:

            raise RuntimeError(
                "No floating-point parameters were found "
                "in the Qwen model."
            )


        if floating_parameter_dtypes != {
            torch.bfloat16
        }:

            observed = sorted(
                str(dtype)
                for dtype
                in floating_parameter_dtypes
            )

            raise RuntimeError(
                "Qwen BF16 parameter validation failed. "
                "Expected all floating-point parameters to be "
                "torch.bfloat16, but observed: "
                f"{observed}"
            )


        self._floating_parameter_dtypes = (
            floating_parameter_dtypes
        )


        # ------------------------------------------------------------
        # Record floating buffer precision.
        #
        # Buffers are provenance information and are not automatically
        # rejected because some implementations can legitimately keep
        # particular buffers in another dtype.
        # ------------------------------------------------------------

        self._floating_buffer_dtypes = {
            buffer.dtype

            for buffer
            in underlying_model.buffers()

            if buffer.is_floating_point()
        }


        # ------------------------------------------------------------
        # Validate model placement.
        #
        # Prevent unnoticed CPU offloading or multi-GPU distribution.
        # ------------------------------------------------------------

        parameter_devices = {
            parameter.device

            for parameter
            in underlying_model.parameters()
        }


        self._parameter_devices = (
            parameter_devices
        )


        if len(parameter_devices) != 1:

            observed_devices = sorted(
                str(device)
                for device
                in parameter_devices
            )

            raise RuntimeError(
                "Qwen parameters are distributed across "
                "multiple devices. "
                "This benchmark requires one fixed CUDA device. "
                f"Observed: {observed_devices}"
            )


        parameter_device = next(
            iter(parameter_devices)
        )


        if parameter_device.type != "cuda":

            raise RuntimeError(
                "Qwen parameters are not entirely on CUDA. "
                f"Observed device: {parameter_device}."
            )


        parameter_device_index = (
            parameter_device.index
        )


        if parameter_device_index is None:

            parameter_device_index = (
                torch.cuda.current_device()
            )


        if (
            parameter_device_index
            != expected_index
        ):

            raise RuntimeError(
                "Qwen parameter placement does not match "
                "the requested CUDA device. "
                f"Expected cuda:{expected_index}, "
                f"observed cuda:{parameter_device_index}."
            )


        # ------------------------------------------------------------
        # Parameter count
        # ------------------------------------------------------------

        self._parameter_count = sum(
            parameter.numel()

            for parameter
            in underlying_model.parameters()
        )


        # ============================================================
        # OFFICIAL GENERATION CONFIG VALIDATION
        # ============================================================

        generation_config = getattr(
            underlying_model,
            "generation_config",
            None,
        )


        if generation_config is None:

            raise RuntimeError(
                "Underlying Qwen model has no generation_config."
            )


        # ------------------------------------------------------------
        # Do not modify official generation settings here.
        #
        # Record and validate them.
        # ------------------------------------------------------------

        self._generation_do_sample = bool(
            getattr(
                generation_config,
                "do_sample",
                False,
            )
        )


        self._generation_num_beams = int(
            getattr(
                generation_config,
                "num_beams",
                1,
            )
        )


        generation_temperature = getattr(
            generation_config,
            "temperature",
            None,
        )


        if generation_temperature is None:

            self._generation_temperature = None

        else:

            self._generation_temperature = float(
                generation_temperature
            )


        # Official checkpoint uses deterministic non-sampling
        # generation.
        if self._generation_do_sample:

            raise RuntimeError(
                "Official Qwen generation configuration "
                "unexpectedly has do_sample=True."
            )


        if self._generation_num_beams != 1:

            raise RuntimeError(
                "Expected Qwen beam size 1 for the official "
                "greedy condition, but received "
                f"num_beams={self._generation_num_beams}."
            )


        # ------------------------------------------------------------
        # Record resolved attention implementation.
        #
        # FlashAttention 2 is deliberately NOT forced.
        # ------------------------------------------------------------

        model_config = getattr(
            underlying_model,
            "config",
            None,
        )


        if model_config is not None:

            self._attention_implementation = getattr(
                model_config,
                "_attn_implementation",
                None,
            )


        # ------------------------------------------------------------
        # Complete model-loading CUDA work before warm-up/timing.
        # ------------------------------------------------------------

        self._cuda_sync()


        print(
            "Qwen3-ASR 0.6B BF16 loaded successfully."
        )

        print(
            "Floating parameter dtypes: "
            f"{sorted(str(x) for x in self._floating_parameter_dtypes)}"
        )

        print(
            "Parameter devices: "
            f"{sorted(str(x) for x in self._parameter_devices)}"
        )

        print(
            f"max_new_tokens: {self.model.max_new_tokens}"
        )

        print(
            "max_inference_batch_size: "
            f"{self.model.max_inference_batch_size}"
        )

        print(
            "Generation do_sample: "
            f"{self._generation_do_sample}"
        )

        print(
            "Generation num_beams: "
            f"{self._generation_num_beams}"
        )

        print(
            "Attention implementation: "
            f"{self._attention_implementation}"
        )


        return self.model


    # ================================================================
    # TRANSCRIPTION
    # ================================================================

    def transcribe(
        self,
        audio_path: str | Path,
    ) -> Dict[str, Any]:
        """
        Transcribe one local audio file using the official Qwen3-ASR
        Transformers pathway.

        Adapter latency includes Qwen's:
            - local audio loading
            - normalisation
            - resampling where required
            - audio chunk handling
            - prompt construction
            - feature processing
            - host-to-device transfer
            - BF16 model inference
            - autoregressive generation
            - output decoding
            - ASR output parsing
            - final text extraction

        Model loading is excluded.
        """

        if self.model is None:

            raise RuntimeError(
                "Qwen3-ASR model is not loaded. "
                "Call load() before transcribe()."
            )


        # ------------------------------------------------------------
        # Validate audio file
        # ------------------------------------------------------------

        path = Path(
            audio_path
        ).expanduser()


        if not path.exists():

            raise FileNotFoundError(
                f"Audio file not found: {path}"
            )


        if not path.is_file():

            raise ValueError(
                f"Audio path is not a regular file: {path}"
            )


        # ------------------------------------------------------------
        # Ensure previous CUDA work is complete.
        # ------------------------------------------------------------

        self._cuda_sync()


        start_time = time.perf_counter()


        # ------------------------------------------------------------
        # Official Qwen transcription call.
        #
        # Qwen currently decorates transcribe() with torch.no_grad().
        # We also use no_grad() explicitly as defensive protection
        # against future upstream changes while preserving the same
        # execution semantics.
        #
        # language="English":
        #   forces known-English text-only ASR output.
        #
        # context="":
        #   prevents banking-domain prompting / hotword bias.
        #
        # return_time_stamps=False:
        #   avoids loading/executing the separate forced aligner.
        # ------------------------------------------------------------

        with torch.no_grad():

            results = self.model.transcribe(
                audio=str(path),

                context=self.context,

                language=self.language,

                return_time_stamps=(
                    self.return_time_stamps
                ),
            )


        # ------------------------------------------------------------
        # Complete asynchronous CUDA execution before stopping timing.
        # ------------------------------------------------------------

        self._cuda_sync()


        # ------------------------------------------------------------
        # Official API returns List[ASRTranscription]
        # even for one audio input.
        # ------------------------------------------------------------

        if not isinstance(
            results,
            list,
        ):

            raise TypeError(
                "Qwen3ASRModel.transcribe() returned "
                f"{type(results).__name__}; expected list."
            )


        if len(results) != 1:

            raise RuntimeError(
                "Batch-one inference expected exactly one "
                f"Qwen transcription result; received {len(results)}."
            )


        result = results[0]


        # ------------------------------------------------------------
        # Extract final transcription
        # ------------------------------------------------------------

        raw_text = getattr(
            result,
            "text",
            None,
        )


        if raw_text is None:

            text = ""

        elif isinstance(
            raw_text,
            str,
        ):

            text = raw_text.strip()

        else:

            raise TypeError(
                "Qwen result.text has an unexpected type: "
                f"{type(raw_text).__name__}."
            )


        # ------------------------------------------------------------
        # Ensure timestamp condition was respected
        # ------------------------------------------------------------

        returned_timestamps = getattr(
            result,
            "time_stamps",
            None,
        )


        if returned_timestamps is not None:

            raise RuntimeError(
                "Qwen returned timestamps even though "
                "return_time_stamps=False."
            )


        # ------------------------------------------------------------
        # Final file-to-text adapter latency
        # ------------------------------------------------------------

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
                "Invalid Qwen latency measurement: "
                f"{latency_ms}"
            )


        # ------------------------------------------------------------
        # Returned language
        # ------------------------------------------------------------

        result_language = getattr(
            result,
            "language",
            None,
        )


        if (
            result_language is not None
            and not isinstance(
                result_language,
                str,
            )
        ):

            result_language = str(
                result_language
            )


        # ============================================================
        # SHARED ADAPTER CONTRACT
        # ============================================================

        return {

            "text": text,

            "latency_ms": float(
                latency_ms
            ),

            "meta": {

                # ----------------------------------------------------
                # Model identity
                # ----------------------------------------------------

                "implementation": "qwen-asr",

                "backend": self.BACKEND,

                "model_id": self.model_id,

                "architecture": "Qwen3-ASR",

                "model_variant": "0.6B",

                "adapter_name": self.name,


                # ----------------------------------------------------
                # Hardware / precision
                # ----------------------------------------------------

                "device": self.device,

                "precision": self.PRECISION,

                "torch_dtype": "bfloat16",

                "parameter_count": (
                    self._parameter_count
                ),

                "floating_parameter_dtypes": sorted(
                    str(dtype).replace(
                        "torch.",
                        "",
                    )

                    for dtype
                    in self._floating_parameter_dtypes
                ),

                "floating_buffer_dtypes": sorted(
                    str(dtype).replace(
                        "torch.",
                        "",
                    )

                    for dtype
                    in self._floating_buffer_dtypes
                ),

                "parameter_devices": sorted(
                    str(device)

                    for device
                    in self._parameter_devices
                ),


                # ----------------------------------------------------
                # Inference configuration
                # ----------------------------------------------------

                "batch_size": (
                    self.batch_size
                ),

                "max_inference_batch_size": (
                    self.max_inference_batch_size
                ),

                "max_new_tokens": (
                    self.max_new_tokens
                ),

                "language_forced": (
                    self.language
                ),

                "returned_language": (
                    result_language
                ),

                "context": (
                    self.context
                ),

                "context_prompting_enabled": False,

                "timestamps_enabled": False,

                "forced_aligner_loaded": False,

                "streaming": False,


                # ----------------------------------------------------
                # Generation
                # ----------------------------------------------------

                "generation_mode": "greedy",

                "generation_do_sample": (
                    self._generation_do_sample
                ),

                "generation_num_beams": (
                    self._generation_num_beams
                ),

                "generation_temperature": (
                    self._generation_temperature
                ),


                # ----------------------------------------------------
                # Attention backend
                # ----------------------------------------------------

                "attention_implementation": (
                    self._attention_implementation
                ),

                "flash_attention_2_explicitly_enabled": False,


                # ----------------------------------------------------
                # Timing
                # ----------------------------------------------------

                "latency_type": (
                    "adapter_file_to_final_text_wall_clock"
                ),

                "model_loading_included": False,

                "latency_includes": [

                    "local_audio_loading",

                    "audio_normalisation",

                    "resampling_if_required",

                    "audio_chunk_handling",

                    "prompt_construction",

                    "feature_preprocessing",

                    "host_to_device_transfer",

                    "bf16_model_inference",

                    "autoregressive_generation",

                    "token_decoding",

                    "asr_output_parsing",

                    "final_text_extraction",
                ],


                # ----------------------------------------------------
                # Software provenance
                # ----------------------------------------------------

                "qwen_asr_version": (
                    self._qwen_asr_version
                ),

                "transformers_version": (
                    self._transformers_version
                ),

                "accelerate_version": (
                    self._accelerate_version
                ),


                # ----------------------------------------------------
                # Research interpretation
                # ----------------------------------------------------

                "comparison_scope": (
                    "system_level_external_asr_comparator"
                ),

                "precision_note": (
                    "Official Qwen3-ASR inference precision: "
                    "model explicitly loaded with "
                    "dtype=torch.bfloat16."
                ),

                "rtf_note": (
                    "Audio duration and real-time factor should "
                    "be calculated by the benchmark runner "
                    "outside the timed adapter call."
                ),
            },
        }


    # ================================================================
    # CLEANUP
    # ================================================================

    def close(self) -> None:
        """
        Release Qwen model resources and clear the CUDA cache
        on the exact device used by this adapter.
        """

        if self.model is not None:

            self._cuda_sync()

            self.model = None


        self._parameter_count = None

        self._floating_parameter_dtypes.clear()

        self._floating_buffer_dtypes.clear()

        self._parameter_devices.clear()

        self._attention_implementation = None

        self._generation_do_sample = None

        self._generation_num_beams = None

        self._generation_temperature = None


        gc.collect()


        if torch.cuda.is_available():

            with torch.cuda.device(
                self._torch_device
            ):

                torch.cuda.empty_cache()