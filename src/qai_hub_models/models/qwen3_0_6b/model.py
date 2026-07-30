# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Qwen3-0.6B - PreSplit-Part architecture for LLM deployment.

The generic PreSplit/Part/Collection machinery lives in
``qai_hub_models.models._shared.llm.model`` (family-agnostic) and
``qai_hub_models.models._shared.qwen3.model`` (Qwen3-coupled: RoPE embedding,
dynamo encoding adaptation, explicit head_dim, attention-mask multiply, and the
tied-embedding encoding fix). This module supplies the 0.6B-specific architecture
constants and the small concrete subclasses (Part classes + the Collection,
whose ``parts`` mapping registers the Part classes).

Quantization uses the SpinQuant (R2+R3) -> AdaScale -> Calibration recipe; the
SpinQuant rotations are configured via ``spinquant_config`` on the Quantizable
PreSplit and applied automatically inside ``quantize()``. AdaScale runs on
Wikitext/train (via ``get_weight_optimization_data``) and the final calibration
runs on an interleave of self-generated text + Wikitext/train (via
``get_calibration_data``).
"""

from __future__ import annotations

import logging

# LLMIOType is re-exported from this module so the CLI input-spec parser can
# resolve the inherited get_input_spec's "llm_io_type" annotation, which it
# looks up in the concrete model's module.
from qai_hub.public_rest_api import DatasetEntries

from qai_hub_models import Precision
from qai_hub_models.datasets.wikitext import WikiText
from qai_hub_models.models._shared.llm.common import LLMIOType  # noqa: F401
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
    SplitForwardMixin,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_CONTEXT_LENGTHS as GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_EXPORT_SEQUENCE_LENGTHS as GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS,
)
from qai_hub_models.models._shared.lm_driver.generator import HubCompatibleGenerator
from qai_hub_models.models._shared.qwen3.model import (
    Qwen3PartBase,
    Qwen3PreSplitBase,
    Qwen3PreSplitCollectionBase,
    Qwen3QuantizablePreSplitBase,
)
from qai_hub_models.models.qwen3_0_6b.dataset import InterleavedGeneratedWikitext
from qai_hub_models.utils.input_spec import InputSpec

logger = logging.getLogger(__name__)

DEFAULT_EXPORT_CONTEXT_LENGTHS = GLOBAL_DEFAULT_EXPORT_CONTEXT_LENGTHS
DEFAULT_EXPORT_SEQUENCE_LENGTHS = GLOBAL_DEFAULT_EXPORT_SEQUENCE_LENGTHS

# Model identification
MODEL_ID = __name__.split(".")[-2]
MODEL_ASSET_VERSION = 1

# Model architecture constants (from Qwen3-0.6B)
NUM_LAYERS = 28
# Part 1 = embedding, Part 2 = 28 layers + LM head. 2 splits fit the HTP at this
# size; bump NUM_SPLITS if on-device load hits err 1002 (see qwen3_1_7b).
NUM_SPLITS = 2
NUM_LAYERS_PER_SPLIT = 28
HIDDEN_SIZE = 1024
NUM_KEY_VALUE_HEADS = 8
NUM_ATTN_HEADS = 16
# Qwen3 uses an explicit head_dim that differs from hidden_size // num_attn_heads.
HEAD_DIM = 128

# Hugging Face repo
HF_REPO_NAME = "Qwen/Qwen3-0.6B"

# Memory requirements
MIN_MEMORY_RECOMMENDED = 8

# Precision settings
DEFAULT_PRECISION = Precision.w4a16
SUPPORTED_PRECISIONS = [Precision.w4a16]
DEFAULT_CHECKPOINT = {
    Precision.w4a16: "qwen3_0_6b_w4a16",
}

# SpinQuant recipe: R2 (V/O Hadamard) + R3 (online Hadamard on Q/K). R2 on the
# dynamo/SHA export needs fix_node_names_pass first (see tetracode#20426).
SPINQUANT_CONFIG = {"enable_r1": False, "enable_r2": True, "enable_r3": True}

# Name used for split ONNX file basenames (e.g. Qwen3_0_6B_1_of_2.onnx)
SPLIT_MODEL_NAME = "Qwen3_0_6B"


class Qwen3_0_6B_PreSplit(Qwen3PreSplitBase):
    """FP PreSplit for Qwen3-0.6B."""

    GeneratorClass = HubCompatibleGenerator
    num_layers = NUM_LAYERS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    hf_repo_name = HF_REPO_NAME

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    min_memory_recommended = MIN_MEMORY_RECOMMENDED
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    default_precision = DEFAULT_PRECISION


class Qwen3_0_6B_QuantizablePreSplit(Qwen3QuantizablePreSplitBase[Qwen3_0_6B_PreSplit]):
    """Quantizable PreSplit for Qwen3-0.6B."""

    FPModel = Qwen3_0_6B_PreSplit
    GeneratorClass = HubCompatibleGenerator

    num_layers = NUM_LAYERS
    model_id = MODEL_ID
    model_asset_version = MODEL_ASSET_VERSION
    default_checkpoint = DEFAULT_CHECKPOINT
    supported_precisions = SUPPORTED_PRECISIONS
    default_precision = DEFAULT_PRECISION

    split_model_name = SPLIT_MODEL_NAME
    num_splits = NUM_SPLITS
    num_layers_per_split = NUM_LAYERS_PER_SPLIT

    # AdaScale config (16 attn heads + 8 KV heads + 1).
    ada_scale_num_rmsnorm_per_blk = NUM_ATTN_HEADS + NUM_KEY_VALUE_HEADS + 1
    # SpinQuant (R2+R3) is applied in-place before calibration inside quantize().
    spinquant_config = SPINQUANT_CONFIG
    supports_thinking = True

    def get_calibration_data(  # type: ignore[override]
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        image_size: tuple[int, int] | None = None,
    ) -> DatasetEntries | None:
        # Final calibration runs on an interleave of self-generated text +
        # Wikitext/train (AdaScale weight-opt uses plain Wikitext, below).
        return self._prefill_from_dataset_cls(
            InterleavedGeneratedWikitext,
            num_samples=num_samples,
            sequence_length=sequence_length,
            context_length=context_length,
        )

    def get_weight_optimization_data(
        self,
        num_samples: int = 0,
        input_spec: InputSpec | None = None,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        context_length: int = DEFAULT_CONTEXT_LENGTH,
        image_size: tuple[int, int] | None = None,
    ) -> DatasetEntries | None:
        # AdaScale runs on plain Wikitext/train.
        return self._prefill_from_dataset_cls(
            WikiText,
            num_samples=num_samples,
            sequence_length=sequence_length,
            context_length=context_length,
        )


class Qwen3_0_6B_PartBase(Qwen3PartBase):
    """Unified Part base for Qwen3-0.6B."""

    num_splits = NUM_SPLITS
    hidden_size = HIDDEN_SIZE
    num_attention_heads = NUM_ATTN_HEADS
    num_key_value_heads = NUM_KEY_VALUE_HEADS
    head_dim = HEAD_DIM
    default_precision = DEFAULT_PRECISION
    fp_presplit_cls = Qwen3_0_6B_PreSplit
    quant_presplit_cls = Qwen3_0_6B_QuantizablePreSplit


class Qwen3_0_6B_Part1_Of_2(Qwen3_0_6B_PartBase):
    """Part 1: Embedding."""

    part_id = 1


class Qwen3_0_6B_Part2_Of_2(Qwen3_0_6B_PartBase):
    """Part 2: Transformer layers + LM head."""

    part_id = 2


_SPLIT_PART_CLASSES: list[type] = [
    Qwen3_0_6B_Part1_Of_2,
    Qwen3_0_6B_Part2_Of_2,
]


class QuantizedSplitModelWrapper(  # type: ignore[misc]
    SplitForwardMixin, Qwen3_0_6B_QuantizablePreSplit
):
    """Quantized eval via split Parts instead of monolithic QuantSim."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class FPSplitModelWrapper(SplitForwardMixin, Qwen3_0_6B_PreSplit):
    """FP eval via split Parts instead of monolithic torch model."""

    def get_split_part_classes(self) -> list[type]:
        return _SPLIT_PART_CLASSES


class Qwen3_0_6B_Collection(Qwen3PreSplitCollectionBase):
    """Unified Collection with 2 Parts for Qwen3-0.6B."""

    hf_repo_name = HF_REPO_NAME
    fp_presplit_cls = Qwen3_0_6B_PreSplit
    part_base_cls = Qwen3_0_6B_PartBase
    supports_thinking = True
    parts = {
        "part1_of_2": Qwen3_0_6B_Part1_Of_2,
        "part2_of_2": Qwen3_0_6B_Part2_Of_2,
    }
