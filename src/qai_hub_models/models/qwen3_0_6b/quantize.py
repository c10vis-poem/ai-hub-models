# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from qai_hub_models.models._shared.llm.quantize import llm_quantize
from qai_hub_models.models.qwen3_0_6b.model import (
    MODEL_ID,
    SUPPORTED_PRECISIONS,
    Qwen3_0_6B_PreSplit,
    Qwen3_0_6B_QuantizablePreSplit,
)

if __name__ == "__main__":
    llm_quantize(
        quantized_model_cls=Qwen3_0_6B_QuantizablePreSplit,
        fp_model_cls=Qwen3_0_6B_PreSplit,
        model_id=MODEL_ID,
        supported_precisions=SUPPORTED_PRECISIONS,
    )
