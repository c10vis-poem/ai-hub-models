# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import sys

from qai_hub_models.models._shared.llm.evaluate import llm_evaluate
from qai_hub_models.models._shared.llm.model import LLM_QNN
from qai_hub_models.models.qwen3_0_6b.model import (
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_0_6B_PreSplit,
    Qwen3_0_6B_QuantizablePreSplit,
)

if __name__ == "__main__":
    # Consume --use-presplit before llm_evaluate's strict argparse. Presplit
    # evals the monolithic QuantSim (CUDA-accelerated) vs the split sessions.
    use_presplit = "--use-presplit" in sys.argv
    if use_presplit:
        sys.argv.remove("--use-presplit")
    llm_evaluate(
        quantized_model_cls=Qwen3_0_6B_QuantizablePreSplit
        if use_presplit
        else QuantizedSplitModelWrapper,
        fp_model_cls=FPSplitModelWrapper if use_presplit else Qwen3_0_6B_PreSplit,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
    )
