# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

from qai_hub_models.models._shared.llm.demo import llm_chat_demo
from qai_hub_models.models._shared.llm.model import LLM_QNN, LLM_AIMETOnnx, LLMBase
from qai_hub_models.models._shared.qwen3.model import END_TOKENS
from qai_hub_models.models.qwen3_0_6b.model import (
    HF_REPO_NAME,
    MODEL_ID,
    QuantizedSplitModelWrapper,
    Qwen3_0_6B_PreSplit,
)
from qai_hub_models.utils.checkpoint import CheckpointSpec

HF_REPO_URL = f"https://huggingface.co/{HF_REPO_NAME}"


def qwen3_0_6b_chat_demo(
    model_cls: type[LLM_AIMETOnnx] = QuantizedSplitModelWrapper,
    fp_model_cls: type[LLMBase] = Qwen3_0_6B_PreSplit,
    qnn_model_cls: type[LLM_QNN] = LLM_QNN,
    model_id: str = MODEL_ID,
    end_tokens: set = END_TOKENS,
    hf_repo_name: str = HF_REPO_NAME,
    hf_repo_url: str = HF_REPO_URL,
    default_prompt: str | None = None,
    test_checkpoint: CheckpointSpec | None = None,
    enable_thinking: bool | None = None,
) -> None:
    llm_chat_demo(
        model_cls=model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=qnn_model_cls,
        model_id=model_id,
        end_tokens=end_tokens,
        hf_repo_name=hf_repo_name,
        hf_repo_url=hf_repo_url,
        default_prompt=default_prompt,
        test_checkpoint=test_checkpoint,
        supports_thinking=True,
        enable_thinking=enable_thinking,
    )


if __name__ == "__main__":
    qwen3_0_6b_chat_demo()
