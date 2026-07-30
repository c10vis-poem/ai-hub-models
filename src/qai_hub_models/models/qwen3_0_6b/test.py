# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from typing import Any

import pytest
from transformers import AutoConfig

from qai_hub_models.models._shared.llm.llm_helpers import create_genie_config
from qai_hub_models.models.qwen3_0_6b.model import HF_REPO_NAME


@pytest.mark.unmarked
def test_create_genie_config() -> None:
    context_length = 1024
    llm_config = AutoConfig.from_pretrained(HF_REPO_NAME)
    model_list = [f"qwen3_0_6b_part_{i}_of_2.bin" for i in range(1, 3)]
    actual_config = create_genie_config(context_length, llm_config, "rope", model_list)
    expected_config: dict[str, Any] = {
        "dialog": {
            "version": 1,
            "type": "basic",
            "context": {
                "version": 1,
                "size": 1024,
                "n-vocab": 151936,
                "bos-token": 151643,
                "eos-token": 151645,
            },
            "sampler": {
                "version": 1,
                "seed": 42,
                "temp": 0.8,
                "top-k": 40,
                "top-p": 0.95,
            },
            "tokenizer": {"version": 1, "path": "tokenizer.json"},
            "engine": {
                "version": 1,
                "n-threads": 3,
                "backend": {
                    "version": 1,
                    "type": "QnnHtp",
                    "QnnHtp": {
                        "version": 1,
                        "use-mmap": True,
                        "spill-fill-bufsize": 0,
                        "mmap-budget": 0,
                        "poll": True,
                        "cpu-mask": "0xe0",
                        "kv-dim": 128,
                        "pos-id-dim": 64,
                        "allow-async-init": False,
                        "rope-theta": 1000000,
                    },
                    "extensions": "htp_backend_ext_config.json",
                },
                "model": {
                    "version": 1,
                    "type": "binary",
                    "binary": {
                        "version": 1,
                        "ctx-bins": model_list,
                    },
                },
            },
        }
    }

    assert expected_config == actual_config
