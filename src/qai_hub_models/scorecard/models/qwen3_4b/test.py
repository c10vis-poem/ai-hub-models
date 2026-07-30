# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.model import DEFAULT_CONTEXT_LENGTH
from qai_hub_models.models.qwen3_4b import Model
from qai_hub_models.models.qwen3_4b.demo import qwen3_4b_chat_demo
from qai_hub_models.models.qwen3_4b.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_4B_PreSplit,
    Qwen3_4B_QuantizablePreSplit,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.device import cs_8_elite_qrd
from qai_hub_models.scorecard.utils.testing_export_eval import run_llm_compile
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.export.dispatch import resolve_model, select_pipeline
from qai_hub_models.utils.export.result import MultiGraphCollectionExportResult

export_model = select_pipeline(resolve_model(MODEL_ID))

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained(checkpoint)


# qwen3_1_7b is the qwen nightly canary (its SpinQuant R1 path surfaces
# quantization regressions). This model runs the full matrix weekly and keeps
# only one cheap row -- the W4A16 MMLU headline metric -- on @pytest.mark.nightly
# for a nightly regression signal.
@pytest.mark.evaluate
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        ("DEFAULT_W4A16", "wikitext", 13.72, 0),
        pytest.param("DEFAULT_W4A16", "mmlu", 0.646, 1000, marks=pytest.mark.nightly),
        ("DEFAULT_UNQUANTIZED", "wikitext", 12.76, 0),
        ("DEFAULT_UNQUANTIZED", "tiny_mmlu", 0.72, 0),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
) -> None:
    dataset_cls = next(
        d
        for d in FPSplitModelWrapper.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    test.run_llm_evaluate_test(
        task=task,
        checkpoint=checkpoint,
        expected_metric=expected_metric,
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        quantized_split_cls=QuantizedSplitModelWrapper,
        fp_split_cls=FPSplitModelWrapper,
        quantized_presplit_cls=Qwen3_4B_QuantizablePreSplit,
        fp_presplit_cls=Qwen3_4B_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        model_id=MODEL_ID,
    )


# Full W4A16 quantize + demo runs on qwen3_1_7b as the nightly canary; running it
# here too was expensive and flaky, so it's dropped.
# Weekly-only (no @pytest.mark.nightly); nightly demo coverage is on qwen3_1_7b.
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    qwen3_4b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


@pytest.mark.nightly
@pytest.mark.parametrize(
    ("precision", "scorecard_path", "device", "checkpoint"),
    [
        (Precision.w4a16, ScorecardCompilePath.GENIE, cs_8_elite_qrd, "DEFAULT_W4A16"),
    ],
)
@pytest.mark.compile_ram_intensive
def test_compile(
    precision: Precision,
    scorecard_path: ScorecardCompilePath,
    device: ScorecardDevice,
    checkpoint: CheckpointSpec,
) -> None:
    Qwen3_4B_PreSplit.release()
    Qwen3_4B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    result = run_llm_compile(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint=checkpoint,
            _skip_quantsim_creation=True,
            output_dir=test.GENIE_BUNDLES_ROOT,
        ),
        skip_compile_options=True,
        skip_downloading=False,
    )
    assert os.path.exists(test.GENIE_BUNDLES_ROOT)
    genie_bundle_path = Path(
        test.GENIE_BUNDLES_ROOT
    ) / ASSET_CONFIG.get_release_asset_name(
        MODEL_ID, TargetRuntime.GENIE, precision, device.chipset
    )
    assert (genie_bundle_path / "tokenizer.json").exists()
    assert (genie_bundle_path / "genie_config.json").exists()
    assert (genie_bundle_path / "htp_backend_ext_config.json").exists()
    assert (genie_bundle_path / "sample_prompt.txt").exists()

    assert isinstance(result, MultiGraphCollectionExportResult)
