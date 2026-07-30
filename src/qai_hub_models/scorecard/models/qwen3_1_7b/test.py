# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
import torch

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models._shared.llm import test
from qai_hub_models.models._shared.llm.llm_helpers import (
    log_perf_on_device_result,
)
from qai_hub_models.models._shared.llm.model import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_SEQUENCE_LENGTH,
)
from qai_hub_models.models._shared.llm.perf_collection import (
    LLMPerfConfig,
    get_llm_perf_parametrization,
)
from qai_hub_models.models.qwen3_1_7b import Model
from qai_hub_models.models.qwen3_1_7b.demo import qwen3_1_7b_chat_demo
from qai_hub_models.models.qwen3_1_7b.export import (
    export_model,
)
from qai_hub_models.models.qwen3_1_7b.model import (
    MODEL_ID,
    FPSplitModelWrapper,
    QuantizedSplitModelWrapper,
    Qwen3_1_7B_PreSplit,
    Qwen3_1_7B_QuantizablePreSplit,
)
from qai_hub_models.scorecard import (
    ScorecardCompilePath,
    ScorecardDevice,
)
from qai_hub_models.scorecard.device import cs_8_elite_qrd
from qai_hub_models.scorecard.utils.testing_export_eval import run_llm_compile
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.checkpoint import CheckpointSpec
from qai_hub_models.utils.export.result import MultiGraphCollectionExportResult

DEFAULT_EVAL_SEQLEN = [2048, 128, 1]


@pytest.mark.evaluate
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_W4A16"])
def test_load_encodings_to_quantsim(checkpoint: str) -> None:
    Qwen3_1_7B_PreSplit.release()
    Qwen3_1_7B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    Model.from_pretrained(checkpoint)


# qwen3_1_7b is the qwen nightly canary: its SpinQuant R1 rotation makes it the
# model that surfaces quantization regressions (e.g. SpinQuant being reapplied),
# which R2+R3-only models like qwen3_0_6b tolerate silently. It runs the full
# eval matrix + quantize/demo on nightly; qwen3_0_6b/qwen3_4b are weekly and keep
# only one cheap W4A16 MMLU row on nightly.
@pytest.mark.evaluate
@pytest.mark.nightly
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize(
    ("checkpoint", "task", "expected_metric", "num_samples"),
    [
        # Validated recipe (SpinQuant R1+R3 -> AdaScale -> Calibration):
        # QT (w4a16): PPL 16.59, MMLU 56.65%, AutogradedPrompts 97.8%.
        ("DEFAULT_W4A16", "wikitext", 16.59, 0),
        ("DEFAULT_W4A16", "mmlu", 0.5665, 1000),
        # Prompt-generation + LLM-grader smoke test (5 samples). Always runs on
        # the FP PreSplit regardless of checkpoint, so both rows share a floor.
        ("DEFAULT_W4A16", "prompts", 0.75, 5),
        ("DEFAULT_UNQUANTIZED", "wikitext", 15.63, 0),
        ("DEFAULT_UNQUANTIZED", "mmlu", 0.5996, 1000),
        ("DEFAULT_UNQUANTIZED", "prompts", 0.75, 5),
    ],
)
def test_evaluate(
    checkpoint: str,
    task: str,
    expected_metric: float,
    num_samples: int,
    tmp_path: Path,
) -> None:
    dataset_cls = next(
        d
        for d in FPSplitModelWrapper.get_eval_dataset_classes()
        if d.dataset_name() == task
    )
    Qwen3_1_7B_PreSplit.release()
    Qwen3_1_7B_QuantizablePreSplit.release()
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
        quantized_presplit_cls=Qwen3_1_7B_QuantizablePreSplit,
        fp_presplit_cls=Qwen3_1_7B_PreSplit,
        prompt_sequence_length=DEFAULT_EVAL_SEQLEN,
        context_length=DEFAULT_CONTEXT_LENGTH,
        tmp_path=tmp_path,
        model_id=MODEL_ID,
    )


# Nightly quantize canary. This is the model that surfaces SpinQuant issues:
# it quantizes from scratch (DEFAULT_UNQUANTIZED) with R1+R3, so a bug like
# SpinQuant being reapplied compounds R1's full-residual rotation and shows up
# here, whereas qwen3_0_6b's R2+R3-only recipe would tolerate it silently.
@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
def test_quantize_and_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Quantize the model and verify it can respond with 'Paris'."""
    Qwen3_1_7B_PreSplit.release()
    Qwen3_1_7B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    # Quantize from scratch: start from the FP weights (DEFAULT_UNQUANTIZED), not
    # the pre-quantized AIMET checkpoint that "DEFAULT" resolves to. SpinQuant
    # isn't applied on its own, so pass the R1+R3 config explicitly -- without it
    # the quantized model emits garbage instead of a usable response.
    checkpoint_path = test.setup_test_quantization(
        Qwen3_1_7B_QuantizablePreSplit,
        Qwen3_1_7B_PreSplit,
        str(tmp_path),
        precision=Precision.w4a16,
        checkpoint="DEFAULT_UNQUANTIZED",
        use_seq_mse=False,
        use_ada_scale=True,
        ada_scale_num_samples=128,
        ada_scale_num_iterations=2048,
        spinquant_config={"enable_r1": True, "enable_r2": False, "enable_r3": True},
    )
    # Disable thinking mode: the 1.7B model otherwise loops in an unterminated
    # reasoning trace and never emits the answer within the token budget.
    qwen3_1_7b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint_path,
        enable_thinking=False,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out
    Qwen3_1_7B_PreSplit.release()
    Qwen3_1_7B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()


# Nightly demo coverage lives on this canary (see test_quantize_and_demo above).
@pytest.mark.nightly
@pytest.mark.demo
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test can be run on GPU only."
)
@pytest.mark.parametrize("checkpoint", ["DEFAULT", "DEFAULT_UNQUANTIZED"])
def test_demo_default(
    checkpoint: CheckpointSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    Qwen3_1_7B_PreSplit.release()
    Qwen3_1_7B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    qwen3_1_7b_chat_demo(
        fp_model_cls=FPSplitModelWrapper,
        default_prompt="What is the capital of France?",
        test_checkpoint=checkpoint,
    )
    captured = capsys.readouterr()
    assert "Paris" in captured.out


@pytest.mark.nightly
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="This test can be run on GPU only.",
)
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
    Qwen3_1_7B_PreSplit.release()
    Qwen3_1_7B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()
    # Pass both prompt (ar128) and token (ar1) sequence lengths so the
    # genie bundle includes both model types. Without ar1, Genie must use
    # the ar128 model for token generation, halving TPS on-device.
    result = run_llm_compile(
        export_model,
        MODEL_ID,
        precision,
        scorecard_path,
        device,
        extra_model_arguments=dict(
            checkpoint=checkpoint,
            sequence_length=[DEFAULT_SEQUENCE_LENGTH, 1],
            context_length=[DEFAULT_CONTEXT_LENGTH],
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
    print(f"[provenance] precision={precision} bundle={genie_bundle_path}")
    for compile_key, compile_job in (result.compile_jobs or {}).items():
        print(f"[provenance] compile_job[{compile_key}]={compile_job.job_id}")
    for link_key, link_job in (result.link_jobs or {}).items():
        print(f"[provenance] link_job[{link_key}]={link_job.job_id}")


def _get_llm_perf_params() -> list[tuple[Precision, ScorecardDevice]]:
    params = get_llm_perf_parametrization(
        MODEL_ID,
        default_devices=[cs_8_elite_qrd],
        default_precisions=[Precision.w4a16],
    )
    return params if params else [(Precision.w4a16, cs_8_elite_qrd)]


@pytest.fixture(scope="session")
def llm_perf_config() -> LLMPerfConfig:
    return LLMPerfConfig.from_environment()


@pytest.mark.llm_perf
@pytest.mark.skipif(
    not importlib.util.find_spec("qualcomm_device_cloud_sdk"),
    reason="This test requires the qualcomm_device_cloud_sdk package.",
)
@pytest.mark.parametrize(("precision", "device"), _get_llm_perf_params())
def test_llm_perf(
    precision: Precision,
    device: ScorecardDevice,
    llm_perf_config: LLMPerfConfig,
) -> None:
    Qwen3_1_7B_PreSplit.release()
    Qwen3_1_7B_QuantizablePreSplit.release()
    FPSplitModelWrapper.release()
    QuantizedSplitModelWrapper.release()

    tps, ttft, prefill_tps = test.run_llm_perf_test(
        model_id=MODEL_ID,
        device=device,
        precision=precision,
        output_dir=test.GENIE_BUNDLES_ROOT,
        qairt_sdk_path=llm_perf_config.qairt_sdk_path,
        skip_perf_update=llm_perf_config.skip_perf_update,
    )
    log_perf_on_device_result(
        model_name=MODEL_ID,
        precision=str(precision),
        device=device.name,
        tps=tps,
        prefill_tps=prefill_tps,
        ttft_ms=ttft,
    )
