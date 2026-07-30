# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard.envvars import SpecialModelSetting
from qai_hub_models.scorecard.scorecard_config_yaml import (
    LLMWeekendGroup,
    get_downloadable_llm_model_ids,
    get_llm_model_ids,
    get_week_model_ids,
    validate_llm_weekend_coverage,
)
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.path_helpers import MODEL_IDS

# Regional variants are skip_scorecard and must not appear in the LLM set.
REGIONAL_MODELS = {
    "llama_v3_1_sea_lion_3_5_8b_r",
    "llama_v3_elyza_jp_8b",
    "llama_v3_taide_8b_chat",
}


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_pip_commands_require_global_requirements_incompatible(model_id: str) -> None:
    """If manifest.yaml sets any pre/post pip install commands, the model's
    scorecard-config.yaml must have global_requirements_incompatible: true.
    """
    manifest = QAIHMModelManifest.from_model(model_id)
    if not manifest.pre_pip_install_commands and not manifest.post_pip_install_commands:
        return
    assert manifest.scorecard_config.global_requirements_incompatible, (
        f"{model_id}: pre_pip_install_commands/post_pip_install_commands is set in "
        "manifest.yaml, but global_requirements_incompatible is not true in "
        "scorecard-config.yaml."
    )


def test_all_llms_have_weekend_group() -> None:
    """Every scorecard-eligible test_split: llm model must set weekend_group."""
    validate_llm_weekend_coverage()


def test_weeks_are_disjoint_and_cover_all_llms() -> None:
    w1 = get_week_model_ids(LLMWeekendGroup.WEEK1)
    w2 = get_week_model_ids(LLMWeekendGroup.WEEK2)
    assert w1 and w2
    assert w1.isdisjoint(w2)
    assert w1 | w2 == get_llm_model_ids()


def test_downloadable_is_nonempty_subset() -> None:
    downloadable = get_downloadable_llm_model_ids()
    assert downloadable
    assert downloadable <= get_llm_model_ids()


def test_regional_models_excluded() -> None:
    llm_ids = get_llm_model_ids()
    for model_id in REGIONAL_MODELS:
        assert model_id not in llm_ids


def test_tokens_resolve_to_expected_llm_sets() -> None:
    for token, expected in [
        (SpecialModelSetting.LLM_WEEK1, get_week_model_ids(LLMWeekendGroup.WEEK1)),
        (SpecialModelSetting.LLM_WEEK2, get_week_model_ids(LLMWeekendGroup.WEEK2)),
        (SpecialModelSetting.LLM_DOWNLOADABLE, get_downloadable_llm_model_ids()),
    ]:
        torch_ids, static_ids = validate_and_split_enabled_models({token})
        assert torch_ids == expected
        assert not static_ids


def test_combined_weekend_tokens_round_trip() -> None:
    """The workflow's 'pytorch_no_llm,static,llm_week1,llm_downloadable' string is valid."""
    tokens: set = {
        SpecialModelSetting.PYTORCH_NO_LLM,
        SpecialModelSetting.STATIC,
        SpecialModelSetting.LLM_WEEK1,
        SpecialModelSetting.LLM_DOWNLOADABLE,
    }
    torch_ids, static_ids = validate_and_split_enabled_models(tokens)
    assert get_week_model_ids(LLMWeekendGroup.WEEK1) <= torch_ids
    assert get_downloadable_llm_model_ids() <= torch_ids
    assert static_ids
