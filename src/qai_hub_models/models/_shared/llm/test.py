# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from inspect import signature
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, _patch, patch

import numpy as np
import pytest
import qai_hub as hub

from qai_hub_models import Precision, QAIRTVersion, TargetRuntime
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.configs.tool_versions import ToolVersions
from qai_hub_models.models._shared.llm.common import (
    DEFAULT_ATTEMPTS,
    JobOutcome,
    JobRecord,
    cleanup,
    get_qdc_api_token,
    make_key,
    save_job,
)
from qai_hub_models.models._shared.llm.evaluate import evaluate
from qai_hub_models.models._shared.llm.llm_helpers import log_evaluate_test_result
from qai_hub_models.models._shared.llm.model import (
    LLM_QNN,
    LLM_AIMETOnnx,
    LLMBase,
)
from qai_hub_models.models._shared.llm.perf_collection import (
    load_release_assets_for_model,
    update_perf_yaml,
)
from qai_hub_models.models._shared.llm.quantize import quantize
from qai_hub_models.scorecard import ScorecardDevice, ScorecardProfilePath
from qai_hub_models.scorecard.device import DEFAULT_QDC_DEVICE
from qai_hub_models.scorecard.utils.fetch_prerelease_assets import (
    download_prerelease_asset,
)
from qai_hub_models.scorecard.utils.testing import patch_qai_hub
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.model_cache import CacheMode
from qai_hub_models.utils.onnx.helpers import ONNXBundle

GENIE_BUNDLES_ROOT = "genie_bundles"


@contextmanager
def stub_llm_checkpoint_resolution(model_cls: type) -> Iterator[pytest.MonkeyPatch]:
    """Patch resolve_default_checkpoint (base + Llama override) plus
    get_component_graph_input_spec / _hub_compile_options on model_cls so
    LLM pytest tests skip the FP HuggingFace load.
    """
    from transformers import AutoConfig, AutoTokenizer

    from qai_hub_models.models._shared.llama3.model import (
        LlamaDynamicQuantizablePreSplitMixin,
    )
    from qai_hub_models.models._shared.llm.model import DynamicQuantizablePreSplitMixin
    from qai_hub_models.utils.asset_loaders import CachedWebModelAsset

    def _ensure_tokenizer_and_config(cls: Any, ckpt: Path) -> None:
        if not (ckpt / "tokenizer.json").exists():
            AutoTokenizer.from_pretrained(cls.FPModel.hf_repo_name).save_pretrained(
                ckpt
            )
        if not (ckpt / "config.json").exists():
            AutoConfig.from_pretrained(cls.FPModel.hf_repo_name).save_pretrained(ckpt)

    def _stub_resolve_zip_checkpoint(
        cls: Any, precision: Precision, host_device: object, fp_model: object
    ) -> tuple[str, None]:
        # Qwen3 (base DynamicQuantizablePreSplitMixin) publishes the full .zip
        # archive (dynamic ONNX + encodings + weights + tokenizer + config), so
        # reuse the real fetch_default_checkpoint rather than fetching a bare
        # model.encodings object that was never uploaded. Only the FP
        # HuggingFace load (skipped by not passing fp_model) needs stubbing.
        return cls.fetch_default_checkpoint(precision), None

    def _stub_resolve_encodings_checkpoint(
        cls: Any, precision: Precision, host_device: object, fp_model: object
    ) -> tuple[str, None]:
        # Llama fetches encodings only and re-exports ONNX from the FP torch
        # model, so its asset store publishes a bare model.encodings.
        precision_checkpoint = cls.default_checkpoint[precision]
        encodings_path = Path(
            CachedWebModelAsset.from_asset_store(
                cls.model_id,
                cls.model_asset_version,
                f"{precision_checkpoint}/model.encodings",
            ).fetch()
        )
        ckpt = encodings_path.parent
        _ensure_tokenizer_and_config(cls, ckpt)
        return str(ckpt), None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            DynamicQuantizablePreSplitMixin,
            "resolve_default_checkpoint",
            classmethod(_stub_resolve_zip_checkpoint),
        )
        mp.setattr(
            LlamaDynamicQuantizablePreSplitMixin,
            "resolve_default_checkpoint",
            classmethod(_stub_resolve_encodings_checkpoint),
        )
        mp.setattr(
            model_cls,
            "get_component_graph_input_spec",
            lambda self, component_name, graph_name, *a, **kw: {},
        )
        mp.setattr(
            model_cls,
            "get_component_graph_hub_compile_options",
            lambda self, *a, **kw: "",
        )
        yield mp


def _mock_from_pretrained(
    model_cls: type[LLM_AIMETOnnx], context_length: int, sequence_length: int
) -> Mock:
    model = MagicMock()
    model.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained = Mock()
    mock_from_pretrained.__signature__ = signature(model_cls.from_pretrained)
    mock_from_pretrained.return_value = model
    return mock_from_pretrained


def from_bundle_path_patch(bundle_path: str | os.PathLike) -> ONNXBundle:
    return ONNXBundle(
        bundle_path=Path(bundle_path),
        onnx_graph_name="model.onnx",
        aimet_encodings_name="model.encodings",
    )


def split_onnx_patch(*args: Any, num_splits: int, **kwargs: Any) -> list[ONNXBundle]:
    return [from_bundle_path_patch(f"{i}") for i in range(num_splits)]


# reusable patching function
def _create_patches(
    model_cls: type[LLM_AIMETOnnx],
    base_name: str,
    context_length: int,
    sequence_length: int,
    tmp_path: Path,
) -> tuple[
    Mock,
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
    _patch[Mock],
]:
    mock_from_pretrained = _mock_from_pretrained(
        model_cls, context_length, sequence_length
    )
    mock_from_pretrained.return_value.sample_inputs.return_value = {
        "input0": [np.array([1.0, 2.0])]
    }
    # patching to not download from huggingface.
    patch_model = patch(
        f"qai_hub_models.models.{base_name}.Model.from_pretrained",
        mock_from_pretrained,
    )

    patch_fp_model = patch(
        f"qai_hub_models.models.{base_name}.FP_Model.from_pretrained",
        return_value=Mock(),
    )

    patch_onnx_checker = patch("onnx.checker.check_model")
    patch_onnx_load = patch("onnx.load")

    patch_split_onnx = patch(
        "qai_hub_models.models._shared.llm.split_onnx_utils.utils.split_onnx",
        side_effect=split_onnx_patch,
    )

    patch_onnx_files = patch.object(
        ONNXBundle, "from_bundle_path", side_effect=from_bundle_path_patch
    )

    patch_get_or_create_cached_model = patch(
        "qai_hub_models.models._shared.llm.export.get_or_create_cached_model",
        return_value=Mock(),
    )
    patch_tool_versions = patch(
        "qai_hub_models.configs.tool_versions.ToolVersions.from_job",
        return_value=ToolVersions(
            qairt=QAIRTVersion("2.34", validate_exists_on_ai_hub=False)
        ),
    )

    return (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    )


def test_cli_device_with_skips_unsupported_precision_device(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
) -> None:
    (
        _,
        patch_model,
        patch_fp_model,
        _,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    os.makedirs("build", exist_ok=True)
    with (
        patch_model,
        patch_fp_model,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            "SA8295P ADP",
            "--skip-profiling",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--checkpoint",
            "DEFAULT_W4A16",
        ]

        with pytest.raises(
            ValueError,
            match=r"The selected precision \(w4a16\) is not supported on this target device",
        ):
            export_main()  # Call the main function to submit the compile jobs


def test_cli_device_with_skips_unsupported_context_length(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
) -> None:
    (
        _,
        patch_model,
        patch_fp_model,
        _,
        _,
        _,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, 4096, 128, tmp_path)

    os.makedirs("build", exist_ok=True)
    with (
        patch_model,
        patch_fp_model,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            "SA8295P ADP",
            "--skip-profiling",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--checkpoint",
            "DEFAULT_W4",
        ]

        with pytest.raises(
            ValueError,
            match=rf"The {base_name}'s context length is too large to deploy on SA8295P\. Please set the context length to 1024 or lower\.",
        ):
            export_main()  # Call the main function to submit the compile jobs


def test_cli_device_with_skips(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    skip_inferencing: bool,
    skip_profiling: bool,
    target_runtime: TargetRuntime,
    precision: Precision,
) -> None:
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            device.name,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            str(context_length),
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]
        if not skip_inferencing:
            sys.argv.append("--do-inferencing")
        if skip_profiling:
            sys.argv.append("--skip-profiling")

        export_main()  # Call the main function to submit the compile jobs

        # Compile is called parts * 2 times (num_splits token parts, num_splits prompt parts)
        assert mock_hub.submit_compile_job.call_count == parts * 2
        call_args_list = mock_hub.submit_compile_job.call_args_list

        assert all(c.kwargs["device"].name == device.name for c in call_args_list)

        # Link parts times - num_splits
        assert mock_hub.submit_link_job.call_count == parts

        call_args_list = mock_hub.submit_link_job.call_args_list

        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(call_args_list)
        )

        # Skip profile and inference combinations.
        if skip_inferencing:
            mock_hub.submit_inference_job.assert_not_called()
        else:
            mock_hub.submit_inference_job.assert_called()

        if skip_profiling:
            mock_hub.submit_profile_job.assert_not_called()
        else:
            mock_hub.submit_profile_job.assert_called()
        assert tmp_path.exists()
        assert tmp_path.is_dir()


def test_cli_multiple_context_lengths_link_jobs(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    target_runtime: TargetRuntime,
    precision: Precision = Precision.w4a16,  # noqa: PT028
) -> None:
    """Link jobs should combine across both sequence and context lengths.

    With num_splits=N and M context lengths, there should still be exactly N
    link jobs (one per split), each combining all (seq_len, ctx_len) variants.
    """
    context_lengths = [1024, 4096]
    sequence_lengths = [1, 128]
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(
        model_cls, base_name, context_lengths[0], sequence_lengths[0], tmp_path
    )

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, max(context_lengths))
        }

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--device",
            device.name,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            ",".join(str(cl) for cl in context_lengths),
            "--sequence-length",
            ",".join(str(sl) for sl in sequence_lengths),
            "--skip-profiling",
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]

        export_main()

        num_instantiations = len(context_lengths) * len(sequence_lengths)
        assert mock_hub.submit_compile_job.call_count == parts * num_instantiations

        # Regardless of how many context lengths are used, there must be exactly
        # `parts` link jobs — one per split, combining all (seq, ctx) variants.
        assert mock_hub.submit_link_job.call_count == parts
        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(mock_hub.submit_link_job.call_args_list)
        )

        # Each link job must receive compile-job outputs for every instantiation.
        for call in mock_hub.submit_link_job.call_args_list:
            linked_models = call.args[0]
            assert len(linked_models) == num_instantiations


def test_cli_chipset_with_options(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    chipset: str,
    context_length: int,
    sequence_length: int,
    target_runtime: TargetRuntime,
    precision: Precision = Precision.w4a16,  # noqa: PT028
) -> None:
    (
        mock_from_pretrained,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker as mock_onnx_checker,
        patch_onnx_load,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        compile_options = "compile_extra"
        profile_options = "profile_extra"
        link_options = "link_extra"

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",  # script name
            "--chipset",
            chipset,
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--compile-options",
            compile_options,
            "--profile-options",
            profile_options,
            "--link-options",
            link_options,
            "--sequence-length",
            f"{sequence_length},1",
            "--context-length",
            str(context_length),
            "--target-runtime",
            target_runtime.value,
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
            "--do-inferencing",
        ]

        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        export_main()  # Call the main function to submit the compile jobs

        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_profile_job.call_count == parts * 2
        assert mock_hub.submit_inference_job.call_count == parts * 2

        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
            mock_hub.submit_inference_job,
        ]:
            assert all(
                f"chipset:{chipset}" in call.kwargs["device"].attributes
                for call in mock.call_args_list
            )

        # Link parts times - num_splits
        assert mock_hub.submit_link_job.call_count == parts
        expected_model_name = f"{base_name}_{precision}"
        assert all(
            call.kwargs["name"] == f"{expected_model_name}_part_{i + 1}_of_{parts}"
            for i, call in enumerate(mock_hub.submit_link_job.call_args_list)
        )
        mock_get_hub_link_options = (
            mock_from_pretrained.return_value.get_hub_link_options
        )
        assert mock_get_hub_link_options.call_count == parts
        assert all(
            call.args == (target_runtime, link_options)
            for call in mock_get_hub_link_options.call_args_list
        )

        mock_get_hub_compile_options = (
            mock_from_pretrained.return_value.get_hub_compile_options
        )

        assert mock_get_hub_compile_options.call_count == parts * 2
        for call in mock_get_hub_compile_options.call_args_list:
            assert call.args == (target_runtime, precision, compile_options)
            assert "context_graph_name" in call.kwargs
            assert (
                call.kwargs["context_graph_name"]._mock_new_parent._mock_name
                == mock_from_pretrained.return_value.get_qnn_context_graph_name._mock_name
            )

        # Profile parts * 2 times
        assert mock_hub.submit_profile_job.call_count == parts * 2
        mock_get_hub_profile_options = (
            mock_from_pretrained.return_value.get_hub_profile_options
        )
        assert mock_get_hub_profile_options.call_count == parts * 2
        for call in mock_get_hub_profile_options.call_args_list:
            assert len(call.args) == 3
            assert call.args[:2] == (target_runtime, profile_options)
            assert (
                call.args[2]._mock_new_parent._mock_name
                == mock_from_pretrained.return_value.get_qnn_context_graph_name._mock_name
            )

        assert mock_hub.submit_inference_job.call_count == parts * 2

        assert tmp_path.exists()
        assert tmp_path.is_dir()

        # TODO (#15224): Remove from_pretrained as part of inference?
        assert mock_from_pretrained.call_count == 4
        assert (
            mock_from_pretrained.call_args_list[0].kwargs["context_length"]
            == context_length
        )
        assert (
            mock_from_pretrained.call_args_list[1].kwargs["context_length"]
            == context_length
        )
        assert (
            mock_from_pretrained.call_args_list[0].kwargs["sequence_length"]
            == sequence_length
        )
        # In instantiations list (160) from _shared/llm/export.py
        assert mock_from_pretrained.call_args_list[1].kwargs["sequence_length"] == 1


# for llama3 all components are tested, i.e. no option to select individual components 'part_1_of_5', 'part_2_of_5', 'part_3_of_5', 'part_4_of_5', 'part_5_of_5'
def test_cli_default_device_select_component(
    export_main: Callable,
    model_cls: type[LLM_AIMETOnnx],
    tmp_path: Path,
    base_name: str,
    parts: int,
    device: hub.Device,
    cache_mode: CacheMode,
    skip_download: bool,
    skip_summary: bool,
    target_runtime: TargetRuntime,
    decode_sequence_length: int,
    precision: Precision,
) -> None:
    context_length = 4096
    sequence_length = 128
    (
        _,
        patch_model,
        patch_fp_model,
        patch_onnx_checker,
        patch_onnx_load,
        patch_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model,
        patch_tool_versions,
    ) = _create_patches(model_cls, base_name, context_length, sequence_length, tmp_path)

    patch_torch_inference = patch("qai_hub_models.utils.compare.torch_inference")

    with (
        patch_qai_hub() as mock_hub,
        patch_model,
        patch_fp_model,
        patch_onnx_checker as mock_onnx_checker,
        patch_onnx_load,
        patch_split_onnx as mock_split_onnx,
        patch_onnx_files,
        patch_get_or_create_cached_model as mock_get_or_create_cached_model,
        patch_torch_inference as mock_torch_inference,
        patch_tool_versions,
    ):
        mock_onnx_checker.return_value = None
        mock_split_onnx.return_value = None
        mock_torch_inference.return_value = [
            np.array([1.0, 2.0])
        ]  # return mock value for torch inference.

        os.makedirs("build", exist_ok=True)
        sys.argv = [
            "export.py",
            "--output-dir",
            os.path.join("build", tmp_path.name),
            "--model-cache-mode",
            str(cache_mode.name.lower()),
            "--device",
            device.name,
            "--target-runtime",
            target_runtime.value,
            "--context-length",
            str(context_length),
            "--checkpoint",
            f"DEFAULT_{str(precision).upper()}",
        ]
        if skip_download:
            sys.argv.append("--skip-downloading")
        if skip_summary:
            sys.argv.append("--skip-summary")

        mock_hub.submit_compile_job.return_value.target_shapes = {
            "input_ids": (1, context_length)
        }

        export_main()  # Call the main function to submit the compile jobs

        assert mock_hub.submit_compile_job.call_count == parts * 2
        assert mock_hub.submit_link_job.call_count == parts
        assert mock_hub.submit_profile_job.call_count == parts * 2

        # Check names
        expected_model_name = f"{base_name}_{precision}"
        for mock in [
            mock_hub.submit_compile_job,
            mock_hub.submit_profile_job,
        ]:
            for i, call in enumerate(mock.call_args_list):
                instantiation_name = (
                    f"ar{sequence_length}_cl{context_length}"
                    if i < parts
                    else f"ar{decode_sequence_length}_cl{context_length}"
                )
                assert (
                    call.kwargs["name"]
                    == f"{expected_model_name}_{instantiation_name}_{(i % parts) + 1}_of_{parts}"
                )

        assert mock_get_or_create_cached_model.call_count == parts * 2
        # Check cache mode is correct.
        for call in mock_get_or_create_cached_model.call_args_list:
            assert call.kwargs["model_name"] == expected_model_name
            assert call.kwargs["cache_mode"] == cache_mode

        # check compile jobs have correct device name.
        assert all(
            call.kwargs["device"].name == device.name
            for call in mock_hub.submit_compile_job.call_args_list
        )

        assert tmp_path.exists()
        assert tmp_path.is_dir()


def setup_test_quantization(
    model_cls: type[LLM_AIMETOnnx],
    fp_model_cls: type[LLMBase],
    output_path: str,
    precision: Precision,
    checkpoint: str | None = None,
    num_samples: int = 0,
    use_seq_mse: bool = False,
    use_ada_scale: bool = False,
    ada_scale_num_samples: int | None = None,
    ada_scale_num_iterations: int | None = None,
    image_size: tuple[int, int] | None = None,
    spinquant_config: dict | None = None,
) -> str:
    if not (
        (Path(output_path) / "model.encodings").exists()
        and (Path(output_path) / "model.data").exists()
        and (Path(output_path) / "model_dynamic.onnx").exists()
    ):
        quantize(
            quantized_model_cls=model_cls,
            fp_model_cls=fp_model_cls,
            context_length=4096,
            seq_len=2048,
            precision=precision,
            output_dir=output_path,
            checkpoint=checkpoint,
            num_samples=num_samples,
            use_seq_mse=use_seq_mse,
            use_ada_scale=use_ada_scale,
            ada_scale_num_samples=ada_scale_num_samples,
            ada_scale_num_iterations=ada_scale_num_iterations,
            image_size=image_size,
            spinquant_config=spinquant_config,
        )
        cleanup()
    return output_path


PROMPTS_TASKS = {"prompts", "multimodal_prompts"}


def run_llm_evaluate_test(
    task: str,
    checkpoint: str,
    expected_metric: float,
    num_samples: int,
    dataset_cls: type,
    quantized_split_cls: type,
    fp_split_cls: type,
    quantized_presplit_cls: type,
    fp_presplit_cls: type,
    prompt_sequence_length: int | list[int],
    context_length: int,
    model_id: str,
    tmp_path: Path | None = None,
    rtol: float = 0.03,
    log_checkpoint: str | None = None,
    evaluate_kwargs: dict[str, Any] | None = None,
    add_unquantized_extra_kwargs: bool = True,
    fp_baseline_uses_presplit: bool = False,
) -> float:
    """Shared body for the split-model ``test_evaluate`` parametrization.

    Forward-only metrics (wikitext, mmlu, ...) run through the split-Parts
    wrappers so the reported degradation isolates the quantization effect.
    Prompt-generation tasks grade greedily-decoded output, which is
    nondeterministic on the split-Parts ORT-CUDA path, so they always run on
    the monolithic FP PreSplit (deterministic torch) regardless of checkpoint.

    Returns the measured metric and asserts it against ``expected_metric`` (a
    floor for prompt tasks, a two-sided tolerance otherwise).

    ``add_unquantized_extra_kwargs`` threads the ``_skip_quantsim_creation`` /
    ``fp_model`` kwargs the split LLMs pass for the unquantized baseline; the
    VLMs don't take them. ``fp_baseline_uses_presplit`` evaluates the FP
    baseline on the monolithic PreSplit rather than the split wrapper.
    """
    is_prompts = task in PROMPTS_TASKS
    is_unquantized = checkpoint == "DEFAULT_UNQUANTIZED"
    if is_prompts:
        assert tmp_path is not None, "tmp_path is required for prompt-generation tasks"
        eval_checkpoint = "DEFAULT_UNQUANTIZED"
        quantized_model_cls = quantized_presplit_cls
        fp_model_cls = fp_presplit_cls
    else:
        eval_checkpoint = checkpoint
        quantized_model_cls = quantized_split_cls
        # Some models evaluate the FP baseline on the monolithic PreSplit while
        # the quantized rows still go through the split wrapper.
        fp_model_cls = (
            fp_presplit_cls
            if fp_baseline_uses_presplit and is_unquantized
            else fp_split_cls
        )

    extra_kwargs = (
        {"_skip_quantsim_creation": False, "fp_model": None}
        if add_unquantized_extra_kwargs and eval_checkpoint == "DEFAULT_UNQUANTIZED"
        else {}
    )
    task_kwargs = {"output_dir": str(tmp_path)} if is_prompts else None

    actual_metric, _ = evaluate(
        quantized_model_cls=quantized_model_cls,
        fp_model_cls=fp_model_cls,
        qnn_model_cls=LLM_QNN,  # type: ignore[type-abstract]
        num_samples=num_samples,
        dataset_cls=dataset_cls,
        prompt_sequence_length=prompt_sequence_length,
        context_length=context_length,
        kwargs=dict(checkpoint=eval_checkpoint, **extra_kwargs),
        task_kwargs=task_kwargs,
        **(evaluate_kwargs or {}),
    )
    log_evaluate_test_result(
        model_name=model_id,
        checkpoint=log_checkpoint or checkpoint,
        metric=task,
        value=actual_metric,
    )
    if is_prompts:
        assert actual_metric >= expected_metric, (
            f"{task} grader score {actual_metric:.3f} below floor {expected_metric}"
        )
    else:
        np.testing.assert_allclose(actual_metric, expected_metric, rtol=rtol, atol=0)
    return actual_metric


# ---------------------------------------------------------------------------
# LLM performance collection helpers
# ---------------------------------------------------------------------------


def fetch_genie_bundle_for_perf(
    model_id: str,
    precision: Precision,
    chipset: str,
    output_dir: Path,
) -> Path:
    """Download and extract the pre-compiled genie bundle for this model.

    Looks up release-assets.yaml for (precision, chipset, GENIE), downloads
    the zip from S3, and extracts it into output_dir. Returns the extracted
    bundle directory.

    Raises a clear error if no matching asset exists.
    """
    assets = load_release_assets_for_model(model_id)
    asset = assets.get_asset(precision, chipset, ScorecardProfilePath.GENIE)
    if asset is None:
        available_chipsets: list[str] = []
        prec_details = assets.precisions.get(precision)
        if prec_details is not None:
            available_chipsets = sorted(prec_details.chipset_assets.keys())
        raise RuntimeError(
            f"No genie release asset found in release-assets.yaml for "
            f"model_id={model_id!r}, precision={precision!s}, chipset={chipset!r}. "
            f"Available chipsets for this precision: {available_chipsets or '<none>'}. "
            "Build and update release-assets.yaml before running LLM perf collection."
        )

    bundle_dir = output_dir / ASSET_CONFIG.get_release_asset_name(
        model_id, TargetRuntime.GENIE, precision, chipset
    )
    if bundle_dir.exists():
        # Already fetched on a previous test in this session.
        return bundle_dir

    zip_path = download_prerelease_asset(
        asset,
        model_id=model_id,
        runtime=TargetRuntime.GENIE,
        precision=precision,
        chipset=chipset,
        output_folder=output_dir,
        verbose=True,
    )
    shutil.unpack_archive(str(zip_path), extract_dir=str(output_dir))
    if not bundle_dir.exists():
        raise RuntimeError(
            f"Extracted genie bundle missing expected directory {bundle_dir}; "
            f"contents of {output_dir}: {sorted(p.name for p in output_dir.iterdir())}"
        )
    return bundle_dir


def submit_llm_perf_job(
    model_id: str,
    device: ScorecardDevice,
    precision: Precision,
    output_dir: Path | str,
    jobs_file: str,
    qairt_sdk_path: str | None = None,
    skip_perf_update: bool = False,
) -> str:
    """Fetch the genie bundle, submit one QDC job, upsert its record.

    Does not wait. Returns the QDC job id. The collect side re-derives
    the bundle from (model_id, precision, chipset) via
    ``fetch_genie_bundle_for_perf`` -- nothing about local paths is
    persisted.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genie_bundle_path = fetch_genie_bundle_for_perf(
        model_id, precision, device.chipset, output_dir
    )

    from qai_hub_models.models._shared.llm.qdc.genie_jobs import (
        _USE_DEFAULT_PROMPTS,
        submit_genie_bundle_only,
    )

    api_token = get_qdc_api_token(device)

    # Eval is expensive; only run on the default scorecard device.
    run_eval = (
        os.environ.get("QAIHM_RUN_EVAL", "true").lower() == "true"
        and device == DEFAULT_QDC_DEVICE
    )
    eval_prompts = _USE_DEFAULT_PROMPTS if run_eval else None
    job_name = f"Genie {model_id} {precision}"

    job_id = submit_genie_bundle_only(
        api_token,
        device.reference_device.name,
        str(genie_bundle_path),
        job_name=job_name,
        qairt_sdk_path=qairt_sdk_path,
        eval_prompts=eval_prompts,
        model_id=model_id,
    )

    key = make_key(model_id, str(precision), "GENIE", device.name)
    save_job(jobs_file, key, job_id, attempts_left=DEFAULT_ATTEMPTS)
    return job_id


def collect_llm_perf_job(
    model_id: str,
    device: ScorecardDevice,
    precision: Precision,
    record: JobRecord,
    jobs_file: str,
    output_dir: Path | str,
    qairt_sdk_path: str | None = None,
    skip_perf_update: bool = False,
) -> tuple[float | None, float | None, float | None]:
    """Poll a submitted Genie job. On retryable failure, re-fetch the
    bundle from release-assets.yaml and resubmit; the jobs_file row is
    rewritten with the new job id and one fewer attempt.

    ``record`` is the current entry read from jobs_file (job_id +
    attempts_left). Everything else is re-derived from (model_id,
    precision, device) so the collect runner is independent of whatever
    filesystem state the submit runner had.
    """
    from qai_hub_models.models._shared.llm.common import poll_and_retry
    from qai_hub_models.models._shared.llm.qdc.genie_jobs import (
        _USE_DEFAULT_PROMPTS,
        collect_genie_bundle_result,
        save_eval_metadata_json,
        save_eval_results_json,
        submit_genie_bundle_only,
    )

    api_token = get_qdc_api_token(device)
    run_eval = (
        os.environ.get("QAIHM_RUN_EVAL", "true").lower() == "true"
        and device == DEFAULT_QDC_DEVICE
    )
    eval_prompts = _USE_DEFAULT_PROMPTS if run_eval else None
    job_name = f"Genie {model_id} {precision}"
    key = make_key(model_id, str(precision), "GENIE", device.name)
    hub_device_name = device.reference_device.name

    def _resubmit() -> str:
        bundle_path = fetch_genie_bundle_for_perf(
            model_id, precision, device.chipset, Path(output_dir)
        )
        return submit_genie_bundle_only(
            api_token,
            hub_device_name,
            str(bundle_path),
            job_name=job_name,
            qairt_sdk_path=qairt_sdk_path,
            eval_prompts=eval_prompts,
            model_id=model_id,
        )

    def _collect(job_id: str) -> tuple[tuple, JobOutcome, str | None]:
        tps, prefill_tps, ttft, eval_results, outcome, reason = (
            collect_genie_bundle_result(
                api_token, hub_device_name, job_id, eval_prompts=eval_prompts
            )
        )
        return (tps, prefill_tps, ttft, eval_results), outcome, reason

    tps, prefill_tps, ttft, eval_results = poll_and_retry(
        initial_job_id=record.job_id,
        attempts_left=record.attempts_left,
        collect_fn=_collect,
        resubmit_fn=_resubmit,
        on_new_job_id=lambda new_id, left: save_job(
            jobs_file, key, new_id, attempts_left=left
        ),
    )

    metadata = ModelMetadata.from_json(
        fetch_genie_bundle_for_perf(
            model_id, precision, device.chipset, Path(output_dir)
        )
        / "metadata.json"
    )
    assert metadata is not None and metadata.genie is not None
    context_lengths = metadata.genie.context_lengths

    if not skip_perf_update and tps is not None and ttft is not None:
        update_perf_yaml(
            model_id,
            device.reference_device_name,
            precision,
            max(context_lengths),
            tps,
            ttft,
            prefill_tps,
        )

    if eval_results:
        base = f"{model_id}_{device.chipset}_{precision}_eval"
        save_eval_results_json(eval_results, f"{base}.json")
        save_eval_metadata_json(
            model_id,
            device.chipset,
            str(precision),
            f"{base}.meta.json",
            path=ScorecardProfilePath.GENIE,
        )

    return tps, ttft, prefill_tps


def run_llm_perf_test(
    model_id: str,
    device: ScorecardDevice,
    precision: Precision,
    output_dir: Path | str,
    qairt_sdk_path: str | None = None,
    skip_perf_update: bool = False,
) -> tuple[float | None, float | None, float | None]:
    """Compose submit + collect over an ephemeral jobs_file.

    Returns (tokens_per_second, time_to_first_token_ms, prefill_tokens_per_second).
    """
    with tempfile.NamedTemporaryFile(
        prefix="genie_jobs_", suffix=".yaml", delete=False
    ) as tmp:
        jobs_file = tmp.name
    try:
        submit_llm_perf_job(
            model_id=model_id,
            device=device,
            precision=precision,
            output_dir=output_dir,
            jobs_file=jobs_file,
            qairt_sdk_path=qairt_sdk_path,
            skip_perf_update=skip_perf_update,
        )
        from qai_hub_models.models._shared.llm.common import load_jobs

        key = make_key(model_id, str(precision), "GENIE", device.name)
        record = load_jobs(jobs_file)[key]
        return collect_llm_perf_job(
            model_id=model_id,
            device=device,
            precision=precision,
            record=record,
            jobs_file=jobs_file,
            output_dir=output_dir,
            qairt_sdk_path=qairt_sdk_path,
            skip_perf_update=skip_perf_update,
        )
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(jobs_file)
