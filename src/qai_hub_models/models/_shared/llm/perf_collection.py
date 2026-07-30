# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
Shared utilities for LLM performance collection.

Provides:
- LLMPerfConfig: env-var driven configuration dataclass
- get_llm_perf_parametrization: generates (precision, device) pytest params
- update_perf_yaml: writes TPS/TTFT metrics into a model's perf.yaml

The compile/QDC test logic lives in _shared/llm/test.py (run_llm_perf_test).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from filelock import FileLock

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard import ScorecardDevice
from qai_hub_models.scorecard.device import (
    LLM_COMPILE_DEVICES,
    LLM_W4FP16_COMPILE_DEVICES,
    get_canonical_chipset_name,
)
from qai_hub_models.scorecard.envvars import (
    LLMPerfPrecisionsEnvvar,
    LLMPerfReleaseAssetsEnvvar,
    LLMPerfUpdatesEnvvar,
    SpecialLLMPerfPrecisionSetting,
)
from qai_hub_models.scorecard.path_profile import ScorecardProfilePath
from qai_hub_models.scorecard.perf_yaml import QAIHMModelPerf
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.yaml import ScorecardAssetYaml
from qai_hub_models.utils.path_helpers import QAIHM_MODELS_ROOT


@dataclass
class LLMPerfConfig:
    """Configuration for LLM performance collection.

    Loads configuration from environment variables:
    - QAIHM_LLM_MODELS: Comma-separated model IDs or "all"
    - QAIHM_TEST_DEVICES: Comma-separated device names
    - SKIP_PERF_UPDATE: If set, skip updating perf.yaml files
    - QAIRT_SDK_PATH: Path to QAIRT SDK for auto devices
    """

    models: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)
    skip_perf_update: bool = False
    qairt_sdk_path: str | None = None

    @classmethod
    def from_environment(cls) -> LLMPerfConfig:
        """Create config from environment variables."""
        models_str = os.environ.get("QAIHM_LLM_MODELS", "")
        devices_str = os.environ.get("QAIHM_TEST_DEVICES", "")

        models = [m.strip() for m in models_str.split(",") if m.strip()]
        devices = [d.strip() for d in devices_str.split(",") if d.strip()]

        return cls(
            models=models,
            devices=devices,
            skip_perf_update=bool(os.environ.get("SKIP_PERF_UPDATE")),
            qairt_sdk_path=os.environ.get("QAIRT_SDK_PATH"),
        )


def get_supported_precisions(model_id: str) -> list[Precision]:
    """Get the supported precisions for a model from manifest.yaml."""
    manifest = QAIHMModelManifest.from_model(model_id)
    return manifest.supported_precisions


def load_release_assets_for_model(model_id: str) -> QAIHMModelReleaseAssets:
    """Load release assets for ``model_id``, preferring an in-flight workflow artifact.

    When ``LLMPerfReleaseAssetsEnvvar`` is set to a combined release-assets.yaml
    (the kind uploaded as a per-split workflow artifact, with
    ``models: {<model_id>: ...}``), pull this model's entry from there instead of
    the committed per-model copy. Lets a workflow run consume the release-assets.yaml
    the LLM asset-upload step just produced before the consolidated PR has merged.
    """
    if not LLMPerfReleaseAssetsEnvvar.is_default():
        path = LLMPerfReleaseAssetsEnvvar.get()
        if path.exists():
            entry = ScorecardAssetYaml.from_yaml(path).models.get(model_id)
            if entry is not None:
                return entry
    return QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)


def _get_devices_for_precision(
    precision: Precision,
    override_devices: list[ScorecardDevice] | None,
) -> list[ScorecardDevice]:
    """Return the devices applicable to a given precision.

    If override_devices is provided (from QAIHM_TEST_DEVICES), intersects
    that list with the compile-device sets so only valid combos are returned.
    Otherwise uses LLM_COMPILE_DEVICES (+ LLM_W4FP16_COMPILE_DEVICES for w4).
    """
    compile_devices: list[ScorecardDevice] = list(LLM_COMPILE_DEVICES)
    if precision == Precision.w4:
        compile_devices += LLM_W4FP16_COMPILE_DEVICES

    if override_devices is None:
        return compile_devices

    compile_set = set(compile_devices)
    return [d for d in override_devices if d in compile_set]


def get_llm_perf_parametrization(
    model_id: str,
    default_devices: list[ScorecardDevice] | None = None,
    default_precisions: list[Precision] | None = None,
) -> list[tuple[Precision, ScorecardDevice]]:
    """Generate pytest parametrization for LLM performance tests.

    Selects devices per precision based on LLM_COMPILE_DEVICES (all precisions)
    and LLM_W4FP16_COMPILE_DEVICES (w4 only).

    Environment variables:
    - QAIHM_LLM_MODELS: Comma-separated model IDs or "all". If set and this
      model is not in the list, returns [] so the test is skipped.
    - QAIHM_TEST_DEVICES: Comma-separated device names. When set, acts as a
      filter over the compile-device sets (only devices in both lists are used).
    - QAIHM_LLM_PERF_PRECISIONS: See :class:`LLMPerfPrecisionsEnvvar`.
      ``default`` (the envvar default) honors the test's ``default_precisions``
      arg, falling back to supported_precisions when unset; ``all`` uses every
      supported precision; explicit precisions are intersected with the
      model's supported_precisions.
    """
    models_str = os.environ.get("QAIHM_LLM_MODELS", "")
    if models_str and models_str.strip().lower() != "all":
        allowed = [m.strip() for m in models_str.split(",") if m.strip()]
        if model_id not in allowed:
            return []

    devices_str = os.environ.get("QAIHM_TEST_DEVICES", "")
    override_devices: list[ScorecardDevice] | None
    if devices_str and devices_str.strip().lower() == "all":
        override_devices = None
    elif devices_str:
        device_names = [d.strip() for d in devices_str.split(",") if d.strip()]
        override_devices = [
            ScorecardDevice._registry[name]
            for name in device_names
            if name in ScorecardDevice._registry
        ]
    else:
        override_devices = default_devices

    supported_precisions = get_supported_precisions(model_id)
    precision_setting = LLMPerfPrecisionsEnvvar.get()
    if SpecialLLMPerfPrecisionSetting.ALL in precision_setting:
        precisions = supported_precisions
    elif SpecialLLMPerfPrecisionSetting.DEFAULT in precision_setting:
        precisions = default_precisions or supported_precisions
    else:
        supported_set = set(supported_precisions)
        precisions = [
            Precision.parse(p)
            for p in precision_setting
            if isinstance(p, str) and Precision.parse(p) in supported_set
        ]

    # Drop precisions Genie can't run (e.g. q4_0 is GENIEX_LLAMACPP-only).
    precisions = [p for p in precisions if TargetRuntime.GENIE.supports_precision(p)]

    result: list[tuple[Precision, ScorecardDevice]] = []
    for precision in precisions:
        result.extend(
            (precision, device)
            for device in _get_devices_for_precision(precision, override_devices)
        )
    return result


def _record_perf_update(entry: dict) -> None:
    """Append one update_perf_yaml call to the updates log, if one is configured.

    JSON-lines append guarded by a FileLock so concurrent xdist workers don't
    interleave partial writes.
    """
    if LLMPerfUpdatesEnvvar.is_default():
        return
    updates_path = LLMPerfUpdatesEnvvar.get()
    with FileLock(f"{updates_path}.lock"), open(updates_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def update_perf_yaml(
    model_id: str,
    device_name: str,
    precision: Precision,
    context_length: int,
    tps: float,
    ttft_ms: float,
    prefill_tps: float | None = None,
    profile_path: ScorecardProfilePath = ScorecardProfilePath.GENIE,
    ttft_max_ms: float | None = None,
    desired_compute_unit: str = "npu",
) -> None:
    """Upsert one LLM metric into the model's perf.yaml.

    ttft_max_ms: written to time_to_first_token_range.max. Both genie and
    geniex-bench callers extrapolate it as ttft_ms * (context_length / 128);
    when omitted, the same formula is applied here as a fallback. The
    measured TTFT (ttft_ms) is always on the bundle's short sample_prompt.txt,
    not at full context length.
    desired_compute_unit: written to the entry; "npu" by default.
    FileLock guards the read-modify-write against concurrent xdist workers.

    When LLMPerfUpdatesEnvvar is set, the call is also recorded to that
    log so it can be replayed later (see _record_perf_update).
    """
    _record_perf_update(
        dict(
            model_id=model_id,
            device_name=device_name,
            precision=str(precision),
            context_length=context_length,
            tps=tps,
            ttft_ms=ttft_ms,
            prefill_tps=prefill_tps,
            profile_path=profile_path.value,
            ttft_max_ms=ttft_max_ms,
            desired_compute_unit=desired_compute_unit,
        )
    )
    perf_path = QAIHM_MODELS_ROOT / model_id / "perf.yaml"
    with FileLock(f"{perf_path}.lock"):
        _update_perf_yaml_locked(
            model_id,
            device_name,
            precision,
            context_length,
            tps,
            ttft_ms,
            prefill_tps,
            profile_path,
            ttft_max_ms,
            desired_compute_unit,
        )


def _update_perf_yaml_locked(
    model_id: str,
    device_name: str,
    precision: Precision,
    context_length: int,
    tps: float,
    ttft_ms: float,
    prefill_tps: float | None = None,
    profile_path: ScorecardProfilePath = ScorecardProfilePath.GENIE,
    ttft_max_ms: float | None = None,
    desired_compute_unit: str = "npu",
) -> None:
    perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)

    manifest = QAIHMModelManifest.from_model(model_id)
    assert manifest.name is not None
    component_name = manifest.name

    device = ScorecardDevice.get(device_name, return_unregistered=True)
    if device not in perf.supported_devices:
        perf.supported_devices.append(device)

    chipset = get_canonical_chipset_name(device.chipset)
    if chipset not in perf.supported_chipsets:
        perf.supported_chipsets.append(chipset)

    if precision not in perf.precisions:
        perf.precisions[precision] = QAIHMModelPerf.PrecisionDetails()

    precision_details = perf.precisions[precision]

    if component_name not in precision_details.components:
        precision_details.components[component_name] = QAIHMModelPerf.ComponentDetails()

    component_details = precision_details.components[component_name]

    if device not in component_details.performance_metrics:
        component_details.performance_metrics[device] = {}

    device_metrics = component_details.performance_metrics[device]

    if profile_path not in device_metrics:
        device_metrics[profile_path] = QAIHMModelPerf.PerformanceDetails()

    perf_details = device_metrics[profile_path]

    if ttft_max_ms is None:
        # Legacy genie scaling; remove when GENIE retires.
        ttft_max_ms = ttft_ms * (context_length / 128)
    llm_metric = QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength(
        context_length=context_length,
        tokens_per_second=tps,
        time_to_first_token_range_milliseconds=QAIHMModelPerf.PerformanceDetails.TimeToFirstTokenRangeMilliseconds(
            min=ttft_ms,
            max=ttft_max_ms,
        ),
        prefill_tokens_per_second=prefill_tps,
        desired_compute_unit=desired_compute_unit,
    )

    if perf_details.llm_metrics is None:
        perf_details.llm_metrics = []
    _upsert_metric(perf_details.llm_metrics, llm_metric)

    perf.to_model_yaml(model_id)
    print(f"Updated perf.yaml for {model_id}")


def _upsert_metric(
    bucket: list[QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength],
    metric: QAIHMModelPerf.PerformanceDetails.LLMMetricsPerContextLength,
) -> None:
    """Replace the existing entry at the same (context_length, desired_compute_unit) or append."""
    for i, existing in enumerate(bucket):
        if (
            existing.context_length == metric.context_length
            and existing.desired_compute_unit == metric.desired_compute_unit
        ):
            bucket[i] = metric
            return
    bucket.append(metric)


def clear_llm_metrics_for_profile_path(
    model_id: str,
    profile_path: ScorecardProfilePath,
    device_name: str | None = None,
    precision: Precision | None = None,
) -> None:
    """Empty the llm_metrics list for a profile_path so a re-measured bucket
    doesn't keep orphaned context-lengths / compute-units from a prior run.

    device_name / precision scope the clear. When both are None every
    precision/component/device bucket for profile_path is cleared. When set,
    only the matching (device, precision) buckets are cleared -- callers
    replaying a partial run (a subset of devices/precisions) pass them so
    committed metrics for the devices/precisions this run did NOT measure are
    preserved instead of silently wiped.
    """
    perf_path = QAIHM_MODELS_ROOT / model_id / "perf.yaml"
    if not perf_path.exists():
        return
    target_device = (
        ScorecardDevice.get(device_name, return_unregistered=True)
        if device_name is not None
        else None
    )
    with FileLock(f"{perf_path}.lock"):
        perf = QAIHMModelPerf.from_model(model_id, not_exists_ok=True)
        changed = False
        for prec, precision_details in perf.precisions.items():
            if precision is not None and prec != precision:
                continue
            for component_details in precision_details.components.values():
                for (
                    device,
                    device_metrics,
                ) in component_details.performance_metrics.items():
                    if target_device is not None and device != target_device:
                        continue
                    perf_details = device_metrics.get(profile_path)
                    if perf_details is not None and perf_details.llm_metrics:
                        perf_details.llm_metrics = []
                        changed = True
        if changed:
            perf.to_model_yaml(model_id)
