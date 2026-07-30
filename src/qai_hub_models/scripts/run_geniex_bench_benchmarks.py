# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.configs.model_metadata import ModelMetadata
from qai_hub_models.models._shared.llm.common import (
    DEFAULT_ATTEMPTS,
    JobOutcome,
    JobRecord,
    get_qdc_api_token,
    load_jobs,
    make_key,
    poll_and_retry,
    save_job,
)
from qai_hub_models.models._shared.llm.perf_collection import (
    load_release_assets_for_model,
    update_perf_yaml,
)
from qai_hub_models.models._shared.llm.qdc.geniex_jobs import (
    GenieXBenchMetrics,
    collect_geniex_bench_result,
    load_default_eval_prompts,
    save_eval_metadata_json,
    save_eval_results_json,
    submit_geniex_bench_only,
)
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.device import (
    DEFAULT_QDC_DEVICE,
    ScorecardDevice,
    get_canonical_chipset_name,
)
from qai_hub_models.scorecard.envvars import (
    LLMPerfPrecisionsEnvvar,
    SpecialLLMPerfPrecisionSetting,
)
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.utils.fetch_prerelease_assets import (
    download_prerelease_asset,
)
from qai_hub_models.utils.asset_loaders import ASSET_CONFIG
from qai_hub_models.utils.path_helpers import MODEL_IDS

DEFAULT_DEVICES = "cs_x2_elite,cs_x_elite"

# One device per platform bucket in GenieXBenchQDCJobs.add_job_artifacts.
ALL_GENIEX_DEVICES = (
    "cs_x_elite",
    "cs_x2_elite",
    "cs_9075",
    "cs_8_elite_qrd",
    "cs_8_elite_gen_5_qrd",
)
LLAMACPP_DEVICE_ALIASES = ("cpu", "gpu", "npu")
LLAMACPP_CONTEXT_LENGTHS = [512, 4096]

# Number of accuracy-eval prompts to run (of the built-in 100-prompt set).
_EVAL_NUM_PROMPTS = 100


def _qairt_precisions(model_id: str) -> list[Precision]:
    # Mirror Genie: filter on runtime capability only, ignore disabled_paths.
    cg = QAIHMModelManifest.from_model(model_id)
    return [
        p
        for p in cg.supported_precisions
        if TargetRuntime.GENIEX_QAIRT.supports_precision(p)
    ]


def _llamacpp_assets(model_id: str) -> dict[Precision, str]:
    """Return {precision: gguf_url} for each precision with a geniex_llamacpp
    asset in release-assets.yaml. Quants vary across models (q4_0 for most,
    mxfp4 for gpt_oss_20b, etc.).
    """
    assets = QAIHMModelReleaseAssets.from_model(model_id, not_exists_ok=True)
    out: dict[Precision, str] = {}
    for precision, prec_details in assets.precisions.items():
        asset = prec_details.universal_assets.get(ScorecardProfilePath.GENIEX_LLAMACPP)
        if asset and asset.download_url:
            out[precision] = asset.download_url
    return out


def _candidate_model_ids(filter_models: str | None) -> list[str]:
    if filter_models and filter_models.lower() != "all":
        return [m.strip() for m in filter_models.split(",") if m.strip()]
    return list(MODEL_IDS)


def discover_qairt_models(filter_models: str | None) -> list[str]:
    return [
        mid for mid in _candidate_model_ids(filter_models) if _qairt_precisions(mid)
    ]


def discover_llamacpp_models(filter_models: str | None) -> list[str]:
    return [mid for mid in _candidate_model_ids(filter_models) if _llamacpp_assets(mid)]


def _resolve_precisions(
    setting: set[str | SpecialLLMPerfPrecisionSetting],
    candidates: list[Precision],
) -> list[Precision]:
    if not candidates:
        return []
    if SpecialLLMPerfPrecisionSetting.ALL in setting:
        return candidates
    if SpecialLLMPerfPrecisionSetting.DEFAULT in setting:
        return [candidates[0]]
    candidate_set = set(candidates)
    return [
        Precision.parse(p)
        for p in setting
        if isinstance(p, str) and Precision.parse(p) in candidate_set
    ]


def fetch_geniex_qairt_bundle(
    model_id: str, precision: Precision, chipset: str, output_dir: Path
) -> tuple[Path, list[int]]:
    """Download/extract the CI-built geniex_qairt bundle. Returns (bundle_dir, context_lengths)."""
    # release-assets.yaml keys chipset_assets by canonical name, so canonicalize
    # the variant-suffixed workbench chipset (e.g. "-for-galaxy") before lookup,
    # matching the Genie perf path.
    chipset = get_canonical_chipset_name(chipset)
    assets = load_release_assets_for_model(model_id)
    asset = assets.get_asset(precision, chipset, ScorecardProfilePath.GENIEX_QAIRT)
    if asset is None:
        available: list[str] = []
        prec_details = assets.precisions.get(precision)
        if prec_details is not None:
            available = sorted(prec_details.chipset_assets.keys())
        raise RuntimeError(
            f"No geniex_qairt release asset for {model_id!r} precision={precision} "
            f"chipset={chipset!r}. Available: {available or '<none>'}. "
            "Build release-assets.yaml with GENIEX_QAIRT before running geniex-bench QAIRT."
        )

    bundle_dir = output_dir / ASSET_CONFIG.get_release_asset_name(
        model_id, TargetRuntime.GENIEX_QAIRT, precision, chipset
    )
    if not bundle_dir.exists():
        zip_path = download_prerelease_asset(
            asset,
            model_id=model_id,
            runtime=TargetRuntime.GENIEX_QAIRT,
            precision=precision,
            chipset=chipset,
            output_folder=output_dir,
            verbose=True,
        )
        shutil.unpack_archive(str(zip_path), extract_dir=str(output_dir))
        # Delete the zip once extracted so it doesn't fill the runner disk.
        zip_path.unlink(missing_ok=True)
        if not bundle_dir.exists():
            raise RuntimeError(
                f"Extracted geniex_qairt bundle missing expected directory {bundle_dir}; "
                f"contents of {output_dir}: {sorted(p.name for p in output_dir.iterdir())}"
            )

    metadata = ModelMetadata.from_json(bundle_dir / "metadata.json")
    if metadata is None or metadata.genie is None:
        raise RuntimeError(
            f"GenieX-QAIRT bundle for {model_id!r} has no genie metadata at "
            f"{bundle_dir / 'metadata.json'}"
        )
    return bundle_dir, metadata.genie.context_lengths


def _scorecard_device(name: str) -> ScorecardDevice:
    return ScorecardDevice.get(name)


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "Model",
                "Plugin",
                "Precision",
                "Device",
                "Ctx",
                "Decode TPS",
                "Prefill TPS",
                "TTFT (ms)",
                "Status",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["model"],
                    r.get("plugin", ""),
                    r.get("precision", ""),
                    r["device"],
                    r.get("ctx", ""),
                    r.get("decode_tps", ""),
                    r.get("prefill_tps", ""),
                    r.get("ttft_ms", ""),
                    r["status"],
                ]
            )


def write_summary(rows: list[dict]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return

    def _format_values(v: object, spec: str = ".2f") -> str:
        return format(v, spec) if isinstance(v, (int, float)) else "-"

    with open(summary, "a") as f:
        f.write("## GenieX On-device Results\n\n")
        f.write(
            "| Model | Plugin | Precision | Device | Ctx | Decode TPS | Prefill TPS | TTFT (ms) | Status |\n"
        )
        f.write(
            "|-------|--------|-----------|--------|----:|-----------:|------------:|----------:|--------|\n"
        )
        f.writelines(
            f"| {r['model']} | {r.get('plugin', '-')} | {r.get('precision', '-')} | "
            f"{r['device']} | {r.get('ctx', '-')} | "
            f"{_format_values(r.get('decode_tps'))} | {_format_values(r.get('prefill_tps'))} | "
            f"{_format_values(r.get('ttft_ms'), '.1f')} | {r['status']} |\n"
            for r in rows
        )
        f.write("\n")


def run_geniex_bench_job(
    model_id: str,
    model_ref: str,
    device_token: str,
    context_lengths: list[int],
    save_dir_root: str,
    plugin: str,
    geniex_version: str | None,
    llamacpp_quant: str | None = None,
    eval_prompts: list[str] | None = None,
    run_perf: bool = True,
) -> tuple[list[GenieXBenchMetrics], list[dict]]:
    """Submit-and-wait shortcut used by the ``run`` subcommand.

    ``submit`` / ``collect`` subcommands drive
    ``submit_geniex_bench_only`` / ``collect_geniex_bench_result``
    directly instead.
    """
    sd = _scorecard_device(device_token)
    api_token = get_qdc_api_token(sd)
    device_alias = ",".join(LLAMACPP_DEVICE_ALIASES) if plugin == "llama_cpp" else "npu"
    _print_job_banner(
        model_id,
        sd,
        plugin,
        device_alias,
        model_ref,
        context_lengths,
        geniex_version,
        run_perf,
        eval_prompts,
    )

    save_dir = os.path.join(save_dir_root, model_id, sd.name)
    job_name = f"geniex-bench {plugin} {model_id}"

    def _submit() -> str:
        job_id, _, _ = submit_geniex_bench_only(
            api_token=api_token,
            hub_device_name=sd.reference_device_name,
            chipset=sd.chipset,
            model_rows=[(model_id, model_ref)],
            context_lengths=context_lengths,
            plugin=plugin,
            device_alias=device_alias,
            job_name=job_name,
            geniex_version=geniex_version,
            llamacpp_quant=llamacpp_quant,
            eval_prompts=eval_prompts,
            model_id=model_id,
            run_perf=run_perf,
        )
        return job_id

    def _collect(job_id: str) -> tuple[tuple, JobOutcome, str | None]:
        metrics, eval_results, outcome, reason = collect_geniex_bench_result(
            api_token,
            sd.reference_device_name,
            job_id,
            save_results_dir=save_dir,
            eval_prompts=eval_prompts,
            run_perf=run_perf,
        )
        return (metrics, eval_results), outcome, reason

    return poll_and_retry(
        initial_job_id=_submit(),
        attempts_left=DEFAULT_ATTEMPTS - 1,
        collect_fn=_collect,
        resubmit_fn=_submit,
    )


def _print_job_banner(
    model_id: str,
    sd: ScorecardDevice,
    plugin: str,
    device_alias: str,
    model_ref: str,
    context_lengths: list[int],
    geniex_version: str | None,
    run_perf: bool = True,
    eval_prompts: list[str] | None = None,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Model:   {model_id}")
    print(f"Device:  {sd.name} ({sd.reference_device_name}, chipset={sd.chipset})")
    print(f"Plugin:  {plugin} (alias={device_alias})")
    print(f"Ref:     {model_ref}")
    print(f"Ctx:     {context_lengths}")
    print(f"GenieX:  {geniex_version or 'latest stable mirror'}")
    print(f"Perf:    {'on' if run_perf else 'off'}")
    print(f"Eval:    {'on' if eval_prompts else 'off'}")
    print(f"{'=' * 60}")


def _submit_one(
    model_id: str,
    model_ref: str,
    device_token: str,
    context_lengths: list[int],
    save_dir_root: str,
    plugin: str,
    geniex_version: str | None,
    precision: Precision,
    jobs_file: str,
    llamacpp_quant: str | None = None,
    eval_prompts: list[str] | None = None,
    run_perf: bool = True,
) -> str:
    """Submit one geniex-bench job and upsert a jobs_file entry.

    Returns the QDC job id. The collect side re-derives everything else
    from (model_id, precision, chipset) via
    ``fetch_geniex_qairt_bundle`` -- nothing local-path is persisted.
    """
    sd = _scorecard_device(device_token)
    api_token = get_qdc_api_token(sd)
    device_alias = ",".join(LLAMACPP_DEVICE_ALIASES) if plugin == "llama_cpp" else "npu"
    _print_job_banner(
        model_id,
        sd,
        plugin,
        device_alias,
        model_ref,
        context_lengths,
        geniex_version,
        run_perf,
        eval_prompts,
    )

    job_name = f"geniex-bench {plugin} {model_id}"
    job_id, _, _ = submit_geniex_bench_only(
        api_token=api_token,
        hub_device_name=sd.reference_device_name,
        chipset=sd.chipset,
        model_rows=[(model_id, model_ref)],
        context_lengths=context_lengths,
        plugin=plugin,
        device_alias=device_alias,
        job_name=job_name,
        geniex_version=geniex_version,
        llamacpp_quant=llamacpp_quant,
        eval_prompts=eval_prompts,
        model_id=model_id,
        run_perf=run_perf,
    )
    runtime = "GENIEX_QAIRT" if plugin == "qairt" else "GENIEX_LLAMACPP"
    key = make_key(model_id, str(precision), runtime, sd.name)
    save_job(jobs_file, key, job_id, attempts_left=DEFAULT_ATTEMPTS)
    return job_id


def _collect_one(
    model_id: str,
    precision: Precision,
    device_name: str,
    plugin: str,
    record: JobRecord,
    jobs_file: str,
    save_dir_root: str,
    geniex_version: str | None,
    llamacpp_urls: dict[Precision, str] | None = None,
    eval_prompts: list[str] | None = None,
    run_perf: bool = True,
) -> tuple[list[GenieXBenchMetrics], list[dict], str]:
    """Poll a submitted geniex-bench job. On retryable failure, re-fetch
    the bundle from release-assets.yaml (qairt) or the HF URL (llama_cpp)
    and resubmit; the jobs_file row is rewritten with the new job id and
    one fewer attempt.

    Returns ``(metrics, eval_results, status)`` where status is
    ``"success"``, ``"eval_only"``, ``"eval_incomplete (...)"``,
    ``"no_metrics"``, or ``"failed"``.
    """
    sd = ScorecardDevice.get(device_name)
    api_token = get_qdc_api_token(sd)
    device_alias = ",".join(LLAMACPP_DEVICE_ALIASES) if plugin == "llama_cpp" else "npu"
    job_name = f"geniex-bench {plugin} {model_id}"
    save_dir = os.path.join(save_dir_root, model_id, sd.name)
    runtime = "GENIEX_QAIRT" if plugin == "qairt" else "GENIEX_LLAMACPP"
    key = make_key(model_id, str(precision), runtime, sd.name)

    if plugin == "llama_cpp" and (
        llamacpp_urls is None or precision not in llamacpp_urls
    ):
        print(
            f"ERROR: cannot resubmit llama_cpp job for {model_id} "
            f"@ {device_name}: no GGUF URL available",
            file=sys.stderr,
        )
        return [], [], "failed"

    def _resubmit() -> str:
        if plugin == "qairt":
            bundle_dir, ctx_list = fetch_geniex_qairt_bundle(
                model_id, precision, sd.chipset, Path(save_dir_root) / "qairt_bundles"
            )
            if ctx_list:
                ctx_list = [max(ctx_list)]
            model_ref: str = str(bundle_dir)
            llamacpp_quant: str | None = None
        else:
            assert llamacpp_urls is not None
            model_ref = llamacpp_urls[precision]
            ctx_list = LLAMACPP_CONTEXT_LENGTHS
            llamacpp_quant = str(precision)
        new_job_id, _, _ = submit_geniex_bench_only(
            api_token=api_token,
            hub_device_name=sd.reference_device_name,
            chipset=sd.chipset,
            model_rows=[(model_id, model_ref)],
            context_lengths=ctx_list,
            plugin=plugin,
            device_alias=device_alias,
            job_name=job_name,
            geniex_version=geniex_version,
            llamacpp_quant=llamacpp_quant,
            eval_prompts=eval_prompts,
            model_id=model_id,
            run_perf=run_perf,
        )
        return new_job_id

    def _collect(job_id: str) -> tuple[tuple, JobOutcome, str | None]:
        metrics, eval_results, outcome, reason = collect_geniex_bench_result(
            api_token,
            sd.reference_device_name,
            job_id,
            save_results_dir=save_dir,
            eval_prompts=eval_prompts,
            run_perf=run_perf,
        )
        return (metrics, eval_results), outcome, reason

    try:
        metrics, eval_results = poll_and_retry(
            initial_job_id=record.job_id,
            attempts_left=record.attempts_left,
            collect_fn=_collect,
            resubmit_fn=_resubmit,
            on_new_job_id=lambda new_id, left: save_job(
                jobs_file, key, new_id, attempts_left=left
            ),
        )
    except RuntimeError as e:
        print(
            f"ERROR: {e} for {model_id} @ {device_name}",
            file=sys.stderr,
        )
        return [], [], "failed"

    if metrics:
        return metrics, eval_results, "success"
    # With perf off the device runs eval only, so an empty metrics list is
    # expected rather than a failure. But the eval must still return one result
    # per prompt sent; a short count means the run crashed or was truncated.
    if run_perf:
        status = "no_metrics"
    elif eval_prompts and len(eval_results) != len(eval_prompts):
        status = f"eval_incomplete ({len(eval_results)}/{len(eval_prompts)})"
    else:
        status = "eval_only"
    return metrics, eval_results, status


def _rows_and_updates_from_metrics(
    model_id: str,
    sd: ScorecardDevice,
    plugin: str,
    precision: Precision,
    metrics: list[GenieXBenchMetrics],
    skip_perf_update: bool,
) -> tuple[list[dict], list[dict]]:
    """Extracted formatting so ``run`` and ``collect`` share output shape."""
    csv_rows: list[dict] = []
    perf_updates: list[dict] = []
    for m in metrics:
        base_plugin = m.plugin or plugin
        plugin_label = (
            f"{base_plugin}_{m.device_alias}"
            if base_plugin == "llama_cpp" and m.device_alias
            else base_plugin
        )
        csv_rows.append(
            {
                "model": model_id,
                "plugin": plugin_label,
                "precision": str(precision),
                "device": sd.name,
                "ctx": m.context_length,
                "decode_tps": m.decode_tps,
                "prefill_tps": m.prefill_tps,
                "ttft_ms": m.ttft_ms,
                "status": "success",
            }
        )
        if not skip_perf_update:
            profile_path = (
                ScorecardProfilePath.GENIEX_QAIRT
                if plugin == "qairt"
                else ScorecardProfilePath.GENIEX_LLAMACPP
            )
            assert m.prompt_tokens > 0, (
                f"prompt_tokens must be > 0 for TTFT range scaling, "
                f"got {m.prompt_tokens}"
            )
            ttft_min = m.ttft_ms
            ttft_max = m.ttft_ms * (m.context_length / 128)
            update_kwargs = dict(
                model_id=model_id,
                device_name=sd.reference_device_name,
                precision=str(precision),
                context_length=m.context_length,
                tps=m.decode_tps,
                ttft_ms=ttft_min,
                prefill_tps=m.prefill_tps,
                ttft_max_ms=ttft_max,
                profile_path=profile_path.value,
                desired_compute_unit=m.device_alias,
            )
            perf_updates.append(update_kwargs)
            update_perf_yaml(
                model_id=model_id,
                device_name=sd.reference_device_name,
                precision=precision,
                context_length=m.context_length,
                tps=m.decode_tps,
                ttft_ms=ttft_min,
                prefill_tps=m.prefill_tps,
                ttft_max_ms=ttft_max,
                profile_path=profile_path,
                desired_compute_unit=m.device_alias,
            )
    return csv_rows, perf_updates


def _iter_work(
    models_setting: str,
    devices_setting: str,
    plugin_setting: str,
    precisions_setting: str,
    results_dir: str,
    geniex_version: str | None,
) -> Iterator[tuple[str, str, Precision, str, str, list[int], str | None]]:
    """Yield (plugin, model_id, precision, device_token, model_ref, ctx_list,
    llamacpp_quant) tuples for every model x precision x device combo.

    Shared by ``submit`` and ``run``. Handles all the discovery and
    bundle-fetch logic so the two paths stay in step.
    """
    plugins = ["llama_cpp", "qairt"] if plugin_setting == "all" else [plugin_setting]
    precision_setting = LLMPerfPrecisionsEnvvar.parse(precisions_setting)

    if devices_setting.strip().lower() == "all":
        devices = list(ALL_GENIEX_DEVICES)
    else:
        devices = [d.strip() for d in devices_setting.split(",") if d.strip()]

    for plugin in plugins:
        if plugin == "qairt":
            models = discover_qairt_models(models_setting)
            if not models:
                print("No models support the GENIEX_QAIRT runtime.")
                continue
        else:
            models = discover_llamacpp_models(models_setting)
            if not models:
                print("No models support the GENIEX_LLAMACPP runtime.")
                continue

        print(f"Plugin:  {plugin}")
        print(f"Models:  {models}")
        print(f"Devices: {devices}")

        for model_id in models:
            if plugin == "qairt":
                candidates = _qairt_precisions(model_id)
                llamacpp_urls: dict[Precision, str] = {}
            else:
                llamacpp_urls = _llamacpp_assets(model_id)
                candidates = list(llamacpp_urls.keys())

            precisions = _resolve_precisions(precision_setting, candidates)
            if not precisions:
                print(
                    f"Skipping {model_id} on {plugin}: no requested precision "
                    f"available (candidates={[str(p) for p in candidates]})."
                )
                continue

            for precision in precisions:
                for device_token in devices:
                    sd = _scorecard_device(device_token)
                    if plugin == "qairt":
                        bundle_dir, ctx_list = fetch_geniex_qairt_bundle(
                            model_id,
                            precision,
                            sd.chipset,
                            Path(results_dir) / "qairt_bundles",
                        )
                        # Only bench max(ctx); perf.yaml stores only that.
                        if ctx_list:
                            ctx_list = [max(ctx_list)]
                        yield (
                            plugin,
                            model_id,
                            precision,
                            device_token,
                            str(bundle_dir),
                            ctx_list,
                            None,
                        )
                    else:
                        yield (
                            plugin,
                            model_id,
                            precision,
                            device_token,
                            llamacpp_urls[precision],
                            LLAMACPP_CONTEXT_LENGTHS,
                            str(precision),
                        )


def _write_final_outputs(
    rows: list[dict],
    perf_updates: list[dict],
    csv_path: str,
    perf_updates_json: str,
) -> int:
    print(f"\n{'=' * 60}\nRESULTS SUMMARY\n{'=' * 60}")
    for r in rows:
        prec = r.get("precision", "-")
        if r["status"] == "success":
            print(
                f"  {r['model']} [{prec}] @ {r['device']} ctx={r['ctx']}: "
                f"decode={r['decode_tps']:.2f} prefill={r['prefill_tps']:.2f} "
                f"TTFT={r['ttft_ms']:.1f}ms"
            )
        else:
            print(f"  {r['model']} [{prec}] @ {r['device']}: {r['status']}")

    write_csv(rows, csv_path)
    print(f"\nResults saved to {csv_path}")
    # JSON-lines (one update per line), matching the format emitted by
    # update_perf_yaml so apply_llm_perf_updates.py has a single format to read.
    with open(perf_updates_json, "w") as f:
        f.writelines(json.dumps(u) + "\n" for u in perf_updates)
    print(f"Wrote {len(perf_updates)} perf.yaml updates to {perf_updates_json}")
    write_summary(rows)
    # eval_only rows are successful eval-only runs (perf disabled), not failures.
    ok_statuses = {"success", "eval_only"}
    failed = [r for r in rows if r["status"] not in ok_statuses]
    if not rows:
        print("No models were benchmarked (all skipped or no candidates).")
        return 0
    return 1 if failed else 0


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models", default="all", help='Comma-separated model IDs or "all".'
    )
    parser.add_argument(
        "--devices",
        default=DEFAULT_DEVICES,
        help="Comma-separated cs_* names or hub device names, "
        'or "all" for every geniex-bench-supported device.',
    )
    parser.add_argument(
        "--plugin",
        default="all",
        choices=["all", "llama_cpp", "qairt"],
    )
    parser.add_argument(
        "--precisions",
        default="default",
        help='Comma-separated precisions (e.g. "w4,w4a16"), "all" for every '
        'supported precision, or "default" to use the model\'s default precision.',
    )
    parser.add_argument("--results-dir", default="geniex_bench_results")
    parser.add_argument(
        "--geniex-version",
        default=None,
        help='GenieX release tag (e.g. "v0.3.1") to pin geniex-bench/APK '
        'downloads to. Defaults to the unversioned "latest stable" mirror.',
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        default=os.environ.get("QAIHM_RUN_EVAL", "").lower() == "true",
        help="Also collect the 100-prompt accuracy eval (geniex-bench "
        "--accuracy) and write *_eval.json for grading. Defaults to the "
        "QAIHM_RUN_EVAL env var.",
    )
    parser.add_argument(
        "--no-run-perf",
        dest="run_perf",
        action="store_false",
        default=os.environ.get("QAIHM_RUN_PERF", "true").lower() != "false",
        help="Skip the TPS/TTFT perf sweep (and perf.yaml updates), collecting "
        "only the accuracy eval. Defaults to the QAIHM_RUN_PERF env var (on "
        "unless set to 'false').",
    )


def _resolve_eval_prompts(run_eval: bool) -> list[str] | None:
    """Load the eval prompt set, capped to the first _EVAL_NUM_PROMPTS."""
    if not run_eval:
        return None
    return load_default_eval_prompts()[:_EVAL_NUM_PROMPTS]


def _eval_prompts_for_device(
    eval_prompts: list[str] | None, sd: ScorecardDevice
) -> list[str] | None:
    """Restrict the accuracy eval to the default LLM scorecard device.

    Mirrors the genie path (see ``_shared/llm/test.py``): eval is expensive
    (~100 prompts x per-prompt DSP attach), so run it only on
    ``DEFAULT_QDC_DEVICE`` (cs_8_elite_qrd) and leave the other devices
    doing perf only.
    """
    return eval_prompts if sd == DEFAULT_QDC_DEVICE else None


def _save_eval_results(
    eval_results: list[dict],
    model_id: str,
    sd: ScorecardDevice,
    precision: Precision,
    plugin: str,
) -> None:
    if not eval_results:
        return
    profile_path = (
        ScorecardProfilePath.GENIEX_QAIRT
        if plugin == "qairt"
        else ScorecardProfilePath.GENIEX_LLAMACPP
    )
    base = f"{model_id}_{sd.chipset}_{precision}_{plugin}_eval"
    save_eval_results_json(eval_results, f"{base}.json")
    # Identity sidecar for collect_llm_accuracy_csv; without it the grade file
    # is dropped and no accuracy.csv row is written.
    save_eval_metadata_json(
        model_id,
        sd.chipset,
        str(precision),
        f"{base}.meta.json",
        path=profile_path,
    )


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--csv", default="geniex_bench_results.csv")
    parser.add_argument("--skip-perf-update", action="store_true")
    parser.add_argument(
        "--perf-updates-json",
        default="geniex_perf_updates.jsonl",
        help="JSON-lines log of update_perf_yaml calls; replayed by apply_llm_perf_updates.py.",
    )


def _cmd_submit(args: argparse.Namespace) -> int:
    eval_prompts = _resolve_eval_prompts(args.run_eval)
    if os.path.exists(args.jobs_file):
        os.unlink(args.jobs_file)
    submitted = 0
    for (
        plugin,
        model_id,
        precision,
        device_token,
        model_ref,
        ctx_list,
        llamacpp_quant,
    ) in _iter_work(
        args.models,
        args.devices,
        args.plugin,
        args.precisions,
        args.results_dir,
        args.geniex_version,
    ):
        try:
            _submit_one(
                model_id,
                model_ref,
                device_token,
                ctx_list,
                args.results_dir,
                plugin,
                args.geniex_version,
                precision,
                args.jobs_file,
                llamacpp_quant=llamacpp_quant,
                eval_prompts=_eval_prompts_for_device(
                    eval_prompts, _scorecard_device(device_token)
                ),
                run_perf=args.run_perf,
            )
            submitted += 1
        except Exception as e:  # noqa: PERF203
            print(
                f"ERROR: submission failed for {model_id} @ {device_token} "
                f"(plugin={plugin}, precision={precision}): {e}",
                file=sys.stderr,
            )
    print(f"Submitted {submitted} geniex-bench job(s) to {args.jobs_file}")
    return 0 if submitted else 1


def _cmd_collect(args: argparse.Namespace) -> int:
    """Re-iterate the submit-side work list and pick up each job by key.

    The jobs_file carries just (job_id, attempts_left), so we re-run the
    same _iter_work discovery the submit side used and look each
    (model, precision, plugin, device) tuple up in the jobs_file by key.
    """
    eval_prompts = _resolve_eval_prompts(args.run_eval)
    rows: list[dict] = []
    perf_updates: list[dict] = []
    records = load_jobs(args.jobs_file)
    llamacpp_cache: dict[str, dict[Precision, str]] = {}
    for (
        plugin,
        model_id,
        precision,
        device_token,
        _model_ref,
        _ctx_list,
        _llamacpp_quant,
    ) in _iter_work(
        args.models,
        args.devices,
        args.plugin,
        args.precisions,
        args.results_dir,
        args.geniex_version,
    ):
        sd = _scorecard_device(device_token)
        runtime = "GENIEX_QAIRT" if plugin == "qairt" else "GENIEX_LLAMACPP"
        key = make_key(model_id, str(precision), runtime, sd.name)
        record = records.get(key)
        if record is None:
            print(
                f"jobs_file has no entry for {key}; skipping (was it submitted?)",
                file=sys.stderr,
            )
            continue

        llamacpp_urls: dict[Precision, str] | None = None
        if plugin == "llama_cpp":
            llamacpp_urls = llamacpp_cache.setdefault(
                model_id, _llamacpp_assets(model_id)
            )

        try:
            metrics, eval_results, status = _collect_one(
                model_id=model_id,
                precision=precision,
                device_name=sd.name,
                plugin=plugin,
                record=record,
                jobs_file=args.jobs_file,
                save_dir_root=args.results_dir,
                geniex_version=args.geniex_version,
                llamacpp_urls=llamacpp_urls,
                eval_prompts=_eval_prompts_for_device(eval_prompts, sd),
                run_perf=args.run_perf,
            )
        except Exception as e:
            print(
                f"ERROR: collection failed for {model_id} @ {sd.name}: {e}",
                file=sys.stderr,
            )
            rows.append(
                {
                    "model": model_id,
                    "plugin": plugin,
                    "precision": str(precision),
                    "device": sd.name,
                    "status": "failed",
                }
            )
            continue

        _save_eval_results(eval_results, model_id, sd, precision, plugin)

        if status != "success":
            rows.append(
                {
                    "model": model_id,
                    "plugin": plugin,
                    "precision": str(precision),
                    "device": sd.name,
                    "status": status,
                }
            )
            continue

        csv_rows, updates = _rows_and_updates_from_metrics(
            model_id, sd, plugin, precision, metrics, args.skip_perf_update
        )
        rows.extend(csv_rows)
        perf_updates.extend(updates)

    return _write_final_outputs(rows, perf_updates, args.csv, args.perf_updates_json)


def _cmd_run(args: argparse.Namespace) -> int:
    """Legacy behavior: submit-and-wait per job, no jobs_file artifact.

    Preserved so local users and existing scripts keep working. CI drives
    the submit/collect subcommands instead.
    """
    eval_prompts = _resolve_eval_prompts(args.run_eval)
    rows: list[dict] = []
    perf_updates: list[dict] = []
    for (
        plugin,
        model_id,
        precision,
        device_token,
        model_ref,
        ctx_list,
        llamacpp_quant,
    ) in _iter_work(
        args.models,
        args.devices,
        args.plugin,
        args.precisions,
        args.results_dir,
        args.geniex_version,
    ):
        sd = _scorecard_device(device_token)
        device_eval_prompts = _eval_prompts_for_device(eval_prompts, sd)
        try:
            metrics, eval_results = run_geniex_bench_job(
                model_id,
                model_ref,
                device_token,
                ctx_list,
                args.results_dir,
                plugin,
                args.geniex_version,
                llamacpp_quant=llamacpp_quant,
                eval_prompts=device_eval_prompts,
                run_perf=args.run_perf,
            )
        except Exception as e:
            print(
                f"ERROR: geniex-bench job failed for {model_id} @ {sd.name} "
                f"(plugin={plugin}, precision={precision}): {e}",
                file=sys.stderr,
            )
            rows.append(
                {
                    "model": model_id,
                    "plugin": plugin,
                    "precision": str(precision),
                    "device": sd.name,
                    "status": "failed",
                }
            )
            continue

        _save_eval_results(eval_results, model_id, sd, precision, plugin)

        if not metrics:
            # With perf off the device runs eval only, so an empty metrics list
            # is expected rather than a failure. But the eval must still return
            # one result per prompt sent; a short count means the run crashed or
            # was truncated.
            if args.run_perf:
                status = "no_metrics"
            elif device_eval_prompts and len(eval_results) != len(device_eval_prompts):
                status = (
                    f"eval_incomplete ({len(eval_results)}/{len(device_eval_prompts)})"
                )
            else:
                status = "eval_only"
            rows.append(
                {
                    "model": model_id,
                    "plugin": plugin,
                    "precision": str(precision),
                    "device": sd.name,
                    "status": status,
                }
            )
            continue

        csv_rows, updates = _rows_and_updates_from_metrics(
            model_id, sd, plugin, precision, metrics, args.skip_perf_update
        )
        rows.extend(csv_rows)
        perf_updates.extend(updates)

    return _write_final_outputs(rows, perf_updates, args.csv, args.perf_updates_json)


def _warn_geniex_version(version: str | None) -> None:
    if version and not version.startswith("v"):
        print(
            f"WARNING: --geniex-version={version!r} does not start "
            f'with "v"; release tags are SemVer-prefixed (e.g. "v0.3.1"). The '
            f"S3 download will likely 404.",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run geniex-bench benchmarks on QDC.")
    sub = ap.add_subparsers(dest="cmd")

    p_submit = sub.add_parser(
        "submit", help="Submit jobs; append one row per submission to a jobs file."
    )
    _add_shared_args(p_submit)
    p_submit.add_argument(
        "--jobs-file",
        default="geniex_jobs.yaml",
        help="Path to write the YAML jobs file for the collect step.",
    )

    p_collect = sub.add_parser(
        "collect", help="Poll jobs listed in the jobs file and emit CSV/JSONL."
    )
    _add_shared_args(p_collect)
    p_collect.add_argument(
        "--jobs-file",
        default="geniex_jobs.yaml",
        help="Path to the YAML jobs file produced by ``submit``.",
    )
    _add_output_args(p_collect)

    p_run = sub.add_parser(
        "run", help="Legacy submit-and-wait mode (default when no subcommand given)."
    )
    _add_shared_args(p_run)
    _add_output_args(p_run)

    # No subcommand -> ``run`` (backward-compat for flat-arg invocations).
    argv = sys.argv[1:]
    if argv and argv[0] not in {"submit", "collect", "run", "-h", "--help"}:
        argv = ["run", *argv]
    args = ap.parse_args(argv)

    _warn_geniex_version(getattr(args, "geniex_version", None))

    if not args.run_perf and not args.run_eval:
        print(
            "ERROR: nothing to do -- perf is disabled (--no-run-perf) and "
            "--run-eval was not given.",
            file=sys.stderr,
        )
        return 1

    if args.cmd == "submit":
        return _cmd_submit(args)
    if args.cmd == "collect":
        return _cmd_collect(args)
    return _cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
