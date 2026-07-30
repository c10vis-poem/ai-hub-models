# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""On-device geniex-bench scorecard run for QDC Android phones.

Mirrors run_android.py's robustness pattern: network preflight, single
retry per cell, and explicit on-device output check.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import pytest

HOST_ARTIFACT_ROOT = "/qdc/appium"
HOST_QAIRT_BUNDLES = f"{HOST_ARTIFACT_ROOT}/qairt_bundles"
HOST_ROWS = f"{HOST_ARTIFACT_ROOT}/matrix_rows.txt"
HOST_CHIPSET = f"{HOST_ARTIFACT_ROOT}/chipset.txt"
HOST_PROMPTS = f"{HOST_ARTIFACT_ROOT}/prompts"
HOST_STAGE = f"{HOST_ARTIFACT_ROOT}/_stage"
HOST_BUNDLE = f"{HOST_STAGE}/bundle"

DEVICE_BUNDLE = "/data/local/tmp/pkg-geniex"
DEVICE_QAIRT_BUNDLES = f"{DEVICE_BUNDLE}/qairt_bundles"
DEVICE_MM_CACHE = "/data/local/tmp/geniex-cache"
DEVICE_QDC_LOGS = "/data/local/tmp/QDC_logs"
DEVICE_RESULTS = f"{DEVICE_QDC_LOGS}/results"
DEVICE_PROMPTS = "/data/local/tmp/eval_prompts"
DEVICE_EVAL_OUT = f"{DEVICE_QDC_LOGS}/geniex_eval_outputs.txt"
DEVICE_EVAL_ERR = f"{DEVICE_QDC_LOGS}/geniex_eval_stderr.txt"

CTXS = tuple(int(c) for c in "{CTX_LIST}".split(","))
ANDROID_BENCH_URL = "{ANDROID_BENCH_URL}"
PLUGIN = "{PLUGIN}"
N_GEN = int("{N_GEN}")
EVAL_CTX = int("{EVAL_CTX}")
EVAL_N_GEN = int("{EVAL_N_GEN}")
EVAL_TIMEOUT_S = int("{EVAL_TIMEOUT_S}")
EVAL_SLEEP_S = int("{EVAL_SLEEP_S}")
# Placeholder substituted with the literal "1"/"0" host-side; the string
# comparison keeps the template statically typed (a bare `= {RUN_PERF}` reads as
# a self-referential set literal to mypy).
RUN_PERF = "{RUN_PERF}" == "1"  # noqa: PLR0133
RUN_EVAL = "{RUN_EVAL}" == "1"  # noqa: PLR0133


def adb(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    # adb shell drops the remote exit code; recover it via __RC__ trailer.
    raw = subprocess.run(
        ["adb", "shell", f"{cmd}; echo __RC__:$?"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )
    stdout, rc = raw.stdout, raw.returncode
    lines = stdout.rstrip("\n").split("\n") if stdout else []
    if lines and lines[-1].startswith("__RC__:"):
        try:
            rc = int(lines[-1][7:])
            stdout = "\n".join(lines[:-1]) + "\n"
        except ValueError:
            pass
    print(stdout)
    result = subprocess.CompletedProcess(raw.args, rc, stdout=stdout)
    if check:
        assert rc == 0, f"adb command failed (exit {rc}): {cmd}"
    return result


def _preflight_network() -> None:
    # QDC phones occasionally boot with degraded wifi; fail fast instead
    # of letting stage_bundle() hang silently on a stalled fetch. HEAD the
    # actual asset (a bucket-root probe returns 403 on S3 list-bucket).
    preflight = subprocess.run(
        [
            "adb",
            "shell",
            f"curl -sSI -o /dev/null -w '%{{http_code}}' --max-time 15 "
            f"{ANDROID_BENCH_URL}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    http_code = preflight.stdout.strip()
    if preflight.returncode != 0 or not http_code.startswith(("2", "3")):
        pytest.fail(
            f"Device cannot reach {ANDROID_BENCH_URL} (rc={preflight.returncode}, "
            f"http_code={http_code!r}, stderr={preflight.stderr!r}). "
            "Likely QDC device-side wifi failure — file a QDC infra ticket and re-run."
        )


def stage_bundle() -> None:
    if os.path.exists(os.path.join(HOST_BUNDLE, "bin", "geniex-bench")):
        return
    os.makedirs(HOST_STAGE, exist_ok=True)

    print(f"Fetching {ANDROID_BENCH_URL}")
    with urllib.request.urlopen(ANDROID_BENCH_URL) as resp:
        bench_tgz = resp.read()
    with tarfile.open(fileobj=io.BytesIO(bench_tgz), mode="r:gz") as tf:
        members = tf.getmembers()
        top = members[0].name.split("/", 1)[0] if members else ""
        for m in members:
            if not m.name.startswith(top + "/") or m.name == top + "/":
                continue
            rel = m.name[len(top) + 1 :]
            dst = os.path.join(HOST_BUNDLE, rel)
            real_dst = os.path.realpath(dst)
            real_base = os.path.realpath(HOST_BUNDLE)
            if not real_dst.startswith(real_base + os.sep):
                raise ValueError(f"Refusing unsafe tar member path: {m.name!r}")
            if m.isdir():
                os.makedirs(dst, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            f = tf.extractfile(m)
            if f is None:
                continue
            with open(dst, "wb") as out:
                shutil.copyfileobj(f, out)
            os.chmod(dst, m.mode | 0o400)


def push_bundle() -> None:
    # QDC reflashes the phone every session, so always push.
    stage_bundle()
    subprocess.run(["adb", "push", HOST_BUNDLE, DEVICE_BUNDLE], check=True)
    adb(f"find {DEVICE_BUNDLE}/bin -type f -exec chmod 755 {{}} +")
    adb(f"cp {DEVICE_BUNDLE}/lib/qairt/htp-files/*.so {DEVICE_BUNDLE}/lib/")
    adb(f"cp {DEVICE_BUNDLE}/lib/llama_cpp/*.so {DEVICE_BUNDLE}/lib/")


def _run_bench(
    ctx: int, env: str, tsv_path: str, chipset: str, bundle_name: str | None
) -> int:
    if PLUGIN == "qairt":
        assert bundle_name is not None
        size_flags = (
            f"-c {ctx} -n {N_GEN} "
            f"--prompt-file {DEVICE_QAIRT_BUNDLES}/{bundle_name}/sample_prompt.txt"
        )
    else:
        size_flags = f"-c {ctx + N_GEN} -p {ctx} -n {N_GEN}"
    cmd = (
        f"cd {DEVICE_BUNDLE} && {env} ./bin/geniex-bench "
        f"--matrix-file {tsv_path} --output-json-dir {DEVICE_RESULTS} -r 3 "
        f"{size_flags} "
        f"--mm-data-dir {DEVICE_MM_CACHE} --chipset '{chipset}' "
        f"2>>{DEVICE_QDC_LOGS}/geniex_bench_stderr.log"
    )
    res = adb(cmd, check=False)
    if res.returncode == 0:
        return 0
    print(f"geniex-bench ctx={ctx} failed (rc={res.returncode}); retrying once.")
    return adb(cmd, check=False).returncode


def _cleanup_device() -> None:
    # Drop per-job state (dedicated-pool devices are reused across jobs; leftover
    # bundles / caches / matrix TSVs would leak into the next tenant's run).
    # QDC_logs is left alone so retrieval still sees results/logs.
    adb(f"rm -rf {DEVICE_BUNDLE} {DEVICE_MM_CACHE}", check=False)
    adb("rm -f /data/local/tmp/matrix-*.tsv", check=False)


def _run_eval(env: str, model_ref: str, device_alias: str, chipset: str) -> int:
    """Run `geniex-bench --accuracy` once per staged prompt, collecting stdout.

    Output for each prompt is marked with ===EVAL_IDX_NNN=== so the host can
    split and grade it. Returns the number of prompts that produced output; a
    context-length overflow keeps its partial [gen] output and counts as run.
    """
    assert os.path.isdir(HOST_PROMPTS), f"eval requested but {HOST_PROMPTS} missing"
    prompt_files = sorted(
        f for f in os.listdir(HOST_PROMPTS) if f.startswith("prompt_")
    )
    adb(f"mkdir -p {DEVICE_PROMPTS}")
    subprocess.run(["adb", "push", f"{HOST_PROMPTS}/.", DEVICE_PROMPTS], check=True)
    adb(f"rm -f {DEVICE_EVAL_OUT} {DEVICE_EVAL_ERR}")
    eval_ran = 0
    for fn in prompt_files:
        idx = fn[len("prompt_") : -len(".txt")]
        pf = f"{DEVICE_PROMPTS}/{fn}"
        perr = f"{DEVICE_QDC_LOGS}/geniex_eval_prompt_stderr.txt"
        pout = f"{DEVICE_QDC_LOGS}/geniex_eval_prompt_stdout.txt"
        # Each attempt writes to fresh per-prompt temp files on device; the
        # accepted attempt's stdout is concatenated under the ===EVAL_IDX===
        # marker exactly once so retries do not concatenate two generations at
        # the grader. timeout kills a crashed/hung DSP run so the loop advances;
        # a context-length overflow is a legit long generation (partial output
        # kept), so force rc=0 and skip retry.
        cmd = (
            f"cd {DEVICE_BUNDLE} && "
            f"rm -f {pout} {perr} && "
            f"{env} timeout {EVAL_TIMEOUT_S} ./bin/geniex-bench --plugin {PLUGIN} "
            f"--device {device_alias} -m {model_ref} --accuracy --prompt-file {pf} "
            f"-c {EVAL_CTX} -n {EVAL_N_GEN} --mm-data-dir {DEVICE_MM_CACHE} "
            f"--chipset '{chipset}' >{pout} 2>{perr}; "
            f"rc=$?; grep -q 'Context length exceeded' {perr} && rc=0; exit $rc"
        )
        # Retry a random device flake once; the second run's fresh pout/perr
        # overwrite the first attempt's output.
        res = adb(cmd, check=False)
        if res.returncode != 0:
            print(f"eval idx {idx} failed (rc={res.returncode}); retrying once")
            res = adb(cmd, check=False)
        if res.returncode == 0:
            eval_ran += 1
        else:
            print(f"eval idx {idx} failed (rc={res.returncode})")
        # Emit marker + accepted attempt's output once, then wipe the tmpfiles.
        adb(f"echo '===EVAL_IDX_{idx}===' >> {DEVICE_EVAL_OUT}", check=False)
        adb(f"echo '===EVAL_IDX_{idx}===' >> {DEVICE_EVAL_ERR}", check=False)
        adb(f"cat {pout} >> {DEVICE_EVAL_OUT} 2>/dev/null", check=False)
        adb(f"cat {perr} >> {DEVICE_EVAL_ERR} 2>/dev/null", check=False)
        adb(f"rm -f {pout} {perr}", check=False)
        adb(f"sleep {EVAL_SLEEP_S}", check=False)
    n = adb(f"grep -c '===EVAL_IDX_' {DEVICE_EVAL_OUT}", check=False)
    print(f"eval done ({n.stdout.strip()} prompts, {eval_ran} ran)")
    return eval_ran


def test_scorecard() -> None:
    _preflight_network()
    push_bundle()
    # QDC reuses device cells across jobs; wipe stale cell JSONs from a prior
    # run so compute_metrics doesn't ingest another model/plugin's results.
    adb(f"rm -rf {DEVICE_RESULTS}", check=False)
    adb(f"mkdir -p {DEVICE_MM_CACHE} {DEVICE_RESULTS}")
    try:
        bundle_name: str | None = None
        if PLUGIN == "qairt" and os.path.isdir(HOST_QAIRT_BUNDLES):
            adb(f"mkdir -p {DEVICE_QAIRT_BUNDLES}")
            subprocess.run(
                ["adb", "push", f"{HOST_QAIRT_BUNDLES}/.", DEVICE_QAIRT_BUNDLES],
                check=True,
            )
            names = [
                d
                for d in os.listdir(HOST_QAIRT_BUNDLES)
                if os.path.isdir(os.path.join(HOST_QAIRT_BUNDLES, d))
            ]
            assert len(names) == 1, f"expected one qairt bundle, got {names}"
            bundle_name = names[0]

        chipset = Path(HOST_CHIPSET).read_text().strip()
        rows = [r for r in Path(HOST_ROWS).read_text().splitlines() if r.strip()]
        tsv_by_ctx: dict[int, list[str]] = {ctx: [] for ctx in CTXS}
        for row in rows:
            name, plugin, devs, model_id, vlm, _image = row.split("|")
            for d in devs.split(","):
                for ctx in CTXS:
                    tsv_by_ctx[ctx].append(
                        f"{name}-{plugin}-{d}-c{ctx}\t{plugin}\t{d}\t{model_id}"
                        f"\t\t\t\t{vlm}"
                    )
        assert any(tsv_by_ctx.values()), "no model rows produced"

        lib = f"{DEVICE_BUNDLE}/lib"
        env = (
            f"LD_LIBRARY_PATH={lib}:{lib}/llama_cpp:{lib}/qairt "
            f"ADSP_LIBRARY_PATH={lib} "
            f"GENIEX_PLUGIN_PATH={lib}"
        )
        # Perf sweep records failures but defers the fatal check to the end so a
        # perf failure never skips the accuracy eval below.
        failures = []
        cell_json_count = 0
        results_listing = ""
        if RUN_PERF:
            for ctx in CTXS:
                tsv_path = f"/data/local/tmp/matrix-{ctx}.tsv"
                adb(
                    "printf '%s\\n' "
                    + " ".join(f"'{ln}'" for ln in tsv_by_ctx[ctx])
                    + f" > {tsv_path}"
                )
                if _run_bench(ctx, env, tsv_path, chipset, bundle_name) != 0:
                    failures.append(ctx)
            # Confirm cell JSONs exist; adb hides on-device exit codes.
            results_listing = adb(f"ls -l {DEVICE_RESULTS}", check=False).stdout
            count_proc = adb(f"ls {DEVICE_RESULTS} | wc -l", check=False)
            cell_json_count = (
                int(count_proc.stdout.strip().split()[-1])
                if count_proc.stdout.strip()
                else 0
            )
        else:
            print("perf sweep disabled (RUN_PERF=False); eval only")

        eval_ran = 0
        if RUN_EVAL:
            # Single model per job: derive the -m ref from the first row.
            _first_name, _, _first_devs, first_model = rows[0].split("|")[:4]
            eval_model = (
                f"{DEVICE_QAIRT_BUNDLES}/{bundle_name}"
                if PLUGIN == "qairt" and bundle_name
                else first_model
            )
            # Accuracy eval always runs on the NPU (the on-device target).
            eval_ran = _run_eval(env, eval_model, "npu", chipset)

        if RUN_PERF and (failures or cell_json_count == 0):
            pytest.fail(
                f"geniex-bench produced no usable output (failed ctxs={failures}, "
                f"cell_json_count={cell_json_count}).\n--- {DEVICE_RESULTS} ---\n"
                f"{results_listing}"
            )
        # Every prompt failed => fail loudly so the host retry loop resubmits
        # instead of adb silently hiding the device drop.
        if RUN_EVAL and eval_ran == 0:
            pytest.fail("accuracy eval produced no output (all prompts failed)")
    finally:
        _cleanup_device()


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    )
