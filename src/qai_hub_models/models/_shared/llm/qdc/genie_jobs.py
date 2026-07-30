# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import os
import pathlib
import re
import shutil
import statistics
import tempfile
import zipfile
from abc import ABC, abstractmethod

from qualcomm_device_cloud_sdk.models import ArtifactType
from transformers import AutoTokenizer

from qai_hub_models.models._shared.llm.common import JobOutcome
from qai_hub_models.models._shared.llm.model import LLMBase
from qai_hub_models.models._shared.llm.qdc.qdc_jobs import (
    HUB_DEVICE_TO_QDC_DEVICE_MAP,
    QDCDevice,
    QDCJobs,
    create_zip_from_entries,
)
from qai_hub_models.scorecard import ScorecardProfilePath

GENIE_JOB_TIMEOUT = 21600  # 6 hours

DEFAULT_LLM_SYSTEM_PROMPT = LLMBase.default_system_prompt


DEFAULT_EVAL_PROMPTS_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "eval_prompts.json",
    )
)


def _write_eval_prompts_to_dir(
    prompts_dir: str,
    prompts: list[str],
    tokenizer_fallback_source: str,
    model_id: str | None = None,
) -> None:
    """Write chat-templated prompt files into ``prompts_dir``.

    Loads the tokenizer from the model's HF repo (HF_REPO_NAME) to apply
    the correct chat template. Falls back to ``tokenizer_fallback_source``
    (typically the Genie bundle path) only when ``model_id`` is not
    provided or the model module lacks ``HF_REPO_NAME``.

    Thinking mode is disabled for the eval prompts so the model returns a
    direct answer within the on-device token budget instead of spending it
    on a reasoning trace that may be truncated before any answer is produced.
    Passing enable_thinking=False is safe for non-thinking models -- their
    chat templates simply ignore the unused variable.
    """
    hf_repo: str | None = None
    if model_id:
        model_module = importlib.import_module(f"qai_hub_models.models.{model_id}")
        hf_repo = getattr(model_module, "HF_REPO_NAME", None)
    # Loading from the bundle dir can fail on newer chat_template formats
    # (list-of-dicts) that some transformers versions mishandle with
    # "'list' object has no attribute 'keys'". Prefer the HF repo tokenizer,
    # which is what apply_chat_template is exercised against upstream.
    tokenizer = AutoTokenizer.from_pretrained(
        hf_repo if hf_repo else tokenizer_fallback_source
    )

    os.makedirs(prompts_dir, exist_ok=True)
    for idx, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": DEFAULT_LLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        prompt_file = os.path.join(prompts_dir, f"prompt_{idx:03d}.txt")
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(formatted)  # type: ignore[arg-type, unused-ignore]


def _walk_dir_entries(src_dir: str, arcname_prefix: str = "") -> list[tuple[str, str]]:
    """List (abs_path, arcname) pairs for every file under ``src_dir``.

    ``arcname_prefix`` (if non-empty) is prepended to each arcname so the
    zip preserves a directory layout without needing a filesystem copy.
    """
    entries: list[tuple[str, str]] = []
    src_dir = os.fspath(src_dir)
    for root, _, files in os.walk(src_dir):
        for fn in files:
            abs_path = os.path.join(root, fn)
            rel = os.path.relpath(abs_path, src_dir)
            arcname = os.path.join(arcname_prefix, rel) if arcname_prefix else rel
            entries.append((abs_path, arcname))
    return entries


class GenieArtifactHandler(ABC):
    """Abstract base class for Genie artifact handlers."""

    @abstractmethod
    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
    ) -> str:
        """Create artifact bundle and return path to the zip file.

        ``eval_prompts_dir`` (if provided) points at a small tmpdir holding
        the chat-templated ``prompt_NNN.txt`` files; the handler streams
        those into the zip under the appropriate on-device path. Callers
        avoid duplicating the multi-GB genie bundle by keeping the prompt
        files in a separate directory from the bundle itself.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def entry_script(self) -> str | None:
        raise NotImplementedError


class GenieAndroidArtifactHandler(GenieArtifactHandler):
    def __init__(self, test_script: str) -> None:
        self.test_script: str = test_script

    @property
    def entry_script(self) -> str | None:
        return None

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
    ) -> str:
        # Write only the small placeholder-substituted files (test_appium.py,
        # requirements.txt) to dest_dir. The multi-GB genie bundle stays put
        # and is streamed into the zip directly via arcname prefix -- copying
        # it here would triple /scratch usage during zip creation.
        test_folder = os.path.join(dest_dir, "tests")
        os.makedirs(test_folder, exist_ok=True)

        test_appium_path = os.path.join(test_folder, "test_appium.py")
        with open(
            os.path.join(curr_dirname, "device_scripts", self.test_script),
            encoding="utf-8",
        ) as f:
            file_content = f.read()
        with open(test_appium_path, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("<<HEXAGON_VERSION>>", hexagon_version)
                .replace("<<QAIRT_VERSION>>", qairt_version)
                .replace("<<NUM_TRIALS>>", str(num_trials))
            )

        requirements_dest = os.path.join(dest_dir, "requirements.txt")
        shutil.copy(
            os.path.join(curr_dirname, "device_scripts", "requirements.txt"),
            requirements_dest,
        )

        # Assemble the zip entry list without copying the bundle: reference
        # every genie bundle file from its original path with a "genie_bundle/"
        # arcname prefix, and append eval-prompt files (if any) under
        # "genie_bundle/prompts/".
        entries: list[tuple[str, str]] = [
            (test_appium_path, os.path.join("tests", "test_appium.py")),
            (requirements_dest, "requirements.txt"),
        ]
        entries.extend(
            _walk_dir_entries(
                os.fspath(genie_bundle_path), arcname_prefix="genie_bundle"
            )
        )
        if eval_prompts_dir:
            entries.extend(
                _walk_dir_entries(
                    eval_prompts_dir,
                    arcname_prefix=os.path.join("genie_bundle", "prompts"),
                )
            )

        zip_path = os.path.join(dest_dir, "test.zip")
        create_zip_from_entries(zip_path, entries)
        return zip_path


class GenieAutoArtifactHandler(GenieAndroidArtifactHandler):
    """Artifact handler for automotive (auto) devices.

    Extends the Android handler by bundling the QAIRT SDK into the artifact,
    since auto devices cannot download it at runtime.
    """

    def __init__(self, test_script: str, qairt_sdk_path: str) -> None:
        """
        Parameters
        ----------
        test_script
            Filename of the Appium/PyTest script to bundle (e.g., ``run_auto_android.py``).
        qairt_sdk_path
            Path to the QAIRT SDK zip file to bundle with the artifact.
            Must be an accessible, valid zip file.
        """
        super().__init__(test_script)
        if not os.path.isfile(qairt_sdk_path):
            raise FileNotFoundError(
                f"QAIRT SDK path '{qairt_sdk_path}' does not exist or is not a file. "
                "Please verify the --qairt-sdk-path argument."
            )
        self.qairt_sdk_path: str = qairt_sdk_path

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
    ) -> str:
        # Build the standard Android artifact first
        zip_path = super().create_artifact(
            curr_dirname,
            genie_bundle_path,
            dest_dir,
            hexagon_version,
            qairt_version,
            num_trials,
            eval_prompts_dir=eval_prompts_dir,
        )

        # Append the QAIRT SDK into the artifact zip under genie_bundle/
        print(
            f"[QDC] Adding QAIRT SDK from {self.qairt_sdk_path} to {zip_path}...",
            flush=True,
        )
        arcname = os.path.join("genie_bundle", "qairt_sdk.zip")
        # force_zip64 so a >2 GiB SDK doesn't abort mid-write.
        with (
            zipfile.ZipFile(zip_path, "a", allowZip64=True) as zf,
            open(self.qairt_sdk_path, "rb") as src,
            zf.open(arcname, "w", force_zip64=True) as dest,
        ):
            shutil.copyfileobj(src, dest)
        print("[QDC] QAIRT SDK addition to zip complete", flush=True)
        return zip_path


class GenieLinuxArtifactHandler(GenieArtifactHandler):
    """Artifact handler for Linux IoT devices (e.g., IQ9).
    Uses Bash test framework — no Appium wrapper needed.
    """

    @property
    def entry_script(self) -> str:
        return "/bin/bash /data/local/tmp/TestContent/run_linux.sh"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
    ) -> str:
        # Write the version-substituted run_linux.sh into dest_dir; stream
        # the multi-GB genie bundle into the zip via arcname prefix rather
        # than copying it -- see GenieAndroidArtifactHandler for context.
        script_name = "run_linux.sh"
        script_dest = os.path.join(dest_dir, script_name)
        with open(
            os.path.join(curr_dirname, "device_scripts", script_name),
            encoding="utf-8",
        ) as f:
            file_content = f.read()
        with open(script_dest, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("{HEXAGON_VERSION}", hexagon_version)
                .replace("{QAIRT_VERSION}", qairt_version)
                .replace("{NUM_TRIALS}", str(num_trials))
            )

        entries: list[tuple[str, str]] = [(script_dest, script_name)]
        entries.extend(
            _walk_dir_entries(
                os.fspath(genie_bundle_path), arcname_prefix="genie_bundle"
            )
        )
        if eval_prompts_dir:
            entries.extend(
                _walk_dir_entries(
                    eval_prompts_dir,
                    arcname_prefix=os.path.join("genie_bundle", "prompts"),
                )
            )

        zip_path = os.path.join(dest_dir, "test.zip")
        create_zip_from_entries(zip_path, entries)
        return zip_path


class GenieWindowsArtifactHandler(GenieArtifactHandler):
    @property
    def entry_script(self) -> str:
        return "C:\\Temp\\TestContent\\run_windows.ps1"

    def create_artifact(
        self,
        curr_dirname: os.PathLike | str,
        genie_bundle_path: os.PathLike | str,
        dest_dir: os.PathLike | str,
        hexagon_version: str,
        qairt_version: str,
        num_trials: int = 25,
        eval_prompts_dir: str | None = None,
    ) -> str:
        # Windows layout: run_windows.ps1 + bundle contents live at the top
        # of the zip (no genie_bundle/ prefix). Substitute placeholders in
        # the script; stream the bundle via arcnames without copying it.
        script_name = "run_windows.ps1"
        dest_script = os.path.join(dest_dir, script_name)
        with open(
            os.path.join(curr_dirname, "device_scripts", script_name),
            encoding="utf-8",
        ) as f:
            file_content = f.read()
        with open(dest_script, "w", encoding="utf-8") as f:
            f.write(
                file_content.replace("{HEXAGON_VERSION}", hexagon_version)
                .replace("{QAIRT_VERSION}", qairt_version)
                .replace("{NUM_TRIALS}", str(num_trials))
            )

        entries: list[tuple[str, str]] = [(dest_script, script_name)]
        entries.extend(_walk_dir_entries(os.fspath(genie_bundle_path)))
        if eval_prompts_dir:
            entries.extend(
                _walk_dir_entries(eval_prompts_dir, arcname_prefix="prompts")
            )

        zip_path = os.path.join(dest_dir, "test.zip")
        create_zip_from_entries(zip_path, entries)
        return zip_path


class GenieQDCJobs(QDCJobs):
    """
    QDC job handler for Genie workloads.

    Handles uploading Genie bundles and parsing performance metrics
    from Genie benchmark logs.
    """

    def _get_artifact_handler(
        self,
        qdc_device: QDCDevice,
        qairt_sdk_path: str | None = None,
    ) -> GenieArtifactHandler:
        """Get the appropriate artifact handler based on device platform.

        Parameters
        ----------
        qdc_device
            QDCDevice instance (passed to avoid redundant instantiation).
        qairt_sdk_path
            Path to the QAIRT SDK zip file. Required for auto devices.

        Returns
        -------
        genie_artifact_handler: GenieArtifactHandler
            Instance of the appropriate GenieArtifactHandler subclass.
        """
        if qdc_device.windows_platform:
            return GenieWindowsArtifactHandler()
        if qdc_device.iot_platform:
            return GenieLinuxArtifactHandler()
        if qdc_device.auto_platform:
            if qairt_sdk_path is None:
                raise ValueError(
                    "qairt_sdk_path is required for auto devices. "
                    "Please provide the path to the automotive QAIRT SDK zip file."
                )
            return GenieAutoArtifactHandler(
                test_script="run_auto_android.py", qairt_sdk_path=qairt_sdk_path
            )
        if qdc_device.mobile_platform:
            return GenieAndroidArtifactHandler(test_script="run_android.py")
        raise ValueError("Unsupported platform type for Genie artifact handler.")

    def add_job_artifacts(
        self,
        qdc_device: QDCDevice,
        genie_bundle_path: str,
        qairt_sdk_path: str | None = None,
        qairt_version: str = "2.45.40.260406",
        eval_prompts: list[str] | None = None,
        num_trials: int = 25,
        model_id: str | None = None,
    ) -> tuple[list[str], str | None]:
        """Prepare and upload Genie artifacts for the job submission.

        Parameters
        ----------
        qdc_device
            QDCDevice instance for the target device.
        genie_bundle_path
            Directory path containing the genie bundle.
        qairt_sdk_path
            Path to the QAIRT SDK zip file. Required for auto devices.
        qairt_version
            QAIRT SDK version to download on-device (e.g. ``"2.45.40.260406"``).
        eval_prompts
            If provided, list of prompts to evaluate. Each prompt is formatted
            using the bundle's tokenizer and run sequentially on device.
        num_trials
            Number of profiling trials to run.
        model_id
            Model identifier used to load the HF tokenizer if the bundle
            tokenizer lacks a chat template.

        Returns
        -------
        job_artifacts: list[str]
            List of artifact IDs returned by QDC upload.
        entry_script: str | None
            Optional entry script path used by the test framework.
        """
        curr_dirname = os.path.dirname(os.path.abspath(__file__))
        artifact_handler = self._get_artifact_handler(qdc_device, qairt_sdk_path)

        # Write chat-templated eval prompts into a small tmpdir (~100 KB
        # of .txt files) rather than cloning the multi-GB genie bundle just
        # to add a prompts/ subdir. The handler streams these into the zip
        # alongside references to the original bundle files.
        eval_prompts_dir: str | None = None
        if eval_prompts:
            eval_prompts_dir = tempfile.mkdtemp(prefix="genie_eval_prompts_")
            _write_eval_prompts_to_dir(
                eval_prompts_dir, eval_prompts, genie_bundle_path, model_id
            )

        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                zip_path = artifact_handler.create_artifact(
                    curr_dirname,
                    genie_bundle_path,
                    tmpdirname,
                    qdc_device.hexagon_version,
                    qairt_version,
                    num_trials,
                    eval_prompts_dir=eval_prompts_dir,
                )
                upload_response = self.upload_file(zip_path, ArtifactType.TESTSCRIPT)
        finally:
            if eval_prompts_dir:
                shutil.rmtree(eval_prompts_dir, ignore_errors=True)

        return [upload_response], artifact_handler.entry_script

    def compute_metrics(
        self,
        job_log_files: list,
    ) -> tuple[float | None, float | None, float | None]:
        """Compute and print performance metrics from job logs.

        Parameters
        ----------
        job_log_files
            List of job log files retrieved from QDC.

        Returns
        -------
        avg_tokens_per_second : float | None
            Average tokens per second.
        min_time_to_first_token: float | None
            Minimum time to first token in ms.
        prefill_tokens_per_second : float | None
            Prefill (prompt-processing) tokens per second.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            tps: list[float] = []
            ttft: list[float] = []
            prefill_tps: list[float] = []

            if job_log_files:
                for job_log in job_log_files:
                    target_path = os.path.join(
                        tmpdirname, "logs", f"{job_log.filename}.zip"
                    )
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    self.download_job_log_files(job_log.filename, target_path)

                    if "genie" in job_log.filename:
                        print("On device output (genie.log):")
                        shutil.unpack_archive(target_path, tmpdirname, "zip")
                        genie_log_path = os.path.join(tmpdirname, "genie.log")
                        displayed = False
                        for encoding in ("utf-8", "utf-16", "utf-16-le"):
                            try:
                                with open(genie_log_path, encoding=encoding) as file:
                                    genie_content = file.read()
                                    print(genie_content)
                                    displayed = True
                                    break
                            except Exception:
                                pass
                        if not displayed:
                            print(f"Warning: Could not read {genie_log_path}")

                    if fnmatch.fnmatch(
                        os.path.basename(job_log.filename), "profile*.json"
                    ):
                        shutil.unpack_archive(target_path, tmpdirname, "zip")
                        profile_path = os.path.join(
                            tmpdirname, job_log.filename.split("/")[-1]
                        )
                        with open(profile_path, encoding="utf-8") as file:
                            file_content = json.loads(file.read())

                        components = file_content.get("components", [])
                        if (
                            isinstance(components, list)
                            and len(components) > 0
                            and isinstance(components[0], dict)
                            and "events" in components[0]
                            and isinstance(components[0]["events"], list)
                            and len(components[0]["events"]) > 1
                        ):
                            component = components[0]["events"][1]
                            tps.append(
                                float(component["token-generation-rate"]["value"])
                            )
                            ttft.append(
                                float(component["time-to-first-token"]["value"])
                            )
                            prefill_tps.append(
                                float(component["prompt-processing-rate"]["value"])
                            )
                        else:
                            print(
                                "Warning: Unexpected profile log structure, "
                                "skipping metrics for this file."
                            )

        if len(tps) > 0:
            # TTFT in profile logs is in microseconds, convert to milliseconds
            ttft_ms = [t / 1000.0 for t in ttft]

            print("Perf metrics:")
            print(f"  Tokens Per Second (all trials): {tps}")
            print(f"  Time to First Token ms (all trials): {ttft_ms}")
            print(f"  Prefill Tokens Per Second (all trials): {prefill_tps}")
            print(
                f"  Tokens Per Second — average: {statistics.mean(tps):.2f}, median: {statistics.median(tps):.2f}"
            )
            print(
                f"  Time to First Token (ms) — average: {statistics.mean(ttft_ms):.2f}, median: {statistics.median(ttft_ms):.2f}"
            )
            print(
                f"  Prefill Tokens Per Second — average: {statistics.mean(prefill_tps):.2f}, median: {statistics.median(prefill_tps):.2f}"
            )
            return (
                statistics.median(tps),
                statistics.median(prefill_tps),
                statistics.median(ttft_ms),
            )

        print("No performance metrics found.")
        if job_log_files:
            print("Available log files:")
            for job_log in job_log_files:
                print(f"  {job_log.filename}")
        return None, None, None

    @staticmethod
    def _parse_eval_outputs(content: str) -> dict[int, str]:
        """Parse a single eval_outputs.txt file with delimiter markers.

        Format: ===EVAL_IDX_NNN=== followed by the model output for that prompt.
        """
        outputs: dict[int, str] = {}
        parts = re.split(r"===EVAL_IDX_(\d+)===\n?", content)
        for i in range(1, len(parts) - 1, 2):
            idx = int(parts[i])
            outputs[idx] = parts[i + 1].strip()
        return outputs

    @staticmethod
    def _extract_model_output(raw_output: str) -> str:
        """Extract just the model's response from raw genie-t2t-run output.

        The raw output mixes debug logs, the chat-templated prompt echo, and
        the actual response between ``[BEGIN]:`` and ``[END]`` markers. We
        return only the text between those markers; if neither is present,
        fall back to the raw output stripped.
        """
        begin_marker = "[BEGIN]:"
        end_marker = "[END]"
        begin_idx = raw_output.find(begin_marker)
        if begin_idx == -1:
            return raw_output.strip()
        text = raw_output[begin_idx + len(begin_marker) :]
        end_idx = text.find(end_marker)
        if end_idx != -1:
            text = text[:end_idx]
        return text.strip()

    def compute_eval_results(
        self,
        job_log_files: list,
        prompts: list[str],
    ) -> list[dict]:
        """Parse eval outputs from job logs.

        The device scripts write a single eval_outputs.txt file with
        delimiter markers (===EVAL_IDX_NNN===) separating each prompt's
        output.

        Parameters
        ----------
        job_log_files
            List of job log files retrieved from QDC.
        prompts
            Original list of prompts (used to attach prompt text to results).

        Returns
        -------
        results: list[dict]
            List of dicts with keys: idx, prompt, output.
        """
        outputs: dict[int, str] = {}

        with tempfile.TemporaryDirectory() as tmpdirname:
            for job_log in job_log_files:
                if "eval_outputs" not in job_log.filename:
                    continue

                target_path = os.path.join(
                    tmpdirname, "logs", f"{job_log.filename}.zip"
                )
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                self.download_job_log_files(job_log.filename, target_path)

                safe_root = pathlib.Path(tmpdirname).resolve()
                with zipfile.ZipFile(target_path) as zf:
                    for member in zf.namelist():
                        dest = (safe_root / member).resolve()
                        if not str(dest).startswith(str(safe_root) + os.sep):
                            raise ValueError(
                                f"Zip slip detected in log archive: {member}"
                            )
                    zf.extractall(safe_root)

                extracted_name = job_log.filename.split("/")[-1]
                extracted_path = os.path.join(tmpdirname, extracted_name)
                if not os.path.exists(extracted_path):
                    continue

                content = None
                for encoding in ("utf-8", "utf-16", "utf-16-le"):
                    try:
                        with open(extracted_path, encoding=encoding) as f:
                            content = f.read()
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        pass

                if content is None:
                    print(f"Warning: Could not decode {extracted_name}")
                    continue

                outputs = self._parse_eval_outputs(content)

        results: list[dict] = [
            {
                "idx": idx,
                "prompt": prompts[idx] if idx < len(prompts) else "",
                "output": self._extract_model_output(outputs.get(idx, "")),
            }
            for idx in sorted(outputs.keys())
        ]

        if not results:
            print("Warning: No eval results found in job logs.")
            print("Available log files:")
            for job_log in job_log_files:
                print(f"  {job_log.filename}")

        return results


def save_eval_results_json(results: list[dict], output_path: str) -> None:
    """Save evaluation results to a JSON file."""
    if not results:
        print("No results to save.")
        return

    results.sort(key=lambda r: r.get("idx", 0))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


def save_eval_metadata_json(
    model_id: str,
    chipset: str,
    precision: str,
    output_path: str,
    path: ScorecardProfilePath,
    dataset_name: str = "prompts",
) -> None:
    """Save a sidecar identifying which (model, chipset, precision, path, dataset) an eval JSON belongs to.

    collect_llm_accuracy_csv reads this to write the accuracy row under the
    correct runtime; ``path`` is required so a caller can't silently mislabel
    a non-genie result as GENIE.
    """
    metadata = {
        "model_id": model_id,
        "chipset": chipset,
        "precision": precision,
        "path": path.value,
        "dataset_name": dataset_name,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Eval metadata saved to: {output_path}")


_USE_DEFAULT_PROMPTS = object()


def _resolve_eval_prompts(
    eval_prompts: list[str] | None | object,
) -> list[str] | None:
    if eval_prompts is _USE_DEFAULT_PROMPTS:
        with open(DEFAULT_EVAL_PROMPTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    if isinstance(eval_prompts, list):
        return eval_prompts
    return None


def submit_genie_bundle_only(
    api_token: str,
    device: str,
    genie_bundle_path: str,
    job_name: str = "LLM Genie",
    qairt_sdk_path: str | None = None,
    qairt_version: str = "2.45.40.260406",
    eval_prompts: list[str] | None | object = None,
    num_trials: int = 25,
    model_id: str | None = None,
) -> str:
    """Upload artifacts and submit a Genie job, returning the QDC job id.

    Companion to ``collect_genie_bundle_result``. Does no waiting or
    result parsing — the caller records the job id (typically to a
    jobs_file) and polls later.
    """
    prompts_to_use = _resolve_eval_prompts(eval_prompts)

    qdc_device = QDCDevice(device)
    genie_job = GenieQDCJobs(
        api_key=api_token,
        app_name_header="GenieQDCJobApp",
    )

    job_artifacts, entry_script = genie_job.add_job_artifacts(
        qdc_device,
        genie_bundle_path,
        qairt_sdk_path,
        qairt_version,
        eval_prompts=prompts_to_use,
        num_trials=num_trials,
        model_id=model_id,
    )

    job_id = genie_job.submit_automated_job(
        qdc_device,
        job_artifacts,
        entry_script,
        job_name=job_name,
        timeout=GENIE_JOB_TIMEOUT,
    )
    if job_id is None:
        raise RuntimeError("Job submission failed.")
    print(f"Submitted QDC job with ID: {job_id}")
    return job_id


def collect_genie_bundle_result(
    api_token: str,
    device: str,
    job_id: str,
    eval_prompts: list[str] | None | object = None,
) -> tuple[
    float | None,
    float | None,
    float | None,
    list[dict],
    JobOutcome,
    str | None,
]:
    """Poll a submitted Genie job and, on success, download + parse logs.

    Returns ``(tps, prefill_tps, ttft, eval_results, outcome, reason)``.
    On non-SUCCESS outcomes, the metric fields are None and ``reason`` is
    a human-readable failure description. eval_prompts is only consulted
    on success to attach prompt text to the parsed outputs.
    """
    prompts_to_use = _resolve_eval_prompts(eval_prompts)
    genie_job = GenieQDCJobs(
        api_key=api_token,
        app_name_header="GenieQDCJobApp",
    )

    job_status = genie_job.status(job_id)
    job_result = genie_job.result(job_id)
    print(f"QDC job {job_id} completed with status: {job_status}, result: {job_result}")

    if job_result is not None and job_result != "Successful":
        reason = (
            f"QDC job {job_id} on device '{device}' finished with "
            f"status='{job_status}', result='{job_result}'"
        )
        outcome = (
            JobOutcome.RETRYABLE_ERROR
            if job_result == "Error"
            else JobOutcome.RETRYABLE_UNSUCCESSFUL
        )
        print(f"[result={job_result}] {reason}")
        return None, None, None, [], outcome, reason

    genie_job.log_upload_status(job_id)
    # The file listing lags log-upload-status on the QDC backend, so wait
    # for it to populate -- otherwise a successful job yields no metrics.
    job_log_files = genie_job.get_job_log_files(job_id, wait_for_logs=True)

    if not job_log_files:
        reason = (
            f"QDC job {job_id} on device '{device}' reported result="
            f"'{job_result}' but produced no retrievable log files"
        )
        print(f"[empty logs] {reason}")
        return None, None, None, [], JobOutcome.RETRYABLE_EMPTY_LOGS, reason

    tps, prefill_tps, ttft = genie_job.compute_metrics(job_log_files)

    eval_results: list[dict] = []
    if prompts_to_use:
        eval_results = genie_job.compute_eval_results(job_log_files, prompts_to_use)

    return tps, prefill_tps, ttft, eval_results, JobOutcome.SUCCESS, None


def submit_genie_bundle_to_qdc_device(
    api_token: str,
    device: str,
    genie_bundle_path: str,
    job_name: str = "LLM Genie",
    qairt_sdk_path: str | None = None,
    qairt_version: str = "2.45.40.260406",
    eval_prompts: list[str] | None | object = None,
    num_trials: int = 25,
    model_id: str | None = None,
) -> tuple[float | None, float | None, float | None, list[dict]]:
    """
    Submit a Genie bundle to QDC for execution on the specified device.

    Runs profiling and (optionally) evaluation in a single job. Eval is
    skipped by default; pass ``_USE_DEFAULT_PROMPTS`` for the built-in 100
    questions, or a list of prompts to use a custom set.

    Composed wrapper over ``submit_genie_bundle_only`` +
    ``collect_genie_bundle_result``. Retries retryable outcomes up to
    ``DEFAULT_ATTEMPTS`` times; the CI submit/collect split owns retries
    independently via the jobs_file.
    """
    from qai_hub_models.models._shared.llm.common import (
        DEFAULT_ATTEMPTS,
        poll_and_retry,
    )

    def _submit() -> str:
        return submit_genie_bundle_only(
            api_token,
            device,
            genie_bundle_path,
            job_name=job_name,
            qairt_sdk_path=qairt_sdk_path,
            qairt_version=qairt_version,
            eval_prompts=eval_prompts,
            num_trials=num_trials,
            model_id=model_id,
        )

    def _collect(job_id: str) -> tuple[tuple, JobOutcome, str | None]:
        tps, prefill_tps, ttft, eval_results, outcome, reason = (
            collect_genie_bundle_result(api_token, device, job_id, eval_prompts)
        )
        return (tps, prefill_tps, ttft, eval_results), outcome, reason

    return poll_and_retry(
        initial_job_id=_submit(),
        attempts_left=DEFAULT_ATTEMPTS - 1,
        collect_fn=_collect,
        resubmit_fn=_submit,
    )


def _add_bundle_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-token", type=str, required=True)
    p.add_argument(
        "--device",
        type=str,
        required=True,
        choices=HUB_DEVICE_TO_QDC_DEVICE_MAP.keys(),
    )
    p.add_argument("--genie-bundle-path", type=str, required=True)
    p.add_argument("--job-name", type=str, default="LLM Genie")
    p.add_argument("--qairt-sdk-path", type=str, default=None)
    p.add_argument("--qairt-version", type=str, default="2.45.40.260406")
    p.add_argument(
        "--eval-prompts",
        type=str,
        default=None,
        help="Path to JSON file with prompts (defaults to built-in eval_prompts.json).",
    )
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument("--num-trials", type=int, default=25)


def _cmd_run_bundle(args: argparse.Namespace) -> int:
    """One-shot bundle submit-and-wait against a specific device."""
    eval_prompts_val: list[str] | None = None
    if args.eval_prompts:
        with open(args.eval_prompts, encoding="utf-8") as f:
            eval_prompts_val = json.load(f)
        assert eval_prompts_val is not None
        print(f"Loaded {len(eval_prompts_val)} eval prompts from {args.eval_prompts}")

    if not os.path.exists(os.path.join(args.genie_bundle_path, "sample_prompt.txt")):
        raise FileNotFoundError(
            f"sample_prompt.txt not found in {args.genie_bundle_path}. "
            "Please add a file with prompt to run on-device."
        )

    _, _, _, eval_results = submit_genie_bundle_to_qdc_device(
        args.api_token,
        args.device,
        args.genie_bundle_path,
        args.job_name,
        args.qairt_sdk_path,
        args.qairt_version,
        eval_prompts=eval_prompts_val,
        num_trials=args.num_trials,
    )
    if args.output_json:
        save_eval_results_json(eval_results, args.output_json)
    return 0


def _write_junit(junit_path: str, cases: list[tuple[str, str | None]]) -> None:
    """Minimal junit XML compatible with generate_test_summary."""
    import xml.etree.ElementTree as ET

    root = ET.Element(
        "testsuite",
        {
            "name": "llm_perf",
            "tests": str(len(cases)),
            "failures": str(sum(1 for _, m in cases if m)),
        },
    )
    for name, msg in cases:
        tc = ET.SubElement(root, "testcase", {"classname": "llm_perf", "name": name})
        if msg:
            fail = ET.SubElement(tc, "failure", {"message": msg[:200]})
            fail.text = msg
    pathlib.Path(junit_path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(junit_path, encoding="utf-8", xml_declaration=True)


def _cmd_submit(args: argparse.Namespace) -> int:
    import sys

    from qai_hub_models.models._shared.llm import test as llm_test
    from qai_hub_models.models._shared.llm.perf_collection import LLMPerfConfig
    from qai_hub_models.scorecard.test.test_llm_perf import _build_params

    if os.path.exists(args.jobs_file):
        os.unlink(args.jobs_file)
    cfg = LLMPerfConfig.from_environment()
    submitted = 0
    for model_id, precision, device in _build_params():
        try:
            llm_test.submit_llm_perf_job(
                model_id=model_id,
                device=device,
                precision=precision,
                output_dir=os.path.join(model_id, llm_test.GENIE_BUNDLES_ROOT),
                jobs_file=args.jobs_file,
                qairt_sdk_path=cfg.qairt_sdk_path,
                skip_perf_update=cfg.skip_perf_update,
            )
            submitted += 1
        except Exception as e:  # noqa: PERF203
            print(
                f"ERROR: submission failed for {model_id}/{precision}/"
                f"{device.name}: {e}",
                file=sys.stderr,
            )
    print(f"Submitted {submitted} genie job(s) to {args.jobs_file}")
    return 0 if submitted else 1


def _cmd_collect(args: argparse.Namespace) -> int:
    import sys

    from qai_hub_models.models._shared.llm import test as llm_test
    from qai_hub_models.models._shared.llm.common import load_jobs, make_key
    from qai_hub_models.models._shared.llm.llm_helpers import log_perf_on_device_result
    from qai_hub_models.models._shared.llm.perf_collection import LLMPerfConfig
    from qai_hub_models.scorecard.test.test_llm_perf import _build_params

    if not os.path.exists(args.jobs_file):
        print(f"jobs file not found: {args.jobs_file}", file=sys.stderr)
        return 1

    cfg = LLMPerfConfig.from_environment()
    records = load_jobs(args.jobs_file)
    cases: list[tuple[str, str | None]] = []
    for model_id, precision, device in _build_params():
        key = make_key(model_id, str(precision), "GENIE", device.name)
        record = records.get(key)
        case_name = f"{model_id}-{precision}-{device.name}"
        if record is None:
            print(f"jobs_file has no entry for {key}; skipping", file=sys.stderr)
            continue
        try:
            tps, ttft, prefill_tps = llm_test.collect_llm_perf_job(
                model_id=model_id,
                device=device,
                precision=precision,
                record=record,
                jobs_file=args.jobs_file,
                output_dir=os.path.join(model_id, llm_test.GENIE_BUNDLES_ROOT),
                qairt_sdk_path=cfg.qairt_sdk_path,
                skip_perf_update=cfg.skip_perf_update,
            )
        except Exception as e:
            print(
                f"ERROR: collection failed for {case_name} (job {record.job_id}): {e}",
                file=sys.stderr,
            )
            cases.append((case_name, str(e)))
            continue
        log_perf_on_device_result(
            model_name=model_id,
            precision=str(precision),
            device=device.name,
            tps=tps,
            prefill_tps=prefill_tps,
            ttft_ms=ttft,
        )
        cases.append((case_name, None))

    if args.junit_xml:
        _write_junit(args.junit_xml, cases)

    failed = [name for name, msg in cases if msg]
    if failed:
        print(f"FAILED cases: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_bundle = sub.add_parser(
        "run-bundle", help="One-shot bundle submit-and-wait for a specific device."
    )
    _add_bundle_args(p_bundle)

    p_submit = sub.add_parser(
        "submit", help="Submit one QDC job per (model, precision, device)."
    )
    p_submit.add_argument("--jobs-file", required=True)

    p_collect = sub.add_parser(
        "collect", help="Poll jobs listed in the jobs file and update perf.yaml."
    )
    p_collect.add_argument("--jobs-file", required=True)
    p_collect.add_argument("--junit-xml", default=None)

    ns = ap.parse_args()
    if ns.cmd == "run-bundle":
        sys.exit(_cmd_run_bundle(ns))
    if ns.cmd == "submit":
        sys.exit(_cmd_submit(ns))
    sys.exit(_cmd_collect(ns))
