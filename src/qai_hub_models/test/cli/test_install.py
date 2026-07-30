# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Tests for ``qai-hub-models install`` (DFS installer + heavy-side wiring)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from qai_hub_models.cli import install as install_mod
from qai_hub_models.cli.install import (
    Node,
    NodeKind,
    build_install_order,
    plan_install,
)


def _write_manifest(folder: Path, body: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.yaml").write_text(body)


def _write_requirements(folder: Path, body: str = "somepkg==1.0\n") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "requirements.txt").write_text(body)


@pytest.fixture
def fake_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect DATASETS_ROOT / SHARED_ROOT / MODELS_ROOT to a scratch tree.

    Also stubs MODEL_IDS so ``plan_install`` accepts our fake models. Returns
    the temp root; individual tests populate the subfolders they need.
    """
    datasets = tmp_path / "datasets"
    shared = tmp_path / "models" / "_shared"
    models = tmp_path / "models"
    datasets.mkdir(parents=True)
    shared.mkdir(parents=True)
    monkeypatch.setattr(install_mod, "DATASETS_ROOT", datasets)
    monkeypatch.setattr(install_mod, "SHARED_ROOT", shared)
    monkeypatch.setattr(install_mod, "MODELS_ROOT", models)
    monkeypatch.setattr(install_mod, "MODEL_IDS", {"root_model", "leaf_model"})
    monkeypatch.setattr(install_mod, "_has_cuda_gpu", lambda: False)
    install_mod._load_manifest.cache_clear()
    return tmp_path


class TestBuildInstallOrder:
    def test_leaf_model_alone(self, fake_tree: Path) -> None:
        """A model with no deps installs itself alone."""
        _write_manifest(fake_tree / "models" / "root_model", "name: root_model\n")
        order = build_install_order(Node(NodeKind.MODEL, "root_model"))
        assert order == [Node(NodeKind.MODEL, "root_model")]

    def test_post_order_dataset_template_model(self, fake_tree: Path) -> None:
        """Datasets appear before the template that uses them, which appears
        before the model. Post-order: children first, node last.
        """
        _write_manifest(fake_tree / "datasets" / "ds1", "")
        _write_manifest(
            fake_tree / "models" / "_shared" / "shared1",
            "datasets:\n  - ds1\n",
        )
        _write_manifest(
            fake_tree / "models" / "root_model",
            "templates:\n  - shared1\n",
        )
        order = build_install_order(Node(NodeKind.MODEL, "root_model"))
        assert order == [
            Node(NodeKind.DATASET, "ds1"),
            Node(NodeKind.TEMPLATE, "shared1"),
            Node(NodeKind.MODEL, "root_model"),
        ]

    def test_shared_dep_installed_once(self, fake_tree: Path) -> None:
        """A dataset reached via two paths appears exactly once in the plan."""
        _write_manifest(fake_tree / "datasets" / "shared_ds", "")
        _write_manifest(
            fake_tree / "models" / "_shared" / "t1",
            "datasets:\n  - shared_ds\n",
        )
        _write_manifest(
            fake_tree / "models" / "_shared" / "t2",
            "datasets:\n  - shared_ds\n",
        )
        _write_manifest(
            fake_tree / "models" / "root_model",
            "templates:\n  - t1\n  - t2\n",
        )
        order = build_install_order(Node(NodeKind.MODEL, "root_model"))
        # shared_ds appears exactly once, and before t1, t2, root_model.
        ds_count = sum(1 for n in order if n.name == "shared_ds")
        assert ds_count == 1
        idx = {n: i for i, n in enumerate(order)}
        assert (
            idx[Node(NodeKind.DATASET, "shared_ds")]
            < idx[Node(NodeKind.TEMPLATE, "t1")]
        )
        assert (
            idx[Node(NodeKind.DATASET, "shared_ds")]
            < idx[Node(NodeKind.TEMPLATE, "t2")]
        )
        assert (
            idx[Node(NodeKind.TEMPLATE, "t1")] < idx[Node(NodeKind.MODEL, "root_model")]
        )
        assert (
            idx[Node(NodeKind.TEMPLATE, "t2")] < idx[Node(NodeKind.MODEL, "root_model")]
        )

    def test_cross_model_dep(self, fake_tree: Path) -> None:
        """A model that depends on another model installs the dep model first."""
        _write_manifest(fake_tree / "models" / "leaf_model", "")
        _write_manifest(
            fake_tree / "models" / "root_model",
            "models:\n  - leaf_model\n",
        )
        order = build_install_order(Node(NodeKind.MODEL, "root_model"))
        assert order == [
            Node(NodeKind.MODEL, "leaf_model"),
            Node(NodeKind.MODEL, "root_model"),
        ]

    def test_dataset_of_dataset(self, fake_tree: Path) -> None:
        """Dataset manifests may transitively pull in more datasets."""
        _write_manifest(fake_tree / "datasets" / "base_ds", "")
        _write_manifest(
            fake_tree / "datasets" / "derived_ds",
            "datasets:\n  - base_ds\n",
        )
        _write_manifest(
            fake_tree / "models" / "root_model",
            "datasets:\n  - derived_ds\n",
        )
        order = build_install_order(Node(NodeKind.MODEL, "root_model"))
        assert order == [
            Node(NodeKind.DATASET, "base_ds"),
            Node(NodeKind.DATASET, "derived_ds"),
            Node(NodeKind.MODEL, "root_model"),
        ]

    def test_missing_dep_folder_raises(self, fake_tree: Path) -> None:
        """A declared dep whose folder doesn't exist raises with a clear message."""
        _write_manifest(
            fake_tree / "models" / "root_model",
            "datasets:\n  - typo_dataset\n",
        )
        with pytest.raises(ValueError, match="typo_dataset"):
            build_install_order(Node(NodeKind.MODEL, "root_model"))

    def test_path_traversal_dep_name_rejected(self, fake_tree: Path) -> None:
        """A dep name containing path separators or '..' is rejected."""
        _write_manifest(
            fake_tree / "models" / "root_model",
            "datasets:\n  - ../../etc/passwd\n",
        )
        with pytest.raises(ValueError, match="invalid"):
            build_install_order(Node(NodeKind.MODEL, "root_model"))

    def test_models_field_ignored_on_non_model_nodes(self, fake_tree: Path) -> None:
        """A stray ``models:`` on a template/dataset manifest is not traversed."""
        _write_manifest(
            fake_tree / "models" / "_shared" / "t",
            "models:\n  - leaf_model\n",
        )
        _write_manifest(
            fake_tree / "models" / "root_model",
            "templates:\n  - t\n",
        )
        order = build_install_order(Node(NodeKind.MODEL, "root_model"))
        assert Node(NodeKind.MODEL, "leaf_model") not in order

    def test_cycle_is_broken(self, fake_tree: Path) -> None:
        """A cycle between two templates doesn't loop forever."""
        _write_manifest(
            fake_tree / "models" / "_shared" / "a",
            "templates:\n  - b\n",
        )
        _write_manifest(
            fake_tree / "models" / "_shared" / "b",
            "templates:\n  - a\n",
        )
        _write_manifest(
            fake_tree / "models" / "root_model",
            "templates:\n  - a\n",
        )
        order = build_install_order(Node(NodeKind.MODEL, "root_model"))
        names = [n.name for n in order]
        assert names.count("a") == 1
        assert names.count("b") == 1
        assert names[-1] == "root_model"


class TestPlanInstall:
    def test_per_node_pre_requirements_post_order(self, fake_tree: Path) -> None:
        """A single model runs pre commands, then requirements.txt, then post."""
        model_dir = fake_tree / "models" / "root_model"
        _write_manifest(
            model_dir,
            "pre_pip_install_commands:\n"
            "  - command: pip install before\n"
            "post_pip_install_commands:\n"
            "  - command: pip install after\n",
        )
        _write_requirements(model_dir)
        plan = plan_install("root_model")
        assert len(plan) == 1
        _, commands = plan[0]
        # pre  -> requirements.txt  -> post
        assert commands[0] == ["pip", "install", "before"]
        assert commands[1][:3] == ["pip", "install", "-r"]
        assert commands[1][3].endswith("requirements.txt")
        assert commands[2] == ["pip", "install", "after"]

    def test_no_requirements_txt_omits_install(self, fake_tree: Path) -> None:
        """A node without requirements.txt only runs its pre/post commands."""
        _write_manifest(
            fake_tree / "models" / "root_model",
            "post_pip_install_commands:\n  - command: pip install onlyafter\n",
        )
        plan = plan_install("root_model")
        assert plan[0][1] == [["pip", "install", "onlyafter"]]

    def test_machine_gpu_filtered_on_cpu_host(self, fake_tree: Path) -> None:
        """Commands tagged ``machine: gpu`` are skipped on non-GPU hosts."""
        _write_manifest(
            fake_tree / "models" / "root_model",
            "post_pip_install_commands:\n"
            "  - command: pip install cpubuild\n"
            "    machine: cpu\n"
            "  - command: pip install gpubuild\n"
            "    machine: gpu\n"
            "  - command: pip install always\n",
        )
        plan = plan_install("root_model")
        commands = plan[0][1]
        assert ["pip", "install", "cpubuild"] in commands
        assert ["pip", "install", "always"] in commands
        assert ["pip", "install", "gpubuild"] not in commands

    def test_machine_cpu_filtered_on_gpu_host(
        self, fake_tree: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Commands tagged ``machine: cpu`` are skipped on GPU hosts."""
        monkeypatch.setattr(install_mod, "_has_cuda_gpu", lambda: True)
        _write_manifest(
            fake_tree / "models" / "root_model",
            "post_pip_install_commands:\n"
            "  - command: pip install cpubuild\n"
            "    machine: cpu\n"
            "  - command: pip install gpubuild\n"
            "    machine: gpu\n",
        )
        plan = plan_install("root_model")
        commands = plan[0][1]
        assert ["pip", "install", "gpubuild"] in commands
        assert ["pip", "install", "cpubuild"] not in commands

    def test_only_model_manifest_carries_pip_commands(self, fake_tree: Path) -> None:
        """pre/post pip commands on template/dataset manifests are ignored."""
        _write_manifest(
            fake_tree / "models" / "_shared" / "t",
            "pre_pip_install_commands:\n  - command: pip install ignored\n",
        )
        _write_manifest(
            fake_tree / "models" / "root_model",
            "templates:\n  - t\n",
        )
        plan = plan_install("root_model")
        template_entry = next(entry for entry in plan if entry[0].name == "t")
        assert template_entry[1] == []

    def test_unknown_model_id_rejected(self, fake_tree: Path) -> None:
        with pytest.raises(ValueError, match="Unknown model id"):
            plan_install("does_not_exist")


class TestInstallModel:
    def test_dry_run_does_not_invoke_subprocess(self, fake_tree: Path) -> None:
        """--dry-run prints the plan without shelling out."""
        model_dir = fake_tree / "models" / "root_model"
        _write_manifest(
            model_dir,
            "post_pip_install_commands:\n  - command: pip install foo\n",
        )
        with patch("qai_hub_models.cli.install.subprocess.run") as mock_run:
            install_mod.install_model("root_model", dry_run=True)
        mock_run.assert_not_called()

    def test_wet_run_shells_out_per_command(self, fake_tree: Path) -> None:
        """A real run invokes subprocess.run for each planned command."""
        model_dir = fake_tree / "models" / "root_model"
        _write_manifest(
            model_dir,
            "post_pip_install_commands:\n"
            "  - command: pip install first\n"
            "  - command: pip install second\n",
        )
        with patch("qai_hub_models.cli.install.subprocess.run") as mock_run:
            install_mod.install_model("root_model", dry_run=False)
        argvs = [call.args[0] for call in mock_run.call_args_list]
        assert argvs == [["pip", "install", "first"], ["pip", "install", "second"]]


class TestHasCudaGpu:
    def test_returns_false_when_nvidia_smi_hangs(self) -> None:
        """A wedged nvidia-smi shouldn't hang the CLI forever."""
        import subprocess

        def _hang(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd=["nvidia-smi"], timeout=5)

        with patch("qai_hub_models.cli.install.subprocess.run", side_effect=_hang):
            assert install_mod._has_cuda_gpu() is False


class TestDispatchWiring:
    def test_run_model_script_install_calls_install_main(self) -> None:
        """run_model_script('install', ...) forwards to install.main."""
        from qai_hub_models.cli.dispatch import run_model_script

        # dispatch.py imports install_main at module load, so patch the local
        # binding in dispatch (not the original in qai_hub_models.cli.install).
        with patch("qai_hub_models.cli.dispatch.install_main") as mock_install_main:
            run_model_script("mobilenet_v2", "install", ["--dry-run"])
        mock_install_main.assert_called_once_with(["mobilenet_v2", "--dry-run"])
