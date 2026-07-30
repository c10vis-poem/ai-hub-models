# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Heavy-side ``qai-hub-models install <model_id>`` implementation.

Walks a model's dependency graph (datasets -> templates -> other models)
via DFS, installing each node's ``requirements.txt`` and any manifest
``pre_pip_install_commands`` / ``post_pip_install_commands`` exactly
once. Traversal is post-order: a node's dependencies install before the
node itself, so pip resolves in the order the manifest declares.

Dependencies are declared in each folder's ``manifest.yaml``:

  - Model manifests (``models/<id>/manifest.yaml``) may declare
    ``datasets``, ``templates``, ``models`` (cross-model), plus the pre/post
    pip command lists.
  - Shared-template manifests (``models/_shared/<name>/manifest.yaml``)
    may declare ``datasets`` and ``templates`` transitively.
  - Dataset manifests (``datasets/<name>/manifest.yaml``) may declare
    ``datasets``.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from typing import Any

from qai_hub_models.configs.manifest_yaml import PipCommand
from qai_hub_models.utils.asset_loaders import load_yaml
from qai_hub_models.utils.path_helpers import MODEL_IDS, QAIHM_PACKAGE_ROOT

DATASETS_ROOT = QAIHM_PACKAGE_ROOT / "datasets"
SHARED_ROOT = QAIHM_PACKAGE_ROOT / "models" / "_shared"
MODELS_ROOT = QAIHM_PACKAGE_ROOT / "models"

_NVIDIA_SMI_TIMEOUT_SECONDS = 5


class NodeKind(Enum):
    DATASET = "dataset"
    TEMPLATE = "template"
    MODEL = "model"


@dataclass(frozen=True)
class Node:
    """A single installable unit in the dependency graph."""

    kind: NodeKind
    name: str

    @property
    def folder(self) -> Path:
        if self.kind is NodeKind.DATASET:
            return DATASETS_ROOT / self.name
        if self.kind is NodeKind.TEMPLATE:
            return SHARED_ROOT / self.name
        return MODELS_ROOT / self.name

    @property
    def manifest_path(self) -> Path:
        return self.folder / "manifest.yaml"

    @property
    def requirements_path(self) -> Path:
        return self.folder / "requirements.txt"


def _has_cuda_gpu() -> bool:
    """Return True if ``nvidia-smi`` runs successfully.

    Proxy for CUDA GPU availability, matching the check used by the
    internal ``scripts/tasks/venv.py`` install path. A short timeout guards
    against a wedged GPU driver hanging the CLI indefinitely.
    """
    try:
        return (
            subprocess.run(
                ["nvidia-smi"],
                check=False,
                capture_output=True,
                timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


def _validate_dep_name(name: str, kind: NodeKind, referrer: Node) -> None:
    r"""Reject dep names that could escape their intended root.

    ``Node.folder`` builds ``<ROOT> / self.name`` directly from strings read
    out of a manifest, so a name containing ``/``, ``\``, or ``..`` would
    resolve outside its expected root. Since ``manifest.yaml`` and
    ``requirements.txt`` files at that escaped location would then be loaded
    and run, every dep name has to be a plain folder name.
    """
    if not name or "/" in name or "\\" in name or name in {"..", "."}:
        raise ValueError(
            f"{referrer.kind.value} {referrer.name!r} declares an invalid "
            f"{kind.value} dependency name {name!r}. Dep names must be plain "
            "folder names (no path separators, no '..')."
        )


@cache
def _load_manifest(node: Node) -> dict[str, Any]:
    """Return the parsed ``manifest.yaml`` for ``node``, or ``{}`` if missing.

    Cached so ``_load_deps`` and both ``_load_pip_commands`` calls per model
    node share a single parse instead of re-opening the file three times.
    """
    manifest_path = node.manifest_path
    if not manifest_path.exists():
        return {}
    return load_yaml(manifest_path) or {}


def _load_deps(node: Node) -> tuple[list[str], list[str], list[str]]:
    """Return ``(datasets, templates, models)`` from ``node``'s manifest.

    Missing manifest yields empty lists (used by nodes that have a
    ``requirements.txt`` but no ``manifest.yaml``, e.g. datasets without
    transitive deps). Missing keys yield empty lists. Only model manifests
    may declare cross-model deps; other node kinds always return
    ``models=[]`` regardless of what their manifest says. Every declared
    name is validated so it cannot escape its intended root.
    """
    data = _load_manifest(node)
    datasets = list(data.get("datasets") or [])
    templates = list(data.get("templates") or [])
    models = list(data.get("models") or []) if node.kind is NodeKind.MODEL else []
    for name in datasets:
        _validate_dep_name(name, NodeKind.DATASET, node)
    for name in templates:
        _validate_dep_name(name, NodeKind.TEMPLATE, node)
    for name in models:
        _validate_dep_name(name, NodeKind.MODEL, node)
    return datasets, templates, models


def _load_pip_commands(node: Node, field: str) -> list[PipCommand]:
    """Return ``PipCommand`` entries from ``field`` on ``node``'s manifest.

    Only model manifests may declare pip commands; other node kinds return
    an empty list (their manifests don't carry these fields).
    """
    if node.kind is not NodeKind.MODEL:
        return []
    raw = _load_manifest(node).get(field) or []
    return [PipCommand.model_validate(entry) for entry in raw]


def build_install_order(root: Node) -> list[Node]:
    """DFS the dependency graph and return nodes in install order.

    Post-order traversal: each node's dependencies appear before the node
    itself. Nodes already seen are skipped so a shared dep is emitted
    exactly once. Cycles are broken by the visited set (a node in progress
    is treated as already emitted from the perspective of a back-edge).
    """
    order: list[Node] = []
    visited: set[Node] = set()
    in_progress: set[Node] = set()

    def _visit(node: Node) -> None:
        if node in visited or node in in_progress:
            return
        if node is not root and not node.folder.exists():
            raise ValueError(
                f"{node.kind.value} {node.name!r} is declared as a dependency "
                f"but its folder does not exist at {node.folder}. Check for a "
                "typo in the referencing manifest.yaml."
            )
        in_progress.add(node)
        datasets, templates, models = _load_deps(node)
        for name in datasets:
            _visit(Node(NodeKind.DATASET, name))
        for name in templates:
            _visit(Node(NodeKind.TEMPLATE, name))
        for name in models:
            _visit(Node(NodeKind.MODEL, name))
        in_progress.discard(node)
        visited.add(node)
        order.append(node)

    _visit(root)
    return order


def _filter_pip_commands(
    commands: Iterable[PipCommand], on_gpu: bool
) -> list[PipCommand]:
    """Drop commands whose ``machine`` doesn't apply to the current host."""
    kept: list[PipCommand] = []
    for cmd in commands:
        if cmd.machine == "gpu" and not on_gpu:
            continue
        if cmd.machine == "cpu" and on_gpu:
            continue
        kept.append(cmd)
    return kept


def _node_install_commands(node: Node, on_gpu: bool) -> list[list[str]]:
    """Return the argv list for each command needed to install ``node``.

    Order per node:
      1. every ``pre_pip_install_commands`` entry that applies to this host
      2. ``pip install -r requirements.txt`` (only if the file exists)
      3. every ``post_pip_install_commands`` entry that applies to this host
    """
    pre = _filter_pip_commands(
        _load_pip_commands(node, "pre_pip_install_commands"), on_gpu
    )
    post = _filter_pip_commands(
        _load_pip_commands(node, "post_pip_install_commands"), on_gpu
    )

    commands: list[list[str]] = [shlex.split(c.command) for c in pre]
    if node.requirements_path.exists():
        commands.append(["pip", "install", "-r", str(node.requirements_path)])
    commands.extend(shlex.split(c.command) for c in post)
    return commands


def plan_install(model_id: str) -> list[tuple[Node, list[list[str]]]]:
    """Return the full install plan for ``model_id``.

    Each entry pairs a graph node with the argv list of every command
    needed to install it, in the order they should run.
    """
    if model_id not in MODEL_IDS:
        raise ValueError(f"Unknown model id: {model_id!r}")
    root = Node(NodeKind.MODEL, model_id)
    on_gpu = _has_cuda_gpu()
    return [
        (node, _node_install_commands(node, on_gpu))
        for node in build_install_order(root)
    ]


def _run(argv: list[str], dry_run: bool) -> None:
    """Print ``argv`` and (unless ``dry_run``) invoke it via subprocess."""
    print("+", shlex.join(argv), flush=True)
    if dry_run:
        return
    subprocess.run(argv, check=True)


def install_model(model_id: str, dry_run: bool = False) -> None:
    """Install ``model_id`` and every declared dependency in DFS order.

    Parameters
    ----------
    model_id
        The model id to install (must exist under ``qai_hub_models/models/``).
    dry_run
        If True, print each command that would run instead of executing it.
        Useful for previewing the plan.
    """
    plan = plan_install(model_id)
    for node, commands in plan:
        if not commands:
            print(f"# {node.kind.value} {node.name}: nothing to install", flush=True)
            continue
        print(f"# {node.kind.value} {node.name}", flush=True)
        for argv in commands:
            _run(argv, dry_run)


def main(argv: list[str] | None = None) -> None:
    """Entry point called from the lean-CLI dispatcher.

    Kept intentionally tiny — the lean CLI resolves the model id and
    forwards the remaining args here. Options mirror ``pip install`` where
    they overlap.
    """
    parser = argparse.ArgumentParser(
        prog="qai-hub-models install",
        description="Install the dependencies for a model, "
        "including its declared datasets, shared templates, and cross-model deps.",
    )
    parser.add_argument("model_id", help="Model id, e.g. mobilenet_v2.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the install plan without running any pip commands.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    install_model(args.model_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
