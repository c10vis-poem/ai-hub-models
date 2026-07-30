# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from qai_hub_models.configs.manifest_yaml import PipCommand

ACCEPTED = [
    "pip install foo",
    "pip install foo bar",
    'pip install "torch>=2.1,<2.12.0" "setuptools>=80"',
    "pip install git+https://github.com/foo/bar.git@abc --no-build-isolation --use-pep517",
    "pip install k2==1.24.4.dev20251029+cpu.torch2.9.0 -f https://k2-fsa.github.io/k2/cpu.html",
    "pip uninstall -y onnxruntime",
]

REJECTED = [
    "conda install foo",
    "python -m pip install foo",
    "pip install foo; rm -rf /",
    "pip install foo && rm -rf /",
    "pip install foo || echo nope",
    "pip install foo | tee log.txt",
    "pip install foo > /tmp/out",
    "pip install foo < /tmp/in",
    "pip install `id`",
    "pip install $(id)",
    "pip install ${HOME}/wheel.whl",
    'pip install "$(id)"',
    'pip install "`id`"',
    'pip install "${HOME}/wheel.whl"',
    "pip install $HOME/wheel.whl",
    "pip install foo &",
    "pip install foo\nrm -rf /",
    "pip",
    "",
    "   ",
    'pip install "unclosed',
    "pip install \\evil",
]


@pytest.mark.parametrize("cmd", ACCEPTED)
def test_pip_command_accepts(cmd: str) -> None:
    PipCommand(command=cmd)


@pytest.mark.parametrize("cmd", REJECTED)
def test_pip_command_rejects(cmd: str) -> None:
    with pytest.raises(ValidationError):  # pydantic wraps ValueError as ValidationError
        PipCommand(command=cmd)


def test_machine_default() -> None:
    assert PipCommand(command="pip install foo").machine == "any"


def test_machine_values() -> None:
    for m in ("any", "gpu", "cpu"):
        assert PipCommand(command="pip install foo", machine=m).machine == m


def test_machine_rejects_bad_value() -> None:
    with pytest.raises(ValidationError):
        PipCommand(command="pip install foo", machine="tpu")  # type: ignore[arg-type]


def test_bare_string_lifts_to_pip_command() -> None:
    pc = PipCommand.model_validate("pip install foo")
    assert pc.command == "pip install foo"
    assert pc.machine == "any"


def test_bare_string_still_validates_shape() -> None:
    with pytest.raises(ValidationError):
        PipCommand.model_validate("rm -rf /")


def test_list_accepts_mixed_bare_and_mapping_forms() -> None:
    adapter = TypeAdapter(list[PipCommand])
    entries = adapter.validate_python(
        [
            "pip install foo",
            {"command": "pip install bar", "machine": "gpu"},
        ]
    )
    assert [(e.command, e.machine) for e in entries] == [
        ("pip install foo", "any"),
        ("pip install bar", "gpu"),
    ]


def test_serializes_bare_string_when_machine_default() -> None:
    assert PipCommand(command="pip install foo").model_dump() == "pip install foo"


def test_serializes_mapping_when_machine_set() -> None:
    assert PipCommand(command="pip install foo", machine="gpu").model_dump() == {
        "command": "pip install foo",
        "machine": "gpu",
    }
