# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
# THIS FILE WAS AUTO-GENERATED. DO NOT EDIT MANUALLY.


from __future__ import annotations

import argparse
import warnings

from qai_hub_models import Precision, TargetRuntime
from qai_hub_models.models.qwen3_vl_4b_instruct import (
    DEFAULT_PRECISION,
    MODEL_ID,
    Model,
)
from qai_hub_models.utils.args import export_parser
from qai_hub_models.utils.checkpoint import CheckpointType
from qai_hub_models.utils.export.dispatch import resolve_model, select_pipeline

SUPPORTED_PRECISION_RUNTIMES: dict[Precision, list[TargetRuntime]] = {
    Precision.w4a16: [
        TargetRuntime.GENIEX_QAIRT,
        TargetRuntime.GENIE,
    ],
    Precision.q4_0: [
        TargetRuntime.GENIEX_LLAMACPP,
    ],
}


DEFAULT_EXPORT_DEVICE = "Samsung Galaxy S25 (Family)"

export_model = select_pipeline(resolve_model(MODEL_ID))


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for this model's export script.

    Exposed so the qai-hub-models CLI dispatcher can reuse the model's native
    parser without re-running main().
    """
    return export_parser(
        model_cls=Model,
        export_fn=export_model,
        supported_precision_runtimes=SUPPORTED_PRECISION_RUNTIMES,
        default_export_device=DEFAULT_EXPORT_DEVICE,
        omit_precision=True,
    )


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        warnings.warn(
            "Running `python -m qai_hub_models.models.qwen3_vl_4b_instruct.export` is "
            "deprecated and will be removed in a future release. "
            "Use `qai-hub-models export qwen3_vl_4b_instruct` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = build_parser().parse_args()
    # export_parser is called with omit_precision=True, so args.precision is
    # never set by CLI parsing — resolve it from the checkpoint instead.
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is not None:
        args.precision = CheckpointType.from_checkpoint(checkpoint).precision(
            DEFAULT_PRECISION, checkpoint=checkpoint
        )
    else:
        args.precision = DEFAULT_PRECISION
    export_model(MODEL_ID, **vars(args))


if __name__ == "__main__":
    main()
