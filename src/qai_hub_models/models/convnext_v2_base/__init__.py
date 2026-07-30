# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from qai_hub_models.models._shared.imagenet_classifier.app import (
    ImagenetClassifierApp as App,
)

from .model import MODEL_ID
from .model import ConvNextV2Base as Model

__all__ = ["MODEL_ID", "App", "Model"]
