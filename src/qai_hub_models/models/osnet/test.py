# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

import numpy as np

from qai_hub_models.models.osnet.app import OSNetApp
from qai_hub_models.models.osnet.demo import (
    PERSON_A_IMAGE_1,
    PERSON_A_IMAGE_2,
    PERSON_B_IMAGE,
)
from qai_hub_models.models.osnet.demo import main as demo_main
from qai_hub_models.models.osnet.model import MODEL_ASSET_VERSION, MODEL_ID, OSNet
from qai_hub_models.utils.asset_loaders import (
    CachedWebModelAsset,
    load_image,
    load_numpy,
)
from qai_hub_models.utils.test_helpers import assert_most_close

IMAGE_A1 = PERSON_A_IMAGE_1
IMAGE_A2 = PERSON_A_IMAGE_2
IMAGE_B = PERSON_B_IMAGE
EXPECTED_OUT = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "expected_out.npy"
)


def test_task() -> None:
    model = OSNet.from_pretrained(pretrained=True)
    input_spec = model.get_input_spec()
    app = OSNetApp(model, input_spec)

    embeddings = app.predict_features(
        load_image(IMAGE_A1), load_image(IMAGE_A2), load_image(IMAGE_B)
    )
    emb_a1, emb_a2, emb_b = embeddings[0], embeddings[1], embeddings[2]

    expected = load_numpy(EXPECTED_OUT)
    assert_most_close(emb_a1, expected, diff_tol=0.0, rtol=0.0, atol=1e-4)

    sim_same = float(np.dot(emb_a1, emb_a2))
    sim_diff_a1_b = float(np.dot(emb_a1, emb_b))
    sim_diff_a2_b = float(np.dot(emb_a2, emb_b))

    assert sim_same > sim_diff_a1_b, (
        f"Expected same-person similarity ({sim_same:.4f}) > "
        f"Person A img1 vs Person B ({sim_diff_a1_b:.4f})"
    )
    assert sim_same > sim_diff_a2_b, (
        f"Expected same-person similarity ({sim_same:.4f}) > "
        f"Person A img2 vs Person B ({sim_diff_a2_b:.4f})"
    )


def test_demo() -> None:
    demo_main(is_test=True)
