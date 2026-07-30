# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import numpy as np

from qai_hub_models.models.osnet.app import OSNetApp
from qai_hub_models.models.osnet.model import MODEL_ASSET_VERSION, MODEL_ID, OSNet
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)
from qai_hub_models.utils.asset_loaders import CachedWebModelAsset, load_image

PERSON_A_IMAGE_1 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "person_a_1.jpg"
)
PERSON_A_IMAGE_2 = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "person_a_2.jpg"
)
PERSON_B_IMAGE = CachedWebModelAsset.from_asset_store(
    MODEL_ID, MODEL_ASSET_VERSION, "person_b_0.jpg"
)


def osnet_demo(model_cls: type[OSNet], is_test: bool = False) -> None:
    parser = get_model_cli_parser(model_cls)
    parser = get_on_device_demo_parser(parser)
    parser.add_argument(
        "--image-person-a1",
        type=str,
        default=PERSON_A_IMAGE_1,
        help="First image of Person A (path or URL).",
    )
    parser.add_argument(
        "--image-person-a2",
        type=str,
        default=PERSON_A_IMAGE_2,
        help="Second image of Person A (path or URL).",
    )
    parser.add_argument(
        "--image-person-b",
        type=str,
        default=PERSON_B_IMAGE,
        help="Image of Person B (path or URL).",
    )
    args = parser.parse_args([] if is_test else None)
    validate_on_device_demo_args(args, MODEL_ID)

    model = demo_model_from_cli_args(model_cls, MODEL_ID, args)
    input_spec = model.get_input_spec()
    app = OSNetApp(model, input_spec)  # type: ignore[arg-type]
    print("Model loaded.\n")

    img_a1 = load_image(args.image_person_a1)
    img_a2 = load_image(args.image_person_a2)
    img_b = load_image(args.image_person_b)

    embeddings = app.predict_features(img_a1, img_a2, img_b)
    emb_a1, emb_a2, emb_b = embeddings[0], embeddings[1], embeddings[2]

    sim_same = float(np.dot(emb_a1, emb_a2))
    sim_diff_a1_b = float(np.dot(emb_a1, emb_b))
    sim_diff_a2_b = float(np.dot(emb_a2, emb_b))

    if not is_test:
        print("Re-ID Cosine Similarity Results")
        print("=" * 40)
        print(f"  Person A (img1) vs Person A (img2) : {sim_same:.4f}  [same person]")
        print(
            f"  Person A (img1) vs Person B        : {sim_diff_a1_b:.4f}  [different person]"
        )
        print(
            f"  Person A (img2) vs Person B        : {sim_diff_a2_b:.4f}  [different person]"
        )
        print("=" * 40)
        if sim_same > sim_diff_a1_b and sim_same > sim_diff_a2_b:
            print(
                "[PASS] Same-person pair scores higher than cross-person pairs — re-ID working as expected."
            )
        else:
            print(
                "[WARN] Unexpected ordering — placeholder images may not reflect real re-ID behaviour."
            )

        print(
            f"Input images: {args.image_person_a1}, {args.image_person_a2}, {args.image_person_b}"
        )


def main(is_test: bool = False) -> None:
    osnet_demo(OSNet, is_test=is_test)


if __name__ == "__main__":
    main()
