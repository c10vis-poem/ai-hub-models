# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import re
from pathlib import Path

from qai_hub_models.datasets.reid.reid import BaseReidDataset, ReidSample
from qai_hub_models.utils.base_dataset import DatasetMetadata, DatasetSplit
from qai_hub_models.utils.input_spec import InputSpec
from qai_hub_models.utils.private_asset_loaders import CachedPrivateDatasetAsset

DATASET_ID = "entire_id"
DATASET_ASSET_VERSION = 1

# ENTIRe-ID filename convention: <pid>_c<camid>s<seq>_<frame>.(jpg|png)
# e.g. 00009_c021s0_760487.jpg
FILENAME_PATTERN = re.compile(r"^(?P<pid>\d+)_c(?P<camid>\d+)s\d+_.*\.(jpg|png)$")

ENTIRE_ID_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/entire_id/entire_id_data.zip",
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "entire_id_data.zip",
    installation_steps=[
        "Open the Google Drive folder: "
        "https://drive.google.com/drive/folders/1elx1plYw0BSOH9qON9-1dNUgxFtgDJTj",
        "Download the two sub-folders: bounding_box_test/ (gallery) and query/ (query images)",
        "Zip them together into a single archive, e.g.:\n"
        "    zip -r entire_id_data.zip bounding_box_test/ query/",
        "Run: python -m qai_hub_models.scripts.configure_dataset "
        "--class qai_hub_models.datasets.reid.entire_id.ENTIReIDDataset "
        "--files /path/to/entire_id_data.zip",
    ],
    local_cache_extracted_path="entire_id_data",
)


class ENTIReIDDataset(BaseReidDataset):
    r"""
    ENTIRe-ID dataset wrapper for person re-identification evaluation.

    The dataset is publicly available :
        https://serdaryildiz.com/ENTIRe-ID/

    Images follow the naming convention: <person_id>_c<camera_id>s<seq>_<frame>.jpg
    e.g. 00009_c021s0_760487.jpg
    """

    @staticmethod
    def get_private_asset() -> CachedPrivateDatasetAsset:
        return ENTIRE_ID_PRIVATE_ASSET

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        max_eval_samples: int = 500,
        input_data_path: str | None = None,
    ) -> None:
        super().__init__(
            split=split,
            input_spec=input_spec,
            max_eval_samples=max_eval_samples,
            input_data_path=input_data_path,
        )

    @classmethod
    def dataset_name(cls) -> str:
        return DATASET_ID

    def _get_data_root(self) -> Path:
        """Return the directory that contains bounding_box_test/ and query/."""
        return self.dataset_path

    @staticmethod
    def default_samples_per_job() -> int:
        return 491

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://serdaryildiz.github.io/ENTIRe-ID/",
            split_description=(
                "Evaluation split using the bounding_box_test/ (gallery) and "
                "query/ folders from the ENTIRe-ID Google Drive download "
                "(max 500 images by default)."
            ),
        )

    def _build_samples(self, split: DatasetSplit) -> list[ReidSample]:
        """Parse gallery and query images into a flat list of ReidSample objects."""
        gallery_root = self.dataset_path / "bounding_box_test"
        query_root = self.dataset_path / "query"
        gallery_paths = sorted(gallery_root.rglob("*.jpg")) + sorted(
            gallery_root.rglob("*.png")
        )
        query_paths = sorted(query_root.rglob("*.jpg")) + sorted(
            query_root.rglob("*.png")
        )
        if not gallery_paths and not query_paths:
            return []

        query_path_set = set(query_paths)
        parsed: list[tuple[Path, int, int, bool]] = []
        for path in gallery_paths + query_paths:
            match = FILENAME_PATTERN.match(path.name)
            if not match:
                continue
            person_id = int(match.group("pid"))
            camera_id = int(match.group("camid"))
            parsed.append((path, person_id, camera_id, path in query_path_set))

        if not parsed:
            raise ValueError(
                "Found ENTIRe-ID images but none matched the expected naming format "
                "'<person_id>_c<camera_id>s<seq>_<frame>.(jpg|png)' "
                "(e.g. 00009_c021s0_760487.jpg)."
            )

        samples = [
            ReidSample(
                image_path=path,
                person_id=person_id,
                camera_id=camera_id,
                is_query=is_query,
            )
            for path, person_id, camera_id, is_query in parsed
        ]
        # Queries first for deterministic small-sample slicing.
        samples.sort(key=lambda s: (not s.is_query, s.image_path.name))

        if not any(s.is_query for s in samples) or not any(
            not s.is_query for s in samples
        ):
            raise ValueError(
                "ENTIRe-ID data must contain images in both bounding_box_test/ "
                "(gallery) and query/ folders."
            )
        return samples
