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

DATASET_ID = "market1501"
DATASET_ASSET_VERSION = 1

# Market-1501 filename convention: <pid>_c<camid>s<seq>_<frame>_<det>.jpg
# e.g. 0001_c1s1_000151_01.jpg
# junk images use pid == -1 (stored as 0000 or -1 in some splits)
FILENAME_PATTERN = re.compile(r"^(?P<pid>-?\d+)_c(?P<camid>\d+)s\d+_\d+_\d+\.jpg$")

MARKET1501_PRIVATE_ASSET = CachedPrivateDatasetAsset(
    "qai-hub-models/datasets/market1501/market1501_data.zip",
    DATASET_ID,
    DATASET_ASSET_VERSION,
    "market1501_data.zip",
    installation_steps=[
        "Visit the dataset page: https://zheng-lab-anu.github.io/Project/project_reid.html",
        "Download Market-1501-v15.09.15.zip from the Google Drive link on that page: "
        "https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view",
        "Rename or copy the zip so it is named market1501_data.zip, e.g.:\n"
        "    cp Market-1501-v15.09.15.zip market1501_data.zip",
        "Images follow the naming convention: "
        "<person_id>_c<camera_id>s<seq>_<frame>_<det>.jpg "
        "(e.g. 0001_c1s1_000151_01.jpg)",
        "Run: python -m qai_hub_models.scripts.configure_dataset "
        "--class qai_hub_models.datasets.reid.market1501.Market1501Dataset "
        "--files /path/to/market1501_data.zip",
    ],
    local_cache_extracted_path="market1501_data",
)


class Market1501Dataset(BaseReidDataset):
    r"""
    Market-1501 dataset wrapper for person re-identification evaluation.

    The dataset is available from the project page:
        https://zheng-lab-anu.github.io/Project/project_reid.html
    """

    @staticmethod
    def get_private_asset() -> CachedPrivateDatasetAsset:
        return MARKET1501_PRIVATE_ASSET

    def __init__(
        self,
        split: DatasetSplit = DatasetSplit.VAL,
        input_spec: InputSpec | None = None,
        max_eval_samples: int = 1000,
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
        """Return the directory that contains bounding_box_test/ and query/.

        The Market-1501 zip extracts to a versioned subfolder
        (``Market-1501-v15.09.15/``); fall back to the dataset root if that
        subfolder is absent.
        """
        versioned = self.dataset_path / "Market-1501-v15.09.15"
        return versioned if versioned.exists() else self.dataset_path

    @staticmethod
    def default_samples_per_job() -> int:
        return 1000

    @staticmethod
    def get_dataset_metadata() -> DatasetMetadata:
        return DatasetMetadata(
            link="https://zheng-lab-anu.github.io/Project/project_reid.html",
            split_description=(
                "Evaluation split using bounding_box_test/ (gallery) and query/ "
                "from Market-1501-v15.09.15 (max 1000 images by default)."
            ),
        )

    def _build_samples(self, split: DatasetSplit) -> list[ReidSample]:
        """Parse gallery, query, and (optionally) train images into ReidSample objects."""
        data_root = self._get_data_root()

        if split == DatasetSplit.TRAIN:
            train_root = data_root / "bounding_box_train"
            if not train_root.exists():
                return []
            samples: list[ReidSample] = []
            for path in sorted(train_root.glob("*.jpg")):
                match = FILENAME_PATTERN.match(path.name)
                if not match:
                    continue
                person_id = int(match.group("pid"))
                if person_id <= 0:
                    continue
                samples.append(
                    ReidSample(
                        image_path=path,
                        person_id=person_id,
                        camera_id=int(match.group("camid")),
                        is_query=False,
                    )
                )
            return samples

        gallery_root = data_root / "bounding_box_test"
        query_root = data_root / "query"
        gallery_paths = sorted(gallery_root.glob("*.jpg"))
        query_paths = sorted(query_root.glob("*.jpg"))

        if not gallery_paths and not query_paths:
            return []

        query_path_set = set(query_paths)
        parsed: list[tuple[Path, int, int, bool]] = []
        for path in gallery_paths + query_paths:
            match = FILENAME_PATTERN.match(path.name)
            if not match:
                continue
            person_id = int(match.group("pid"))
            if person_id <= 0:
                continue
            camera_id = int(match.group("camid"))
            parsed.append((path, person_id, camera_id, path in query_path_set))

        if not parsed:
            raise ValueError(
                "Found Market-1501 images but none matched the expected naming format "
                "'<person_id>_c<camera_id>s<seq>_<frame>_<det>.jpg' "
                "(e.g. 0001_c1s1_000151_01.jpg)."
            )

        result = [
            ReidSample(
                image_path=path,
                person_id=person_id,
                camera_id=camera_id,
                is_query=is_query,
            )
            for path, person_id, camera_id, is_query in parsed
        ]
        result.sort(key=lambda s: (not s.is_query, s.image_path.name))
        return result
