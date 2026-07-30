# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

from qai_hub_models import Precision
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scripts.collect_scorecard_assets_results import (
    _preserve_llamacpp_gguf_assets,
)


def _asset(
    s3_key: str | None = None, download_url: str | None = None
) -> QAIHMModelReleaseAssets.AssetDetails:
    return QAIHMModelReleaseAssets.AssetDetails(
        s3_key=s3_key, download_url=download_url
    )


def _make_assets(entries: dict) -> QAIHMModelReleaseAssets:
    out = QAIHMModelReleaseAssets()
    for precision, kinds in entries.items():
        for path, asset in kinds.get("universal", {}).items():
            out.add_asset(asset, precision=precision, chipset=None, path=path)
        for chipset, path_map in kinds.get("chipset", {}).items():
            for path, asset in path_map.items():
                out.add_asset(asset, precision=precision, chipset=chipset, path=path)
    return out


def test_preserves_committed_llamacpp_gguf() -> None:
    """A genie/geniex_qairt-only scorecard must not drop the committed
    q4_0/geniex_llamacpp GGUF entry (download_url, no s3_key).
    """
    scorecard = _make_assets(
        {
            Precision.w4a16: {
                "chipset": {
                    "qualcomm-snapdragon-8-elite": {
                        ScorecardProfilePath.GENIE: _asset(s3_key="s3/genie.zip"),
                    }
                }
            }
        }
    )
    committed = _make_assets(
        {
            Precision.q4_0: {
                "universal": {
                    ScorecardProfilePath.GENIEX_LLAMACPP: _asset(
                        download_url="https://hf/model.gguf"
                    ),
                },
            },
        }
    )

    merged = _preserve_llamacpp_gguf_assets(scorecard, committed)

    kept = merged.get_asset(
        Precision.q4_0, chipset=None, path=ScorecardProfilePath.GENIEX_LLAMACPP
    )
    assert kept is not None
    assert kept.download_url == "https://hf/model.gguf"
    assert kept.s3_key is None

    fresh = merged.get_asset(
        Precision.w4a16,
        chipset="qualcomm-snapdragon-8-elite",
        path=ScorecardProfilePath.GENIE,
    )
    assert fresh is not None and fresh.s3_key == "s3/genie.zip"


def test_scorecard_llamacpp_wins_over_committed() -> None:
    """If the scorecard did produce a llamacpp entry (unlikely today but a
    fresh entry should still win), it is preserved over the committed one.
    """
    scorecard = _make_assets(
        {
            Precision.q4_0: {
                "universal": {
                    ScorecardProfilePath.GENIEX_LLAMACPP: _asset(
                        download_url="https://hf/new.gguf"
                    ),
                },
            },
        }
    )
    committed = _make_assets(
        {
            Precision.q4_0: {
                "universal": {
                    ScorecardProfilePath.GENIEX_LLAMACPP: _asset(
                        download_url="https://hf/old.gguf"
                    ),
                },
            },
        }
    )

    merged = _preserve_llamacpp_gguf_assets(scorecard, committed)

    kept = merged.get_asset(
        Precision.q4_0, chipset=None, path=ScorecardProfilePath.GENIEX_LLAMACPP
    )
    assert kept is not None
    assert kept.download_url == "https://hf/new.gguf"


def test_does_not_preserve_non_llamacpp_download_url() -> None:
    """Only geniex_llamacpp entries are preserved. A stray download_url on
    another runtime is dropped (scorecard is authoritative for those).
    """
    scorecard = _make_assets({})
    committed = _make_assets(
        {
            Precision.w4a16: {
                "chipset": {
                    "cs": {
                        ScorecardProfilePath.GENIE: _asset(
                            download_url="https://stray/genie.zip"
                        ),
                    }
                }
            }
        }
    )

    merged = _preserve_llamacpp_gguf_assets(scorecard, committed)

    assert merged.empty
