# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from qai_hub_models.configs.manifest_yaml import QAIHMModelManifest
from qai_hub_models.scorecard import ScorecardProfilePath
from qai_hub_models.scorecard.artifacts import ScorecardArtifact
from qai_hub_models.scorecard.envvars import (
    ArtifactsDirEnvvar,
    DeploymentEnvvar,
    EnabledDevicesEnvvar,
    EnabledModelsEnvvar,
    EnabledPathsEnvvar,
    EnabledPrecisionsEnvvar,
    SpecialModelSetting,
)
from qai_hub_models.scorecard.release_assets_yaml import QAIHMModelReleaseAssets
from qai_hub_models.scorecard.results.code_gen import (
    remove_asset_failures,
    update_model_publish_status,
)
from qai_hub_models.scorecard.results.yaml import ScorecardAssetYaml
from qai_hub_models.scorecard.static.list_models import (
    validate_and_split_enabled_models,
)
from qai_hub_models.utils.hub_clients import get_default_hub_deployment


def _preserve_llamacpp_gguf_assets(
    scorecard_assets: QAIHMModelReleaseAssets,
    committed_assets: QAIHMModelReleaseAssets,
) -> QAIHMModelReleaseAssets:
    """Preserve committed geniex_llamacpp GGUF entries (download_url, no s3_key)
    when scorecard only re-produced other runtimes' assets. Without this, running
    a genie/geniex_qairt-only scorecard would silently drop the llamacpp asset
    from the model's release-assets.yaml.
    """
    merged = copy.deepcopy(scorecard_assets)
    for precision, prec_details in committed_assets.precisions.items():
        for path, asset in prec_details.universal_assets.items():
            if path != ScorecardProfilePath.GENIEX_LLAMACPP:
                continue
            if asset.s3_key or not asset.download_url:
                continue
            if merged.get_asset(precision, None, path) is None:
                merged.add_asset(
                    copy.deepcopy(asset),
                    precision=precision,
                    chipset=None,
                    path=path,
                )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    EnabledModelsEnvvar.add_arg(parser, {SpecialModelSetting.PYTORCH})
    DeploymentEnvvar.add_arg(parser, default=get_default_hub_deployment())
    EnabledPrecisionsEnvvar.add_arg(parser)
    EnabledPathsEnvvar.add_arg(parser)
    EnabledDevicesEnvvar.add_arg(parser)
    ArtifactsDirEnvvar.add_arg(parser)
    parser.add_argument(
        "--release-assets-yaml",
        type=str,
        default=str(ScorecardArtifact.RELEASE_ASSETS.path),
    )
    return parser.parse_args()


def main() -> None:
    # Verify args are compatible with the chosen deployment.
    args = parse_args()
    pytorch_models, _ = validate_and_split_enabled_models(args.models)

    assets_path = Path(args.release_assets_yaml)
    if not assets_path.exists() or assets_path.stat().st_size == 0:
        print("No scorecard release assets found. Not updating any files.")
        return

    modified_files: list[str] = []
    scorecard_assets = ScorecardAssetYaml.from_yaml(args.release_assets_yaml)
    for model_id in sorted(pytorch_models):
        try:
            manifest = QAIHMModelManifest.from_model(model_id)
            sc = manifest.scorecard_config
            if (
                sc.skip_hub_tests_and_scorecard
                or sc.skip_scorecard
                or sc.freeze_perf_yaml
            ):
                continue

            if scorecard_model_assets := scorecard_assets.models.get(model_id):
                if sc.is_llm:
                    committed_assets = QAIHMModelReleaseAssets.from_model(
                        model_id, not_exists_ok=True
                    )
                    scorecard_model_assets = _preserve_llamacpp_gguf_assets(
                        scorecard_model_assets, committed_assets
                    )
                scorecard_model_assets = remove_asset_failures(
                    scorecard_model_assets, manifest.disabled_paths
                )

                if scorecard_model_assets.has_ephemeral_s3_keys:
                    print(
                        f"Skipping {model_id} release-assets.yaml: assets were "
                        "uploaded to ephemeral_test_assets/ (auto-purged)."
                    )
                else:
                    # Write updated assets
                    modified_files.append(
                        str(scorecard_model_assets.to_model_yaml(model_id))
                    )
            else:
                QAIHMModelReleaseAssets().to_model_yaml(
                    model_id
                )  # deletes existing file

            # Update model status & reason, if applicable
            if update_model_publish_status(manifest):
                manifest_path = manifest.to_model_yaml()
                print(f"Updated publish status at {manifest_path}")

        except Exception as e:
            raise ValueError(
                f"Failed to collect accuracy results for {model_id}"
            ) from e


if __name__ == "__main__":
    main()
