# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------

from __future__ import annotations

import torch

from qai_hub_models.utils.base_evaluator import BaseEvaluator
from qai_hub_models.utils.metrics import MEAN_AVERAGE_PRECISION, MetricMetadata


class ReIDEvaluator(BaseEvaluator):
    """Evaluator for person re-identification using mAP and Rank-1."""

    def __init__(self) -> None:
        """Initialize empty accumulators for embeddings and ground-truth labels."""
        self.embeddings: list[torch.Tensor] = []
        self.person_ids: list[torch.Tensor] = []
        self.camera_ids: list[torch.Tensor] = []
        self.query_mask: list[torch.Tensor] = []
        self._metrics: tuple[float, float] | None = None

    def add_batch(
        self,
        output: torch.Tensor,
        gt: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
    ) -> None:
        """Accumulate a batch of embeddings and ground-truth labels.

        Parameters
        ----------
        output:
            Model output embeddings of shape [N, feature_dim].
        gt:
            Tuple of (person_ids, camera_ids, is_query) each of shape [N].
        """
        assert isinstance(output, torch.Tensor)
        assert isinstance(gt, (tuple, list)) and len(gt) == 3
        pids, camids, is_query = gt
        self.embeddings.append(output.detach().cpu())
        self.person_ids.append(torch.as_tensor(pids).reshape(-1).cpu())
        self.camera_ids.append(torch.as_tensor(camids).reshape(-1).cpu())
        self.query_mask.append(torch.as_tensor(is_query).reshape(-1).bool().cpu())
        self._metrics = None

    def reset(self) -> None:
        self.embeddings = []
        self.person_ids = []
        self.camera_ids = []
        self.query_mask = []
        self._metrics = None

    def _compute_metrics(self) -> tuple[float, float]:
        if self._metrics is not None:
            return self._metrics
        if not self.embeddings:
            self._metrics = (0.0, 0.0)
            return self._metrics

        embeddings = torch.cat(self.embeddings, dim=0)
        pids = torch.cat(self.person_ids, dim=0)
        camids = torch.cat(self.camera_ids, dim=0)
        query_mask = torch.cat(self.query_mask, dim=0)
        gallery_mask = ~query_mask

        if query_mask.sum() == 0 or gallery_mask.sum() == 0:
            self._metrics = (0.0, 0.0)
            return self._metrics

        query_emb = embeddings[query_mask]
        query_pids = pids[query_mask]
        query_camids = camids[query_mask]
        gallery_emb = embeddings[gallery_mask]
        gallery_pids = pids[gallery_mask]
        gallery_camids = camids[gallery_mask]

        distance = torch.cdist(query_emb, gallery_emb)

        rank1_hits = 0
        map_sum = 0.0
        valid_queries = 0

        for query_index in range(query_emb.shape[0]):
            order = torch.argsort(distance[query_index], dim=0)
            qpid = query_pids[query_index]
            qcam = query_camids[query_index]

            keep = ~((gallery_pids[order] == qpid) & (gallery_camids[order] == qcam))
            if keep.sum() == 0:
                continue
            matches = gallery_pids[order][keep] == qpid
            if matches.sum() == 0:
                continue

            valid_queries += 1
            rank1_hits += int(matches[0].item())

            hit_positions = torch.nonzero(matches, as_tuple=False).flatten()
            cum_matches = torch.cumsum(matches.int(), dim=0).float()
            precision_at_hits = cum_matches[hit_positions] / (
                hit_positions.float() + 1.0
            )
            map_sum += float(precision_at_hits.mean().item())

        if valid_queries == 0:
            self._metrics = (0.0, 0.0)
            return self._metrics

        mean_ap = 100.0 * map_sum / valid_queries
        rank1 = 100.0 * rank1_hits / valid_queries
        self._metrics = (mean_ap, rank1)
        return self._metrics

    def get_accuracy_score(self) -> float:
        mean_ap, _ = self._compute_metrics()
        return mean_ap

    def formatted_accuracy(self) -> str:
        mean_ap, rank1 = self._compute_metrics()
        return f"{mean_ap:.2f}% (mAP), {rank1:.2f}% (R1)"

    def get_metric_metadata(self) -> MetricMetadata:
        return MEAN_AVERAGE_PRECISION.with_description(
            "Mean Average Precision for person re-identification retrieval."
        )
