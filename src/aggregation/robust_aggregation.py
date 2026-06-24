"""Robust aggregation primitives used by the unified revision experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


TensorUpdate = list[torch.Tensor]


def _validate_updates(updates: Sequence[TensorUpdate]) -> None:
    if not updates:
        raise ValueError("At least one client update is required")
    parameter_count = len(updates[0])
    if parameter_count == 0:
        raise ValueError("Client updates cannot be empty")
    if any(len(update) != parameter_count for update in updates):
        raise ValueError("All client updates must have identical structures")


def flatten_update(update: TensorUpdate) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in update])


def mean_aggregate(
    updates: Sequence[TensorUpdate],
    weights: Sequence[float] | None = None,
) -> TensorUpdate:
    _validate_updates(updates)
    n_clients = len(updates)
    if weights is None:
        normalized = torch.full(
            (n_clients,),
            1.0 / n_clients,
            device=updates[0][0].device,
            dtype=updates[0][0].dtype,
        )
    else:
        if len(weights) != n_clients:
            raise ValueError("weights must match the number of client updates")
        normalized = torch.as_tensor(
            weights,
            device=updates[0][0].device,
            dtype=updates[0][0].dtype,
        )
        normalized = normalized / normalized.sum()

    aggregated: TensorUpdate = []
    for parameter_index in range(len(updates[0])):
        stacked = torch.stack(
            [update[parameter_index] for update in updates],
            dim=0,
        )
        view_shape = (n_clients,) + (1,) * (stacked.ndim - 1)
        aggregated.append((stacked * normalized.view(view_shape)).sum(dim=0))
    return aggregated


def coordinate_trimmed_mean(
    updates: Sequence[TensorUpdate],
    trim_count: int,
) -> TensorUpdate:
    _validate_updates(updates)
    n_clients = len(updates)
    if trim_count < 0:
        raise ValueError("trim_count cannot be negative")
    if 2 * trim_count >= n_clients:
        raise ValueError("trim_count must leave at least one client update")

    aggregated: TensorUpdate = []
    for parameter_index in range(len(updates[0])):
        stacked = torch.stack(
            [update[parameter_index] for update in updates],
            dim=0,
        )
        sorted_values = torch.sort(stacked, dim=0).values
        kept = (
            sorted_values[trim_count : n_clients - trim_count]
            if trim_count
            else sorted_values
        )
        aggregated.append(kept.mean(dim=0))
    return aggregated


def krum(
    updates: Sequence[TensorUpdate],
    byzantine_bound: int,
) -> tuple[TensorUpdate, dict]:
    """Return the Krum-selected update using the original neighbor count.

    Krum requires n >= 2f + 3 and scores each update using its n-f-2 nearest
    *other* updates. The diagonal is explicitly excluded.
    """

    _validate_updates(updates)
    n_clients = len(updates)
    if byzantine_bound < 0:
        raise ValueError("byzantine_bound cannot be negative")
    if n_clients < 2 * byzantine_bound + 3:
        raise ValueError(
            f"Krum requires n >= 2f + 3; received n={n_clients}, "
            f"f={byzantine_bound}"
        )

    matrix = torch.stack([flatten_update(update) for update in updates], dim=0)
    squared_norms = (matrix * matrix).sum(dim=1, keepdim=True)
    distances = squared_norms + squared_norms.T - 2.0 * matrix @ matrix.T
    distances = distances.clamp_min(0.0)
    distances.fill_diagonal_(float("inf"))

    neighbor_count = n_clients - byzantine_bound - 2
    nearest = torch.topk(
        distances,
        k=neighbor_count,
        dim=1,
        largest=False,
    ).values
    scores = nearest.sum(dim=1)
    selected_index = int(torch.argmin(scores).item())

    return [tensor.clone() for tensor in updates[selected_index]], {
        "selected_client_position": selected_index,
        "neighbor_count": neighbor_count,
        "scores": [float(value) for value in scores.detach().cpu()],
    }


@dataclass
class AdaptiveTrimmedMean:
    """Adaptive coordinate-wise trimmed mean.

    The method estimates the fraction of anomalous client updates from robust
    distances to the coordinate-wise median. It then smooths the observed
    outlier fraction into a per-round trimming ratio.
    """

    initial_trim_ratio: float = 0.10
    min_trim_ratio: float = 0.05
    max_trim_ratio: float = 0.20
    adaptation_rate: float = 0.50
    outlier_z_threshold: float = 3.5

    def __post_init__(self) -> None:
        if not 0 <= self.min_trim_ratio <= self.initial_trim_ratio:
            raise ValueError("initial_trim_ratio must be >= min_trim_ratio")
        if self.initial_trim_ratio > self.max_trim_ratio:
            raise ValueError("initial_trim_ratio must be <= max_trim_ratio")
        if not 0 < self.adaptation_rate <= 1:
            raise ValueError("adaptation_rate must be in (0, 1]")
        self.trim_ratio = self.initial_trim_ratio
        self.round_index = 0

    def aggregate(
        self,
        updates: Sequence[TensorUpdate],
        client_ids: Sequence[int],
    ) -> tuple[TensorUpdate, dict]:
        _validate_updates(updates)
        if len(client_ids) != len(updates):
            raise ValueError("client_ids must match the number of updates")

        self.round_index += 1
        matrix = torch.stack([flatten_update(update) for update in updates], dim=0)
        coordinate_median = matrix.median(dim=0).values
        distances = torch.linalg.vector_norm(
            matrix - coordinate_median.unsqueeze(0),
            dim=1,
        )

        distance_median = distances.median()
        mad = (distances - distance_median).abs().median()
        robust_scale = 1.4826 * mad + torch.finfo(distances.dtype).eps
        robust_z = (distances - distance_median) / robust_scale
        flagged_mask = robust_z > self.outlier_z_threshold
        flagged_positions = torch.nonzero(flagged_mask, as_tuple=False).flatten()
        flagged_ids = [int(client_ids[index]) for index in flagged_positions]

        observed_ratio = float(flagged_mask.float().mean().item())
        desired_ratio = min(
            self.max_trim_ratio,
            max(self.min_trim_ratio, observed_ratio),
        )
        self.trim_ratio += self.adaptation_rate * (
            desired_ratio - self.trim_ratio
        )

        n_clients = len(updates)
        trim_count = max(1, int(round(self.trim_ratio * n_clients)))
        trim_count = min(trim_count, (n_clients - 1) // 2)
        aggregated = coordinate_trimmed_mean(updates, trim_count=trim_count)

        metadata = {
            "round": self.round_index,
            "trim_ratio": float(self.trim_ratio),
            "trim_count_each_tail": trim_count,
            "observed_outlier_ratio": observed_ratio,
            "outlier_z_threshold": self.outlier_z_threshold,
            "flagged_client_ids": flagged_ids,
            "robust_z_scores": [
                float(value) for value in robust_z.detach().cpu()
            ],
            "distances": [
                float(value) for value in distances.detach().cpu()
            ],
        }
        return aggregated, metadata
