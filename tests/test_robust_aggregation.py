import unittest

import torch

from src.aggregation.robust_aggregation import (
    AdaptiveTrimmedMean,
    coordinate_trimmed_mean,
    krum,
)
from src.experiments.transparency_paradox_experiment import (
    LedgerFeedbackController,
)


def update(value: float) -> list[torch.Tensor]:
    return [torch.tensor([value, value + 0.01], dtype=torch.float32)]


class RobustAggregationTests(unittest.TestCase):
    def test_krum_excludes_self_and_selects_honest_cluster(self):
        updates = [update(value) for value in [0.0, 0.02, -0.01, 0.01, 8.0]]
        selected, metadata = krum(updates, byzantine_bound=1)
        self.assertLess(abs(float(selected[0][0])), 0.1)
        self.assertEqual(metadata["neighbor_count"], 2)

    def test_trimmed_mean_removes_extreme_tails(self):
        updates = [update(value) for value in [-10, 0, 1, 2, 20]]
        aggregated = coordinate_trimmed_mean(updates, trim_count=1)
        self.assertAlmostEqual(float(aggregated[0][0]), 1.0, places=5)

    def test_atma_flags_outliers_and_increases_trimming(self):
        aggregator = AdaptiveTrimmedMean(
            initial_trim_ratio=0.05,
            min_trim_ratio=0.05,
            max_trim_ratio=0.20,
            adaptation_rate=1.0,
        )
        updates = [update(value) for value in [0, 0.01, -0.01, 0.02, 8, 9]]
        _, metadata = aggregator.aggregate(updates, client_ids=list(range(6)))
        self.assertTrue({4, 5}.issubset(set(metadata["flagged_client_ids"])))
        self.assertGreater(metadata["trim_ratio"], 0.05)

    def test_atma_uses_configured_safety_bound_not_known_byzantine_ratio(self):
        aggregator = AdaptiveTrimmedMean(
            initial_trim_ratio=0.10,
            min_trim_ratio=0.05,
            max_trim_ratio=0.25,
            adaptation_rate=1.0,
        )
        updates = [update(value) for value in [0, 0.01, -0.01, 0.02, 8, 9]]

        _, metadata = aggregator.aggregate(updates, client_ids=list(range(6)))

        self.assertEqual(metadata["trim_ratio"], 0.25)

    def test_ledger_feedback_controller_reduces_scale_after_detection(self):
        controller = LedgerFeedbackController(initial_scale=5.0)
        self.assertAlmostEqual(controller.update(1.0), 3.5)
        self.assertAlmostEqual(controller.update(1.0), 2.45)

    def test_ledger_feedback_controller_respects_bounds(self):
        controller = LedgerFeedbackController(initial_scale=1.0)
        for _ in range(20):
            controller.update(0.0)
        self.assertLessEqual(controller.scale, controller.maximum_scale)
        for _ in range(20):
            controller.update(1.0)
        self.assertGreaterEqual(controller.scale, controller.minimum_scale)


if __name__ == "__main__":
    unittest.main()
