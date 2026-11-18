from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from src.model.combined import ROOT, SOURCE_REPO, SPLIT_POLICY, inspect
from src.model.data import load_prepared
from src.model.train import _campaign_weights


class CombinedDataTests(unittest.TestCase):
    def test_archive_compatibility_and_unified_split(self):
        legacy = ROOT / "data/grid16_legacy_joint_dataset_m0_m4"
        wide = ROOT / "data/wide16_v3_joint_dataset"
        if not (legacy.exists() and wide.exists() and SOURCE_REPO.exists()):
            self.skipTest("source campaigns not available")
        _, summary = inspect(legacy, wide, SOURCE_REPO)
        self.assertEqual(summary["rows"], {"train": 225470, "validation": 17360, "test": 53690})
        self.assertEqual(summary["total_rows"], 296520)
        self.assertEqual(summary["exact_feature_row_overlap"], 0)

    def test_prepared_groups_do_not_cross_splits(self):
        prepared = ROOT / "result/combined_legacy_wide_v3/prepared"
        if not prepared.exists():
            self.skipTest("combined prepared dataset not available")
        manifest, arrays = load_prepared(prepared)
        self.assertEqual(len(arrays["features"]), 296520)
        self.assertEqual(arrays["tau_friction"].shape, (296520, 6))
        self.assertEqual(manifest["split_policy"], SPLIT_POLICY)
        for trajectory in range(1, 17):
            mask = arrays["trajectory"] == trajectory
            self.assertEqual(len(np.unique(arrays["split"][mask])), 1)
        self.assertEqual(len(np.unique(arrays["campaign"])), 2)

    def test_campaign_weights_equalize_total_weight(self):
        campaigns = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.uint8)
        indices = np.arange(len(campaigns))
        weights = _campaign_weights({"campaign": campaigns}, indices).numpy()
        self.assertAlmostEqual(float(weights[campaigns == 0].sum()), 3.0)
        self.assertAlmostEqual(float(weights[campaigns == 1].sum()), 3.0)


if __name__ == "__main__":
    unittest.main()
