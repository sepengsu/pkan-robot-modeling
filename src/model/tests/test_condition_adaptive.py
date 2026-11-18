from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from src.model.condition_adaptive import (
    CONDITIONS, CONFIG_PATH, DEFAULT_OUTPUT, DEFAULT_PREPARED, KAN22, PROTOCOLS,
    _clean_validation, _split_duplicate_audit,
)
from src.model.data import load_prepared
from src.model.networks import ResidualMLP, parameter_count


class ConditionAdaptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_PREPARED.exists():
            raise unittest.SkipTest("combined prepared dataset unavailable")
        cls.manifest, cls.arrays = load_prepared(DEFAULT_PREPARED)

    def test_exact_test_split_is_disjoint_and_validation_is_cleaned(self):
        overlap = _split_duplicate_audit(self.arrays)
        self.assertEqual(overlap[1]["unique_duplicate_states"], 0)
        self.assertEqual(overlap[2]["unique_duplicate_states"], 0)
        self.assertEqual(overlap[0]["second_rows_matching"], 182)
        for protocol, heldout in PROTOCOLS.items():
            excluded = np.isin(self.arrays["condition"],
                               [CONDITIONS.index(name) for name in heldout])
            train = np.flatnonzero((self.arrays["split"] == 0) & ~excluded)
            raw_val = np.flatnonzero((self.arrays["split"] == 1) & ~excluded)
            clean_val, removed = _clean_validation(train, raw_val, self.arrays["features"])
            self.assertFalse(np.any(excluded[train]))
            self.assertFalse(np.any(excluded[clean_val]))
            self.assertEqual(removed, 182 if protocol == "pooled" else 130)

    def test_physical_scale_basis_matches_all_labels(self):
        path = DEFAULT_OUTPUT / "physics_basis.npz"
        if not path.exists():
            self.skipTest("physical basis not prepared")
        with np.load(path, allow_pickle=False) as saved:
            basis = saved["inertia_scale_derivative_nm"]
        prediction = self.arrays["tau_nominal"].astype(np.float64) + 0.2 * basis
        self.assertLess(float(np.max(np.abs(prediction - self.arrays["tau_reference"]))), 5e-5)

    def test_mlp_and_kan_have_equal_parameter_counts(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        mlp = ResidualMLP(22, 30)
        kan = KAN22(config["grid_intervals"])
        self.assertEqual(parameter_count(mlp), 1806)
        self.assertEqual(parameter_count(kan), 1806)


if __name__ == "__main__":
    unittest.main()
