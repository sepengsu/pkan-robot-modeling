from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from src.model.collect_friction import (
    CONFIG_PATH, SOURCE_REPO, _asymmetric_friction, _build_chunk, _lugre_friction,
    _model_ids, _perturb_plan, _read_plan, _scene_models, _smooth_friction,
    _source_paths, _split_lookup, scenarios,
)


class FrictionCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_three_distinct_trials_and_disjoint_splits(self):
        self.assertEqual(len(scenarios(self.config)), 15)
        self.assertEqual(len(_split_lookup(self.config)), 16)
        _, plan_path = _source_paths(SOURCE_REPO, "T1", "bare_p0")
        scene_path, _ = _source_paths(SOURCE_REPO, "T1", "ag145_p0")
        plan = _read_plan(plan_path, self.config["stride_from_100hz"])
        limits = _model_ids(_scene_models(scene_path, SOURCE_REPO, self.config)[0][0])[2]
        trials = [_perturb_plan(plan, "T1", trial, self.config, limits)
                  for trial in (1, 2, 3)]
        self.assertEqual(len({trial[4] for trial in trials}), 3)
        for left, right in ((0, 1), (0, 2), (1, 2)):
            self.assertGreater(float(np.max(np.abs(trials[left][1] - trials[right][1]))), 1e-4)
        for trial in trials:
            self.assertTrue(np.allclose(trial[1][[0, -1]], plan[1][[0, -1]], atol=1e-12))
            self.assertTrue(np.allclose(trial[2][[0, -1]], plan[2][[0, -1]], atol=1e-12))

    def test_friction_laws_are_finite_and_have_expected_symmetry(self):
        velocity = np.asarray([[-0.1] * 6, [0.0] * 6, [0.1] * 6])
        fc = np.asarray(self.config["friction_base_nm"])
        b = np.asarray(self.config["viscous_base_nm_s_rad"])
        smooth = _smooth_friction(velocity, fc, b, self.config)
        asymmetric = _asymmetric_friction(velocity, fc, b, self.config)
        lugre = _lugre_friction(np.asarray([0.0, 0.1, 0.2]), velocity, fc, b, self.config)
        self.assertTrue(np.isfinite(lugre).all())
        self.assertTrue(np.allclose(smooth[0], -smooth[2]))
        self.assertTrue(np.allclose(smooth[1], 0))
        self.assertFalse(np.allclose(asymmetric[0], -asymmetric[2]))

    def test_smoke_chunk_has_all_fifteen_targets(self):
        scene_path, plan_path = _source_paths(SOURCE_REPO, "T1", "ag145_p0")
        plan = _read_plan(plan_path, self.config["stride_from_100hz"])
        models = _scene_models(scene_path, SOURCE_REPO, self.config)
        arrays, meta = _build_chunk("T1", "ag145_p0", 1, plan, models, self.config, 20)
        self.assertEqual(arrays["features"].shape, (20, 22))
        self.assertEqual(arrays["tau_reference_nm"].shape, (20, 6))
        self.assertEqual(arrays["delta_friction_nm"].shape, (20, 15, 6))
        self.assertTrue(np.isfinite(arrays["delta_friction_nm"]).all())
        self.assertEqual(meta["rows"], 20)


if __name__ == "__main__":
    unittest.main()
