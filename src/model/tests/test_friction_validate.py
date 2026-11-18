from __future__ import annotations

import unittest

import numpy as np

from src.model.friction_validate import _fit_closed_form, _predict_closed, _scores


class FrictionValidationTests(unittest.TestCase):
    def test_closed_form_recovers_velocity_friction_on_unseen_states(self):
        rng = np.random.default_rng(5)
        states = np.zeros((300, 22), dtype=np.float32)
        states[:, 6:12] = rng.uniform(-0.12, 0.12, size=(300, 6))
        targets = np.empty((300, 15, 6), dtype=np.float32)
        for joint in range(6):
            velocity = states[:, 6 + joint]
            for scenario in range(15):
                targets[:, scenario, joint] = (0.02 * (scenario + 1)
                                                * np.tanh(velocity / 0.005)
                                                + 0.3 * velocity)
        data = {"train": {"features": states, "delta": targets}}
        linear, _, timings = _fit_closed_form(data, {"ridge_lambda": 1e-8, "spline_intervals": 5})
        self.assertGreater(timings["Linear_friction"], 0)
        new = np.zeros((60, 22), dtype=np.float32)
        new[:, 6:12] = rng.uniform(-0.12, 0.12, size=(60, 6))
        prediction = _predict_closed("Linear_friction", linear, new)
        expected = np.empty_like(prediction)
        for joint in range(6):
            for scenario in range(15):
                expected[:, scenario, joint] = (0.02 * (scenario + 1)
                                                * np.tanh(new[:, 6 + joint] / 0.005)
                                                + 0.3 * new[:, 6 + joint])
        self.assertLess(float(np.sqrt(np.mean((prediction - expected) ** 2))), 1e-5)

    def test_scores_use_all_scenarios_and_conditions(self):
        truth = np.zeros((7, 15, 6), dtype=np.float32)
        predicted = np.ones_like(truth)
        data = {"delta": truth, "condition": np.arange(7, dtype=np.int8)}
        scenarios = [{"name": str(i), "model": "synthetic", "level": "one"} for i in range(15)]
        overall, by_scenario, by_condition, by_joint = _scores(
            "test", "11", "test", data, predicted, scenarios, 1, 0.0, 1)
        self.assertEqual(overall["scenario_labels"], 105)
        self.assertAlmostEqual(overall["torque_rmse_nm"], 1.0)
        self.assertEqual(len(by_scenario), 15)
        self.assertEqual(len(by_condition), 7)
        self.assertEqual(len(by_joint), 6)


if __name__ == "__main__":
    unittest.main()
