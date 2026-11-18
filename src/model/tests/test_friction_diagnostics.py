from __future__ import annotations

import unittest

import numpy as np

from src.model.friction_history_diagnostic import LAGS, _design, _spline_design
from src.model.networks import SplineBasis


class FrictionDiagnosticTests(unittest.TestCase):
    def test_history_design_uses_only_current_and_past_velocity(self):
        first = np.linspace(-0.1, 0.1, 40)
        changed_future = first.copy()
        changed_future[21:] = 10.0
        self.assertTrue(np.array_equal(_design(first, True)[:21],
                                       _design(changed_future, True)[:21]))
        self.assertEqual(_design(first, True).shape[1], 3 + 2 * len(LAGS))

    def test_current_velocity_spline_matches_lagged_parameter_count(self):
        v = np.linspace(-0.1, 0.1, 40)
        current = _spline_design(v, 0.0, 0.04, SplineBasis(intervals=9))
        lagged = _design(v, True)
        self.assertEqual(current.shape, lagged.shape)


if __name__ == "__main__":
    unittest.main()
