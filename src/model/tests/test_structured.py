from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import torch

from src.model.data import EXPECTED_SPLITS, load_archived, load_prepared
from src.model.networks import CoupledKAN, JointwiseSpline, PhysicsStructured, SplineBasis


ROOT = Path(__file__).resolve().parents[3]


class SplineTests(unittest.TestCase):
    def test_partition_and_gradients(self):
        basis = SplineBasis(5)
        x = torch.linspace(-3, 3, 101).reshape(-1, 1).requires_grad_()
        values = basis(x)
        torch.testing.assert_close(values.sum(-1), torch.ones_like(x), atol=1e-6, rtol=0)
        values.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_jointwise_has_no_cross_joint_dependence(self):
        model = JointwiseSpline(5)
        with torch.no_grad():
            model.coefficients.copy_(torch.randn_like(model.coefficients))
        x = torch.randn(8, 6)
        changed = x.clone(); changed[:, 1] += 1.5
        a, b = model(x), model(changed)
        torch.testing.assert_close(a[:, 0], b[:, 0])
        self.assertGreater(float(torch.max(torch.abs(a[:, 1] - b[:, 1])).detach()), 1e-6)

    def test_staged_decomposition(self):
        joint, coupled = JointwiseSpline(), CoupledKAN()
        model = PhysicsStructured(joint, coupled)
        x = torch.randn(3, 12)
        total, one, two = model(x[:, 6:12], x)
        torch.testing.assert_close(total, one + two)
        self.assertFalse(any(p.requires_grad for p in joint.parameters()))


class DataTests(unittest.TestCase):
    def test_archived_provenance_and_split(self):
        root = Path(r"D:\Code\sdml-doosan-robot\data\grid16_legacy_joint_dataset_m0_m4")
        if not root.exists():
            self.skipTest("archived dataset not present")
        manifest, arrays = load_archived(root)
        self.assertEqual(manifest["split_policy"], EXPECTED_SPLITS)
        self.assertEqual(len(manifest["conditions"]), 7)
        self.assertEqual(int(np.sum(arrays["split"] == 0)), 31766)

    def test_mujoco_scene_consistency_and_friction_effect(self):
        root = ROOT / "result/structured_kan/prepared_d"
        if not root.exists():
            self.skipTest("prepared simulator dataset not present")
        manifest, arrays = load_prepared(root)
        self.assertLess(manifest["scene_reconstruction_max_abs_nm"], 2e-5)
        self.assertGreater(np.sqrt(np.mean((arrays["tau_friction"] - arrays["tau_reference"])**2)), .05)
        self.assertTrue(np.isfinite(arrays["tau_friction"]).all())


if __name__ == "__main__":
    unittest.main()
