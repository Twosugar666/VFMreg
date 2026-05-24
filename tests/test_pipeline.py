"""
tests/test_pipeline.py
======================
端到端冒烟测试：保证 utils / configs / evaluate / tools 能 import & 基本功能可跑。
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


class TestUtilsGeometry(unittest.TestCase):
    def test_rodrigues_roundtrip(self):
        from utils import rodrigues_to_matrix, matrix_to_rodrigues
        rvec = np.array([0.1, -0.2, 0.3])
        R = rodrigues_to_matrix(rvec)
        back = matrix_to_rodrigues(R)
        np.testing.assert_allclose(rvec, back, atol=1e-6)

    def test_6d_roundtrip(self):
        from utils import matrix_to_6d_np, sixd_to_matrix_np
        from utils import rodrigues_to_matrix
        R = rodrigues_to_matrix([0.5, 0.1, -0.3])
        d6 = matrix_to_6d_np(R)
        R2 = sixd_to_matrix_np(d6)
        np.testing.assert_allclose(R, R2, atol=1e-6)

    def test_pose_error(self):
        from utils import pose_error, rodrigues_to_matrix
        R = rodrigues_to_matrix([0.0, 0.0, 0.0])
        t_err, r_err = pose_error(R, np.zeros(3), R, np.zeros(3))
        self.assertAlmostEqual(t_err, 0.0)
        self.assertAlmostEqual(r_err, 0.0)


class TestConfigsLoader(unittest.TestCase):
    def test_load_base(self):
        from configs import load_config
        cfg = load_config(os.path.join(ROOT, "configs", "base.yaml"))
        self.assertIn("seg", cfg)
        self.assertIn("nerf", cfg)
        self.assertIn("vfmreg", cfg)
        self.assertEqual(cfg["seed"], 42)

    def test_expand_variants(self):
        from configs import expand_variants
        variants = list(expand_variants(
            os.path.join(ROOT, "configs", "ablation_vfmreg.yaml")))
        self.assertGreater(len(variants), 1)
        names = [n for _, n in variants]
        self.assertTrue(any("full" in n for n in names))
        self.assertTrue(any("no_render_loss" in n for n in names))


class TestEvaluateMetrics(unittest.TestCase):
    def test_miou(self):
        from evaluate.metrics import confusion_matrix, miou_from_cm
        pred = np.array([0, 1, 2, 1, 0, 2])
        gt = np.array([0, 1, 2, 0, 0, 2])
        cm = confusion_matrix(pred, gt, num_classes=3)
        miou = miou_from_cm(cm)
        self.assertGreater(miou, 0.4)
        self.assertLess(miou, 1.0)

    def test_aggregate_pose(self):
        from evaluate.metrics import aggregate_pose_errors
        out = aggregate_pose_errors([0.4, 0.6, 0.5], [0.3, 0.7, 0.6])
        self.assertEqual(out["n_samples"], 3)
        self.assertAlmostEqual(out["trans_mean_mm"], 0.5, places=2)


class TestTools(unittest.TestCase):
    def test_icp_synth(self):
        from tools.icp_baseline import icp
        rng = np.random.default_rng(0)
        src = rng.normal(size=(500, 3))
        # 已知小幅旋转 + 平移
        angle = np.deg2rad(8)
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [0, 0, 1]])
        t = np.array([2.0, -1.0, 0.5])
        dst = (R @ src.T).T + t
        out = icp(src, dst, max_iters=80)
        from utils import pose_error
        et, er = pose_error(out["R"], out["t"], R, t)
        self.assertLess(et, 0.05)
        self.assertLess(er, 0.5)

    def test_mesh_sampling(self):
        from tools.mesh_utils import sample_surface
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        pts = sample_surface(verts, faces, n_samples=200)
        self.assertEqual(pts.shape, (200, 3))


class TestPaperAlignment(unittest.TestCase):
    def test_alignment_with_real_results(self):
        from evaluate.paper_alignment import align
        results_dir = os.path.join(ROOT, "results")
        if not os.path.isdir(results_dir):
            self.skipTest("results 目录不存在")
        rep = align(results_dir, tol=0.10)
        self.assertIn("details", rep)
        self.assertIn("passed", rep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
