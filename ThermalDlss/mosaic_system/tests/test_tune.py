from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mosaic_system.tune import (
    _compare_evaluations,
    _constraint_values,
    _eligibility,
    _manifest_sha256,
    parse_args,
)


class TuneFairnessTests(unittest.TestCase):
    def test_default_search_budget_is_fixed_ten_epochs_and_seed_42(self):
        args = parse_args([])
        self.assertEqual(args.epochs_per_trial, 10)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.pruner, "none")
        self.assertEqual(args.eval_max_samples, 10)

    def test_quality_gate_rejects_ssim_regression(self):
        baseline = {
            "ssim_model": 0.80,
            "model_phase_mean_std": 0.01,
            "model_clip_ratio": 0.001,
        }
        candidate = {
            "ssim_model": 0.79,
            "model_phase_mean_std": 0.01,
            "model_clip_ratio": 0.001,
        }
        eligible, reasons = _eligibility(candidate, baseline)
        self.assertFalse(eligible)
        self.assertIn("SSIM baseline altı", reasons)
        self.assertGreater(_constraint_values(candidate, baseline)[0], 0)

    def test_comparison_requires_identical_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            candidate = root / "candidate"
            baseline.mkdir()
            candidate.mkdir()
            header = "group_id,psnr_model,ssim_model\n"
            (baseline / "metrics.csv").write_text(
                header + "g1,30.0,0.8\n", encoding="utf-8"
            )
            (candidate / "metrics.csv").write_text(
                header + "g2,31.0,0.81\n", encoding="utf-8"
            )
            (baseline / "evaluated_manifest.jsonl").write_text(
                '{"group_id":"g1"}\n', encoding="utf-8"
            )
            (candidate / "evaluated_manifest.jsonl").write_text(
                '{"group_id":"g2"}\n', encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                _compare_evaluations(baseline, candidate)

    def test_manifest_hash_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            one = root / "one"
            two = root / "two"
            one.mkdir()
            two.mkdir()
            payload = '{"group_id":"same"}\n'
            (one / "evaluated_manifest.jsonl").write_text(payload, encoding="utf-8")
            (two / "evaluated_manifest.jsonl").write_text(payload, encoding="utf-8")
            self.assertEqual(_manifest_sha256(one), _manifest_sha256(two))


if __name__ == "__main__":
    unittest.main()
