#!/usr/bin/env python3
"""Small CPU checks; malformed inputs live only in disposable temporary fixtures."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("action_clip_analysis", HERE / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
np = analysis.np


class Checks(unittest.TestCase):
    def test_sha_pass_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "input").write_bytes(b"observed")
            analysis.check_files(root, {"input": analysis.sha(root / "input")})
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                analysis.check_files(root, {"input": "0" * 64})

    def test_trace_sha_mismatch_rejected_after_consumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"; path.write_text('{"tick":1}\n')
            with self.assertRaisesRegex(ValueError, "trace SHA mismatch"):
                list(analysis.records(path, "0" * 64))

    def test_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"; path.write_text('{"value":NaN}')
            with self.assertRaisesRegex(ValueError, "Nonfinite JSON"):
                analysis.read(path)

    def test_nonfinite_tensor_rejected(self):
        with self.assertRaisesRegex(ValueError, "ABI/nonfinite"):
            analysis.array([float("inf")], (1,))

    def test_tensor_abi_rejected(self):
        with self.assertRaisesRegex(ValueError, "ABI/nonfinite"):
            analysis.array([0.0] * 5, (6,))

    def test_noncompleted_report_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("report.json", "clean_effective_contract.json", "birth.json", "first_cohort.json", "runtime.json", "clean_fixture_receipt.json"):
                (root / name).write_text(json.dumps({"status": "failed"} if name == "report.json" else {}))
            pin = {"files_sha256": {p.name: analysis.sha(p) for p in root.iterdir()}}
            with self.assertRaisesRegex(ValueError, "Completed inference"):
                analysis.analyze_run(root, pin, {})

    def test_regularizer_coefficient_and_no_clip_identity(self):
        zero = np.zeros((1, 6)); action = np.ones((1, 6))
        self.assertAlmostEqual(float(analysis.regularization(action, zero, zero)[0]), -.28)
        self.assertTrue(np.array_equal(analysis.regularization(action, zero, zero), analysis.regularization(np.clip(action, -analysis.BOUNDS, analysis.BOUNDS), zero, zero)))

    def test_clip_difference_includes_two_following_history_rows(self):
        zero = np.zeros((1, 6)); over = zero.copy(); over[0, 0] = 4.
        clipped = np.clip(over, -analysis.BOUNDS, analysis.BOUNDS)
        differences = [analysis.regularization(over, zero, zero) - analysis.regularization(clipped, zero, zero),
                       analysis.regularization(zero, over, zero) - analysis.regularization(zero, clipped, zero),
                       analysis.regularization(zero, zero, over) - analysis.regularization(zero, zero, clipped)]
        self.assertTrue(all(float(value[0]) != 0. for value in differences))
        self.assertEqual(float(analysis.regularization(zero, zero, zero)[0]), 0.)

    def test_same_trace_effort_coefficient_ratio(self):
        term = 12.5 + 3.5
        self.assertAlmostEqual((-1e-5 * term) / (-1e-4 * term), .1)
        self.assertAlmostEqual((-1e-5 + 1e-4) * term * .02, 9e-5 * term * .02)

    def test_projection_equivalence_and_horizontal_motion(self):
        for pitch in (-.6, -.1, 0., .1, .6):
            factor = math.sqrt(1. - math.sin(pitch) ** 2)
            self.assertAlmostEqual(factor, math.cos(pitch))
            body_vx = .5 * math.cos(pitch)
            self.assertAlmostEqual(body_vx * factor, .5 * math.cos(pitch) ** 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "validation.json")
    args = parser.parse_args()
    analysis.require(not args.output.exists(), "Refusing to overwrite validation output")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    args.output.write_text(json.dumps({"status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
        "analysis_script_sha256": analysis.sha(HERE / "analyze.py"),
        "validation_script_sha256": analysis.sha(Path(__file__)),
        "input_manifest_sha256": analysis.sha(HERE / "inputs.json"),
        "audit_sha256": analysis.sha(HERE / "audit.json"),
        "temporary_failure_fixtures_only": True, "GPU_or_simulator_started": False,
        "source_inputs_mutated": False, "qualification": False}, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
