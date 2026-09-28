"""A preflight receipt must come from executed modules and fail on bad behavior."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v6_functional_probes", ROOT / "scripts/validate_v6_functional_probes.py")
PROBES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBES)


def test_probe_entry_emits_success_receipt_with_real_tick_output_and_source_hashes(tmp_path):
    output = tmp_path / "receipt.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/validate_v6_functional_probes.py"),
                             "--output", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["passed"] and len(report["source_sha256"]) >= 6
    assert report["source_sha256"] == PROBES.source_hashes()
    cases = report["transport"]["cases"]
    assert cases["zero_ms"]["output_by_tick"][:3] == [0., 1., 2.]
    assert cases["disabled"]["output_by_tick"][:3] == [1., 2., 3.]
    assert cases["all_drop"]["lost_packets"] == 12
    assert report["signal"]["episode_parameters_constant"]
    assert report["training_schedule"]["busy_repeat_preserves_start"]
    assert report["training_schedule"]["velocity_abs_max"][3] > 0


def test_transport_probe_rejects_an_enabled_zero_delay_identity_shortcut(monkeypatch):
    monkeypatch.setattr(PROBES.UsbCommandTransport, "apply_torque", lambda self, command: command)
    with pytest.raises(PROBES.ProbeFailure, match="CAN output mismatch"):
        PROBES.transport_probe(617)


def test_failed_probe_entry_writes_failure_and_returns_nonzero(tmp_path):
    output = tmp_path / "failed.json"
    result = PROBES.main(["--output", str(output), "--contract", str(tmp_path / "missing.json")])
    report = json.loads(output.read_text())
    assert result == 1 and report["passed"] is False
    assert "FileNotFoundError" in report["error"]
