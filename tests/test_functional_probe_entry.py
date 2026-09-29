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
    phase = report["transport_phase_aware"]
    assert phase["passed"] and phase["delivery_model"] == "phase_aware_hold_v1"
    phase_cases = phase["cases"]
    assert phase_cases["before_phase"]["interval_torque_by_tick"][:3] == [.75, 1.75, 2.75]
    assert phase_cases["after_phase_next_tick"]["interval_torque_by_tick"][:3] == [0., .75, 1.75]
    assert phase_cases["before_phase"]["end_held_by_tick"][:3] == [1., 2., 3.]
    assert phase_cases["iid_all_drop_after_warmup"]["interval_torque_by_tick"] == [.75] + [1.] * 11
    assert phase_cases["iid_all_drop_after_warmup"]["lost_packets"] == 11
    assert phase_cases["fixed_burst"]["interval_torque_by_tick"][4:7] == [4., 4., 6.25]
    assert phase_cases["fixed_burst"]["lost_packets"] == 4
    assert phase["jitter_boundary"]["scalar_event_queue_matches"]
    assert phase["jitter_boundary"]["early_packet_count"] > 0
    assert phase["jitter_boundary"]["late_packet_count"] > 0
    assert report["signal"]["episode_parameters_constant"]
    assert report["training_schedule"]["busy_repeat_preserves_start"]
    assert report["training_schedule"]["velocity_abs_max"][3] > 0


def test_transport_probe_rejects_an_enabled_zero_delay_identity_shortcut(monkeypatch):
    monkeypatch.setattr(PROBES.UsbCommandTransport, "apply_torque", lambda self, command: command)
    with pytest.raises(PROBES.ProbeFailure, match="CAN output mismatch"):
        PROBES.transport_probe(617)


def test_phase_probe_rejects_using_end_cache_as_interval_torque(monkeypatch):
    original = PROBES.UsbCommandTransport._phase_transport_tick

    def wrong_output(self, command):
        original(self, command)
        return self.held

    monkeypatch.setattr(PROBES.UsbCommandTransport, "_phase_transport_tick", wrong_output)
    with pytest.raises(PROBES.ProbeFailure, match="Phase-aware interval torque"):
        PROBES.phase_transport_probe(617)


def test_phase_probe_rejects_silently_disabled_submillisecond_jitter(monkeypatch):
    original = PROBES.UsbCommandTransport._configure_phase_model

    def ignore_jitter(self, *args):
        original(self, *args)
        self.jitter_half_width.zero_()

    monkeypatch.setattr(PROBES.UsbCommandTransport, "_configure_phase_model", ignore_jitter)
    with pytest.raises(PROBES.ProbeFailure, match="jitter did not cross CAN boundary"):
        PROBES.phase_transport_probe(617)


def test_phase_probe_rejects_clearing_held_torque_on_packet_loss(monkeypatch):
    original = PROBES.UsbCommandTransport._phase_transport_tick

    def clear_on_loss(self, command):
        before = self.lost.clone()
        result = original(self, command)
        dropped = self.lost > before
        self.held[dropped] = 0.
        result[dropped] = 0.
        return result

    monkeypatch.setattr(PROBES.UsbCommandTransport, "_phase_transport_tick", clear_on_loss)
    with pytest.raises(PROBES.ProbeFailure, match="Phase-aware interval torque"):
        PROBES.phase_transport_probe(617)


def test_phase_probe_rejects_ignored_burst_profile(monkeypatch):
    original = PROBES.UsbCommandTransport._phase_transport_tick

    def ignore_burst(self, command):
        self.fixed_burst_length.zero_()
        return original(self, command)

    monkeypatch.setattr(PROBES.UsbCommandTransport, "_phase_transport_tick", ignore_burst)
    with pytest.raises(PROBES.ProbeFailure, match="Phase-aware interval torque"):
        PROBES.phase_transport_probe(617)


def test_phase_failure_blocks_the_real_probe_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(PROBES.UsbCommandTransport, "_phase_transport_tick", lambda self, command: command)
    output = tmp_path / "phase_failure.json"
    assert PROBES.main(["--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["passed"] is False
    assert "Phase-aware interval torque" in report["error"]


def test_failed_probe_entry_writes_failure_and_returns_nonzero(tmp_path):
    output = tmp_path / "failed.json"
    result = PROBES.main(["--output", str(output), "--contract", str(tmp_path / "missing.json")])
    report = json.loads(output.read_text())
    assert result == 1 and report["passed"] is False
    assert "FileNotFoundError" in report["error"]
