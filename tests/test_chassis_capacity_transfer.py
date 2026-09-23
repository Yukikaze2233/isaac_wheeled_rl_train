"""Capacity probes retain the selected actor and the actual memory stop reason."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_chassis_capacity as capacity


def test_capacity_probe_transfers_actor_and_stops_before_host_exhaustion(tmp_path, monkeypatch):
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"test actor")
    monkeypatch.setattr(sys, "argv", ["probe", "--plan", str(ROOT / "contracts/v5_adaptive_v56.json"),
        "--output", str(tmp_path / "result"), "--envs", "8192", "--updates", "8", "--transfer", str(checkpoint)])
    samples = iter([
        {"available_ram_kib": 8 * 1024**2, "gpu_free_mib": 16000},
        {"available_ram_kib": 3 * 1024**2, "host_available_ram_kib": 900000, "gpu_free_mib": 12000,
         "gpu_used_mib": 12000, "gpu_utilization_percent": 50},
    ])
    monkeypatch.setattr(capacity, "resources", lambda **kwargs: next(samples))
    calls, signals = [], []

    def popen(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(poll=lambda: None, send_signal=signals.append, wait=lambda **kwargs: None, returncode=-15)

    monkeypatch.setattr(capacity.subprocess, "Popen", popen)
    assert capacity.main() == 1
    assert calls[0][-3:] == ["--transfer", str(checkpoint), "--transfer-actor-only"]
    assert signals == [capacity.signal.SIGTERM]
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert report["stop_reason"] == "host_memory_guard"
    assert report["probes"][0]["min_host_available_ram_kib"] == 900000
