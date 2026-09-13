"""CPU coverage of live identity, actual canonical transforms and selected-env IO."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import io
import json
import math
from pathlib import Path
import socket
import sys
import uuid
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.v40.live_view import (
    BODY_NAMES, LatestState, LiveViewSession, PoseStream, add_arguments,
    clean_viewer_environment, normalize_pose, urdf_visuals, validate_options,
)


def test_default_disabled_needs_no_runtime(monkeypatch):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    assert not args.live_view
    monkeypatch.setattr("subprocess.run", lambda *_a, **_k: pytest.fail("disabled path must not probe dependencies"))
    validate_options(args)
    env = {"PYTHONPATH": "sim", "PYTHONHOME": "sim", "LD_LIBRARY_PATH": "sim", "VIRTUAL_ENV": "sim", "KEEP": "yes"}
    assert clean_viewer_environment(env) == {"KEEP": "yes", "PYTHONNOUSERSITE": "1"}
    assert env["PYTHONPATH"] == "sim"


def test_real_canonical_meshes_keep_local_rotation_and_hashes(tmp_path):
    from wheeled_tasks.v40.contract import load_contract, validate_asset
    asset = validate_asset(load_contract(), allow_research=True)
    visuals = urdf_visuals(asset, sorted(BODY_NAMES), tmp_path)
    assert len(visuals) == 7
    assert {v["body"] for v in visuals} == BODY_NAMES
    base = next(v for v in visuals if v["body"] == "base_link")
    assert base["wxyz"] == pytest.approx([math.sqrt(.5), 0, 0, math.sqrt(.5)])
    for visual in visuals:
        assert visual["scale"] == [1, 1, 1]
        assert (tmp_path / visual["mesh"]).read_bytes() == (asset["directory"] / visual["source"]).read_bytes()
    changed = deepcopy(asset)
    changed["manifest"]["files_sha256"]["meshes/base_link.STL"] = "0" * 64
    with pytest.raises(ValueError, match="mesh changed"):
        urdf_visuals(changed, sorted(BODY_NAMES), tmp_path)


@pytest.mark.parametrize("pose", [[0] * 7, [0, 0, 0, 0, 0, 0, float("nan")], [0] * 6])
def test_bad_quaternion_rejected(pose):
    with pytest.raises(ValueError):
        normalize_pose(pose)


@pytest.mark.skipif(sys.platform != "linux", reason="WSL/Linux abstract Unix socket transport")
def test_publisher_rate_limit_and_single_run_receiver():
    now = [1.0]
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    receiver.bind("\0v40-test-" + uuid.uuid4().hex)
    receiver.settimeout(1)
    evidence = io.StringIO()
    publisher = PoseStream("run-a", "digest", ["body"], receiver.getsockname(), evidence, clock=lambda: now[0])
    try:
        assert publisher.due()
        now[0] += .049
        assert not publisher.due()
        now[0] += .002
        assert publisher.due()
        publisher.publish([[1, 2, 3, 0, 0, 0, 2]], step=3, sim_time=.03,
                          telemetry={"large_packet": "x" * 4000})
        message = json.loads(receiver.recv(65535))
        assert message["poses_xyzw"] == [[1, 2, 3, 0, 0, 0, 1]]
        clock = [message["sent_ns"]]
        state = LatestState({"run_id": "run-a", "names": ["body"]}, "digest", clock=lambda: clock[0])
        for key, value in (("run_id", "other-run"), ("manifest_sha256", "other-asset"),
                           ("names", ["other-body"]), ("seq", 0)):
            assert not state.accept({**message, key: value})
        assert state.accept(message)
        assert not state.accept(message)
        assert state.status()["live"]
        clock[0] += 1_100_000_000
        assert not state.status()["live"]
        ended = {**message, "seq": 2, "sent_ns": clock[0], "phase": "ended"}
        assert state.accept(ended)
        assert not state.status()["live"]
        assert not state.accept({**ended, "seq": 3, "phase": "training"})
        assert json.loads(evidence.getvalue()) == message
    finally:
        publisher.close()
        receiver.close()


def test_capture_indexes_gpu_before_cpu_and_preserves_policy_clock():
    events = []

    class GPU:
        def __init__(self, value, label, selected=False):
            self.value, self.label, self.selected = value, label, selected

        def __getitem__(self, index):
            assert index == 1
            events.append((self.label, "select", index))
            return GPU(self.value, self.label, True)

        def detach(self):
            assert self.selected
            return self

        def cpu(self):
            assert self.selected, "N-env CPU copy forbidden"
            events.append((self.label, "cpu"))
            return self

        def tolist(self):
            return self.value

        def item(self):
            assert self.selected
            return self.value

    poses = [[0, 0, .3, 0, 0, 0, 1]] * 7
    raw = SimpleNamespace(
        robot=SimpleNamespace(data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=GPU(poses, "poses")))),
        commands=GPU([0, 0, .32], "commands"), actions=GPU([0] * 6, "actions"),
        episode_length_buf=GPU(5, "episode"), common_step_counter=123, step_dt=.01,
    )
    sent = []
    session = LiveViewSession.__new__(LiveViewSession)
    session.args = SimpleNamespace(live_view_env=1)
    session.raw_env = raw
    session.stream = SimpleNamespace(publish=lambda *args, **kwargs: sent.append((args, kwargs)))
    session.capture(reward=GPU(.2, "reward"))
    assert sent[0][0] == (poses,)
    assert sent[0][1]["step"] == 123 and sent[0][1]["sim_time"] == 1.23
    assert sent[0][1]["telemetry"]["reward"] == .2
    assert events[:2] == [("poses", "select", 1), ("poses", "cpu")]


def test_strict_cuda_torchvision_target_and_other_pins_unchanged():
    spec = importlib.util.spec_from_file_location("train_live_gate_test", ROOT / "scripts/train_v40.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TARGET_VERSIONS == {
        "isaaclab": "6.1.14", "isaaclab_assets": "0.3.4", "isaaclab_tasks": "1.10.9",
        "isaaclab_rl": "0.5.5", "isaacsim": "6.0.0.1", "rsl-rl-lib": "5.5.1",
        "torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128",
        "onnx": "1.23.0rc1", "onnxruntime": "1.30.0",
    }
    expected = module.TARGET_VERSIONS["torchvision"]
    assert module.runtime_version_matches("0.26.0+cu128", expected)
    for installed in ("0.26.0", "0.26.0+cu130", "0.26.0+cpu", "0.26.0.dev1+cu128"):
        assert not module.runtime_version_matches(installed, expected)
