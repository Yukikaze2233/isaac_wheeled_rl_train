"""Offline recording contracts and real CPU FFmpeg streaming; no Isaac/GPU imports."""
from __future__ import annotations

import builtins
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.v40_metrics import BATCH_FIELDS, REASONS


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("test_record_entry", ROOT / "scripts/record_v40.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(tick, *, terminal=False, timeout=False, bad=False):
    row = {
        "schema_version": 1, "sample_kind": "pre_reset" if tick else "initial",
        "policy_tick": tick, "physics_steps": tick * 2, "episode_step": tick,
        "time_s": tick * .01, "sim_time_s": tick * .01, "episode_time_s": tick * .01,
        "policy_dt_s": .01, "physics_dt_s": .005, "contact_history_samples": 2,
        "command": [0., 0., .32], "root_link_pos_w_m": [0., 0., .32],
        "root_link_quat_wxyz": [1., 0., 0., 0.], "projected_gravity_b": [0., 0., -1.],
        "root_com_lin_vel_b_m_s": [0., 0., 0.], "root_com_ang_vel_b_rad_s": [0., 0., 0.],
        "height_m": float("nan") if bad else .32, "joint_pos_rad": [0.] * 6,
        "joint_vel_rad_s": [0.] * 6, "sim_joint_effort_nm": [0.] * 6,
        "wheel_axis_midpoint_w_m": [0., 0., .05], "wheel_net_force_max_n": [10., 10.],
        "non_wheel_net_force_max_n": 0., "base_visual_clearance_lower_bound_m": .2,
        "terminated": terminal and not timeout, "timeout": terminal and timeout,
        "termination_flags": {reason: terminal and not timeout and reason == "tilt" for reason in REASONS} if tick else {},
    }
    return {key: ({name: [flag] for name, flag in value.items()} if key == "termination_flags"
                  else [value] if key in BATCH_FIELDS else value) for key, value in row.items()}


class FakeEnv:
    """Model the audited Lab order: render physics -> snapshot -> auto-reset live state."""
    def __init__(self, events, *, stop=None, timeout=False, bad_tick=None):
        self.events = events
        self.tick = 0
        self.live_tick = 0
        self.stop = stop
        self.timeout = timeout
        self.bad_tick = bad_tick
        self.command = None
        self.contract_sha256 = "contract"
        self.asset_manifest_sha256 = "asset"
        self.previous_frames = []

    def set_evaluation_command(self, command):
        self.command = command
        self.events.append("command")

    def reset(self):
        assert self.command == (0., 0., .32)
        self.events.append("reset")
        return self.get_observations(), {}

    def get_observations(self):
        return {"policy": [[float(self.live_tick)]], "critic": [[0.]]}

    def capture_evaluation_initial_snapshot(self):
        return snapshot(0)

    def get_evaluation_snapshot(self):
        return snapshot(self.tick, terminal=self.tick == self.stop,
                        timeout=self.timeout, bad=self.tick == self.bad_tick)

    def step(self, action):
        assert action == [[0.]]
        self.tick += 1
        self.live_tick = 0 if self.tick == self.stop else self.tick
        return self.get_observations(), [0.], [self.tick == self.stop], {}

    def render(self, recompute=False):
        assert recompute is False
        # At most the previous frame can remain alive; an accumulating frame list fails.
        assert sum(ref() is not None for ref in self.previous_frames) <= 1
        frame = np.full((720, 960, 3), (self.tick % 250) + 1, dtype=np.uint8)
        self.previous_frames.append(weakref.ref(frame))
        return frame

    def close(self):
        self.events.append("env_close")


class FakeWriter:
    def __init__(self, events, path=None, *, fail=False, empty=False):
        self.events = events
        self.path = path
        self.fail = fail
        self.empty = empty
        self.pixels = []

    def append_data(self, frame):
        self.pixels.append(int(frame[0, 0, 0]))

    def close(self):
        self.events.append("writer_close")
        if self.fail:
            raise RuntimeError("encoder trailer failed")
        if self.path and not self.empty:
            self.path.write_bytes(b"fake encoded mp4")


@pytest.mark.parametrize("steps,stop,timeout,ticks,reason", [
    (8, None, False, [0, 4, 8], "duration_limit"),
    (9, None, False, [0, 4, 8, 9], "duration_limit"),
    (1000, None, False, list(range(0, 1001, 4)), "duration_limit"),
    (1000, 6, False, [0, 4, 6], "terminated"),
    (1000, 1, False, [0, 1], "terminated"),
    (1000, 4, True, [0, 4], "timeout"),
])
def test_stream_sampling_terminal_cached_frame_and_bound(cli, tmp_path, steps, stop, timeout, ticks, reason):
    env = FakeEnv([], stop=stop, timeout=timeout)
    writer = FakeWriter([])
    summary = {"frames": 0, "frame_policy_ticks": [], "policy_steps": 0}
    saved = []
    cli.collect_video(env, env, lambda obs: [[0.]], steps, lambda: True, writer,
                      lambda path, frame: saved.append((path.name, int(frame[0, 0, 0]))), tmp_path, summary)
    assert summary["frame_policy_ticks"] == ticks
    assert writer.pixels == [(tick % 250) + 1 for tick in ticks]
    assert saved == [("final_frame.png", (ticks[-1] % 250) + 1)]
    assert env.tick == (stop or steps)
    assert summary["sim_duration_s"] == env.tick * .01
    assert summary["stop_reason"] == reason
    if stop and not timeout:
        assert env.live_tick == 0 and summary["termination_reasons"] == ["tilt"]


@pytest.mark.parametrize("failure", ["state", "action", "observation", "app_stop"])
def test_invalid_rollout_stops_and_saves_image(cli, tmp_path, failure):
    env = FakeEnv([], bad_tick=2 if failure == "state" else None)
    if failure == "observation":
        env.get_observations = lambda: {"policy": [[float("nan")]]}
    writer = FakeWriter([])
    summary = {"frames": 0, "frame_policy_ticks": [], "policy_steps": 0}
    images = []
    actor = lambda obs: [[float("inf")]] if failure == "action" else [[0.]]
    with pytest.raises((ValueError, RuntimeError)):
        cli.collect_video(env, env, actor, 1000, lambda: failure != "app_stop", writer,
                          lambda path, frame: images.append(int(frame[0, 0, 0])), tmp_path, summary)
    assert env.tick == (2 if failure == "state" else 0)
    assert images == [env.tick + 1]


@pytest.mark.parametrize("duration", ["0", "10.01", "nan", "inf", "0.015", "-1"])
def test_invalid_duration(cli, duration):
    with pytest.raises(SystemExit) as exc:
        cli.parse_arguments(["--duration-s", duration])
    assert exc.value.code == 2


@pytest.mark.parametrize("arguments", [
    ["--apply"], ["--apply", "--research"], ["--num-envs", "2"],
])
def test_explicit_apply_gates(cli, arguments):
    with pytest.raises(SystemExit) as exc:
        cli.parse_arguments(arguments)
    assert exc.value.code == 2


def test_help_and_preflight_work_with_stdlib_only(tmp_path):
    for option in ("--help", "--preflight-only"):
        result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/record_v40.py"), option,
                                 "--research", "--output-dir", str(tmp_path / "unused")],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == (0 if option == "--help" else 2), result.stderr
        if option == "--preflight-only":
            report = json.loads(result.stdout)
            assert report["simulation_started"] is False and report["blockers"]
        assert not (tmp_path / "unused").exists()


def test_ready_preflight_never_imports_runtime(cli, monkeypatch, tmp_path):
    args, train = cli.parse_arguments(["--research", "--preflight-only", "--output-dir", str(tmp_path / "unused")])
    monkeypatch.setattr(cli, "parse_arguments", lambda argv: (args, train))
    monkeypatch.setattr(train, "preflight", lambda args: ({"blockers": []}, None, None))
    original = builtins.__import__

    def no_runtime(name, *args, **kwargs):
        assert not name.startswith(("torch", "numpy", "imageio", "isaac", "omni", "carb", "rsl_rl"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_runtime)
    before = dict(os.environ)
    assert cli.main([]) == 0
    assert dict(os.environ) == before and not (tmp_path / "unused").exists()


@pytest.mark.parametrize("blocker", ["original_gate", "existing_output", "dangling_symlink", "missing_dependency"])
def test_recording_cannot_waive_preflight_gates(cli, monkeypatch, tmp_path, blocker):
    checkpoint = tmp_path / "model_final.pt"
    checkpoint.write_bytes(b"not loaded")
    (tmp_path / "run_manifest.json").write_text("{}")
    output = tmp_path / "output"
    if blocker == "existing_output":
        output.mkdir()
    elif blocker == "dangling_symlink":
        output.symlink_to(tmp_path / "missing")
    args, train = cli.parse_arguments(["--apply", "--research", "--source-case", "source-final",
                                      "--checkpoint", str(checkpoint), "--output-dir", str(output)])
    monkeypatch.setattr(cli, "parse_arguments", lambda argv: (args, train))
    monkeypatch.setattr(train, "preflight", lambda args: (
        {"blockers": ["collision rejected"] if blocker == "original_gate" else []}, None, None))
    monkeypatch.setattr(cli, "record", lambda *args: pytest.fail("blocked recording must not execute"))
    if blocker == "missing_dependency":
        def missing(package):
            raise cli.importlib.metadata.PackageNotFoundError(package)
        monkeypatch.setattr(cli.importlib.metadata, "version", missing)
    assert cli.main([]) == 2
    assert not (output / "summary.json").exists()


def test_finite_check_visits_tensordict_fields_before_detach(cli):
    class FakeTensorDict:
        def items(self):
            return {"policy": [[0.]], "critic": [[float("nan")]]}.items()

        def detach(self):
            pytest.fail("TensorDict must be traversed, not converted as a tensor")

    with pytest.raises(ValueError, match="nonfinite observation.critic"):
        cli.require_finite(FakeTensorDict(), "observation")


@pytest.mark.parametrize("uid", [0, 1000])
def test_explicit_camera_launcher(cli, monkeypatch, uid):
    calls = []
    monkeypatch.setenv("ENABLE_CAMERAS", "invalid")
    monkeypatch.setenv("LIVESTREAM", "2")
    monkeypatch.setattr(cli.os, "geteuid", lambda: uid)
    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=lambda cfg: calls.append(cfg)))
    cli.launch_recording_app(SimpleNamespace(device="cuda:0"))
    expected = {"headless": True, "enable_cameras": True, "livestream": 0, "device": "cuda:0"}
    if uid == 0:
        expected["kit_args"] = "--allow-root"
    assert calls == [expected]
    assert os.environ["ENABLE_CAMERAS"] == "1" and os.environ["LIVESTREAM"] == "0"


def test_viewer_cfg_reuses_task_without_mdp_copy(cli, monkeypatch):
    cfg = SimpleNamespace(scene=SimpleNamespace(), sim=SimpleNamespace(), viewer=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "wheeled_tasks.direct.v40_serial.env_cfg", SimpleNamespace(V40EnvCfg=lambda: cfg))
    monkeypatch.setitem(sys.modules, "wheeled_tasks.direct.v40_serial.env",
                        SimpleNamespace(V40Env=lambda **kwargs: kwargs))
    args = SimpleNamespace(contract=None, usd_cache_dir=None, research=True, stage="stand", seed=40,
                           num_envs=1, device="cuda:0")
    assert cli.make_recording_env(args) == {"cfg": cfg, "render_mode": "rgb_array"}
    assert cfg.rerender_on_reset is False and cfg.viewer.resolution == (960, 720)
    assert cfg.viewer.eye == (1., -1.4, .8) and cfg.viewer.lookat == (0., 0., .3)
    assert cfg.scene.num_envs == 1 and cfg.sim.device == "cuda:0" and cfg.allow_research


def test_warmup_enables_stock_pre_reset_render_without_physics(cli, monkeypatch):
    flags = {}
    events = []
    settings = SimpleNamespace(set_bool=lambda key, value: flags.update({key: value}))
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(settings=SimpleNamespace(get_settings=lambda: settings)))
    sim = SimpleNamespace(has_rtx_sensors=lambda: flags.get("/isaaclab/render/rtx_sensors", False),
                          render=lambda: events.append("render"))
    env = SimpleNamespace(sim=sim, render=lambda **kwargs: np.ones((2, 2, 3), dtype=np.uint8))
    cli.warm_renderer(env)
    assert events == ["render"] and flags["/isaaclab/render/rtx_sensors"]


@pytest.mark.parametrize("failure", [None, "bad_state", "encoder", "empty_video", "checkpoint", "source_change"])
def test_summary_and_video_finalized_before_nonreturning_app_close(cli, monkeypatch, tmp_path, failure):
    args, _ = cli.parse_arguments(["--apply", "--research", "--checkpoint", str(tmp_path / "final.pt"),
                                  "--output-dir", str(tmp_path / "video"), "--source-case", "fixed-repo"])
    events = []
    env = FakeEnv(events, stop=6, bad_tick=2 if failure == "bad_state" else None)
    provenance = {"checkpoint_sha256": "checkpoint", "run_manifest_sha256": "run",
                  "run_manifest": {"stage": "stand"}}
    actor = SimpleNamespace(to=lambda device: SimpleNamespace(eval=lambda: lambda obs: [[0.]]))

    def checked(*args):
        events.append("checked")
        if failure == "checkpoint":
            raise ValueError("checkpoint rejected")
        return actor, provenance

    manifest = {"contract_id": "v40", "contract_sha256": "contract", "asset_manifest_sha256": "asset"}
    train = SimpleNamespace(make_manifest=lambda *args: manifest, checked_checkpoint=checked, METADATA_KEYS=tuple(manifest))
    writer = FakeWriter(events, args.output_dir / "replay.mp4", fail=failure == "encoder", empty=failure == "empty_video")

    def save_image(path, frame):
        events.append("image_saved")
        path.write_bytes(b"fake PNG")

    def wrapper(raw):
        raw.reset()
        return raw

    class FastShutdown(BaseException):
        pass

    def app_close():
        summary = json.loads((args.output_dir / "summary.json").read_text())
        assert "writer_close" in events and "env_close" in events
        assert summary["recording_ok"] is (failure is None)
        assert summary["source_case"] == "fixed-repo" and "NOT footage" in summary["display_label"]
        assert summary["checkpoint_sha256"] == "checkpoint"
        if failure is None:
            assert events.index("image_saved") < events.index("writer_close") < events.index("env_close")
            assert summary["sim_duration_s"] == .06 and summary["frame_policy_ticks"] == [0, 4, 6]
            assert (args.output_dir / "replay.mp4").stat().st_size > 0
            assert summary["termination_reasons"] == ["tilt"]
        raise FastShutdown()

    monkeypatch.setattr(cli, "video_backend", lambda: (lambda path: writer, save_image, {}))
    monkeypatch.setattr(cli, "launch_recording_app", lambda args: SimpleNamespace(app=SimpleNamespace(close=app_close, is_running=lambda: True)))
    monkeypatch.setattr(cli, "make_recording_env", lambda args: env)
    monkeypatch.setattr(cli, "warm_renderer", lambda env: None)
    monkeypatch.setattr(cli, "source_identity", lambda root: {"changed": True} if failure == "source_change" else {})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", SimpleNamespace(RslRlVecEnvWrapper=wrapper))
    contract = {"timing": {"physics_dt": .005, "decimation": 2}}
    if failure == "checkpoint":
        assert cli.record(args, train, contract, {}, {}, {}) == 3
        assert events == ["checked"]
        assert not json.loads((args.output_dir / "summary.json").read_text())["recording_ok"]
    else:
        with pytest.raises(FastShutdown):
            cli.record(args, train, contract, {}, {}, {})
        if failure != "source_change":
            assert events.index("command") < events.index("reset")
            assert events.count("reset") == 2


def test_repo_root_loads_selected_sibling_and_hashes_untracked_source(cli, tmp_path):
    root = tmp_path / "old-source"
    (root / "scripts").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "scripts/train_v40.py").write_text("from pathlib import Path\nREPO_ROOT = Path(__file__).resolve().parents[1]\nVARIANT = 'old'\n")
    (root / "src/physics.py").write_text("WHEEL_LIMIT = 3.14159\n")
    # Existing project imports are intentionally rejected instead of mixing trees.
    with pytest.raises(RuntimeError, match="mixed repository"):
        cli.load_train(root)
    code = ("import importlib.util; from pathlib import Path; "
            f"s=importlib.util.spec_from_file_location('record', {str(ROOT / 'scripts/record_v40.py')!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            f"assert m.load_train(Path({str(root)!r})).VARIANT == 'old'")
    subprocess.run([sys.executable, "-S", "-c", code], check=True, timeout=30)
    before = cli.source_identity(root)
    (root / "src/physics.py").write_text("WHEEL_LIMIT = None\n")
    after = cli.source_identity(root)
    assert before["source_tree_sha256"] != after["source_tree_sha256"]
    assert "src/physics.py" in before["source_files_sha256"]


def test_existing_ffmpeg_streaming_roundtrip(cli, tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    factory, save_image, metadata = cli.video_backend()
    env = FakeEnv([], stop=6)
    summary = {"frames": 0, "frame_policy_ticks": [], "policy_steps": 0}
    with factory(tmp_path / "replay.mp4") as writer:
        cli.collect_video(env, env, lambda obs: [[0.]], 1000, lambda: True,
                          writer, save_image, tmp_path, summary)
    assert metadata["executable"] and (tmp_path / "replay.mp4").stat().st_size > 0
    with imageio.get_reader(str(tmp_path / "replay.mp4"), format="FFMPEG") as reader:
        assert reader.count_frames() == 3
        assert reader.get_meta_data()["fps"] == 25
        assert reader.get_data(2).shape == (720, 960, 3)
    assert int(imageio.imread(tmp_path / "final_frame.png")[0, 0, 0]) == 7


def test_missing_ffmpeg_is_clear_error_without_install(cli, monkeypatch):
    pytest.importorskip("imageio.v2")
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "/nonexistent/ffmpeg"))
    with pytest.raises(RuntimeError, match="existing FFmpeg unavailable; no installation attempted"):
        cli.video_backend()
