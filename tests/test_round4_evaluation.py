"""CSV statistics and mock PhysX event boundaries; no simulator/GPU/PPO run."""
import copy
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("round4_evaluator_test", ROOT / "scripts/round4/evaluate_policy.py")
ev = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ev)


def case(kind="single_push", episodes=1):
    return {"id": kind + "_h030", "kind": kind, "command": [0, 0, .30],
            "episodes": episodes, "episode_seconds": 20,
            "push": None if kind == "standing" else {"time_s": 5, "count_per_episode": 1,
                "method": "env.apply_velocity_impulse(delta_xy, env_ids)", "delta_xy_m_s": [.5, 0.]}}


def request():
    result = {"schema_version": 2, "initialization": "scratch", "parent": None, "policy_quality_verified": False,
              "random_push_enabled": False, "recovery": dict(ev.RECOVERY), "cases": []}
    for kind in ("standing", "single_push"):
        for height in (29, 30, 31, 32):
            item = case(kind, 3 if kind == "standing" else 20)
            item["id"], item["command"][2] = f"{kind}_h{height:03d}", height/100
            result["cases"].append(item)
    result["cases"] += [{"id": name, "kind": "tracking", "command": command.copy(), "episodes": 2,
                         "episode_seconds": 20, "push": None} for name, command in ev.TRACKING_COMMANDS.items()]
    return result


def csv_rows(episode=0, *, band_start=6.9, stop=20, termination=False, break_hold=False):
    for step in range(1, round(stop*100)+1):
        time = step/100
        good = time >= band_start and not (break_hold and 7.1 <= time <= 7.2)
        yield {"episode": episode, "step": step, "time_s": time, "finite": 1,
               "height_error_m": .009 if good else .02, "planar_speed_m_s": .049 if good else .2,
               "tilt_deg": 9.9 if good else 15., "terminated": int(termination and time == stop),
               "timeout": int(not termination and time == 20)}


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ev.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def event(**overrides):
    return {"episode": 0, "ending": "horizon", "push_applied": True, "push_time_s": 5., **overrides}


def test_request_recovery_protocol_is_explicit():
    ev.validate_request(request())
    old = request()
    old["recovery"].update(height_error_band_m=.005, planar_speed_band_m_s=.02,
                           steady_band_hold_s=.5, hold_must_finish_within_deadline=True)
    with pytest.raises(ValueError, match="recovery protocol mismatch"):
        ev.validate_request(old)


@pytest.mark.parametrize("start,expected", [(6.9, 1), (7., 1), (7.01, 0)])
def test_real_csv_hold_start_deadline_not_finish_deadline(tmp_path, start, expected):
    path = tmp_path / "telemetry.csv"
    write_rows(path, csv_rows(band_start=start))
    summary = ev.summarize_csv(path, case(), [event()])
    assert summary["metrics"]["recovered_within_2s_episodes"] == expected
    assert summary["metrics"]["failed_disturbed_episodes"] == 1-expected


def test_hold_must_be_continuous(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, csv_rows(band_start=6.9, break_hold=True))
    summary = ev.summarize_csv(path, case(), [event()])
    assert summary["metrics"]["recovered_within_2s_episodes"] == 0


@pytest.mark.parametrize("ending,termination", [("interrupted", False), ("termination", True)])
def test_failure_or_censoring_never_counts_as_recovered(tmp_path, ending, termination):
    path = tmp_path / "telemetry.csv"
    write_rows(path, csv_rows(band_start=5.1, stop=10, termination=termination))
    result = ev.summarize_csv(path, case(), [event(ending=ending)])
    assert result["metrics"]["recovered_within_2s_episodes"] == 0
    assert result["metrics"]["failed_disturbed_episodes"] == int(termination)
    assert result["metrics"]["censored_disturbed_episodes"] == int(not termination)


def test_pre_push_failure_is_not_in_disturbed_denominator(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, csv_rows(stop=4, termination=True))
    result = ev.summarize_csv(path, case(), [event(push_applied=False, ending="termination")])
    assert result["metrics"]["pre_push_failed_episodes"] == 1
    assert result["metrics"]["disturbed_episodes"] == 0
    assert result["metrics"]["failure_rate_bounds_all_disturbed"] is None


def test_nonfinite_observation_before_any_step_is_a_real_failure(tmp_path):
    path = tmp_path / "telemetry.csv"
    write_rows(path, [])
    result = ev.summarize_csv(path, case(), [event(push_applied=False, ending="nonfinite", observation_nonfinite=True)])
    assert result["metrics"]["pre_push_failed_episodes"] == 1
    assert result["metrics"]["censored_episodes"] == 0


def test_zero_failures_does_not_establish_one_percent():
    confidence = ev.binomial_information(0, 20)
    assert confidence["zero_failure_one_sided_95_upper"] == pytest.approx(.13910834)
    assert confidence["wilson_95"][1] > .01


class MockEnv:
    """Only the audited observation/impulse boundary, not simulated robot physics."""
    step_dt = .01
    device = "cpu"

    def __init__(self, terminate_first=False):
        self.episode = -1
        self.tick = 0
        self.events = []
        self.terminate_first = terminate_first
        self.extras = {"log": {"Push/events_count": 0}}
        self.vel = torch.zeros(1, 6)
        self.pose = torch.tensor([[0., 0., .3, 0., 0., 0., 1.]])

    def set_evaluation_command(self, command):
        self._evaluation_command_override = command
        self.events.append("command")

    def reset(self):
        self.episode += 1
        self.tick = 0
        self.vel.zero_()
        self.events.append("reset")
        return self._get_observations(), {}

    def _get_observations(self):
        self.events.append("observations")
        critic = torch.zeros(1, 29)
        critic[:, 25:28] = self.vel[:, :3]
        return {"policy": torch.zeros(1, 125), "critic": critic}

    def _read_root_impulse_state(self, _ids):
        return self.vel.clone(), self.pose.clone()

    def apply_velocity_impulse(self, delta, ids):
        self.events.append(("push", self.episode, self.tick))
        self.vel[:, :2] += delta
        return {"events_count": 1, "realized_delta_xy_m_s": delta.clone()}

    def step(self, actions):
        self.tick += 1
        term = self.terminate_first and self.episode == 0 and self.tick == 400
        timeout = self.tick == 1999
        height = .1 if term else .3
        self.vel *= .95
        self.snapshot = {"sample_kind": "pre_reset", "state_phase": "pre_reset", "policy_tick": self.tick,
            "root_com_lin_vel_b_m_s": self.vel[:, :3].clone(), "root_link_pos_w_m": torch.tensor([[0., 0., height]]),
            "root_com_ang_vel_b_rad_s": self.vel[:, 3:].clone(), "projected_gravity_b": torch.tensor([[0., 0., -1.]]),
            "wheel_net_force_max_n": torch.tensor([[50., 50.]]), "non_wheel_net_force_max_n": torch.tensor([0.]),
            "base_visual_clearance_lower_bound_m": torch.tensor([.1]), "height_m": torch.tensor([height]),
            "terminated": torch.tensor([term]), "timeout": torch.tensor([timeout]),
            "diagnostic_flags": {"nonfinite": torch.tensor([False])},
            "termination_flags": {"tilt": torch.tensor([term])},
            "command": torch.tensor([self._evaluation_command_override])}
        return self._get_observations(), torch.zeros(1), torch.tensor([term]), torch.tensor([timeout]), {}

    def get_evaluation_snapshot(self):
        return copy.deepcopy(self.snapshot)


class MockPolicy:
    def __init__(self, env):
        self.env = env

    def run(self, outputs, inputs):
        self.env.events.append("policy")
        return [np.zeros((1, 6), dtype=np.float32)]


def test_push_at_five_seconds_once_and_refresh_before_next_action(tmp_path):
    env = MockEnv(terminate_first=True)
    result = ev.run_case(env, MockPolicy(env), case(episodes=2), tmp_path / "case", lambda: None)
    pushes = [item for item in env.events if isinstance(item, tuple)]
    assert pushes == [("push", 1, 500)]
    index = env.events.index(pushes[0])
    assert env.events[index+1:index+3] == ["observations", "policy"]
    assert result["metrics"]["pre_push_failed_episodes"] == 1
    assert result["metrics"]["disturbed_episodes"] == result["metrics"]["recovered_within_2s_episodes"] == 1
    evidence = result["episodes"][1]
    assert evidence["after_com_velocity_w_m_s_rad_s"][0]-evidence["before_com_velocity_w_m_s_rad_s"][0] == .5
    assert evidence["realized_delta_xy_m_s"] == [.5, 0.]
    assert evidence["critic_velocity_refreshed"]
    # The failed pre-reset height was recorded, not an auto-reset's healthy height.
    assert result["episodes"][0]["height_abs_error_max_m"] > .19


def test_timeout_counts_three_independent_standing_windows(tmp_path):
    env = MockEnv()
    result = ev.run_case(env, MockPolicy(env), case("standing", 3), tmp_path / "standing", lambda: None)
    assert result["metrics"]["timeout_events"] == 3
    assert result["metrics"]["requested_accumulated_seconds"] == 60
    assert result["metrics"]["observed_sim_seconds"] == pytest.approx(59.97)
    assert not any(isinstance(item, tuple) for item in env.events)


def test_interruption_preserves_partial_csv_and_censored_trial(tmp_path):
    env = MockEnv()
    def check():
        if env.tick >= 550:
            raise InterruptedError("test timeout")
    result = ev.run_case(env, MockPolicy(env), case(), tmp_path / "interrupted", check)
    assert result["status"] == "failed"
    assert result["metrics"]["disturbed_episodes"] == result["metrics"]["censored_disturbed_episodes"] == 1
    assert result["metrics"]["recovered_within_2s_episodes"] == 0
    assert (tmp_path / "interrupted/telemetry.csv").stat().st_size > 100


def test_stale_critic_after_push_is_rejected():
    env = MockEnv()
    obs, _ = env.reset()
    env._get_observations = lambda: obs
    evidence = {}
    with pytest.raises(RuntimeError, match="stale"):
        ev.apply_push(env, obs, [.5, 0.], evidence=evidence)
    assert evidence["push_applied"] is True
    assert evidence["realized_delta_xy_m_s"] == [.5, 0.]


def test_effective_materials_use_actual_wheel_readback_not_nominal_assumption():
    report = {"passed": True, "bindings": [{"collider": "/World/ground/plane", "usd_combine_modes": ["average", "average"],
                "usd_coefficients": [.5, .5, 0.]}], "wheel_env_ids": [0, 0], "wheel_body_paths": ["left", "right"],
                "wheel_physx_coefficients": [[[.9, .7, 0.]], [[.9, .7, 0.]]]}
    values = ev.effective_materials(report)
    assert values[0]["static_friction"] == .7 and values[0]["dynamic_friction"] == .6


def test_failed_final_gate_publishes_hashed_failure_not_placeholder(tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request()))
    def reject(_path):
        raise ValueError("final export is not verified")
    monkeypatch.setattr(ev, "verify_inputs", reject)
    output = tmp_path / "reports"
    assert ev.main(["--request", str(path), "--output", str(output)]) == 2
    result = json.loads((output / "result.json").read_text())
    marker = json.loads((output / "report_manifest.json").read_text())
    assert result["status"] == "failed" and not result["evaluation_success"]
    assert len(result["cases"]) == 12 and all(item["status"] == "failed" for item in result["cases"])
    assert marker["request_sha256"] == ev.file_record(path)["sha256"]
    assert "report_manifest.json" not in [item["path"] for item in marker["files"]]
    for item in marker["files"]:
        assert ev.file_record(output / item["path"]) == {key: item[key] for key in ("size", "sha256")}
    with pytest.raises(FileExistsError):
        ev.main(["--request", str(path), "--output", str(output)])


@pytest.mark.parametrize("change", ["missing", "push", "command", "episodes", "v1"])
def test_tracking_request_cannot_silently_drop_or_change_cases(change):
    data = request()
    if change == "missing":
        data["cases"].pop()
    elif change == "v1":
        data["schema_version"] = 1
    elif change == "push":
        data["cases"][-1]["push"] = case()["push"]
    elif change == "command":
        data["cases"][-1]["command"] = [0, -2, .32]
    else:
        data["cases"][-1]["episodes"] = 1
    with pytest.raises(ValueError):
        ev.validate_request(data)


def test_tracking_csv_full_and_per_episode_steady_errors_include_failed_censored_samples(tmp_path):
    item = request()["cases"][8]
    rows = []
    for episode, actual in enumerate(([0, 1, 2, 3], [-3, 0, 3, 6])):
        for index, (t, vx) in enumerate(zip([1, 2, 2.01, 3], actual)):
            rows.append({"episode": episode, "step": index+1, "time_s": t, "finite": 1,
                         "height_error_m": 0, "planar_speed_m_s": abs(vx), "tilt_deg": 0,
                         "vx_m_s": vx, "wz_rad_s": index*2*(1 if episode == 0 else -1),
                         "terminated": int(episode == 0 and t == 3), "timeout": 0,
                         "command_vx_m_s": 3, "command_wz_rad_s": 0,
                         "diagnostic_flags_json": json.dumps({"non_wheel_contact": t == 3}),
                         "termination_flags_json": json.dumps({"tilt": episode == 0 and t == 3}),
                         "non_wheel_net_force_n": 12 if t == 3 else 0,
                         "left_wheel_net_force_n": 50, "right_wheel_net_force_n": 60})
    path = tmp_path / "tracking.csv"
    write_rows(path, rows)
    result = ev.summarize_csv(path, item, [event(episode=0, push_applied=False, ending="termination"),
                                             event(episode=1, push_applied=False, ending="interrupted")])
    stats = result["metrics"]["tracking"]
    assert stats["full"]["vx"]["mae"] == 2.25
    assert stats["full"]["vx"]["p95_abs_error"] == pytest.approx(4.95)
    assert stats["full"]["wz"]["actual_mean"] == 0
    assert stats["steady_after_2s"]["valid_samples"] == 4  # Excludes t==2 in BOTH episodes.
    assert stats["steady_after_2s"]["vx"]["mae"] == 1
    assert stats["steady_after_2s"]["vx"]["p95_abs_error"] == pytest.approx(2.7)
    assert stats["steady_after_2s"]["wz"]["mae"] == 5
    assert result["episodes"][0]["tracking"]["steady_after_2s"]["vx"]["mae"] == .5
    assert result["episodes"][1]["tracking"]["steady_after_2s"]["vx"]["mae"] == 1.5
    assert stats["diagnostic_sample_counts"]["non_wheel_contact"] == 2
    assert stats["termination_sample_counts"]["tilt"] == 1
    assert stats["contact_peak_n"]["non_wheel_net_force_n"] == 12
    assert result["status"] == "failed" and result["metrics"]["censored_episodes"] == 1


def test_tracking_runs_without_push_and_records_signed_command(tmp_path):
    item = request()["cases"][-1]
    env = MockEnv()
    result = ev.run_case(env, MockPolicy(env), item, tmp_path / "tracking", lambda: None)
    assert not any(isinstance(entry, tuple) for entry in env.events)
    assert result["metrics"]["disturbed_episodes"] == 0
    stats = result["metrics"]["tracking"]["steady_after_2s"]
    assert stats["wz"]["command"] == -6 and stats["wz"]["actual_mean"] == 0
    assert stats["wz"]["mae"] == stats["wz"]["p95_abs_error"] == 6
    with (tmp_path / "tracking/telemetry.csv").open() as stream:
        first = next(csv.DictReader(stream))
    assert float(first["command_wz_rad_s"]) == -6


def test_tracking_interrupted_before_steady_window_has_no_fake_zero_error(tmp_path):
    item = request()["cases"][8]
    env = MockEnv()
    def check():
        if env.tick >= 100:
            raise InterruptedError("budget exhausted")
    result = ev.run_case(env, MockPolicy(env), item, tmp_path / "tracking", check)
    stats = result["metrics"]["tracking"]["steady_after_2s"]
    assert stats["valid_samples"] == 0 and stats["vx"]["mae"] is None
    assert stats["wz"]["p95_abs_error"] is None
    assert result["status"] == "failed" and result["metrics"]["unstarted_episodes"] == 1
