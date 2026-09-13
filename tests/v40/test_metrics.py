"""CPU synthetic trajectories/clock/tensor-adapter tests, NOT physics or policy results."""
from __future__ import annotations

import ast
import builtins
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

from wheeled_algo.v40_metrics import (
    BATCH_FIELDS, LIMITATIONS, REASONS, InvalidTrajectory, Thresholds, aggregate_cases,
    collect_case, default_cases, diagnostic_json, evaluate_trajectory, require_trained_stage,
    snapshot_to_record, validate_command, write_json_exclusive,
)

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / "src/wheeled_tasks/direct/v40_serial/env.py"


@pytest.fixture
def contract(request):
    # Read-only fixture: bypasses no physical gate and never instantiates an environment.
    version = getattr(request, "param", "v1")
    return json.loads((ROOT / f"contracts/own_v40_{version}.json").read_text())


def record(contract, tick, *, command=(0., 0., .32), x=0.):
    return {
        "schema_version": 1, "sample_kind": "initial" if tick == 0 else "pre_reset",
        "policy_tick": tick + 17, "physics_steps": tick*2 + 38,
        "time_s": (tick+17)*.01, "sim_time_s": (tick*2+38)*.005,
        "episode_step": tick, "episode_time_s": tick*.01,
        "policy_dt_s": .01, "physics_dt_s": .005, "contact_history_samples": 2,
        "command": list(command), "root_link_pos_w_m": [0., 0., command[2]],
        "root_link_quat_wxyz": [1., 0., 0., 0.],
        "root_com_lin_vel_b_m_s": [command[0], 0., 0.],
        "root_com_ang_vel_b_rad_s": [0., 0., command[1]], "projected_gravity_b": [0., 0., -1.],
        "height_m": command[2], "joint_pos_rad": list(contract["joints"]["nominal_positions"]),
        "joint_vel_rad_s": [0.]*6, "sim_joint_effort_nm": [0.]*6,
        "wheel_axis_midpoint_w_m": [x, 0., .05], "wheel_net_force_max_n": [20., 21.],
        "non_wheel_net_force_max_n": 0., "base_visual_clearance_lower_bound_m": .20,
        "terminated": False, "timeout": False,
        "termination_flags": {} if tick == 0 else {reason: False for reason in REASONS},
    }


def trajectory(contract, n=1000, **kwargs):
    return [record(contract, i, **kwargs) for i in range(n+1)]


def test_ten_second_perfect_synthetic_stand(contract):
    result = evaluate_trajectory(trajectory(contract), contract, "stand", (0, 0, .32))
    assert result["passed"] and result["sustained_valid_duration_s"] == 10.
    assert result["observed_duration_s"] == 10. and result["sample_count"] == 1000
    assert result["end_reason"] == "horizon_reached"
    assert result["tracking"]["height"]["rmse"] == 0
    assert result["limitations"]["perturbations"] == "not_tested"
    assert result["limitations"]["video"] == "not_recorded"


@pytest.mark.parametrize("contract", ["v1", "v2"], indirect=True)
def test_initial_anchor_never_rebased_after_spinup(contract):
    rows = trajectory(contract, command=(0, 1., .30))
    for row in rows:
        # Movement happens before steady window then stops; must still fail 5cm goal.
        row["wheel_axis_midpoint_w_m"][0] = 12.0 + min(row["episode_time_s"]/.5, 1.)*.08
    result = evaluate_trajectory(rows, contract, "locomotion", (0, 1., .30))
    assert result["wheel_axis_drift"]["anchor_xy_m"] == [12., 0.]
    assert result["wheel_axis_drift"]["max_m"] == pytest.approx(.08)
    assert result["wheel_axis_drift"]["final_m"] == pytest.approx(.08)
    assert "wheel_axis_drift" in result["failure_reasons"]
    assert not result["passed"]


def test_max_drift_not_only_final_drift(contract):
    rows = trajectory(contract)
    rows[70]["wheel_axis_midpoint_w_m"][0] = .06
    result = evaluate_trajectory(rows, contract, "stand", (0, 0, .32))
    assert result["wheel_axis_drift"]["final_m"] == 0.
    assert not result["passed"]


def test_translating_command_drift_is_descriptive_not_stationary_gate(contract):
    rows = trajectory(contract, command=(.5, 0, .30))
    for row in rows:
        row["wheel_axis_midpoint_w_m"][0] = .5*row["episode_time_s"]
    result = evaluate_trajectory(rows, contract, "locomotion", (.5, 0, .30))
    assert result["passed"] and not result["wheel_axis_drift"]["applicable"]
    assert result["wheel_axis_drift"]["final_m"] == 5.


@pytest.mark.parametrize("contract", ["v1", "v2"], indirect=True)
def test_rmse_and_signed_steady_bias_with_si_units(contract):
    rows = trajectory(contract, command=(.2, .4, .30))
    for row in rows:
        row["root_com_lin_vel_b_m_s"][0] += .1
        row["root_com_ang_vel_b_rad_s"][2] -= .2
        row["height_m"] += .015
    result = evaluate_trajectory(rows, contract, "locomotion", (.2, .4, .30))
    for name, expected in (("vx", .1), ("wz", -.2), ("height", .015)):
        assert result["tracking"][name]["rmse"] == pytest.approx(abs(expected))
        assert result["tracking"][name]["steady_bias"] == pytest.approx(expected)
    assert set(result["failure_reasons"]) == {"vx_steady_bias", "wz_steady_bias", "height_steady_bias"}


@pytest.mark.parametrize("count", [0, 1, 100, 999])
def test_short_episode_cannot_pass(contract, count):
    result = evaluate_trajectory(trajectory(contract, count), contract, "stand", (0, 0, .32))
    assert not result["passed"] and "short_episode" in result["failure_reasons"]
    assert result["observed_duration_s"] == count*.01


@pytest.mark.parametrize("field,value", [
    ("height_m", float("nan")), ("sim_time_s", float("inf")),
    ("joint_vel_rad_s", [0, 0, 0, float("nan"), 0, 0]),
    ("sim_joint_effort_nm", [float("inf")]*6), ("terminated", 0),
])
def test_invalid_numeric_or_types_rejected(contract, field, value):
    rows = trajectory(contract, 2)
    rows[1][field] = value
    with pytest.raises(InvalidTrajectory):
        evaluate_trajectory(rows, contract, "stand", (0, 0, .32))


@pytest.mark.parametrize("field", ["wheel_axis_midpoint_w_m", "sim_joint_effort_nm", "termination_flags"])
def test_missing_fields_rejected(contract, field):
    rows = trajectory(contract, 1)
    del rows[1][field]
    with pytest.raises(InvalidTrajectory, match="missing"):
        evaluate_trajectory(rows, contract, "stand", (0, 0, .32))


@pytest.mark.parametrize("field,value", [
    ("time_s", .01), ("policy_tick", 16), ("physics_steps", 39),
    ("episode_time_s", 10.), ("episode_step", 0), ("policy_dt_s", 10.),
    ("sim_time_s", 200.), ("contact_history_samples", 1),
])
def test_clock_backwards_milliseconds_and_reset_leakage_rejected(contract, field, value):
    rows = trajectory(contract, 2)
    rows[1][field] = value
    with pytest.raises(InvalidTrajectory):
        evaluate_trajectory(rows, contract, "stand", (0, 0, .32))


@pytest.mark.parametrize("terminated,timeout,end_reason", [(True, False, "terminated"), (False, True, "timeout"), (True, True, "terminated_and_timeout")])
def test_termination_and_timeout_always_classified_even_on_last_tick(contract, terminated, timeout, end_reason):
    rows = trajectory(contract)
    rows[-1].update(terminated=terminated, timeout=timeout)
    result = evaluate_trajectory(rows, contract, "stand", (0, 0, .32))
    assert not result["passed"] and result["end_reason"] == end_reason
    assert result["sustained_valid_duration_s"] == pytest.approx(9.99)
    assert result["first_failure_episode_time_s"] == 10.


def test_never_average_failure_away_or_accept_post_terminal_records(contract):
    rows = trajectory(contract)
    passed = evaluate_trajectory(rows, contract, "stand", (0, 0, .32))
    rows[-1]["terminated"] = True
    rows[-1]["non_wheel_net_force_max_n"] = 4.
    rows[-1]["termination_flags"]["non_wheel_contact"] = True
    failed = evaluate_trajectory(rows, contract, "stand", (0, 0, .32))
    suite = aggregate_cases([dict(passed, case_id="good")]*10 + [dict(failed, case_id="fall")])
    assert not suite["passed"] and suite["failed_case_ids"] == ["fall"]
    assert failed["non_wheel_contact_policy_ticks"] == 1
    rows.append(record(contract, 1001))
    with pytest.raises(InvalidTrajectory, match="after terminal"):
        evaluate_trajectory(rows, contract, "stand", (0, 0, .32))
    assert not aggregate_cases([])["passed"]


@pytest.mark.parametrize("contract", ["v1", "v2"], indirect=True)
def test_joint_prior_and_bbox_metrics_not_hardware_claims(contract):
    rows = trajectory(contract)
    rows[1]["sim_joint_effort_nm"][1] = contract["actuators"]["leg"]["effort_limit"]*1.1
    rows[1]["joint_pos_rad"][1] = contract["joints"]["knee_hard_limits"]["L_joint2"][1] + .02
    rows[1]["base_visual_clearance_lower_bound_m"] = -.01
    result = evaluate_trajectory(rows, contract, "stand", (0, 0, .32))
    assert not result["passed"]
    assert result["joints"]["L_joint2"]["peak_effort_prior_utilization"] == pytest.approx(1.1)
    assert result["joints"]["L_joint2"]["minimum_hard_limit_margin_rad"] == pytest.approx(-.02)
    assert {"knee_limit", "effort_prior_utilization", "base_visual_bounds_ground"} <= set(result["failure_reasons"])
    assert "prior" in LIMITATIONS["effort"] and "mesh" in LIMITATIONS["clearance"]


@pytest.mark.parametrize("contract", ["v1", "v2"], indirect=True)
@pytest.mark.parametrize("stage,command", [("stand", (0, 0, .30)), ("height", (.1, 0, .30)), ("locomotion", (0, 12.57, .30)), ("locomotion", (0, 0, .4)), ("locomotion", (float("nan"), 0, .3)), ("unknown", (0, 0, .32))])
def test_out_of_domain_commands_rejected(contract, stage, command):
    with pytest.raises(InvalidTrajectory):
        validate_command(contract, stage, command)


@pytest.mark.parametrize("contract", ["v2"], indirect=True)
@pytest.mark.parametrize("command", [(2., 2., .28), (-2., -2., .32), (2., 0., .30), (0., -2., .30)])
def test_v2_locomotion_edges_pass_synthetic_evaluation(contract, command):
    rows = trajectory(contract, command=command)
    for row in rows:
        row["wheel_axis_midpoint_w_m"][0] = command[0] * row["episode_time_s"]
    result = evaluate_trajectory(rows, contract, "locomotion", command)
    assert result["passed"] and result["stage"] == "locomotion"
    assert result["command"] == list(command)
    assert result["thresholds"] == Thresholds().to_dict()
    assert all(item["rmse"] == item["steady_bias"] == 0 for item in result["tracking"].values())


@pytest.mark.parametrize("contract", ["v2"], indirect=True)
def test_v2_fixed_command_cannot_change_in_records(contract):
    command = (2., 0., .30)
    rows = trajectory(contract, 2, command=command)
    rows[1]["command"][0] = 0.
    with pytest.raises(InvalidTrajectory, match="fixed command or mixed cases"):
        evaluate_trajectory(rows, contract, "locomotion", command)


@pytest.mark.parametrize("contract", ["v1", "v2"], indirect=True)
@pytest.mark.parametrize("component", [0, 1, 2], ids=["vx", "wz", "height"])
@pytest.mark.parametrize("side", [0, 1], ids=["below", "above"])
def test_command_just_outside_actual_stage_rejected(contract, component, side):
    key = ("vx", "wz", "height")[component]
    edge = contract["commands"]["stages"]["locomotion"][key][side]
    command = [0., 0., .30]
    command[component] = edge
    assert validate_command(contract, "locomotion", command) == tuple(command)
    command[component] = math.nextafter(edge, -math.inf if side == 0 else math.inf)
    with pytest.raises(InvalidTrajectory, match=f"command {key}"):
        validate_command(contract, "locomotion", command)


@pytest.mark.parametrize("contract", ["v1", "v2"], indirect=True)
@pytest.mark.parametrize("key,component,bounds", [("vx", 0, [-.2, .3]), ("wz", 1, [-.4, .6]), ("height", 2, [.29, .31])])
def test_reduced_stage_ranges_and_invalid_bounds_are_not_relaxed(contract, key, component, bounds):
    stage = contract["commands"]["stages"]["locomotion"]
    stage[key] = bounds
    command = [0., 0., .30]
    command[component] = bounds[1]
    assert validate_command(contract, "locomotion", command) == tuple(command)
    command[component] += .01
    with pytest.raises(InvalidTrajectory, match=f"command {key}"):
        validate_command(contract, "locomotion", command)
    command[component] = bounds[0]
    for invalid in (bounds[::-1], [float("nan"), bounds[1]], [bounds[0], float("inf")]):
        stage[key] = invalid
        with pytest.raises(InvalidTrajectory):
            validate_command(contract, "locomotion", command)


@pytest.mark.parametrize("key,bounds", [("vx", [-2., 2.]), ("wz", [-2., 2.]), ("height", [.27, .33])])
def test_v1_contract_cannot_expand_absolute_domain(contract, key, bounds):
    contract["commands"]["stages"]["locomotion"][key] = bounds
    with pytest.raises(InvalidTrajectory, match=f"command {key}"):
        validate_command(contract, "locomotion", (0., 0., .30))


@pytest.mark.parametrize("contract", ["v2"], indirect=True)
def test_v2_contract_cannot_expand_approved_height_domain(contract):
    contract["commands"]["stages"]["locomotion"]["height"] = [.27, .33]
    with pytest.raises(InvalidTrajectory, match="command height"):
        validate_command(contract, "locomotion", (2., 0., .30))


@pytest.mark.parametrize("contract", ["v1", "v2"], indirect=True)
@pytest.mark.parametrize("identity", [None, "unknown", "own-v40-jointspace-h5-v3"])
def test_unknown_contract_identity_rejected(contract, identity):
    if identity is None:
        del contract["contract_id"]
    else:
        contract["contract_id"] = identity
    with pytest.raises(InvalidTrajectory, match="contract identity"):
        validate_command(contract, "locomotion", (0., 0., .30))


@pytest.mark.parametrize("contract", ["v1", "v2"], indirect=True)
def test_stage_not_assumed_from_checkpoint(contract):
    require_trained_stage({"stage": "height"}, "height")
    for manifest in ({}, {"stage": "stand"}, {"stage": "locomotion"}):
        with pytest.raises(InvalidTrajectory, match="trained stage"):
            require_trained_stage(manifest, "height")
    assert len(default_cases(contract, "stand")) == 1
    assert all(abs(c["command"][1]) <= 1. for c in default_cases(contract, "locomotion"))


def test_thresholds_and_overwrite_guard(tmp_path):
    for kwargs in ({"duration_s": float("nan")}, {"duration_s": 11.}, {"steady_start_s": 10.}, {"height_rmse_m": -.01}):
        with pytest.raises(InvalidTrajectory):
            Thresholds(**kwargs)
    path = tmp_path / "summary.json"
    write_json_exclusive(path, {"passed": False})
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"passed": True})
    assert json.loads(path.read_text()) == {"passed": False}
    with pytest.raises(ValueError):
        write_json_exclusive(tmp_path / "invalid.json", {"nan": float("nan")})
    assert not (tmp_path / "invalid.json").exists()
    assert diagnostic_json({"nan": float("nan")}) == {"nan": {"invalid_numeric": "nan"}}


def batch(record):
    return {key: ({name: [flag] for name, flag in value.items()} if key == "termination_flags" else
                  [deepcopy(value)] if key in BATCH_FIELDS else value) for key, value in record.items()}


class FakeAutoReset:
    """No physics: reproduce snapshot-before-reset / observations-after-reset ordering."""
    def __init__(self, contract):
        self.contract = contract
        self.calls = 0
        self.live_post_reset_height = .32

    def capture_evaluation_initial_snapshot(self):
        return batch(record(self.contract, 0, x=3.))

    def get_observations(self):
        return {"policy": "fake-observation"}

    def step(self, action):
        self.calls += 1
        end = record(self.contract, self.calls, x=3.08)
        end.update(terminated=True, height_m=.10, base_visual_clearance_lower_bound_m=-.01)
        end["termination_flags"]["low_height"] = True
        self.cached = batch(end)
        self.live_post_reset_height = .32  # Must NEVER be used by collector as terminal state.
        return {"policy": "reset-observation"}, 1000., [1], {}

    def get_evaluation_snapshot(self):
        return deepcopy(self.cached)


def test_collector_uses_terminal_snapshot_and_stops_before_second_reset_episode(contract):
    fake = FakeAutoReset(contract)
    records = []
    reason = collect_case(fake, fake, lambda obs: "mean", 1000, lambda: True, records)
    assert reason == "terminated" and fake.calls == 1
    assert records[-1]["height_m"] == .10 and fake.live_post_reset_height == .32
    result = evaluate_trajectory(records, contract, "stand", (0, 0, .32))
    assert not result["passed"] and "low_height" in result["failure_reasons"]
    assert result["wheel_axis_drift"]["final_m"] == pytest.approx(.08)


def test_fake_application_clock_stop_is_short_not_pass(contract):
    fake = FakeAutoReset(contract)
    records = []
    assert collect_case(fake, fake, lambda _: None, 1000, lambda: False, records) == "application_stopped"
    assert not evaluate_trajectory(records, contract, "stand", (0, 0, .32))["passed"]
    assert fake.calls == 0


def isolated_methods(*names):
    """Execute only source tensor adapters; no Isaac class imported/instantiated."""
    torch = pytest.importorskip("torch")
    cls = next(n for n in ast.parse(ENV.read_text()).body if isinstance(n, ast.ClassDef))
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"torch": torch, "math": math, "deepcopy": deepcopy,
                 "WHEEL_BODY_NAMES": ["L_link3", "R_link3"]}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "cpu_only_v40_adapters", "exec"), namespace)
    return torch, namespace


def test_command_override_sampler_does_not_consume_rng_and_cannot_exceed_stage(contract):
    torch, methods = isolated_methods("set_evaluation_command", "_sample_commands")
    fake = types.SimpleNamespace(contract=contract, cfg=types.SimpleNamespace(stage="height"),
                                 commands=torch.zeros(1, 3), device="cpu", _command_period_ticks=300,
                                 _command_ticks_left=torch.zeros(1, dtype=torch.long),
                                 _commands_due=torch.ones(1, dtype=torch.bool), _evaluation_command_override=None)
    setter, sampler = methods["set_evaluation_command"], methods["_sample_commands"]
    setter(fake, (0., 0., .28))
    assert fake._evaluation_command_pending
    ids = torch.tensor([0])
    before = torch.random.get_rng_state().clone()
    for _ in range(3):
        fake.commands.zero_()
        sampler(fake, ids)  # same code used in reset and when observations mark resample due
        torch.testing.assert_close(fake.commands, torch.tensor([[0., 0., .28]]))
    assert torch.equal(before, torch.random.get_rng_state())
    with pytest.raises(InvalidTrajectory):
        setter(fake, (0., 1., .28))
    setter(fake, None)
    assert fake._evaluation_command_override is None
    sampler(fake, ids)
    assert not torch.equal(before, torch.random.get_rng_state())


def test_snapshot_clones_link_origins_effort_and_termination_before_live_reset(contract):
    torch, methods = isolated_methods("_make_evaluation_snapshot", "get_evaluation_snapshot", "_get_dones")
    # Lab 3 exposes ProxyArray.torch views and XYZW link poses.
    pose = torch.tensor([[1., 2., .10, 0., 2**-.5, 0., 2**-.5]])
    tensors = dict(
        root_link_pose_w=pose, root_link_pos_w=pose[:, :3], root_link_quat_w=pose[:, 3:],
        root_com_vel_w=torch.zeros(1, 6), root_com_lin_vel_b=torch.zeros(1, 3),
        root_com_ang_vel_b=torch.zeros(1, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]]),
        body_link_pos_w=torch.tensor([[[1., 1., .05], [3., 1., .05]]]),
        body_com_pos_w=torch.full((1, 2, 3), 99.), applied_torque=torch.full((1, 6), 2.),
    )
    data = types.SimpleNamespace(**{name: types.SimpleNamespace(torch=value) for name, value in tensors.items()})
    q = torch.tensor([contract["joints"]["nominal_positions"]])
    contact = torch.zeros(1, 7)
    fake = types.SimpleNamespace(
        robot=types.SimpleNamespace(data=data, body_names=["L_link3", "R_link3"]),
        num_envs=1, device="cpu",
        cfg=types.SimpleNamespace(wheel_slip_diagnostics=False),
        contact_sensor=types.SimpleNamespace(data=types.SimpleNamespace(
            net_forces_w_history=types.SimpleNamespace(torch=torch.zeros(1, 2, 7, 3)))),
        _named_indices=lambda actual, requested, kind: torch.tensor([actual.index(n) for n in requested]),
        _joint_state=lambda: (q, torch.zeros_like(q)), _joint_ids=torch.arange(6),
        _wheel_body_ids=torch.tensor([5, 6]), _non_wheel_body_ids=torch.arange(5),
        _base_height=lambda: data.root_link_pos_w.torch[:, 2], _base_visual_clearance=lambda: torch.tensor([-.01]),
        _contact_magnitudes=lambda: contact, contract=contract, _invalid_actions=torch.tensor([False]),
        _knee_ids=torch.tensor([1, 4]), _knee_limits=torch.tensor(list(contract["joints"]["knee_hard_limits"].values())),
        common_step_counter=1, _sim_step_counter=2, episode_length_buf=torch.tensor([1]),
        step_dt=.01, physics_dt=.005, max_episode_length=2000, extras={},
        commands=torch.tensor([[0., 0., .32]]), torques=torch.zeros(1, 6),
        scene=types.SimpleNamespace(env_origins=torch.zeros(1, 3)),
        _evaluation_command_override=(0., 0., .32), _evaluation_snapshot=None,
    )
    fake._make_evaluation_snapshot = types.MethodType(methods["_make_evaluation_snapshot"], fake)
    methods["_get_dones"](fake)
    first = methods["get_evaluation_snapshot"](fake)
    assert first["terminated"].item() and first["height_m"].item() == pytest.approx(.10)
    assert first["wheel_axis_midpoint_w_m"].tolist() == [[2., 1., pytest.approx(.05)]]
    assert first["sim_joint_effort_nm"].tolist() == [[2.]*6]  # not commands/torques or zeros
    torch.testing.assert_close(first["root_link_quat_wxyz"], torch.tensor([[2**-.5, 0., 2**-.5, 0.]]))
    data.root_link_pos_w.torch[:, 2] = .32
    data.applied_torque.torch.zero_()
    fake.episode_length_buf.zero_()  # stand-in for auto-reset mutating live state
    methods["_get_dones"](fake)  # duplicate same-tick request MUST NOT overwrite cached terminal
    second = methods["get_evaluation_snapshot"](fake)
    assert second["height_m"].item() == pytest.approx(.10) and second["episode_step"].item() == 1
    first["height_m"].fill_(99.)
    assert methods["get_evaluation_snapshot"](fake)["height_m"].item() == pytest.approx(.10)
    assert snapshot_to_record(second)["terminated"] is True


def test_pending_override_blocks_observation_before_history_update():
    _, methods = isolated_methods("_get_observations")
    with pytest.raises(RuntimeError, match="reset required"):
        methods["_get_observations"](types.SimpleNamespace(_evaluation_command_pending=True))
    # Existing test_env_contract_static validates repeated same-tick HistoryStack observations.
    cls = next(n for n in ast.parse(ENV.read_text()).body if isinstance(n, ast.ClassDef))
    reset = ast.unparse(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_reset_idx"))
    assert "self._sample_commands(env_ids)" in reset
    assert "self._evaluation_snapshot =" not in reset


def load_cli(monkeypatch, script="evaluate_v40.py"):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("v40_evaluation_cpu_test", ROOT / "scripts" / script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli


def test_cli_defaults_to_preflight_and_requires_explicit_research(monkeypatch):
    cli = load_cli(monkeypatch)
    assert cli.parse_arguments([]).preflight_only
    assert cli.parse_arguments(["--apply", "--research", "--preflight-only"]).preflight_only
    for args in (["--apply"], ["--apply", "--research"], ["--num-envs", "2"]):
        with pytest.raises(SystemExit) as exc:
            cli.parse_arguments(args)
        assert exc.value.code == 2


def test_cli_preflight_never_starts_sim_or_creates_outputs(monkeypatch, tmp_path, contract):
    cli = load_cli(monkeypatch)
    monkeypatch.setattr(cli, "preflight", lambda args: ({"blockers": ["physical gate remains blocked"], "simulation_started": False}, contract, None))
    monkeypatch.setattr(cli, "make_env", lambda args: pytest.fail("Sim launch forbidden"))
    assert cli.main(["--output-dir", str(tmp_path / "new")]) == 2
    assert not (tmp_path / "new").exists()
    assert "isaaclab.app" not in sys.modules


def test_cli_rejects_checkpoint_stage_mismatch_before_sim(monkeypatch, contract, tmp_path):
    cli = load_cli(monkeypatch)
    monkeypatch.setattr(cli, "preflight", lambda args: ({"blockers": [], "simulation_started": False}, contract, {}))
    monkeypatch.setattr(cli, "make_manifest", lambda *args: {})
    monkeypatch.setattr(cli, "checked_checkpoint", lambda *args: (object(), {"run_manifest": {"stage": "height"}}))
    assert cli.main(["--checkpoint", str(tmp_path / "fake.pt")]) == 2
    assert "isaaclab.app" not in sys.modules


@pytest.mark.parametrize("contract", ["v2"], indirect=True)
@pytest.mark.parametrize("script", ["evaluate_v40.py", "record_v40.py"])
@pytest.mark.parametrize("mismatch", [None, "stage", "contract_id", "contract_sha256", "asset_manifest_sha256"])
def test_v2_eval_and_record_require_exact_checkpoint_identity(monkeypatch, contract, tmp_path, script, mismatch, capsys):
    """Real checkpoint validation; stop before simulator/video startup even on success."""
    torch = pytest.importorskip("torch")
    from test_export import make_checkpoint
    from wheeled_tasks.v40.contract import audit_asset

    cli = load_cli(monkeypatch, script)
    checkpoint_path = tmp_path / "model.pt"
    output = tmp_path / "output"
    arguments = ["--apply", "--research", "--stage", "locomotion", "--command", "2", "0", ".30",
                 "--contract", str(ROOT / "contracts/own_v40_v2.json"),
                 "--checkpoint", str(checkpoint_path), "--output-dir", str(output)]
    if script == "record_v40.py":
        arguments += ["--source-case", "fixed-repo"]
        args, train = cli.parse_arguments(arguments)
        monkeypatch.setattr(cli, "parse_arguments", lambda argv: (args, train))
    else:
        args, train = cli.parse_arguments(arguments), cli
    asset = audit_asset(contract)
    manifest = train.make_manifest(contract, asset, args)
    recorded = deepcopy(manifest)
    if mismatch == "stage":
        recorded["stage"] = "stand"
    elif mismatch == "contract_id":
        recorded["contract_id"] = "own-v40-jointspace-h5-v1"
    elif mismatch is not None:
        recorded[mismatch] = "0" * 64
    checkpoint, _ = make_checkpoint(recorded)
    torch.save(checkpoint, checkpoint_path)
    (tmp_path / "run_manifest.json").write_text(json.dumps(recorded))
    monkeypatch.setattr(train, "preflight", lambda args: ({"blockers": []}, contract, asset))
    reached_runtime = []

    def stop_before_runtime(*args):
        reached_runtime.append(True)
        raise RuntimeError("synthetic stop after successful validation")

    hook = "video_backend" if script == "record_v40.py" else "launch_app"
    monkeypatch.setattr(cli, hook, stop_before_runtime)
    expected_code = 2 if mismatch is not None and script == "evaluate_v40.py" else 3
    assert cli.main(arguments) == expected_code
    assert reached_runtime == ([True] if mismatch is None else [])
    if expected_code == 2:
        report = json.loads(capsys.readouterr().out)
        error = " ".join(report["blockers"])
        assert not output.exists()
    else:
        summary = json.loads((output / "summary.json").read_text())
        error = summary["runtime_error"]
        if script == "record_v40.py":
            assert summary["stage"] == "locomotion" and summary["command"] == [2., 0., .30]
        else:
            config = json.loads((output / "evaluation_config.json").read_text())
            assert config["stage"] == "locomotion" and config["contract_id"] == contract["contract_id"]
            assert config["cases"][0]["command"] == [2., 0., .30]
    expected_error = "successful validation" if mismatch is None else "trained stage" if mismatch == "stage" else mismatch
    assert expected_error in error
    assert "isaaclab.app" not in sys.modules


@pytest.mark.parametrize("stage,outcome", [
    ("stand", "passed"), ("stand", "failed"), ("height", "failed"),
    ("height", "startup_error"), ("stand", "wrapper_error"),
    ("stand", "invalid"), ("stand", "env_close_error"),
    ("stand", "aggregate_error"), ("stand", "summary_write_error"),
])
@pytest.mark.parametrize("app_close", ["returned", "system_exit", "error"])
def test_cli_publishes_evidence_before_app_close(monkeypatch, contract, tmp_path, stage, outcome, app_close):
    """Synthetic rollout only: terminal exit(0) is never evaluation success evidence."""
    cli = load_cli(monkeypatch)
    output = tmp_path / "evaluation"
    events = []
    published_at_close = []
    manifest = {"contract_id": contract["contract_id"], "contract_sha256": "contract-hash",
                "asset_manifest_sha256": "asset-hash"}
    provenance = {"checkpoint_sha256": "checkpoint-hash", "run_manifest_sha256": "manifest-hash",
                  "run_manifest": {"stage": stage}}
    actor = types.SimpleNamespace(to=lambda device: actor, eval=lambda: actor)
    monkeypatch.setattr(cli, "preflight", lambda args: ({"blockers": []}, contract, {}))
    monkeypatch.setattr(cli, "make_manifest", lambda *args: manifest)
    monkeypatch.setattr(cli, "checked_checkpoint", lambda *args: (actor, provenance))

    def close_env():
        events.append("env_close")
        if outcome in {"env_close_error", "wrapper_error"}:
            raise OSError("synthetic env close failure")

    raw = types.SimpleNamespace(**manifest, close=close_env)
    raw.set_evaluation_command = lambda command: setattr(raw, "command", command)
    wrapped = types.SimpleNamespace(reset=lambda: None, close=close_env)

    def make_env(args):
        if outcome == "startup_error":
            raise RuntimeError("synthetic startup failure")
        return raw

    def wrap(env):
        assert env is raw
        if outcome == "wrapper_error":
            raise RuntimeError("synthetic wrapper failure")
        return wrapped

    def collect(raw_env, wrapped_env, policy, steps, is_running, records):
        assert raw_env is raw and wrapped_env is wrapped
        events.append("collect")
        if outcome == "invalid":
            raise InvalidTrajectory("synthetic invalid policy output")
        records.extend(trajectory(contract, steps, command=raw.command))
        if outcome == "failed":
            records[-1]["terminated"] = True
            records[-1]["wheel_axis_midpoint_w_m"][0] = .22
        return "terminated" if outcome == "failed" else "horizon_reached"

    publication_fails = outcome in {"aggregate_error", "summary_write_error"}

    def close_app():
        events.append("app_close")
        if not publication_fails:
            # Read from disk AT entry, before a non-returning close can terminate Python.
            summary_path = output / "summary.json"
            published_at_close.append(json.loads(summary_path.read_text()) if summary_path.exists() else None)
        if app_close == "system_exit":
            raise SystemExit(0)
        if app_close == "error":
            raise RuntimeError("synthetic app close failure")

    app = types.SimpleNamespace(close=close_app, is_running=lambda: True)
    monkeypatch.setattr(cli, "launch_app", lambda args: types.SimpleNamespace(app=app))
    monkeypatch.setattr(cli, "make_env", make_env)
    monkeypatch.setattr(cli, "collect_case", collect)
    original_import = builtins.__import__

    def stub_import(name, *args, **kwargs):
        if name == "torch":
            return types.SimpleNamespace(inference_mode=nullcontext)
        if name == "isaaclab_rl.rsl_rl":
            return types.SimpleNamespace(RslRlVecEnvWrapper=wrap)
        assert not name.startswith(("isaaclab", "isaacsim", "omni", "pxr", "rsl_rl")), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stub_import)
    if publication_fails:
        def fail_publication(*args):
            raise OSError("synthetic suite publication failure")

        if outcome == "aggregate_error":
            monkeypatch.setattr(cli, "aggregate_cases", fail_publication)
        else:
            def write(path, data):
                if path.name == "summary.json":
                    fail_publication()
                write_json_exclusive(path, data)
            monkeypatch.setattr(cli, "write_json_exclusive", write)

    arguments = ["--apply", "--research", "--headless", "--device", "cpu", "--stage", stage,
                 "--checkpoint", str(tmp_path / "synthetic.pt"), "--output-dir", str(output)]
    if app_close == "system_exit":
        with pytest.raises(SystemExit) as exc:
            cli.main(arguments)
        assert exc.value.code == 0
    elif publication_fails:
        with pytest.raises(OSError, match="synthetic suite publication failure"):
            cli.main(arguments)
    else:
        assert cli.main(arguments) == (0 if outcome == "passed" and app_close == "returned" else 3)

    assert events[-1] == "app_close" and events.count("app_close") == 1
    assert events.count("env_close") == (0 if outcome == "startup_error" else 1)
    cleanup_path = output / "application_cleanup.json"
    if app_close == "system_exit":
        assert not cleanup_path.exists()  # No invented claim that terminal close completed.
    else:
        cleanup = json.loads(cleanup_path.read_text())
        assert cleanup["status"] == ("failed" if app_close == "error" else "returned")
        if app_close == "error":
            assert "synthetic app close failure" in cleanup["runtime_error"]
    if publication_fails:
        assert not (output / "summary.json").exists()
        return

    suite = json.loads((output / "summary.json").read_text())
    assert published_at_close == [suite]  # Immutable evaluation evidence, including on cleanup errors.
    assert suite["passed"] is (outcome == "passed")
    assert suite["application_cleanup"].startswith("not_observed")
    config = json.loads((output / "evaluation_config.json").read_text())
    assert suite["evaluation_config_sha256"] == config["evaluation_config_sha256"]
    assert suite["thresholds"] == config["thresholds"] == Thresholds().to_dict()
    for key in ("contract_sha256", "asset_manifest_sha256", "checkpoint_sha256", "run_manifest_sha256"):
        assert suite[key] == config[key]
    cases = default_cases(contract, stage)
    assert suite["case_count"] == len(cases)
    assert suite["failed_case_ids"] == [case["case_id"] for case in suite["cases"] if not case["passed"]]
    if outcome in {"startup_error", "wrapper_error"}:
        assert all(case["end_reason"] == "not_tested" for case in suite["cases"])
        assert "synthetic " in suite["runtime_error"]
        assert "collect" not in events
    else:
        first_id = cases[0]["case_id"]
        assert suite["cases"][0] == json.loads((output / (first_id + ".summary.json")).read_text())
        assert (output / (first_id + ".records.json")).exists()
        assert events.count("collect") == 1
        if outcome == "failed":
            assert {"terminated", "wheel_axis_drift"} <= set(suite["cases"][0]["failure_reasons"])
        if outcome == "invalid":
            assert suite["cases"][0]["end_reason"] == "invalid_or_interrupted"
        for case in suite["cases"][1:]:
            assert case["end_reason"] == "not_tested" and not case["passed"]
    if outcome in {"env_close_error", "wrapper_error"}:
        assert "environment close failed: synthetic env close failure" in suite["runtime_error"]
    if outcome == "wrapper_error":
        assert "synthetic wrapper failure" in suite["runtime_error"]


def test_cli_help_without_isaac():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_v40.py"), "--help"],
                          capture_output=True, text=True, timeout=20,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert proc.returncode == 0, proc.stderr
    assert "--apply" in proc.stdout and "--preflight-only" in proc.stdout
    assert "--command" in proc.stdout and "--thresholds" in proc.stdout
