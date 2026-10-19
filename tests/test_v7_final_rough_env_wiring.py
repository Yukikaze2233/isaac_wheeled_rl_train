"""V7 production env reward dispatch on CPU buffers; no simulator or PPO.

Constructor/payment/reset AST comes from env.py. Physical rows and ray hits are
explicit CPU fixtures, not native contact, scan or learned skill qualification.
"""
import ast
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

from test_scut_flat_reward_runtime import env as legacy_env, field, methods
from wheeled_tasks.chassis.scut_flat_reward_runtime import ScutFlatRewardRuntime
from wheeled_tasks.chassis.scut_observation import build_scut35


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/wheeled_tasks/chassis/env.py"
PROFILE = "v7_final_rough_v1"
ASSET = (ROOT / "reports/v7_training_review_20261011/mechanical_stop_candidates_20261011"
         / "stop107_reference_v2/asset")


def node(name):
    tree = ast.parse(SOURCE.read_text())
    cls = next(value for value in tree.body if isinstance(value, ast.ClassDef) and value.name == "ChassisEnv")
    return deepcopy(next(value for value in cls.body if isinstance(value, ast.FunctionDef) and value.name == name))


def execute(nodes, scope):
    scope.update(torch=torch, __name__="wheeled_tasks.chassis.env", __package__="wheeled_tasks.chassis")
    exec(compile(ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[])),
                 str(SOURCE), "exec"), scope)


@pytest.fixture
def cpu_base_scan(monkeypatch):
    """Execute the actual pure base scanner without importing its Warp helpers."""
    path = ROOT / "src/wheeled_tasks/chassis/scut_rough_support.py"
    tree = ast.parse(path.read_text())
    scan = next(value for value in tree.body if isinstance(value, ast.FunctionDef) and value.name == "base_ground_scan")
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(scan)], type_ignores=[])),
                 str(path), "exec"), scope)
    monkeypatch.setitem(sys.modules, "wheeled_tasks.chassis.scut_rough_support", NS(base_ground_scan=scope["base_ground_scan"]))


def asset():
    return (json.loads((ASSET / "model_spec.json").read_text()),
            json.loads((ASSET / "manifest.json").read_text()))


def mapping(body_names, *, rough=True):
    return {"root_velocity_reference": "root_rigid_body_com_in_root_link_frame",
        "root_height_reference": "root_link_z_minus_base_ground_scan" if rough else "root_link_z_minus_env_origin",
        "wheel_position_reference": "wheel_link_origins_in_root_link_frame",
        "undesired_link_names": ["base_link", "L_link1", "L_link2", "R_link1", "R_link2",
            "LL_link1", "LL_link2", "LL_link3", "LL_link4", "RR_link1", "RR_link2", "RR_link3", "RR_link4"],
        "terminal_link_names": ["base_link"]}


@pytest.mark.parametrize("profile", ["scut0719_flat_v1", "scut0730_rough_v1", PROFILE])
@pytest.mark.parametrize("rehearsal", [False, True])
def test_constructor_selects_only_explicit_v7_profile_and_preserves_old_arguments(monkeypatch, profile, rehearsal):
    spec, manifest = asset()
    calls = []

    class Constructor:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def summary(self):
            return {"fixture": profile}

    import wheeled_tasks.chassis.scut_flat_reward_runtime as old_module
    monkeypatch.setattr(old_module, "ScutFlatRewardRuntime", Constructor)
    monkeypatch.setitem(sys.modules, "wheeled_tasks.chassis.final_rough_reward_runtime", NS(V7FinalRoughRewardRuntime=Constructor))
    e = NS(reward_selection={"scut_source_selected": True}, rehearsal_spec=None,
        source_flat_rehearsal_mask=None, robot=NS(body_names=manifest["rigid_body_names"],
            joint_names=manifest["tree_joint_names"]), model_spec=spec, dt=.005, startup_report={})
    e.wheel_ids = [e.robot.body_names.index(name) for name in ("L_link3", "R_link3")]
    if rehearsal:
        from wheeled_tasks.chassis.fixed_flat_rehearsal import SOURCE_VERSION
        e.rehearsal_spec = {"settings": {"version": SOURCE_VERSION, "reward_profile": profile},
                            "case_index": [0, -1]}
        e.source_flat_rehearsal_mask = torch.tensor([True, False])
    cfg = {"reward_profile": profile, "scut_reward_mapping": mapping(e.robot.body_names),
           "formal_command_decoupling": {"version": "fixture", "enabled": False}}
    if profile == PROFILE:
        cfg["formal_terrain"] = {"type": "rough", "source": "scut0730"}
    dispatch = next(value for value in node("__init__").body if isinstance(value, ast.If)
        and "V7FinalRoughRewardRuntime" in ast.unparse(value))
    execute([dispatch], {"self": e, "config": cfg, "manifest": manifest, "num_envs": 2, "device": "cpu"})
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (2, "cpu", e.robot.body_names, cfg["scut_reward_mapping"])
    assert kwargs["physics_dt_s"] == .005
    assert kwargs["command_decoupling"] is cfg["formal_command_decoupling"]
    if rehearsal:
        assert kwargs["flat_rehearsal_mask"] is e.source_flat_rehearsal_mask
        assert kwargs["flat_rehearsal_reward_profile"] == profile
    if profile == PROFILE:
        assert kwargs["terrain_source"] == cfg["formal_terrain"]["source"]
        assert kwargs["model_spec"] is spec and kwargs["manifest"] is manifest
        assert kwargs["native_joint_names"] is e.robot.joint_names
        assert kwargs["wheel_body_names"] == ("L_link3", "R_link3")
        assert kwargs["flat_rehearsal_reward_profile"] == PROFILE
        assert "reward_profile" not in kwargs
    else:
        assert set(kwargs) == ({"reward_profile", "physics_dt_s", "command_decoupling", "tracking_reward_repair"}
            | ({"flat_rehearsal_mask", "flat_rehearsal_reward_profile"} if rehearsal else set()))
        assert kwargs["reward_profile"] == profile
    assert e.startup_report["scut_flat_reward"] == {"fixture": profile}


def v7_env(*, count=2, plane=False, ground_height=0.):
    from wheeled_tasks.chassis.final_rough_reward_runtime import V7FinalRoughRewardRuntime
    spec, manifest = asset()
    e = legacy_env(count)
    names, bodies = manifest["tree_joint_names"], manifest["rigid_body_names"]
    e.device, e.tick, e.policy_dt = "cpu", 1, .02
    e.cfg.update(reward_profile=PROFILE, formal_closure_monitor_only=True,
                 closure_gap_termination_m=None, record_diagnostics=True)
    e.robot.joint_names, e.robot.body_names = names, bodies
    e.manifest, e.model_spec = manifest, spec
    e.scut_reward = V7FinalRoughRewardRuntime(count, "cpu", bodies, mapping(bodies),
        model_spec=spec, manifest=manifest, native_joint_names=names,
        flat_rehearsal_reward_profile=PROFILE)
    e.ids = [names.index(name) for name in manifest["control_joint_names"]]
    e.knee_ids = [names.index(name) for name in ("L_joint2", "R_jonit2")]
    by_name = {joint["name"]: joint for joint in spec["joints"]}
    e.knee_bounds = {name: tuple(float(by_name[name]["limit"][key]) for key in ("lower", "upper"))
                     for name in ("L_joint2", "R_jonit2")}
    e.spring_ids = [names.index(name) for name in manifest["spring_joint_names"]]
    e.wheel_ids = [bodies.index(name) for name in ("L_link3", "R_link3")]
    data = e.robot.data
    data.root_link_pose_w.torch[:, 2] = .34 + ground_height
    data.root_com_lin_vel_b.torch[:, 0] = 1.6
    data.joint_pos = field(torch.tensor([manifest["nominal_joint_pos"][name] for name in names]).repeat(count, 1))
    data.joint_vel, data.joint_acc, data.applied_torque = (field(torch.zeros(count, len(names))) for _ in range(3))
    poses = torch.zeros(count, len(bodies), 7)
    poses[:, :, 6] = 1.
    poses[:, e.wheel_ids, 0] = torch.tensor([.04, -.02])
    poses[:, e.wheel_ids, 1] = torch.tensor([.2, -.2])
    poses[:, e.wheel_ids, 2] = .18 + ground_height
    data.body_link_pose_w = field(poses)
    e.contact_force = torch.zeros(count, len(bodies), 3)
    e.commands[:] = torch.tensor([1.5, .2, .30])
    from wheeled_tasks.chassis.stepup_command import StepUpCommand
    e.v7_stepup = StepUpCommand(count, "cpu", {"version": "v7-stepup-command-v1", "enabled": True,
        "event_source": "external_step_wall_events", "height_bias_m": .1,
        "hold_seconds": .2, "maximum_height_m": .42})
    e.v7_stepup.remaining.fill_(1)
    e.v7_stepup.value.fill_(.3)
    rays = []
    mesh = object()

    def raycast(starts, directions, *bound_mesh):
        assert bound_mesh == (() if plane else (mesh,))
        rays.append((starts.clone(), directions.clone()))
        hits = starts.clone()
        hits[..., 2] = ground_height
        return (hits,)

    e.shared_terrain = NS(warp_mesh=mesh, raycast=raycast,
        boundary=lambda xyz: torch.zeros(count, dtype=torch.bool))
    if plane:
        e.evaluation_surface = "plane_regression_v1"
    e.mode = torch.ones(count, dtype=torch.long)
    e.jump_requested = torch.zeros(count, dtype=torch.bool)
    nominal = torch.tensor([manifest["nominal_joint_pos"][name] for name in manifest["control_joint_names"]])
    e.get_observations = lambda: {"policy": build_scut35(data.root_com_ang_vel_b.torch,
        data.projected_gravity_b.torch, e.commands, data.joint_pos.torch[:, e.ids],
        data.joint_vel.torch[:, e.ids], torch.zeros(count, 6), nominal,
        e.jump_requested, torch.zeros(count), torch.zeros(count))}
    r = e.scut_reward
    r.begin_policy(torch.arange(count * 6, dtype=torch.float32).reshape(count, 6) / 10.)
    for _ in range(4):
        r.observe_physics(torch.tensor([[.1, .2, 4., .3, .4, 8.]]).repeat(count, 1),
                          torch.ones(count, 6), torch.tensor([[.2, .3, .1, .4, .5, .15]]).repeat(count, 1))
        r.finish_contacts()
    return e, rays


@pytest.mark.parametrize("plane", [False, True])
def test_actual_env_forwards_full_physical_frame_and_pays_31_once(cpu_base_scan, plane):
    e, rays = v7_env(plane=plane)
    incoming = e.get_observations()["policy"].clone()
    paid_calls = []
    real_pay = e.scut_reward.pay

    def record(**kwargs):
        paid_calls.append(kwargs)
        return real_pay(**kwargs)

    e.scut_reward.pay = record
    frame, reward, done, extras = e.finish()
    assert not done.any() and len(paid_calls) == 1
    call = paid_calls[0]
    assert call["native_joint_positions"] is e.robot.data.joint_pos.torch
    assert torch.equal(call["wheel_link_pose_world"], e.robot.data.body_link_pose_w.torch[:, e.wheel_ids])
    assert torch.equal(call["wheel_link_positions_world"], call["wheel_link_pose_world"][..., :3])
    assert call["policy_tick"] == 1 and callable(call["raycast"])
    assert call["stair_active"].dtype == torch.bool and not call["stair_active"].any()
    assert "scripted_interval" not in call
    assert e.v7_stepup.active.all()
    assert call["reward_ground_height_world"].tolist() == [0., 0.]
    assert {value[0].shape[1] for value in rays} == {9, 8}
    assert frame["policy"].shape == (2, 35) and torch.equal(frame["policy"], incoming)
    payments = extras["native_reward_components"]
    assert len(payments) == 31 and torch.equal(reward, torch.stack(list(payments.values()), -1).sum(-1))
    expected_zero_torque = math.exp(-(.15 / 1.5) ** 2) * .02 * 20.
    assert payments["airborne_air_wheel_zero_torque_exp"][0].item() == pytest.approx(expected_zero_torque)
    assert not payments["airborne_joint_pos_limits"].any()
    assert "policy_learning_mask" not in extras
    for name, paid in payments.items():
        assert torch.equal(extras["log"]["/reward/" + name], paid.mean())
    for name, value in e.scut_reward.policy_metrics().items():
        assert name in extras["log"] and torch.equal(extras["log"][name], value)


@pytest.mark.parametrize("cause", ["timeout", "confirmed", "numeric"])
def test_actual_paid_transition_precedes_lane_reset_and_returned_actor35(cpu_base_scan, cause):
    e, _ = v7_env()
    r = e.scut_reward
    if cause == "timeout": e.episode_length_buf[0] = 999
    elif cause == "confirmed":
        r.terminal_counter[0] = 9
        r.net_contact[0, e.robot.body_names.index("base_link"), 2] = 2.
    else: r.joint_velocity[0, 0] = 501.
    old_commands, old_action = e.commands.clone(), r.raw_action.clone()
    resets = []
    reset_node = next(value for value in node("reset").body if isinstance(value, ast.If)
                     and ast.unparse(value.test) == "self.scut_reward is not None")

    def advance(done, terminated, extras):
        ids = done.nonzero().flatten()
        resets.append(ids.clone())
        assert len(extras["native_reward_components"]) == 31
        execute([reset_node], {"self": e, "ids": ids})
        e.commands[ids] = torch.tensor([0., 0., .25])
        e.robot.data.root_link_pose_w.torch[ids, 2] = .25
        e.robot.data.body_link_pose_w.torch[ids[:, None], e.wheel_ids, 2] = .06
        e.episode_length_buf[ids] = 0

    e._advance_policy_state = advance
    frame, reward, done, extras = e.finish()
    assert done.tolist() == [True, False] and resets[0].tolist() == [0]
    assert torch.equal(extras["diagnostics"]["commands"], old_commands)
    assert torch.equal(extras["diagnostics"]["raw_actor_actions"], old_action)
    assert not r.raw_action[0].any() and torch.equal(r.raw_action[1], old_action[1])
    assert frame["policy"].shape == (2, 35)
    assert frame["policy"][0, 3].item() == pytest.approx(1.25)
    assert frame["policy"][1, 3].item() == pytest.approx(1.5)
    assert frame["policy"][:, 28].eq(1.).all() and not frame["policy"][:, 32].any()
    assert "policy_learning_mask" not in extras
    termination = extras["native_reward_components"]["termination"][0].item()
    assert termination == (0. if cause in ("timeout", "numeric") else -24.)
    if cause == "numeric":
        assert reward[0].item() == 0.
        assert all(value[0].item() == 0. for value in extras["native_reward_components"].values())


@pytest.mark.parametrize("profile", ["scut0719_flat_v1", "scut0730_rough_v1"])
def test_old_profiles_keep_26_payments_and_receive_no_new_frame_arguments(cpu_base_scan, profile):
    e = legacy_env()
    e.cfg["reward_profile"] = profile
    if profile == "scut0730_rough_v1":
        e.scut_reward = ScutFlatRewardRuntime(2, "cpu", ["base_link", "L_wheel", "R_wheel"],
            {"root_velocity_reference": "root_rigid_body_com_in_root_link_frame",
             "root_height_reference": "root_link_z_minus_base_ground_scan",
             "wheel_position_reference": "wheel_link_origins_in_root_link_frame",
             "undesired_link_names": ["base_link"], "terminal_link_names": ["base_link"]}, reward_profile=profile)
        mesh = object()
        e.shared_terrain = NS(warp_mesh=mesh, raycast=lambda starts, directions, bound: (
            torch.cat((starts[..., :2], torch.zeros_like(starts[..., 2:])), -1),),
            boundary=lambda xyz: torch.zeros(2, dtype=torch.bool))
    calls = []
    real_pay = e.scut_reward.pay
    e.scut_reward.pay = lambda **kwargs: (calls.append(kwargs), real_pay(**kwargs))[1]
    _, reward, done, extras = e.finish()
    assert not done.any() and len(calls) == 1 and len(extras["native_reward_components"]) == 26
    assert not set(calls[0]) & {"native_joint_positions", "wheel_link_pose_world", "policy_tick", "raycast", "stair_active", "scripted_interval"}
    assert torch.equal(reward, torch.stack(list(extras["native_reward_components"].values()), -1).sum(-1))
    assert "policy_learning_mask" not in extras


def test_actual_script_handoff_mask_reaches_termination_and_phase_without_reusing_PPO_mask(cpu_base_scan):
    e, _ = v7_env(count=3)
    e.cfg["record_diagnostics"] = False
    e.nonwheel_ids = e.scut_reward.undesired_ids
    learning = torch.tensor([False, True, True])
    e.recovery = NS(source_flat_handoff=True, enabled=torch.tensor([True, True, False]),
        learning_mask=learning, pure_rl=torch.ones(3, dtype=torch.bool),
        episode_ids=torch.zeros(3, dtype=torch.long), outcome=lambda *args: None,
        failure_reasons=lambda: {}, diagnostics=lambda: {},
        policy_transition=lambda *args: {"episode_ids": torch.zeros(3, dtype=torch.long)})
    e.flat_episode = NS(learning_mask=torch.tensor([True, True, False]), metrics=lambda **kwargs: {},
        commands=NS(observe_motion_interval=lambda *args: None))
    e._advance_flat_episode = lambda *args: None
    received = {}
    real_termination, real_pay = e.scut_reward.termination, e.scut_reward.pay

    def terminate(*args, **kwargs):
        received["terminal_mask"] = kwargs["scripted_interval"]
        return real_termination(*args, **kwargs)

    def pay(**kwargs):
        received["airborne_mask"] = kwargs["scripted_interval"]
        return real_pay(**kwargs)

    e.scut_reward.termination, e.scut_reward.pay = terminate, pay
    frame, _, done, extras = e.finish()
    assert not done.any() and frame["policy"].shape == (3, 35)
    assert received["terminal_mask"] is received["airborne_mask"]
    assert received["airborne_mask"].tolist() == [True, False, False]
    assert e.recovery.pure_rl.all()  # An endpoint handoff does not own the whole interval.
    assert extras["policy_learning_mask"].tolist() == [False, True, False]
    assert e.scut_reward.last_airborne_transition.updated.tolist() == [False, True, True]
    assert e.scut_reward.airborne.active.tolist() == [False, True, True]
    assert extras["log"]["/airborne/scripted_observation_excluded_fraction"].item() == pytest.approx(1. / 3.)
    assert len(extras["native_reward_components"]) == 31
    e.tick += 1
    learning[0] = True
    _, _, done, extras = e.finish()
    assert not done.any() and e.scut_reward.last_airborne_transition.entered.tolist() == [True, False, False]
    assert extras["policy_learning_mask"].tolist() == [True, True, False]


def test_v7_profile_can_use_existing_payment_measurement_without_new_ownership():
    e = NS()
    methods("_configure_reward_episode_measurement")["_configure_reward_episode_measurement"](e,
        {"reward_profile": PROFILE, "policy_dt": .02,
         "reward_episode_measurement": {"version": "reward_episode_measurement_v1", "enabled": True}})
    assert e.reward_episode_measurement_enabled


@pytest.mark.parametrize("actual_runtime", [True, False])
def test_preflight_readback_uses_actual_new_runtime_class_and_summary(actual_runtime):
    e, _ = v7_env()
    if not actual_runtime:
        e.scut_reward = NS(summary=lambda: {"reward_profile": PROFILE})
    selected = ast.parse('config.get("reward_profile") == "v7_final_rough_v1"', mode="eval").body
    branch = next(value for value in ast.walk(node("__init__")) if isinstance(value, ast.If)
                  and ast.dump(value.test) == ast.dump(selected))
    actual = {"modules": {}}
    execute([branch], {"self": e, "config": {"reward_profile": PROFILE}, "actual": actual})
    assert actual["modules"]["v7_final_rough_reward"] is actual_runtime
    if actual_runtime:
        assert actual["v7_final_rough_reward"] == e.scut_reward.summary()
        assert actual["v7_final_rough_reward"]["nonzero_term_count"] == 31
        assert actual["v7_final_rough_reward"]["native_qualified"] is False
    else:
        assert actual["v7_final_rough_reward"] is None


def test_v7_rehearsal_height_metrics_use_scanned_ground_datum(cpu_base_scan):
    e, _ = v7_env(ground_height=.2)
    e.source_flat_rehearsal_mask = torch.tensor([True, False])
    _, _, _, extras = e.finish()
    for group in ("source_flat0719", "source_rough0730"):
        assert extras["log"][f"/rehearsal/{group}/height_mae_m"].item() == pytest.approx(.04)
        assert extras["log"][f"/rehearsal/{group}/height_bias_m"].item() == pytest.approx(.04)


@pytest.mark.parametrize("profile", ["scut0730_rough_v1", PROFILE, "scut0719_flat_v1"])
def test_rough_command_factory_uses_existing_sampler_without_profile_fallback(profile):
    path = ROOT / "src/wheeled_tasks/chassis/flat_episode_runtime.py"
    tree = ast.parse(path.read_text())
    factory = next(value for value in tree.body if isinstance(value, ast.FunctionDef)
                   and value.name == "build_flat_episode_runtime")
    rough_condition = ast.parse('config.get("formal_terrain", {}).get("type") == "rough"', mode="eval").body
    branch = next(value for value in ast.walk(factory) if isinstance(value, ast.If)
                  and ast.dump(value.test) == ast.dump(rough_condition))
    grid = torch.zeros(10, 14, 3)
    scope = {"config": {"formal_terrain": {"type": "rough"}, "reward_profile": profile},
        "rough_terrain": NS(importer=NS(terrain_origins=grid)),
        "environment_origins": torch.zeros(2, 3), "num_envs": 2, "device": "cpu",
        "__name__": "wheeled_tasks.chassis.flat_episode_runtime", "__package__": "wheeled_tasks.chassis"}
    module = ast.fix_missing_locations(ast.Module(body=[deepcopy(branch)], type_ignores=[]))
    if profile == "scut0719_flat_v1":
        with pytest.raises(ValueError, match="rough commands"):
            exec(compile(module, str(path), "exec"), scope)
    else:
        exec(compile(module, str(path), "exec"), scope)
        from wheeled_tasks.chassis.rough_commands import ScutRoughCommands
        assert type(scope["commands"]) is ScutRoughCommands
        assert scope["commands"].physics_steps == 0
        assert not scope["commands"].fixed_request_mask.any()
