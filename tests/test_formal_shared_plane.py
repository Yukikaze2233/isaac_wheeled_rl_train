"""CPU checks for continuous support and the actual pre-reset guard consumer."""
import ast
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.formal_terrain import SharedPlane, plane_origins
from support_nonstandard_start_integration import execute, method


def plane():
    return SharedPlane(plane_origins(2, 4.), ["/World/ground/Plane"],
                       ["/World/ground"], ["/World/ground/physicsMaterial"], 4.)


def env_for_guard(*, monitor=True, threshold=None, formal=True, shared=True, gap=.05):
    tensor = lambda value: NS(torch=value)
    pose = torch.tensor([[50., 0., .3, 0., 0., 0., 1.], [0., 6., .3, 0., 0., 0., 1.]])
    data = NS(root_com_lin_vel_b=tensor(torch.zeros(2, 3)),
              root_com_ang_vel_b=tensor(torch.zeros(2, 3)),
              projected_gravity_b=tensor(torch.tensor([[0., 0., -1.]]).repeat(2, 1)),
              root_link_pose_w=tensor(pose), joint_pos=tensor(torch.zeros(2, 18)))
    return NS(cfg={"formal_v7_flat": formal, "formal_closure_monitor_only": monitor,
                   "closure_gap_termination_m": threshold, "evaluation_long_corridors": True,
                   "flat_half_length_m": 44., "flat_floor_width_m": 8.,
                   "formal_terrain": {"type": "plane"}},
              num_envs=2, device="cpu", robot=NS(data=data), origins=torch.zeros(2, 3),
              reset_positions=torch.zeros(2, 3), reset=lambda ids: None,
              scut_reward=NS(net_contact=torch.zeros(2, 2, 3),
                  termination=lambda *args: tuple(torch.zeros(2, dtype=torch.bool) for _ in range(3))),
              closure_gap_peak=torch.full((2,), gap), max_closure_gap=0., max_closure_gap_all_finite=0.,
              max_closure_gap_numerical_reset=0., tick=0,
              contact_force=torch.tensor([[[0., 0., 12.], [0., 0., 13.]]]).repeat(2, 1, 1),
              wheel_ids=[0, 1], shared_terrain=plane() if shared else None)


def run_pre_reset_guards(env):
    body = method("_finish_scut_flat_step").body
    end = next(i for i, node in enumerate(body) if isinstance(node, ast.For))
    return execute(body[:end], env)


def test_shared_plane_support_and_boundary_continue_beyond_previous_corridor():
    shared = plane()
    points = torch.tensor([[50., 0., .3], [0., 6., .3], [-10000., 10000., .3]])
    assert shared.height(points[:, :2]).eq(0).all()
    assert not shared.boundary(points).any()
    origins = torch.tensor(plane_origins(4, 4.))
    assert origins[:, :2].unique().tolist() == [-2., 2.]
    env = env_for_guard()
    scope = execute([method("ground_height")], env)
    assert scope["ground_height"](env, points[:2, :2]).eq(0).all()
    assert not run_pre_reset_guards(env)["reasons"]["boundary"].any()


@pytest.mark.parametrize("threshold", [None, .003])
def test_monitor_only_keeps_gap_telemetry_without_reset_or_fatal(threshold):
    env = env_for_guard(threshold=threshold)
    scope = run_pre_reset_guards(env)
    assert env.max_closure_gap == pytest.approx(.05)
    assert "closure_gap" not in scope["reasons"]
    assert not scope["terminated"].any()
    env.closure_gap_peak[0] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite reference physical mapping"):
        run_pre_reset_guards(env)


def test_unselected_legacy_guards_and_boundary_are_unchanged():
    with pytest.raises(RuntimeError, match="exceeds 3 mm"):
        run_pre_reset_guards(env_for_guard(monitor=False, formal=False, shared=False, gap=.004))
    with pytest.raises(RuntimeError, match="severe closed-chain gap"):
        run_pre_reset_guards(env_for_guard(monitor=False, threshold=None))
    env = env_for_guard(monitor=False, formal=False, shared=False, gap=.004, threshold=.003)
    reasons = run_pre_reset_guards(env)["reasons"]
    assert reasons["closure_gap"].all() and reasons["boundary"].all()


def test_monitor_only_large_gap_does_not_trigger_repeated_snapshot_io():
    env = env_for_guard()
    env.cfg.update(formal_physics_trace_directory="unused", formal_physics_trace_ticks=[5])
    saved = []
    env._save_formal_physics_state = saved.append
    for env.tick in (1, 2, 3, 4, 5):
        run_pre_reset_guards(env)
    assert saved == ["closure_failure", "physics_step_0005"]
    assert env._closure_failure_saved


def test_scene_probe_records_real_contact_at_pre_reset_outside_positions():
    env = env_for_guard()
    scope = execute([method("prepare_formal_plane_contact_probe"),
                     method("formal_plane_contact_probe_result")], env)
    scope["prepare_formal_plane_contact_probe"](env)
    assert env.reset_positions[:, :2].tolist() == [[50., 0.], [0., 6.]]
    run_pre_reset_guards(env)
    result = scope["formal_plane_contact_probe_result"](env)
    assert result["passed"] and result["outside_frames"] == [1, 1]
    assert result["outside_peak_wheel_force_n"] == [13., 13.]
    # Contact only inside the former floor is insufficient evidence.
    env.robot.data.root_link_pose_w.torch[:, :2].zero_()
    scope["prepare_formal_plane_contact_probe"](env)
    run_pre_reset_guards(env)
    assert not scope["formal_plane_contact_probe_result"](env)["passed"]
