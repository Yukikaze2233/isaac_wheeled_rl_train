"""Actual tensor observers/controllers agree on stopping after platform flight.

These are CPU source-module event/price counterexamples, not Isaac collision
or PPO trials. Ordinary jump and airborne wheel targets retain their snapshots.
"""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import torch

from wheeled_tasks.chassis.jump_landing import JUMP_LANDING_V1, JumpLandingWindow
from wheeled_tasks.chassis.step_jump import StepJumpTask
from wheeled_tasks.chassis.task import Phase

ROOT = Path(__file__).resolve().parents[1]


def window_state(*, stop=False, count=2, vx=.4, yaw=-.2):
    window = JumpLandingWindow(count, "cpu", .02, deepcopy(JUMP_LANDING_V1),
                              step_jump_stop_target=stop)
    state = dict(request=torch.ones(count, dtype=torch.bool),
        phase=torch.full((count,), Phase.GROUND), contacts=torch.ones(count, 2, dtype=torch.bool),
        com_released=torch.zeros(count, dtype=torch.bool), clear_air_time_peak=torch.zeros(count),
        raw_height=torch.full((count,), .305), height=torch.full((count,), .305),
        commands=torch.tensor([[vx, yaw, .305]]).repeat(count, 1),
        goal_apex=torch.full((count,), .28), landing_delta=torch.full((count,), .20),
        com_vz=torch.zeros(count), step_jump=torch.tensor([True] + [False] * (count - 1)))
    return window, state


def fly(window, state):
    window.update(**state)
    state.update(phase=torch.full_like(state["phase"], Phase.FLIGHT),
        contacts=torch.zeros_like(state["contacts"]), com_released=torch.ones_like(state["com_released"]),
        clear_air_time_peak=torch.full_like(state["clear_air_time_peak"], .08),
        com_vz=torch.full_like(state["com_vz"], -.3))
    window.update(**state)


def touchdown(window, state):
    fly(window, state)
    state.update(phase=torch.full_like(state["phase"], Phase.LANDING),
        contacts=torch.ones_like(state["contacts"]), com_vz=torch.zeros_like(state["com_vz"]))
    window.update(**state)


def costs(window, *, vx=0., yaw=0., wheel_velocity=None, com_vz=None):
    count = len(window.elapsed)
    velocity, omega = torch.zeros(count, 3), torch.zeros(count, 3)
    velocity[:, 0], omega[:, 2] = vx, yaw
    h, vz = window.reference()
    if com_vz is not None:
        vz = torch.full_like(vz, com_vz)
    return window.reward_terms(h, vz, velocity, omega,
        torch.tensor([[0., 0., -1.]]).repeat(count, 1),
        torch.zeros(count, 2) if wheel_velocity is None else wheel_velocity,
        torch.zeros(count, 2), torch.zeros(count, 2), torch.full((count,), 135.),
        position_xy=torch.ones(count, 2), wheel_clearance=torch.full((count, 2), .05))


def axis(terms, name):
    return terms[name + "_broad"] + terms[name + "_precision"]


@pytest.mark.parametrize("vx,yaw", [(.4, -.2), (-.4, .2)])
def test_platform_stop_cost_changes_only_explicit_step_lane(vx, yaw):
    legacy, before = window_state(vx=vx, yaw=yaw)
    fixed, after = window_state(stop=True, vx=vx, yaw=yaw)
    touchdown(legacy, before)
    touchdown(fixed, after)
    torch.testing.assert_close(fixed.goal_commands, legacy.goal_commands, rtol=0, atol=0)
    torch.testing.assert_close(fixed.landing_goal_commands[0, :2], torch.zeros(2), rtol=0, atol=0)
    torch.testing.assert_close(fixed.landing_goal_commands[1], legacy.goal_commands[1], rtol=0, atol=0)
    stopped_old, stopped_new = costs(legacy), costs(fixed)
    assert axis(stopped_old, "jump_landing_velocity")[0].item() == pytest.approx(-1.1612496)
    for name in ("jump_landing_velocity", "jump_landing_yaw"):
        assert axis(stopped_new, name)[0] == 0.
        assert axis(stopped_old, name)[0] < 0.
        assert axis(costs(fixed, vx=vx, yaw=yaw), name)[0] < 0.
        assert axis(costs(legacy, vx=vx, yaw=yaw), name)[0] == 0.
    changed = {"jump_landing_velocity_broad", "jump_landing_velocity_precision",
               "jump_landing_yaw_broad", "jump_landing_yaw_precision"}
    for key in stopped_old:
        torch.testing.assert_close(stopped_new[key][1:], stopped_old[key][1:], rtol=0, atol=0)
        if key not in changed:
            torch.testing.assert_close(stopped_new[key], stopped_old[key], rtol=0, atol=0)


@pytest.mark.parametrize("vx", [-.4, .4])
def test_precontact_wheels_still_follow_takeoff_goal_for_step_and_ordinary(vx):
    legacy, before = window_state(vx=vx)
    fixed, after = window_state(stop=True, vx=vx)
    fly(legacy, before)
    fly(fixed, after)
    matching = (fixed.goal_commands[:, :1] / fixed.settings["wheel_radius_m"]
                * torch.tensor([[1., -1.]]))
    for wheels in (matching, matching.flip(-1), torch.zeros_like(matching)):
        left, right = (costs(legacy, wheel_velocity=wheels, com_vz=-.3),
                       costs(fixed, wheel_velocity=wheels, com_vz=-.3))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert axis(costs(fixed, wheel_velocity=matching, com_vz=-.3), "jump_precontact_wheel_velocity").eq(0).all()
    assert axis(costs(fixed, wheel_velocity=torch.zeros_like(matching), com_vz=-.3),
                "jump_precontact_wheel_velocity").lt(0).all()


def test_ordinary_jump_is_bitwise_compatible_through_release_bounce_and_expiry():
    legacy, before = window_state()
    enabled, after = window_state(stop=True)
    before["step_jump"].zero_()
    after["step_jump"].zero_()
    touchdown(legacy, before)
    touchdown(enabled, after)
    for tick in range(151):
        for state in (before, after):
            state["request"].zero_()
            state["commands"][:] = torch.tensor([[-.9, .7, .43]])
            state["contacts"][:] = tick % 3 != 0
            state["phase"][:] = Phase.RECOVERY if tick % 3 else Phase.FLIGHT
        for window, state in ((legacy, before), (enabled, after)):
            window.update(**state)
        for key in ("goal_commands", "landing_goal_commands", "active", "elapsed",
                    "outcome_active", "request_elapsed", "touchdown", "expired", "airborne"):
            torch.testing.assert_close(getattr(legacy, key), getattr(enabled, key), rtol=0, atol=0)
        left, right = costs(legacy, vx=.1, yaw=.05), costs(enabled, vx=.1, yaw=.05)
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    assert not enabled.active.any() and enabled.outcome_active.all()
    assert enabled.duration == 3. and enabled._outcome_ticks * enabled.dt == 6.


def test_platform_target_cannot_be_rewritten_by_release_or_later_mask_then_reset_is_local():
    window, state = window_state(stop=True)
    touchdown(window, state)
    original = window.goal_commands.clone()
    state["request"].zero_()
    state["commands"][:] = torch.tensor([[-1., 1., .43]])
    state["step_jump"].logical_not_()
    window.update(**state)
    torch.testing.assert_close(window.goal_commands, original, rtol=0, atol=0)
    assert window.landing_goal_commands[0, :2].eq(0).all()
    torch.testing.assert_close(window.landing_goal_commands[1], original[1], rtol=0, atol=0)
    elapsed = window.elapsed[1].clone()
    window.reset_accepted(torch.tensor([0]))
    assert window.goal_commands[0].eq(0).all() and window.landing_goal_commands[0].eq(0).all()
    assert window.elapsed[1] == elapsed
    state["request"][0], state["phase"][0] = True, Phase.GROUND
    window.update(**state)
    assert window.cycle_accepted.tolist() == [True, False]
    torch.testing.assert_close(window.landing_goal_commands[0], state["commands"][0], rtol=0, atol=0)


def test_native_step_controller_and_physical_assessor_agree_with_platform_stop_cost():
    task = StepJumpTask(["step"], {"step": {"step_jump": {
        "rises_m": [.20], "treads_m": [.8], "launch_distance_m": .25,
        "approach_speed_m_s": .4, "landing_margin_m": .1, "settle_seconds": .35,
        "landing_clearance_m": .08, "trigger_source": "laser"}}}, "cpu", .02)
    def control(time, *, support, force):
        return task.control(torch.tensor([time]), torch.tensor([.4]),
            torch.tensor([support]), torch.tensor([force]), torch.tensor([.20]),
            torch.tensor([True]), torch.zeros(1))
    assert control(.02, support=True, force=9.81).item()
    launch_vx = task.command_vx.item()
    assert launch_vx > .25
    legacy, before = window_state(count=1, vx=launch_vx, yaw=0.)
    fixed, after = window_state(stop=True, count=1, vx=launch_vx, yaw=0.)
    fly(legacy, before)
    fly(fixed, after)
    control(.10, support=False, force=0.)
    control(.32, support=True, force=9.81)
    assert task.seen_imu_flight.item() and task.command_vx.item() == 0.
    for window, state in ((legacy, before), (fixed, after)):
        state["commands"][:, 0] = task.command_vx
        state["phase"][:] = Phase.LANDING
        state["contacts"][:] = True
        window.update(**state)
    assert axis(costs(legacy), "jump_landing_velocity").item() < 0.
    assert axis(costs(fixed), "jump_landing_velocity").item() == 0.
    assert axis(costs(fixed, vx=launch_vx), "jump_landing_velocity").item() < 0.
    # The unchanged privileged assessor demands two wheels on the correct top
    # after real flight and front crossing; continuing at its old reward target
    # is above its original 0.25 m/s landing threshold.
    def observe(x, bottom, contact, speed):
        centers = torch.zeros(1, 2, 3)
        centers[:, :, 0], centers[:, :, 2] = x, bottom + .06
        return task.observe(centers, torch.full((1, 2), contact, dtype=torch.bool),
            torch.tensor([.3 if not contact else 0.]), torch.tensor([speed]), torch.tensor([[0., 0., -1.]]),
            torch.tensor([True]), torch.tensor([True]))
    observe(-.25, 0., True, 0.)
    for x in (-.18, -.10, -.03, .04):
        observe(x, .26, False, launch_vx)
    for _ in range(20):
        observe(.3, .2, True, launch_vx)
    assert not task.success.item()
    for _ in range(20):
        observe(.3, .2, True, 0.)
    assert task.success.item()
    assert task.settle_seconds.item() == pytest.approx(.35)


@pytest.mark.parametrize("value", [1, "true", None])
def test_stop_target_flag_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        JumpLandingWindow(1, "cpu", .02, deepcopy(JUMP_LANDING_V1), step_jump_stop_target=value)


@pytest.mark.parametrize("mask", [torch.ones(2), torch.ones(2, 1, dtype=torch.bool)])
def test_step_task_identity_is_validated_before_acceptance(mask):
    window, state = window_state(stop=True)
    state["step_jump"] = mask
    with pytest.raises(ValueError, match="step_jump"):
        window.update(**state)
    assert not window.outcome_active.any()


def test_opt_in_runtime_declaration_preserves_default_and_rejects_implicit_flags():
    from wheeled_tasks.chassis.jump_landing_revision import runtime_expectation
    config = dict(jump_landing=deepcopy(JUMP_LANDING_V1), policy_dt=.02)
    legacy, _ = window_state()
    expected = runtime_expectation(config)
    assert legacy.runtime_state() == expected
    config["step_jump_landing_stop_target"] = False
    assert runtime_expectation(config) == expected
    config["step_jump_landing_stop_target"] = True
    fixed, _ = window_state(stop=True)
    assert fixed.runtime_state() == runtime_expectation(config)
    config["step_jump_landing_stop_target"] = 1
    with pytest.raises(ValueError, match="explicit boolean"):
        runtime_expectation(config)


def test_real_source_materialization_declares_and_authenticates_platform_stop_target(tmp_path):
    """Real recipe/design/worker path and independent preflight; no scene/parent."""
    from wheeled_tasks.chassis import integrated_production as production
    from wheeled_tasks.chassis import integrated_runtime as runtime
    from wheeled_tasks.chassis import integrated_sampling as sampling
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import validate_design_contract
    shutil.copytree(ROOT / "contracts", tmp_path / "contracts")
    shutil.copytree(ROOT / production.SCUT_DIRECTORY, tmp_path / production.SCUT_DIRECTORY)
    candidate = production.revised_recipe(ROOT, version=27)
    for file in [*runtime._references(ROOT, sampling.load_sources(ROOT)), runtime.AUDIT[0],
                 candidate["ordinary_terrain_transform"]["file"]]:
        destination = tmp_path / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, destination)
    files = production.artifact_files(27)
    unchanged_graph = {name: (tmp_path / files[name]).read_bytes()
                       for name in ("plan", "lineage", "manifest")}
    production._put(tmp_path, files["recipe"], production.revised_recipe(tmp_path, version=27))
    for domain, _ in production.lineage.CHAIN:
        design = production.revised_design(domain, tmp_path, version=27)
        production._put(tmp_path, production.design_file(domain, version=27), design)
        worker = production.canonical_worker(domain, tmp_path, version=27)
        assert production.validate_static_worker(worker, tmp_path) == domain
        effective = production.static_training_config(worker, tmp_path)
        inventory = production.production_source_files(worker, root=tmp_path)
        assert "src/wheeled_tasks/chassis/jump_landing.py" in inventory
        assert "src/wheeled_tasks/chassis/jump_landing_revision.py" in inventory
        row = design["stages"][domain]["roles"]["train"]
        if domain in ("speed", "terrain"):
            assert "step_jump_landing_stop_target" not in effective
            assert "step_jump_landing_stop_target" not in row["contract"]
            continue
        assert effective["step_jump_landing_stop_target"] is True
        assert row["contract"]["step_jump_landing_stop_target"] is True
        invocation = row["invocation"]
        seed = invocation["seed"].get("root_seed", invocation["seed"].get("value"))
        control = load_chassis_control(tmp_path / effective["control_math_source"])
        def validate(config, table=design):
            return validate_design_contract(config, table, control, stage=domain, role="train",
                seed=seed, num_envs=invocation["num_envs"], block_index=0)
        receipt = validate(effective)
        native = JumpLandingWindow(2, "cpu", effective["policy_dt"], effective["jump_landing"],
            step_jump_stop_target=effective["step_jump_landing_stop_target"])
        assert native.runtime_state() == receipt["runtime_expected"]["jump_landing"]
        for value in (None, False):
            altered = deepcopy(effective)
            if value is None:
                altered.pop("step_jump_landing_stop_target")
            else:
                altered["step_jump_landing_stop_target"] = value
            with pytest.raises(ValueError, match="step_jump_landing_stop_target"):
                validate(altered)
        # Reproduce the old recipe-only mistake: a true effective flag with
        # no independent contract field/runtime marker is rejected before Kit.
        legacy_design = deepcopy(design)
        legacy_row = legacy_design["stages"][domain]["roles"]["train"]
        legacy_row["contract"].pop("step_jump_landing_stop_target")
        legacy_row["runtime"]["jump_landing"].pop("step_jump_landing_stop_target")
        with pytest.raises(ValueError, match="step_jump_landing_stop_target"):
            validate(effective, legacy_design)
    # Only the candidate recipe/designs are materialized in this temporary tree;
    # no production plan, lineage or migration/future parent is manufactured.
    for name, original in unchanged_graph.items():
        assert (tmp_path / files[name]).read_bytes() == original
