"""Native coverage and per-request evidence for the versioned v17 candidates."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis import scut_jump_revision as revision
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.jump_landing import JUMP_LANDING_V1, JumpLandingWindow
from wheeled_tasks.chassis.nonstandard_start import birth_probabilities, validate_nonstandard_start
from wheeled_tasks.chassis.preflight import ROLES, design_preflight
from wheeled_tasks.chassis.step_jump import StepJumpTask
from wheeled_tasks.chassis.task import Phase, choose_scene_groups


ROOT = Path(__file__).resolve().parents[1]


def read(file):
    return json.loads((ROOT / file).read_text())


@pytest.mark.parametrize("domain,group_count", [("jump", 56), ("stepjump", 57)])
def test_complete_flat_coverage_is_permanent_with_native_integer_population(domain, group_count):
    worker = read(revision.worker_file(domain))
    flat = read(revision.FLAT_FILE)
    counts = Counter(name for name, _ in choose_scene_groups(worker["scene_groups"], 4096))
    assert len(counts) == group_count and sum(counts.values()) == 4096
    assert set(flat["skill_specs"]) <= set(counts)
    assert min(counts.values()) >= 1
    for name, spec in flat["skill_specs"].items():
        assert worker["skill_specs"][name] == spec
        group = next(g for g in worker["scene_groups"] if g["name"] == name)
        assert group["terrain"] == ["flat"]
    membership = worker["behavior_pool_membership"]
    flat_count = sum(counts[name] for name in flat["skill_specs"])
    static_count = sum(count for name, count in counts.items()
        if membership[name] in ("stand_anchors", "stand_continuous"))
    assert flat_count == pytest.approx(.30 * 4096, abs=10)
    assert static_count == pytest.approx(.10 * 4096, abs=10)
    for name in counts:
        for partner in (name + "__height_train", name + "__reverse_train",
                        name.replace("forward_", "backward_", 1)):
            if partner != name and partner in counts:
                assert abs(counts[name] - counts[partner]) <= 1
    metadata = worker["task_campaign"]["sampling_revision"]
    assert metadata["allocation_clock"] == "constant_all_actor_updates"
    assert metadata["outcome"] == "unassessed_research_candidate"


@pytest.mark.parametrize("domain", revision.SOURCES)
def test_exact_predecessor_learning_physics_and_success_criteria_are_preserved(domain):
    before = read(revision.SOURCES[domain][0])
    after = read(revision.worker_file(domain))
    assert after == revision.build_worker(domain)
    assert after["design_preflight"]["sha256"] == hashlib.sha256(
        (ROOT / revision.design_file(domain)).read_bytes()).hexdigest()
    changed = {"scene_groups", "skill_specs", "behavior_pool_membership", "behavior_pool_fractions",
        "task_campaign", "design_preflight", "nonstandard_start", "dense_tracking",
        "jump_cycle_reset_on_accepted"}
    assert {k: v for k, v in before.items() if k not in changed} == {
        k: v for k, v in after.items() if k not in changed}
    assert (after["physics_dt"], after["policy_dt"], after["actor_dim"], after["critic_dim"],
            after["target_num_envs"]) == (.005, .02, 35, 81, 4096)
    assert after["learning_rate_schedule"] == "fixed" and after["learning_rate"] == 1e-4
    assert after["task_campaign"]["revision_migration"] == revision.revision_binding(domain)
    assert after["task_campaign"]["clock"] == "domain_actor_updates"
    assert after["task_campaign"]["updates"] == 20000
    assert after["task_campaign"]["evaluation_interval"] == 0
    assert after["evaluation"] == before["evaluation"] and len(after["evaluation"]["cases"]) == 322
    assert after["hardware_deployment_ready"] is False
    assert after["physics_prior_scope"]["nominal_unidentified_axes"] == [
        "L_joint1", "LL_joint1", "R_joint1", "RR_joint1"]


@pytest.mark.parametrize("domain", revision.SOURCES)
def test_original_jump_probability_assist_and_request_clocks_are_not_restarted(domain):
    before = read(revision.SOURCES[domain][0])
    worker = read(revision.worker_file(domain))
    assert worker["new_asset_curriculum"] == before["new_asset_curriculum"]
    assert worker["jump_assist"] == before["jump_assist"]
    curriculum = worker["new_asset_curriculum"]["jump_curriculum"]
    assert curriculum["bins_m"] == [.03, .06, .10, .15, .20, .25, .30]
    assert curriculum["minimum_probability"] == .05
    assert curriculum["clock"] == "domain_actor_updates"
    settings = worker["nonstandard_start"]
    validate_nonstandard_start(settings)
    assert settings["origin_actor_update"] == revision.ORIGINS[domain]
    assert settings["hold_seconds"] == .2
    assert settings["application"] == "zero_all_active_motor_effort_after_response"
    assert birth_probabilities(settings, settings["origin_actor_update"] - int(domain == "jump")) == [.85, .15, 0.]
    assert birth_probabilities(settings, settings["origin_actor_update"] + 1500) == [.65, .20, .15]
    # The deployed sampler excludes jump terrains, step requests and recovery;
    # all new eligible lanes are ordinary flat rehearsal rather than active jumps.
    for group in worker["scene_groups"]:
        spec = worker["skill_specs"][group["name"]]
        active = spec["kind"] in ("jump", "running_jump", "step_jump", "double_step_jump")
        if active:
            assert group["terrain"] != ["flat"] or spec.get("step_jump")


def test_stepjump_binds_the_actual_new_jump_parent_and_keeps_incremental_double_geometry():
    worker = read(revision.worker_file("stepjump"))
    parent = revision.worker_file("jump")
    assert worker["task_campaign"]["source_contract_file"] == parent
    assert worker["task_campaign"]["source_contract_sha256"] == hashlib.sha256((ROOT / parent).read_bytes()).hexdigest()
    assert "jump_landing" not in worker  # raised COM/landing-region objectives remain independent
    assert worker["task_campaign"]["sampling_revision"]["raised_landing_scope"].startswith("legacy_COM_displacement")
    for source in ("manual", "laser"):
        spec = worker["skill_specs"][f"double_step_jump_200_250_{source}"]
        assert spec["step_jump"]["rises_m"] == [.2, .25]
        assert spec["step_jump"]["treads_m"] == [.8, .6]
        assert spec["terrain_limits"]["stair_heights_m"] == [.2, .45]
        assert spec["step_jump"]["trigger_source"] == source
        task = StepJumpTask([f"double_step_jump_200_250_{source}"], worker["skill_specs"], "cpu", .02)
        torch.testing.assert_close(task.tops, torch.tensor([[.20, .45]]))
        assert task.settle_seconds.item() == pytest.approx(.35)
        assert not task.success.item()


@pytest.mark.parametrize("domain", revision.SOURCES)
def test_five_independent_design_roles_keep_old_fixed_criteria_and_reward_paths(domain):
    worker = read(revision.worker_file(domain))
    before = read(revision.SOURCES[domain][0])
    design = read(revision.design_file(domain))
    old = read(before["design_preflight"]["file"])
    assert design == revision.revised_design(domain)
    assert set(design["stages"][domain]["roles"]) == set(ROLES)
    control = load_chassis_control(ROOT / worker["control_math_source"])
    for role in ROLES:
        row = design["stages"][domain]["roles"][role]
        effective = deepcopy(worker)
        seed, count = 290927, 4096
        if role != "train":
            seed = 290930 if role == "confirmation" else 290929
            effective["evaluation"]["seed"] = seed
            names = [c["name"] for c in row["contract"]["evaluation"]["cases"]]
            effective["evaluation"] = select_evaluation_cases(effective["evaluation"], names)
            effective = fixed_suite_contract(effective)
            count = len(names) * effective["evaluation"]["episodes_per_case"]
            for key in ("evaluation", "scene_groups", "skill_specs", "behavior_pool_membership",
                        "behavior_pool_fractions", "dense_tracking", "task_semantics"):
                assert row["contract"][key] == old["stages"][domain]["roles"][role]["contract"][key]
            assert row["contract"]["nonstandard_start"]["enabled"] is False
        receipt = design_preflight(effective, control, ROOT, stage=worker["enabled_stages"][0], role=role,
                                  seed=seed, num_envs=count, block_index=0)
        assert receipt["status"] == "static_passed"


@pytest.mark.parametrize("domain", revision.SOURCES)
def test_bound_source_mutation_fails_without_copying_the_contract_corpus(tmp_path, domain):
    file, sha = revision.SOURCES[domain]
    path = tmp_path / file
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / file).read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA differs"):
        revision.read_bound(file, sha, tmp_path)


def landing_rig(count=1):
    observer = JumpLandingWindow(count, "cpu", .02, deepcopy(JUMP_LANDING_V1))
    state = dict(request=torch.ones(count, dtype=torch.bool), phase=torch.full((count,), Phase.TAKEOFF),
        contacts=torch.ones(count, 2, dtype=torch.bool), com_released=torch.zeros(count, dtype=torch.bool),
        clear_air_time_peak=torch.zeros(count), raw_height=torch.full((count,), .305),
        commands=torch.tensor([[.4, 0., .305]]).repeat(count, 1), goal_apex=torch.full((count,), .28),
        landing_delta=torch.full((count,), .20), height=torch.full((count,), .305), com_vz=torch.zeros(count))
    observer.update(**state)
    state.update(phase=torch.full((count,), Phase.FLIGHT), contacts=torch.zeros(count, 2, dtype=torch.bool),
        com_released=torch.ones(count, dtype=torch.bool), clear_air_time_peak=torch.full((count,), .08))
    observer.update(**state)
    state.update(phase=torch.full((count,), Phase.LANDING), contacts=torch.ones(count, 2, dtype=torch.bool))
    observer.update(**state)
    return observer, state


def test_second_accepted_cycle_before_six_seconds_clears_only_its_lane_and_latches_new_goals():
    observer, state = landing_rig(2)
    for _ in range(20):
        observer.update(**state)
    assert observer.elapsed[0] < 3. and observer.request_elapsed[0] < 6.
    unaffected = {k: v[1].clone() for k, v in vars(observer).items() if isinstance(v, torch.Tensor)}
    observer.reset_accepted(torch.tensor([0]))
    assert not observer.active[0] and not observer.verified_flight[0] and not observer.outcome_active[0]
    assert observer.elapsed[0] == observer.request_elapsed[0] == 0.
    for key, value in unaffected.items():
        assert torch.equal(getattr(observer, key)[1], value)
    state["phase"][0] = Phase.TAKEOFF
    state["com_released"][0] = False
    state["clear_air_time_peak"][0] = 0.
    state["landing_delta"][0] = .25
    state["goal_apex"][0] = .33
    state["raw_height"][0] = .32
    observer.update(**state)
    assert observer.cycle_accepted[0] and observer.armed[0]
    assert observer.landing_delta[0] == pytest.approx(.25)
    assert observer.goal_apex[0] == pytest.approx(.33)
    assert observer.goal_height[0] == pytest.approx(.32)
    # Contact without a new verified flight cannot borrow the first cycle.
    state["phase"][0] = Phase.LANDING
    observer.update(**state)
    assert not observer.active[0]
    state["phase"][0] = Phase.FLIGHT
    state["contacts"][0] = False
    state["com_released"][0] = True
    state["clear_air_time_peak"][0] = .08
    observer.update(**state)
    state["contacts"][0] = True
    state["phase"][0] = Phase.LANDING
    observer.update(**state)
    assert observer.active[0] and observer.elapsed[0] == 0.


def test_legacy_update_only_cannot_renew_first_cycle_on_a_busy_repeat():
    observer, state = landing_rig()
    initial_goal = observer.goal_height.clone()
    initial_delta = observer.landing_delta.clone()
    state["phase"].fill_(Phase.TAKEOFF)
    state["raw_height"].fill_(.43)
    state["landing_delta"].fill_(.25)
    state["accepted_event"] = torch.ones(1, dtype=torch.bool)
    observer.update(**state)
    assert not observer.cycle_accepted.item()
    assert torch.equal(observer.goal_height, initial_goal)
    assert torch.equal(observer.landing_delta, initial_delta)
    assert observer.elapsed.item() == pytest.approx(.02)


def test_new_request_cannot_inherit_full_task_release_apex_or_clear_air_evidence():
    task = FullTaskSemantics(2, "cpu", .02, dict(jump_apex_frame="com_release"))
    position = torch.tensor([[.5, 0., .305], [1., 0., .305]])
    for value in vars(task).values():
        if isinstance(value, torch.Tensor):
            value.fill_(1)
    task.reset_accepted(torch.tensor([0]), position)
    for name in ("com_released", "com_release_pending", "com_start_valid", "com_release_height",
                 "com_start_height", "com_rise", "com_release_speed", "com_displacement",
                 "release_recorded", "clear_air_time", "clear_air_time_peak", "height_peak"):
        assert getattr(task, name)[0] == 0 and getattr(task, name)[1] == 1
    torch.testing.assert_close(task.origin_xy[0], position[0, :2])
    task.observe_com(torch.tensor([4, 4]), torch.tensor([Phase.TAKEOFF, Phase.GROUND]),
        torch.tensor([.75, .305]), torch.zeros(2), requested=torch.tensor([True, False]),
        contacts=torch.ones(2, 2, dtype=torch.bool))
    assert not task.com_released[0] and task.com_rise[0] == 0.
    assert task.com_start_height[0] == pytest.approx(.75)


def test_native_second_riser_accepts_before_old_outcome_timeout_without_borrowing_first_flight():
    from test_step_jump_task import (control, finish_command, fly_and_land, launch,
                                    make_task, observe)
    task = make_task(double=True)
    first_time = launch(task)
    assert fly_and_land(task, index=0) == 1
    assert task.completed[0, 0] and not task.completed[0, 1]
    time = finish_command(task, first_time)
    for _ in range(300):
        time += .02
        if control(task, time).any():
            break
    else:
        raise AssertionError("Native controller did not accept the second riser")
    assert time - first_time < 6.
    assert task.request_count.item() == 2 and task.assessment_index.item() == 1
    assert task.landing_delta.item() == pytest.approx(.25)
    assert not task.air_seen.item() and not task.front_clear.any()
    for _ in range(25):
        observe(task, 1.1, .45)
    assert not task.completed[0, 1] and not task.success.item()
    assert fly_and_land(task, index=1) == 1 and task.success.item()

    # The independent cost observer also needs a new cycle even if the first
    # request's six-second outcome deadline has not expired. This is opt-in;
    # raised v17 workers retain their dedicated COM/platform reward path.
    observer, state = landing_rig()
    for _ in range(round((time - first_time) / .02)):
        observer.update(**state)
    assert observer.outcome_active.item()
    observer.reset_accepted(torch.tensor([0]))
    state.update(phase=torch.tensor([Phase.TAKEOFF]), com_released=torch.tensor([False]),
                 clear_air_time_peak=torch.zeros(1), landing_delta=torch.tensor([.25]),
                 goal_apex=torch.tensor([.33]))
    observer.update(**state)
    assert observer.armed.item() and observer.cycle_accepted.item()
    assert not observer.verified_flight.item() and observer.request_elapsed.item() == 0.


@pytest.mark.parametrize("bad", [torch.tensor([True]), torch.tensor([[0]]), [0]])
def test_accepted_observer_reset_requires_controller_index_ids(bad):
    observer, _ = landing_rig()
    with pytest.raises(ValueError, match="Accepted jump IDs"):
        observer.reset_accepted(bad)
