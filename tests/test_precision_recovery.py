"""Precision, phase isolation, retained sampling and nonrefundable rollback budgets."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract
from wheeled_tasks.chassis.policy_transfer import resume_budget, validate_reward_resume
from wheeled_tasks.chassis.rewards import precision_tracking_terms

ROOT = Path(__file__).resolve().parents[1]


def materialize(name, protected=None):
    load = lambda path: json.loads((ROOT / path).read_text())
    plan = resolve_plan(load(name), load)
    return plan, [stage_contract(load(plan["base_contract"]), plan, recipe, 16384,
                                protected_cases=protected) for recipe in plan["stages"]]


@pytest.fixture(scope="module")
def stages():
    return materialize("contracts/v6_scut35_precision_v1.json")[1]


def terms(settings, errors=(0., .05, .1, .257), *, jumping=False, terrain=0., stationary=False):
    e = torch.tensor(errors, dtype=torch.float64, requires_grad=True)
    zero = torch.zeros_like(e)
    vx = 0. if stationary else .5
    commands = torch.stack((zero + vx, zero, zero + .305), -1)
    velocity = torch.stack((vx - e, zero, zero), -1)
    gravity = torch.stack((zero, zero, zero - 1), -1)
    reference = SimpleNamespace(jumping=torch.full_like(e, jumping, dtype=torch.bool), terrain_mode=zero + terrain)
    wheel = torch.zeros(len(e), 2, 3, dtype=e.dtype)
    wheel[:, 0, 0] = e
    result = precision_tracking_terms(velocity, gravity, .305 + e * .1, commands,
                                     zero + 1, zero + 1, reference, wheel, settings)
    return e, result


def test_precision_distinguishes_tolerance_errors_and_has_finite_gradients(stages):
    e, result = terms(stages[0]["precision_tracking"])
    for name in ("velocity_precision", "height_precision"):
        values = result[name]
        assert values[0] == 0 and (values.diff() < 0).all()
        slope = torch.autograd.grad(values.sum(), e, retain_graph=True)[0]
        assert torch.isfinite(slope).all() and (slope[1:] < 0).all()
    assert result["velocity_precision"][-1] < -.9
    assert result["height_precision"][2] < -.6


def test_special_skills_keep_phase_local_objectives(stages):
    cfg = stages[0]["precision_tracking"]
    for jumping, terrain in ((True, 0.), (False, 1.), (False, -1.)):
        _, result = terms(cfg, jumping=jumping, terrain=terrain)
        assert not result["velocity_precision"].any()
        assert not result["lateral_precision"].any()
        assert not result["stand_precision"].any()
        assert bool(result["height_precision"].any()) is not jumping
    _, stand = terms(cfg, stationary=True)
    assert not stand["velocity_precision"].any() and stand["stand_precision"][1] < -.45


def test_fork_penalty_is_continuous_at_the_original_tolerance(stages):
    e, result = terms(stages[0]["precision_tracking"], (.049999, .05, .050001, .07, .1))
    fork = result["no_fork"]
    assert not fork[:2].any() and abs(float(fork[2].detach())) < 1e-7
    assert (fork[2:].diff() < 0).all() and fork.min() >= -1.
    assert torch.isfinite(torch.autograd.grad(fork.sum(), e)[0]).all()


def test_reward_migration_keeps_acceptance_physics_and_total_budget(stages):
    _, old = materialize("contracts/v6_scut35_continuation_v1.json")
    receipt = validate_reward_resume(old[0], stages[0])
    assert receipt["actor_critic_optimizer"] == "preserved"
    assert sum(s["total_updates"] for s in stages) == 7000
    for stage in stages:
        assert stage["precision_tracking"] == stages[0]["precision_tracking"]
        assert stage["evaluation"]["regression_recovery"]["max_retries"] == 2
        assert stage["evaluation"]["cases"] == old[0]["evaluation"]["cases"]
        assert stage["command_transport"]["delay_ms"] == [.1, 5.]
    for key, value in (("asset_manifest_sha256", "changed"), ("policy_dt", .01)):
        changed = deepcopy(stages[0])
        changed[key] = value
        with pytest.raises(ValueError, match="undeclared semantics"):
            validate_reward_resume(old[0], changed)
    changed = deepcopy(stages[0])
    changed["evaluation"]["height_mae_m_max"] = .1
    with pytest.raises(ValueError, match="acceptance criteria"):
        validate_reward_resume(old[0], changed)


def test_later_stages_reserve_retained_skills_without_shrinking_speed_stand():
    protected = ["stand", "rotate_1rps", "rotate_1rps_perturbed", "rotate_1_reverse",
                 "forward_2__mu_010", "jump_small"]
    _, stages = materialize("contracts/v6_scut35_precision_v1.json", protected)
    for stage in stages:
        mapping = stage["protected_rehearsal"]["case_to_group"]
        assert set(mapping) == set(protected)
        assert mapping["rotate_1rps"] == mapping["rotate_1rps_perturbed"]
        weights = {g["name"]: g["fraction"] for g in stage["scene_groups"]}
        assert sum(weights.values()) == pytest.approx(1.)
        floor = .25 / len(set(mapping.values()))
        assert all(weights[group] >= floor - 1e-9 for group in mapping.values())
        pools = stage["behavior_pool_fractions"]
        if "speed_retention" in pools:
            assert pools["speed_retention"] >= .2 - 1e-9
            assert pools["stand_retention"] >= .15 - 1e-9


def test_discarded_updates_are_charged_and_learning_lineage_is_explicit():
    infos = {"successful_updates_total": 100, "training_transitions": 100 * 16384 * 24}
    budget = resume_budget(infos, 300, 16384 * 24)
    assert budget["parent_updates"] == 300 and budget["learning_lineage_updates"] == 100
    assert budget["parent_training_transitions"] == 300 * 16384 * 24
    assert budget["discarded_updates_charged"] == 200
    for spent, batch in ((99, 16384 * 24), (300, 64 * 24)):
        with pytest.raises(ValueError):
            resume_budget(infos, spent, batch)
    fork = {"successful_updates_total": 112, "learning_lineage_updates": 112,
            "training_transitions": infos["training_transitions"] + 12 * 512 * 24,
            "batch_transitions": 512 * 24}
    assert resume_budget(fork, 120, 512 * 24)["parent_training_transitions"] == fork["training_transitions"] + 8 * 512 * 24


def test_full_passed_catalog_can_be_rehearsed_even_if_every_original_group_is_protected(stages):
    from wheeled_tasks.chassis.task import choose_scene_groups
    names = [case["name"] for case in stages[0]["evaluation"]["cases"]]
    _, protected_stages = materialize("contracts/v6_scut35_precision_v1.json", names)
    for stage in protected_stages:
        assert len(choose_scene_groups(stage["scene_groups"], 16384)) == 16384
        assert all(g["fraction"] > 0 for g in stage["scene_groups"])


def test_materialized_contract_bytes_are_stable_across_process_hash_seeds():
    code = ("import json,hashlib; from pathlib import Path; "
            "from wheeled_tasks.chassis.full_curriculum import resolve_plan,stage_contract; "
            "load=lambda p:json.loads(Path(p).read_text()); "
            "p=resolve_plan(load('contracts/v6_scut35_precision_v1.json'),load); "
            "c=stage_contract(load(p['base_contract']),p,p['stages'][0],16384); "
            "print(hashlib.sha256(json.dumps(c,indent=2).encode()).hexdigest())")
    values = [subprocess.check_output([sys.executable, "-B", "-c", code], cwd=ROOT,
              env={**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONHASHSEED": seed}, text=True)
              for seed in ("17", "618")]
    assert values[0] == values[1]
