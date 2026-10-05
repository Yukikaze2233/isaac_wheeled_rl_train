"""Independent declared-task coverage, including recovery's legacy pool names.

This is a teaching-scope change: 872 recovery rehearsal lanes gain ordinary
tilt/push sampling. Recovery has no instantiated nonstandard leg bank, so this
does not add actual bank hold/capture. The generic mask can retain eligibility
for those mechanisms only in workers that instantiate the optional bank.
It is separate from the performance candidate's fixed-mask equivalence tests.
The oracle uses an explicit 41-group rehearsal catalogue and task semantics;
it neither imports ORDINARY_POOLS nor asks the helper to compute expectations.
"""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.ordinary_posture_curriculum import (
    OrdinaryPostureCurriculum, ordinary_eligibility, posture_settings)
from wheeled_tasks.chassis.task import choose_scene_groups


ROOT = Path(__file__).resolve().parents[1]
POPULATIONS = {"recovery": 1449, "speed": 3928, "terrain": 4096, "jump": 1231, "stepjump": 1228}
TRANSLATION_16 = (
    "forward_05", "forward_05__height_train", "backward_05", "backward_05__height_train",
    "forward_1", "forward_1__height_train", "backward_1", "backward_1__height_train",
    "forward_continuous_low", "forward_continuous_low__height_train",
    "backward_continuous_low", "backward_continuous_low__height_train",
    "curve_low", "curve_low__height_train", "curve_low__reverse_train",
    "curve_low__reverse_train__height_train")
TRANSITIONS_6 = ("start_stop_05", "start_stop_05__height_train", "start_stop_1",
    "start_stop_1__height_train", "velocity_curve_1", "velocity_curve_1__height_train")
YAW_8 = ("rotate_1", "rotate_1__height_train", "rotate_1__reverse_train",
    "rotate_1__reverse_train__height_train", "yaw_continuous_low",
    "yaw_continuous_low__height_train", "yaw_continuous_low__reverse_train",
    "yaw_continuous_low__reverse_train__height_train")
LEGACY_30 = TRANSLATION_16 + TRANSITIONS_6 + YAW_8
CORE_41 = frozenset(LEGACY_30 + tuple("stand_" + str(height) + "mm"
    for height in (230, 250, 280, 305, 320, 350, 380, 400, 430))
    + ("stand_continuous", "height_scan"))


def worker(domain):
    return json.loads((ROOT / f"contracts/v6_task_campaign_{domain}_worker_v18.json").read_text())


def lane_inputs(config):
    rows = choose_scene_groups(config["scene_groups"], 4096)
    groups, kinds = zip(*rows)
    specs = config["skill_specs"]
    recovery = torch.tensor([bool(specs[group].get("recovery_pose")) for group in groups])
    # This is SkillCommands' actual world-spin flag, including spin_translate.
    spin = torch.tensor([specs[group].get("kind") == "spin_translate" for group in groups])
    return groups, kinds, recovery, spin


def asserted_ordinary_spec(spec):
    assert spec.get("mode", 0) != 4
    assert not any(spec.get(key) for key in ("request_events", "command_sequence", "recovery_pose"))
    assert spec["kind"] not in ("jump", "airborne", "landing", "recovery", "spin", "spin_translate")


def independent_oracle(domain, config, groups, kinds):
    selected = []
    for group, terrain in zip(groups, kinds):
        spec = config["skill_specs"][group]
        if domain in ("recovery", "jump", "stepjump"):
            ordinary = group in CORE_41
        elif domain == "speed":
            ordinary = spec["kind"] != "spin_translate"
        else:
            assert domain == "terrain"
            ordinary = True
        if ordinary:
            asserted_ordinary_spec(spec)
            assert terrain != "jump"
        selected.append(ordinary)
    return torch.tensor(selected, dtype=torch.bool)


@pytest.mark.parametrize("group", LEGACY_30)
def test_every_legacy_recovery_motion_rehearsal_is_ordinary_without_requests(group):
    config = worker("recovery")
    expected_pool = ("translation" if group in TRANSLATION_16 else
        "transitions" if group in TRANSITIONS_6 else "yaw")
    assert config["behavior_pool_membership"][group] == expected_pool
    spec = config["skill_specs"][group]
    assert spec["mode"] == 1
    asserted_ordinary_spec(spec)
    scene = next(row for row in config["scene_groups"] if row["name"] == group)
    assert scene["terrain"] == ["flat"]
    assert ordinary_eligibility(config, [group], ["flat"], "cpu").tolist() == [True]


@pytest.mark.parametrize("domain, expected_count", POPULATIONS.items())
def test_actual_4096_population_matches_independent_task_oracle(domain, expected_count):
    config = worker(domain)
    groups, kinds, recovery, spin = lane_inputs(config)
    expected = independent_oracle(domain, config, groups, kinds)
    assert int(expected.sum()) == expected_count
    actual = ordinary_eligibility(config, groups, kinds, "cpu", recovery=recovery, spin=spin)
    assert torch.equal(actual, expected)
    assert not (actual & (recovery | spin)).any()
    if domain == "recovery":
        assert config.get("nonstandard_start") is None
        counts = Counter(groups)
        assert {name for name, pool in config["behavior_pool_membership"].items()
            if pool in ("translation", "transitions", "yaw")} == set(LEGACY_30)
        assert {counts[name] for name in TRANSLATION_16} == {23}
        assert {counts[name] for name in TRANSITIONS_6} == {48}
        assert {counts[name] for name in YAW_8} == {27}
        added = torch.tensor([group in LEGACY_30 for group in groups])
        assert int(added.sum()) == 872 and int((actual & ~added).sum()) == 577
        # This is the generic optional-bank predicate; recovery has no such
        # bank instantiated. Its actual added mechanisms are ordinary tilt/push.
        potential_nonstandard = torch.tensor([terrain != "jump" for terrain in kinds]) & ~recovery & ~spin
        assert torch.equal(potential_nonstandard & actual, expected)


def test_all_54_scripted_recovery_groups_remain_excluded_even_with_a_legacy_pool_name():
    config = worker("recovery")
    groups = [name for name, spec in config["skill_specs"].items() if spec.get("recovery_pose")]
    assert len(groups) == 54
    assert not ordinary_eligibility(config, groups, ["flat"] * len(groups), "cpu").any()
    # Pool alias acceptance cannot bypass a real scripted-task spec.
    config["behavior_pool_membership"].update({group: "translation" for group in groups})
    assert not ordinary_eligibility(config, groups, ["flat"] * len(groups), "cpu").any()


@pytest.mark.parametrize("pool", ("translation", "transitions", "yaw"))
@pytest.mark.parametrize("fault", ("mode4", "request", "sequence", "recovery_pose",
    "jump", "airborne", "landing", "spin", "jump_terrain", "script_mask", "spin_mask"))
def test_legacy_alias_does_not_expand_request_or_scripted_ownership(pool, fault):
    spec = {"kind": "forward", "mode": 1}
    terrain, recovery, spin = "flat", None, None
    if fault == "mode4":
        spec["mode"] = 4
    elif fault == "request":
        spec["request_events"] = [{"time_s": 1.}]
    elif fault == "sequence":
        spec["command_sequence"] = [{"kind": "jump"}]
    elif fault == "recovery_pose":
        spec["recovery_pose"] = "upright"
    elif fault == "jump_terrain":
        terrain = "jump"
    elif fault == "script_mask":
        recovery = torch.tensor([True])
    elif fault == "spin_mask":
        spin = torch.tensor([True])
    else:
        spec["kind"] = fault
    config = {"skill_specs": {"row": spec}, "behavior_pool_membership": {"row": pool}}
    assert ordinary_eligibility(config, ["row"], [terrain], "cpu",
        recovery=recovery, spin=spin).tolist() == [False]


@pytest.mark.parametrize("domain", POPULATIONS)
def test_fixed_evaluation_keeps_posture_and_nonstandard_birth_disabled(domain):
    config = worker(domain)
    before = deepcopy(config)
    fixed = fixed_suite_contract(config)
    assert config == before
    assert fixed["ordinary_posture_curriculum"]["enabled"] is False
    if "nonstandard_start" in fixed:
        assert fixed["nonstandard_start"]["enabled"] is False
    # Even a broad ordinary mask cannot enable the fixed-exam runtime.
    lesson = OrdinaryPostureCurriculum(8, "cpu", fixed["ordinary_posture_curriculum"],
        torch.ones(8, dtype=torch.bool), evaluation=True)
    gen = torch.Generator().manual_seed(51)
    state = gen.get_state().clone()
    lesson.reset(torch.arange(8), 5000, generator=gen)
    assert not lesson.enabled and lesson.birth_population == 0 and not lesson.selected.any()
    assert torch.equal(state, gen.get_state())
    assert not lesson.push_mask(torch.ones(8, dtype=torch.bool), torch.ones(8, dtype=torch.bool),
        torch.ones(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool), torch.zeros(8, dtype=torch.bool)).any()


def test_flat_worker_keeps_its_original_hash_and_no_posture_hook():
    plan = json.loads((ROOT / "contracts/v6_task_campaign_v18.json").read_text())
    flat = next(row for row in plan["domains"] if row["name"] == "flat")
    raw = (ROOT / flat["contract"]).read_bytes()
    assert sha256(raw).hexdigest() == flat["sha256"]
    assert "ordinary_posture_curriculum" not in json.loads(raw)
    lesson = OrdinaryPostureCurriculum(8, "cpu", posture_settings(0, enabled=False),
        torch.ones(8, dtype=torch.bool))
    assert not lesson.enabled and not lesson.eligible.any()
