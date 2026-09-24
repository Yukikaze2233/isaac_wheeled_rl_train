"""Known ability is retained; new frontiers require real boundary evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.adaptive_commands import AdaptiveCommandCurriculum
from wheeled_tasks.chassis.evaluation import summarize_evaluation_tiers
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract

ROOT = Path(__file__).resolve().parents[1]


def configuration(count=12288):
    load = lambda name: json.loads((ROOT / name).read_text())
    plan = resolve_plan(load("contracts/v5_nominal_recovery_v58.json"), load)
    return plan, stage_contract(load(plan["base_contract"]), plan, plan["stages"][0], count)


@pytest.mark.parametrize("count,updates,batches,warmup", [(8192, 1000, 4, 50), (12288, 667, 6, 34), (16384, 500, 8, 25)])
def test_recovery_budget_and_optimizer_batch_scale_together(count, updates, batches, warmup):
    plan, c = configuration(count)
    assert c["total_updates"] == updates
    assert c["num_mini_batches"] == batches
    assert c["critic_warmup_updates"] == warmup
    assert count * c["num_steps_per_env"] // batches == 49152
    assert c["learning_rate"] == 3e-5 and c["learning_rate_schedule"] == "fixed"
    assert c["evaluation"]["mode"] == "monitor"
    assert len(c["evaluation"]["cases"]) == 27
    assert [len(c["evaluation"]["tiers"][name]) for name in ("retain", "learn", "observe")] == [18, 4, 5]
    assert len(c["skill_specs"]) == 20
    retained = set(c["performance_curriculum"]["retained_groups"])
    assert sum(group["fraction"] for group in c["scene_groups"] if group["name"] in retained) == pytest.approx(.5)
    assert not c.get("height_tracking") and not c.get("height_workspace") and not c.get("stationary_tracking")
    assert not c["performance_curriculum"].get("height_course")
    for name, spec in c["skill_specs"].items():
        assert "height_sampling" not in spec
        assert spec["command"][2] == .305
    assert c["skill_specs"]["forward_3"]["sample_amplitude"]
    assert not c["skill_specs"]["rotate_4__reverse_train"]["sample_amplitude"]
    for name in plan["stages"][0]["training_group_exclusions"]:
        assert name not in c["skill_specs"]
    assert c["performance_curriculum"]["min_regression_reference_updates"] >= 250 * count / 4096


def curriculum():
    _, c = configuration()
    settings = deepcopy(c["performance_curriculum"])
    settings.update(window_episodes=2, min_reference_updates=1, min_regression_reference_updates=10,
                    settle_seconds=.02, minimum_scored_seconds=.02, retained_groups=["keep"],
                    frontier_amplitude_range=[1., 1.])
    settings["frontier_evidence"].update(steady_seconds=.02)
    specs = {"keep": {"kind": "forward", "command": [3., 0., .305], "episode_seconds": .1},
             "learn": {"kind": "rotate", "command": [0., -4., .305], "episode_seconds": .1}}
    return AdaptiveCommandCurriculum(["keep", "learn"], specs, settings, "cpu")


def episode(model, update=10, *, failed=False, actual_ratio=1., ticks=5):
    commands = torch.tensor([[3., 0., .305], [0., -4., .305]])
    model.sample_commands("keep", torch.tensor([0]), commands[:1], lambda n: torch.full((n,), .5))
    model.sample_commands("learn", torch.tensor([1]), commands[1:], lambda n: torch.full((n,), .5))
    commands[1, 1] *= actual_ratio
    for index in range(ticks):
        model.observe(torch.full((2,), .001), torch.zeros(2), torch.zeros(2), torch.ones(2, dtype=torch.bool),
            torch.full((2,), index == ticks - 1), torch.full((2,), failed), update, commands=commands)


def test_known_task_bypasses_beginner_caps_while_repair_requires_evidence():
    model = curriculum()
    episode(model)
    episode(model)
    assert model.state["keep"]["caps"] == [3., 0.]
    assert model.state["keep"]["revision"] == 0
    assert not model.windows["keep"]
    assert model.state["learn"]["caps"] == [0., 1.5]
    change = model.drain_changes()[0]
    assert change["old_caps"] == [0., 1.] and change["new_caps"] == [0., 1.5]
    assert change["scored_command_mean"] == [0., 1.]
    assert change["scored_seconds_mean"] > 0
    assert model.drain_changes() == []


def test_easy_or_short_successes_cannot_promote_and_early_failures_are_not_hidden():
    model = curriculum()
    episode(model, actual_ratio=.2)
    episode(model, ticks=2)
    assert model.state["learn"]["unqualified_episodes"] == 2
    assert len(model.windows["learn"]) == 0
    episode(model, actual_ratio=.2, failed=True)
    assert len(model.windows["learn"]) == 1 and model.windows["learn"][0][3] == 0
    assert model.state["learn"]["caps"][1] == 1.


def test_repeated_failure_cannot_move_boundary_faster_than_regression_cooldown():
    model = curriculum()
    model.state["learn"]["caps"][1] = 3.
    for update in (12, 15, 22):
        episode(model, update, failed=True)
        episode(model, update, failed=True)
        assert model.state["learn"]["caps"][1] == (2. if update == 22 else 2.5)
    changes = model.drain_changes()
    assert [change["reference_update"] for change in changes] == [12., 22.]


def test_evidence_and_retention_state_survive_checkpoint_roundtrip():
    model = curriculum()
    episode(model)
    state = model.state_dict()
    assert state["version"] == 3
    assert len(state["groups"]["learn"]["window"][0]) == 8
    restored = curriculum()
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_quantitative_retention_does_not_rewrite_case_grades_or_ignore_failures():
    case = {"command": [0., 0., .305], "passed": True, "height_mae_m": .005,
            "frames": 100, "episodes": 4, "requested_episodes": 4, "survival_rate": 1., "failures": 0}
    baseline = {"cases": {"stand": case}}
    candidate = {"cases": {"stand": {**case, "passed": False, "height_mae_m": .006}}}
    settings = {"cases": [{"name": "stand", "command": case["command"]}],
                "tiers": {"retain": ["stand"], "learn": [], "observe": []},
                "retention_margins": {"height_mae_m": .002}}
    summary = summarize_evaluation_tiers(candidate, settings, baseline)
    assert summary["retention_within_margin"]
    assert summary["tiers"]["retain"]["passed"] == 0
    assert not candidate["cases"]["stand"]["passed"]
    candidate["cases"]["stand"].update(failures=1, survival_rate=.75)
    assert not summarize_evaluation_tiers(candidate, settings, baseline)["retention_within_margin"]
