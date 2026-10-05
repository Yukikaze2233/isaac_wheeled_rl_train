"""Bounded probe authenticates real recipes and reports missing native coverage."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.jump_landing import JUMP_LANDING_V1, JumpLandingWindow

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_scut_curriculum", ROOT / "scripts/probe_scut_curriculum.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize("domain", probe.ORIGINS)
def test_exact_four_train_invocations_and_source_dependencies(domain):
    cfg = json.loads((ROOT / f"contracts/v6_task_campaign_{domain}_worker_v17.json").read_text())
    design = json.loads((ROOT / cfg["design_preflight"]["file"]).read_text())
    stage, seed, block = probe.invocation(design, cfg)
    assert stage == cfg["enabled_stages"][0]
    assert type(seed) is int and block == 0
    sources = probe.source_snapshot(cfg)
    assert all((ROOT / file).is_file() and len(sha) == 64 for file, sha in sources.items())
    assert "scripts/probe_scut_curriculum.py" in sources
    design["stages"][cfg["design_preflight"]["stage"]]["roles"]["train"]["invocation"]["num_envs"] = 256
    with pytest.raises(RuntimeError, match="4096"):
        probe.invocation(design, cfg)


def environment(count=2):
    return NS(recovery=None, nonstandard_start=None, jump_cycle_reset_on_accepted=False,
        num_envs=count, device="cpu", cfg={}, control_audit=NS(last=dict(
            pd_updates=4, feedback_refreshes=4, target_decodes=1, physics_steps=4)))


def test_finite_native_pd_and_interval_masks_are_hard_structural_checks():
    env = environment()
    audit = probe.StructuralAudit(env, "terrain")
    data = dict(gap=torch.tensor([.001, .002]), terminated=torch.zeros(2, dtype=torch.bool),
        **{name: torch.zeros(2, 6) for name in ("motor_position", "motor_velocity", "motor_effort", "velocity", "omega", "height")})
    extras = dict(diagnostics=data, policy_learning_mask=torch.ones(2, dtype=torch.bool))
    audit.observe(1, torch.zeros(2), torch.zeros(2, dtype=torch.bool), extras)
    assert audit.summary()["counts"]["all_policy_intervals"] == 2
    extras["policy_learning_mask"][0] = False
    with pytest.raises(RuntimeError, match="ownership"):
        audit.observe(1, torch.zeros(2), torch.zeros(2, dtype=torch.bool), extras)
    extras["policy_learning_mask"][0] = True
    env.control_audit.last["pd_updates"] = 3
    with pytest.raises(RuntimeError, match="four fresh"):
        audit.observe(1, torch.zeros(2), torch.zeros(2, dtype=torch.bool), extras)


def test_unseen_recovery_handoff_never_becomes_functional_credit():
    audit = probe.StructuralAudit(environment(), "recovery")
    summary = audit.summary()
    assert summary["coverage"]["functional_qualification"] is False
    assert summary["coverage"]["continuous_handoff_observed"] is False
    assert "not functionally qualified" in summary["coverage"]["limitation"]
    audit.counts.update(recovery_blend_intervals=10, recovery_pure_rl_observed_intervals=30)
    assert audit.summary()["coverage"]["continuous_handoff_observed"] is True
    assert audit.summary()["coverage"]["functional_qualification"] is False


def test_request_birth_exclusion_checks_actual_selected_lanes():
    env = environment()
    env.scene_groups = ["stand", "request_forward"]
    env.cfg = {"behavior_pool_membership": {"stand": "stand_anchors", "request_forward": "jump_requests"}}
    env.mode = torch.tensor([0, 4])
    env.nonstandard_start_eligible = torch.tensor([True, False])
    env.nonstandard_start = NS(selected=torch.tensor([True, False]))
    audit = probe.StructuralAudit.__new__(probe.StructuralAudit)
    audit.env, audit.domain, audit.counts = env, "jump", Counter()
    audit.check_birth_exclusion()
    assert audit.counts["birth_eligible_request_lanes"] == 0
    env.nonstandard_start.selected[1] = True
    with pytest.raises(RuntimeError, match="actually selected"):
        audit.check_birth_exclusion()


def test_accepted_cycle_wrap_observes_native_clear_without_injecting_request():
    env = environment(count=3)
    env.jump_cycle_reset_on_accepted = True
    env.full_tasks = FullTaskSemantics(3, "cpu", .02, {})
    env.jump_landing = JumpLandingWindow(3, "cpu", .02, JUMP_LANDING_V1)
    env.scene_groups = ["jump", "stand", "double_step_jump"]
    for module in (env.full_tasks, env.jump_landing):
        for value in vars(module).values():
            if isinstance(value, torch.Tensor):
                value.fill_(1)
    env._reset_jump_evidence = lambda ids, p: (
        env.full_tasks.reset_accepted(ids, p), env.jump_landing.reset_accepted(ids))
    audit = probe.StructuralAudit(env, "jump")
    assert audit.summary()["coverage"]["accepted_cycle_clear_observed"] is False
    env._reset_jump_evidence(torch.tensor([0, 2]), torch.zeros(3, 3))
    assert audit.counts["accepted_cycle_lanes_cleared"] == 2
    assert audit.accepted_by_group == {"jump": 1, "double_step_jump": 1}
    assert env.full_tasks.clearance_peak[1] == 1
    assert audit.summary()["coverage"]["accepted_cycle_clear_observed"] is True
    assert audit.summary()["coverage"]["functional_qualification"] is False
