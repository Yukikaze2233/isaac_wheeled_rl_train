"""CPU-only native RSL loading through the production resume statements."""
import ast
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner

from wheeled_tasks.chassis.policy_transfer import (
    formal_stage_load_kwargs, verify_learning_state_restore,
)

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/train_chassis.py"


def config():
    return {"formal_stage_optimizer": "scut_finetune_v1", "formal_v7_flat": True,
            "formal_learning_state_resume": True, "training_profile": "v6_78d_scratch_v1",
            "curriculum_stage": "terrain", "formal_stage_transition": {
                "from": "foundation", "to": "terrain", "reset_stage_actor_clock": True}}


def algorithm(*, used=False):
    actor, critic = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    actor.register_parameter("std_param", torch.nn.Parameter(torch.tensor([.37, .42])))
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=2e-4 if used else 1e-4)
    if used:
        for parameter in optimizer.param_groups[0]["params"]:
            optimizer.state[parameter] = {"step": torch.tensor(20.),
                "exp_avg": torch.full_like(parameter, .1), "exp_avg_sq": torch.full_like(parameter, .2)}
    alg = NS(_raw_actor=actor, _raw_critic=critic, optimizer=optimizer,
             learning_rate=optimizer.param_groups[0]["lr"], rnd=None)
    alg.load = MethodType(PPO.load, alg)
    alg.save = lambda: deepcopy({"actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(), "optimizer_state_dict": optimizer.state_dict()})
    return alg


def execute_actual_resume_load(tmp_path, checkpoint, settings):
    """Run actual load/verification calls; no Kit, environments or PPO updates."""
    path = tmp_path / "parent.pt"
    torch.save(checkpoint, path)
    runner = NS(alg=algorithm(), current_learning_iteration=0)
    runner.load = MethodType(OnPolicyRunner.load, runner)
    tree = ast.parse(ENTRY.read_text())
    resume = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args.resume"
        and any(isinstance(child, ast.Assign) and ast.unparse(child.targets[0]) == "checkpoint"
                for child in node.body))
    start = next(i for i, node in enumerate(resume.body) if isinstance(node, ast.ImportFrom)
        and any(alias.name == "formal_stage_load_kwargs" for alias in node.names))
    end = next(i for i, node in enumerate(resume.body) if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "restored")
    scope = {"c": settings, "infos": checkpoint["infos"], "checkpoint": checkpoint,
        "args": NS(resume=path, stage=settings["curriculum_stage"], resume_diagnostic=False,
                   rng_mode="inherit", seed=290929),
        "runner": runner,
        "effective_algorithm": {"learning_rate": 1e-4}, "fresh_critic_state": None}
    exec(compile(ast.Module(body=resume.body[start:end + 1], type_ignores=[]), str(ENTRY), "exec"), scope)
    return runner, scope["restored"]


def test_actual_first_handoff_preserves_policy_std_and_starts_empty_adam(tmp_path):
    parent = algorithm(used=True).save()
    parent.update(iter=4500, infos={"stage": "foundation", "successful_updates_total": 4500})
    runner, receipt = execute_actual_resume_load(tmp_path, parent, config())
    assert receipt == {"actor_exact": True, "critic_exact": True, "optimizer_exact": False,
        "optimizer_initialization": "fresh_on_foundation_to_terrain_only",
        "optimizer_state_entries": 0, "initial_learning_rate": 1e-4}
    assert not runner.alg.optimizer.state
    assert runner.alg.learning_rate == 1e-4
    assert torch.equal(runner.alg._raw_actor.std_param, parent["actor_state_dict"]["std_param"])
    # The loader does not inherit source runner iteration; main subsequently
    # binds it to authenticated lineage4500, independently of shared paid5549.
    assert runner.current_learning_iteration == 0


def test_actual_same_rough_restart_retains_adam_lr_and_iteration(tmp_path):
    parent = algorithm(used=True).save()
    parent.update(iter=5000, infos={"stage": "terrain", "successful_updates_total": 5000})
    runner, receipt = execute_actual_resume_load(tmp_path, parent, config())
    assert receipt == {"actor_exact": True, "critic_exact": True, "optimizer_exact": True}
    assert runner.alg.learning_rate == 2e-4
    assert runner.current_learning_iteration == 5000
    assert all(state["step"].item() == 20 for state in runner.alg.optimizer.state.values())




@pytest.mark.parametrize("change", ["actor", "critic", "std", "adam", "lr"])
def test_finetune_readback_rejects_wrong_weights_old_moments_or_rate(change):
    alg = algorithm()
    checkpoint = alg.save()
    if change == "adam":
        alg.optimizer.state[next(iter(alg._raw_actor.parameters()))] = {"step": torch.tensor(20.)}
    elif change == "lr":
        alg.learning_rate = 2e-4
    else:
        parameter = (alg._raw_actor.std_param if change == "std" else
                     next(iter((alg._raw_actor if change == "actor" else alg._raw_critic).parameters())))
        with torch.no_grad():
            parameter.add_(1.)
    with pytest.raises(ValueError):
        verify_learning_state_restore(alg, checkpoint, fresh_optimizer_learning_rate=1e-4)


@pytest.mark.parametrize("stage", [None, "recovery", "mixed"])
def test_finetune_rejects_nonfoundation_nonrough_parent(stage):
    with pytest.raises(ValueError, match="foundation parent"):
        formal_stage_load_kwargs(config(), {"stage": stage}, stage="terrain")


def test_default_restart_and_source_binding_stay_explicit():
    assert formal_stage_load_kwargs({}, {"stage": "foundation"}, stage="foundation") == {}
    tree = ast.parse(ENTRY.read_text())
    source_guard = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and any(isinstance(child, ast.Expr) and '"src/wheeled_tasks/chassis/policy_transfer.py"'
                in ast.unparse(child).replace("'", '"') for child in node.body))
    assert "formal_stage_optimizer" in ast.unparse(source_guard.test)
