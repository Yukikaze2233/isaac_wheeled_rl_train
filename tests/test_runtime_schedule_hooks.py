"""Execute production Python method bodies on CPU tensors, without Isaac/PhysX.

AST extraction avoids simulator imports; it does not replace the tested method
with a handwritten imitation and must not be reported as a physics rollout.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.skill_commands import SkillCommands
from wheeled_tasks.chassis.task import Phase, PhaseTracker

ROOT = Path(__file__).resolve().parents[1]


def production_method(path, cls, name):
    tree = ast.parse((ROOT / path).read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    method = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "Phase": Phase}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return namespace[name]


def environment(events=None, sequence=None, exact=True):
    spec = {"kind": "jump", "command": [0., 0., .305], "mode": 4,
            "semantics": {"jump_apex_delta_m": .1},
            "request_events": events or [{"time_s": .8, "type": "new"}]}
    if sequence:
        spec["command_sequence"] = sequence
    case = {"name": "probe", "skill": spec, "command": [0., 0., .305], "task": "jump"}
    cfg = {"skill_specs": {"probe": spec}, "evaluation_exact_cases": exact, "evaluation": {"cases": [case]},
           "task_semantics": {"jump_request_seconds": .8}, "new_asset_curriculum": {"enabled": True},
           "command_slew": {"vx_m_s2": 1.5, "yaw_rad_s2": 3.},
           "command_reference": {"step_request_range_s": [.8, .8], "step_hold_range_s": [2., 2.]}}
    pose = torch.zeros(1, 7)
    pose[:, 6] = 1.
    generator = torch.Generator().manual_seed(617)
    e = SimpleNamespace(cfg=cfg, num_envs=1, device="cpu", scene_groups=["probe"],
        commands=torch.zeros(1, 3), command_target=torch.zeros(1, 2), mode=torch.zeros(1, dtype=torch.long),
        targets=torch.zeros(1, 4), platform_delta=torch.zeros(1), route_goal=torch.ones(1), origins=torch.zeros(1, 3),
        phase=PhaseTracker(1, "cpu", .02), episode_length_buf=torch.zeros(1), policy_dt=.02,
        jump_requested=torch.zeros(1, dtype=torch.bool), success_hold=torch.zeros(1),
        sequence_success_latched=torch.zeros(1, dtype=torch.bool), contact_force=torch.zeros(1, 2, 3), wheel_ids=[0, 1],
        successful_jump_count=torch.zeros(1, dtype=torch.long),
        robot=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=SimpleNamespace(torch=pose))),
        random=lambda n: torch.rand(n, generator=generator),
        full_tasks=FullTaskSemantics(1, "cpu", .02, {"jump_apex_delta_m": .1}),
        command_clock=torch.zeros(1), height_clock=torch.zeros(1), push_clock=torch.zeros(1),
        push_enabled=torch.zeros(1, dtype=torch.bool), episode_limits=torch.zeros(1),
        case_commands={"probe": case["command"]}, cases={"probe": case}, step_assist=None, references=None,
        raw_height=torch.full((1,), .305))
    e.skills = SkillCommands(e)
    e.skills.sample(torch.tensor([0]))
    return e


def test_production_dispatch_accepts_exact_request_and_clears_previous_jump_evidence():
    method = production_method("src/wheeled_tasks/chassis/env.py", "ChassisEnv", "update_targets")
    e = environment()
    e.episode_length_buf[:] = 40
    e.full_tasks.com_rise[:] = .3
    e.full_tasks.clear_air_time_peak[:] = .5
    e.full_tasks.com_released[:] = True
    e.phase.flew[:] = True
    e.success_hold[:] = 3
    e.sequence_success_latched[:] = True
    method(e)
    assert e.skills.schedule.accepted_request_count.item() == 1
    assert e.jump_requested.item() and e.targets[0, 3].item() == 1
    assert e.full_tasks.com_rise.item() == 0 and e.full_tasks.clear_air_time_peak.item() == 0
    assert not e.full_tasks.com_released.item() and not e.phase.flew.item()
    assert e.success_hold.item() == 0 and not e.sequence_success_latched.item()


def test_production_dispatch_preserves_busy_repeat_reference_start():
    method = production_method("src/wheeled_tasks/chassis/env.py", "ChassisEnv", "update_targets")
    e = environment([{"time_s": .8, "type": "new"}, {"time_s": .9, "type": "repeat"}])
    e.episode_length_buf[:] = 40
    method(e)
    e.phase.phase[:] = Phase.TAKEOFF
    start = e.skills.schedule.request_start.clone()
    e.episode_length_buf[:] = 45
    method(e)
    assert e.skills.schedule.repeat_request_count.item() == 1
    assert torch.equal(e.skills.schedule.request_start, start)
    assert e.skills.sequence_pending().item()


def test_production_reference_uses_request_clock_and_active_route_latch():
    update = production_method("src/wheeled_tasks/chassis/env.py", "ChassisEnv", "update_command_reference")
    e = environment()
    received = []
    e.references = SimpleNamespace(update=lambda *args: received.append(args) or args[0])
    e.command_acceleration = torch.zeros(1)
    e.skills.manual_step_mode[:] = 1
    e.skills.manual_step_start[:] = .8
    e.skills.schedule.request_start[:] = 2.
    e.skills.schedule.request_active[:] = True
    e.episode_length_buf[:] = 150
    update(e)
    assert received[-1][2].item() == 1 and received[-1][3].item() == pytest.approx(1.)
    e.skills.schedule.request_active[:] = False
    update(e)
    assert received[-1][2].item() == 0


@pytest.mark.parametrize("path,cls", [("src/wheeled_tasks/chassis/env.py", "ChassisEnv"),
                                      ("src/wheeled_tasks/chassis/eval_env.py", "FixedCaseEnv")])
def test_production_resample_preserves_mid_sequence_cursor_and_jump_target(path, cls):
    resample = production_method(path, cls, "resample_commands")
    e = environment(sequence=[{"time_s": 0., "command": [.4, 0., .305], "mode": 1},
                              {"time_s": 2., "command": [0., 0., .305], "mode": 1},
                              {"time_s": 3., "command": [0., 0., .305], "mode": 4}])
    e.skills.sequence_cursor[:] = 2
    e.skills.schedule.jump_apex[:] = .1
    e.skills.manual_step_start[:] = 7.
    e.episode_length_buf[:] = 125
    resample(e, torch.tensor([0]), reset_height=False)
    assert e.skills.sequence_cursor.item() == 2
    assert e.skills.schedule.jump_apex.item() == pytest.approx(.1)
    assert e.skills.manual_step_start.item() == 7.


def test_evaluation_entry_collects_episode_ledger_before_grading():
    tree = ast.parse((ROOT / "scripts/evaluate_chassis.py").read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    observe = [node for node in calls if isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name) and node.func.value.id == "request_metrics"
               and node.func.attr == "observe"]
    report = [node for node in calls if isinstance(node.func, ast.Attribute)
              and isinstance(node.func.value, ast.Name) and node.func.value.id == "request_metrics"
              and node.func.attr == "report_into"]
    grades = [node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "grade_fixed_suite"]
    assert len(observe) == len(report) == len(grades) == 1
    assert observe[0].lineno < report[0].lineno < grades[0].lineno
    assert isinstance(grades[0].args[0], ast.Name) and grades[0].args[0].id == report[0].args[0].id
