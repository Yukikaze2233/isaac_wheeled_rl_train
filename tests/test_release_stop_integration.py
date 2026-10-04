"""Run the production resample wrapper without constructing a simulator."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / "src/wheeled_tasks/chassis/env.py"


@pytest.mark.parametrize("reset", [False, True])
def test_resample_preserves_finite_zero_request_and_actual_acceleration_history(reset):
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ChassisEnv")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "resample_commands")
    end = next(i for i, n in enumerate(method.body)
               if isinstance(n, ast.If) and ast.unparse(n.test) == "self.skills is not None")
    method.body = method.body[:end + 1]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {"torch": torch}
    exec(compile(module, str(SOURCE), "exec"), scope)
    env = NS(commands=torch.tensor([[.8, .4, .305], [.2, .3, .305]]),
        command_target=torch.ones(2, 2), step_assist=None, references=None,
        cfg={"command_slew": {"vx_m_s2": 1.5, "yaw_rad_s2": 4.}})
    previous = env.commands.clone()

    def sample(ids, *, reset_height):
        env.command_target[0] = 0.
        if reset_height:
            env.commands[0, :2] = 0.
        env.commands[1, :2] = torch.tensor([-.4, -.5])

    env.skills = NS(sample=sample, release_stop=NS(active=torch.tensor([True, False])))
    scope["resample_commands"](env, torch.arange(2), reset_height=reset)
    assert env.command_target[0].eq(0.).all()
    torch.testing.assert_close(env.command_target[1], torch.tensor([-.4, -.5]))
    torch.testing.assert_close(env.commands[:, :2], torch.zeros(2, 2) if reset else previous[:, :2])


def test_actual_stop_diagnostics_use_current_yaw_height_and_gravity():
    tree = ast.parse(SOURCE.read_text())
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and ast.unparse(n.func) == "self.skills.diagnostics")
    assert {k.arg: ast.unparse(k.value) for k in call.keywords} == {
        "omega": "omega", "height_error": "height - self.commands[:, 2]", "gravity": "gravity"}
