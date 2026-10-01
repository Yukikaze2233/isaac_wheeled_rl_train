"""Exercise the native step's extras producer without starting a PhysX scene."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("configuration", [{}, {"remediation": True},
    {"continuation": True}, {"task_campaign": {"domain": "flat"}}])
def test_computed_native_rewards_are_available_to_every_rollout_entry(configuration):
    # Execute the actual producer block, including its configuration conditions.
    # Stubbing all of ChassisEnv would hide the bug this test must detect.
    path = Path(__file__).resolve().parents[1] / "src/wheeled_tasks/chassis/env.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ChassisEnv")
    step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    block, = [node for node in step.body if isinstance(node, ast.If)
              and ast.unparse(node.test) == "components is not None"]
    code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), str(path), "exec")
    components = {"velocity": torch.tensor([1., 2.]), "height": torch.tensor([3., 4.])}
    original = {name: value.clone() for name, value in components.items()}
    extras = {"log": {}}
    env = SimpleNamespace(cfg=configuration, policy_dt=.02)
    exec(code, {"self": env, "components": components, "extras": extras})
    assert extras["native_reward_components"] is components
    for name, value in original.items():
        torch.testing.assert_close(components[name], value)
        torch.testing.assert_close(extras["log"]["/reward/" + name], value.mean() * .02)
