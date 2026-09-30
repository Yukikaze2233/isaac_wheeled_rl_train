"""Replay the existing phase/reward code on two stationary synthetic states.

This is a deterministic incentive check, not a physical rollout or causal PPO
experiment. Run from the repository with PYTHONPATH=src. The landing block is
compiled directly from env.py so the audit does not substitute a new reward.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.rewards import dense_error_terms
from wheeled_tasks.chassis.task import Phase, PhaseTracker


root = Path(__file__).resolve().parents[2]
source_path = root / "src/wheeled_tasks/chassis/env.py"
source = source_path.read_text()
blocks = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.If)
          and ast.unparse(node.test) == "self.cfg.get('landing_tracking')"]
assert len(blocks) == 1
function = ast.parse("def landing_terms(self, height, vertical_velocity):\n"
                     "    components = {}\n    return components\n").body[0]
function.body.insert(1, blocks[0])
module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
namespace = {"torch": torch, "Phase": Phase}
exec(compile(module, str(source_path), "exec"), namespace)

cfg_path = root / "contracts/v6_s1_continuation_worker_v1.json"
cfg = json.loads(cfg_path.read_text())
errors = torch.tensor([.009, .011])
commands = torch.tensor([[0., 0., .23], [0., 0., .23]])
height = commands[:, 2] + errors
phase = PhaseTracker(2, "cpu", dt=.02)
phase.phase.fill_(Phase.FLIGHT)
contact = torch.ones(2, 2, dtype=torch.bool)
for _ in range(100):
    phase.update(contact, torch.zeros(2, dtype=torch.bool), errors.abs() < .01)
stub = SimpleNamespace(cfg=cfg, phase=phase, full_tasks=FullTaskSemantics,
                       commands=commands, mode=torch.zeros(2, dtype=torch.long),
                       body_mass=torch.ones(2, 1), contact_peak=torch.full((2,), 9.81))
landing = namespace["landing_terms"](stub, height, torch.zeros(2))
broad, fine = dense_error_terms(errors, cfg["dense_tracking"]["height"])
subtotal = broad + fine + sum(landing.values())
result = {
    "purpose": "Check reward discontinuity when stable RECOVERY returns to GROUND",
    "scope": "synthetic upright supported states; no new simulation or PPO; other common reward terms cancel",
    "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [source_path, cfg_path, root / 'src/wheeled_tasks/chassis/task.py',
                                root / 'src/wheeled_tasks/chassis/full_tasks.py',
                                root / 'src/wheeled_tasks/chassis/rewards.py']},
    "initial_phase": "FLIGHT",
    "supported_elapsed_s": 2.,
    "height_errors_m": errors.tolist(),
    "final_phase": [Phase(int(value)).name for value in phase.phase],
    "landing_reward_density": {key: value.tolist() for key, value in landing.items()},
    "dense_height_reward_density": (broad + fine).tolist(),
    "height_plus_landing_density": subtotal.tolist(),
    "worse_height_extra_reward_per_second": float(subtotal[1] - subtotal[0]),
}
assert result["final_phase"] == ["GROUND", "RECOVERY"]
assert result["worse_height_extra_reward_per_second"] > 0
out = Path(__file__).with_name("recovery_reward_probe.json")
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
