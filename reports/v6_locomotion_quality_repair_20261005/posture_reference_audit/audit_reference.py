#!/usr/bin/env python3
"""CPU counterfactual of the frozen, actually configured pitch reward target."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
import torch.nn.functional as F

SOURCE = "b4ce80964d47c0127e0b3d9929ad35a620bbf204"
ROOT = Path(__file__).resolve().parents[3]
REWARD = "src/wheeled_tasks/chassis/rewards.py"
CONTRACT = "contracts/v6_task_campaign_speed_worker_v14.json"


def frozen(path):
    return subprocess.check_output(["git", "show", f"{SOURCE}:{path}"], cwd=ROOT)


def main():
    raw_reward, raw_contract = frozen(REWARD), frozen(CONTRACT)
    contract = json.loads(raw_contract)
    module = ast.parse(raw_reward)
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                    and n.name == "reference_motion_terms")
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[function], type_ignores=[]), REWARD, "exec"), namespace)
    reward = namespace[function.name]
    settings = contract["reference_reward"]
    floor = contract["dense_tracking"]["locomotion_posture"]
    assert settings["visible_reference_only"] is True
    rows = []
    for acceleration in (-1.5, 0., 1.5):
        reference_pitch = max(-contract["command_reference"]["pitch_limit_rad"],
                              min(contract["command_reference"]["pitch_limit_rad"],
                                  math.atan(acceleration / 9.81)))
        for speed in (0., 1., 3., 5.):
            for pitch in sorted({0., reference_pitch}):
                reference = SimpleNamespace(pitch=torch.tensor([reference_pitch]),
                    terrain_mode=torch.tensor([0]), elapsed=torch.tensor([0.]),
                    jumping=torch.tensor([False]))
                gravity = torch.tensor([[math.sin(pitch), 0., -math.cos(pitch)]])
                args = (torch.tensor([[speed, 0., 0.]]), gravity,
                        torch.tensor([[speed, 0., .305]]), torch.tensor([1.]),
                        torch.tensor([True]), reference)
                actual = reward(*args, settings, locomotion_posture=floor)
                hypothetical = reward(*args, {**settings, "visible_reference_only": False},
                                      locomotion_posture=floor)
                keys = ("pitch_exp", "pitch_velocity", "locomotion_pitch_floor")
                rows.append(dict(acceleration_m_s2=acceleration, command_vx_m_s=speed,
                    body_pitch_rad=pitch, body_pitch_deg=math.degrees(pitch),
                    computed_reference_pitch_rad=reference_pitch,
                    actual_visible_target_rad=0.,
                    actual_pitch_density={k:float(actual[k]) for k in keys},
                    actual_sum_per_second=sum(float(actual[k]) for k in keys),
                    hypothetical_acceleration_reference_sum_per_second=
                        sum(float(hypothetical[k]) for k in keys)))
                # At actual zero target, acceleration alone cannot alter payment.
                zero_reference = SimpleNamespace(**{**vars(reference), "pitch": torch.zeros(1)})
                check = reward(*args[:-1], zero_reference, settings, locomotion_posture=floor)
                assert all(torch.equal(actual[k], check[k]) for k in keys)
    receipt = dict(source_commit=SOURCE,
        frozen_reward_sha256=hashlib.sha256(raw_reward).hexdigest(),
        contract_file=CONTRACT, contract_sha256=hashlib.sha256(raw_contract).hexdigest(),
        actual_visible_reference_only=True, locomotion_posture=floor,
        scope="Actual frozen CPU reward function; synthetic states, no physics/PPO or runtime edits.",
        counterfactual="visible_reference_only=False is hypothetical and has NOT been deployed.",
        limitation="atan(ax/g) is the existing software reference, not proof of the optimal physical posture.",
        all_acceleration_independence_assertions_passed=True, rows=rows)
    target = Path(__file__).with_name("reference_counterfactual.json")
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(target), states=len(rows),
                         actual_flat_pitch_target_rad=0., assertions_passed=True)))


if __name__ == "__main__":
    main()
