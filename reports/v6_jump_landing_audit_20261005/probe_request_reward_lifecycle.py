"""CPU-only reproduction using the actually deployed immutable source tree."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

SOURCE = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_122d8db63f37')
sys.path.insert(0, str(SOURCE / 'src'))

import torch
from wheeled_tasks.chassis.references import CommandReference
from wheeled_tasks.chassis.rewards import reference_jump_terms
from wheeled_tasks.chassis.task import Phase
from wheeled_tasks.chassis.training_schedule import TrainingSchedule

torch.set_num_threads(1)
contract = json.loads((SOURCE / 'contracts/v6_task_campaign_jump_worker_v6.json').read_text())
environment = SimpleNamespace(num_envs=1, device='cpu', cfg=contract, scene_groups=['jump_06'],
    stage_actor_update=0, global_actor_update=0, random=lambda count: torch.full((count,), .9))
schedule = TrainingSchedule(environment)
schedule.reset(torch.tensor([0]))
schedule.request_stress.zero_()
schedule.next_request.fill_(torch.inf)
reference = CommandReference(1, 'cpu', contract['command_reference'])
yes = torch.ones(1, dtype=torch.bool)
zero = torch.zeros(1)
height = torch.full((1,), .305)
commands = torch.tensor([[0., 0., .305]])
task = SimpleNamespace(com_released=yes, com_rise=torch.full((1,), .06),
    clear_air_time_peak=torch.full((1,), .15), com_displacement=zero,
    com_release_speed=torch.sqrt(torch.full((1,), 2 * 9.81 * .06)),
    origin_xy=torch.zeros(1, 2))
rows = []

def observe(time, phase):
    request = schedule.update_requests(torch.tensor([time]), torch.tensor([phase]), yes, yes)
    elapsed = torch.where(request, time - schedule.request_start, zero)
    reference.update(height, zero, zero, zero, torch.full((1,), 7.), request,
        elapsed, torch.full((1,), .06))
    contacts = torch.tensor([[phase != Phase.FLIGHT, phase != Phase.FLIGHT]])
    terms = reference_jump_terms(reference, task, torch.tensor([phase]), contacts,
        height, zero, torch.zeros(1, 3), torch.tensor([[0., 0., -1.]]),
        torch.full((1,), .16), torch.zeros(1, 3), commands)
    rows.append(dict(time_s=time, physical_phase=phase.name,
        request_active=bool(request.item()), awaiting_ground=bool(schedule.awaiting_ground.item()),
        reference_phase=int(reference.phase.item()),
        terms={key: float(value.item()) for key, value in terms.items() if 'landing' in key or 'recovery' in key}))

observe(.8, Phase.GROUND)
observe(1.3, Phase.FLIGHT)
touchdown = .8 + float(reference.flight_end.item()) + .02
observe(touchdown, Phase.LANDING)
observe(touchdown + .06, Phase.RECOVERY)
observe(touchdown + .32, Phase.RECOVERY)
observe(touchdown + .52, Phase.GROUND)
assert rows[2]['terms']['jump_landing_velocity'] > 0
assert rows[3]['request_active'] is False and rows[3]['awaiting_ground'] is True
assert all(value == 0 for value in rows[3]['terms'].values())
assert all(value == 0 for value in rows[4]['terms'].values())
assert schedule.accepted_request_count.item() == 1
assert schedule.completed_jump_count.item() == 1
print(json.dumps(dict(source=str(SOURCE), rows=rows,
    success_hold_seconds=contract['task_semantics']['success_hold_seconds'],
    assertion='Physical RECOVERY releases trigger and simultaneously zeroes jump landing/recovery terms; baseline rewards remain separate.',
    limitations='Prescribed physical phases reproduce reward plumbing, not closed-loop dynamics or causal PPO improvement.',
    source_files={name: hashlib.sha256((SOURCE / 'src/wheeled_tasks/chassis' / name).read_bytes()).hexdigest()
        for name in ['training_schedule.py', 'references.py', 'rewards.py']}), indent=2))
