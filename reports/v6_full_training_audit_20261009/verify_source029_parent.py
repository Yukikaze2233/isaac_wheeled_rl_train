"""Verify the exact REC parent locally; never collect rollouts or update PPO."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

source = Path('/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009')
assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == 'e5bd9c1367b875d7f229b9a1930647013d023372'
sys.path.insert(0, str(source / 'src'))
import torch
from wheeled_tasks.chassis.integrated_lineage import _validate_native_learning_state
from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count

root = Path(__file__).resolve().parent
receipt = json.loads((root / 'source029_parent_download001.json').read_text())
for pin in receipt['files'].values():
    content = Path(pin['local_path']).read_bytes()
    assert len(content) == pin['size'] and hashlib.sha256(content).hexdigest() == pin['sha256']
directory = root / 'source029_parent_rec41393'
checkpoint = directory / 'model_final.pt'
seal = json.loads((directory / 'model_final.checkpoint.json').read_text())
contract = json.loads((directory / 'model_final.contract.json').read_text())
saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
infos = saved['infos']
assert checkpoint_update_count(checkpoint) == 41393
assert seal['checkpoint_sha256'] == receipt['files']['model_final.pt']['sha256']
assert seal['contract_sha256'] == infos['contract_sha256'] == receipt['files']['model_final.contract.json']['sha256']
assert seal['task_campaign']['domain'] == infos['task_campaign']['domain'] == contract['task_campaign']['domain'] == 'recovery'
for key, info_key in (('learning_updates', 'successful_updates_total'),
                      ('learning_lineage_updates', 'learning_lineage_updates'),
                      ('stage_actor_update', 'stage_actor_update'),
                      ('task_actor_update', 'task_actor_update'),
                      ('actor_updates_consumed', 'actor_updates_consumed')):
    assert seal[key] == infos[info_key]
for key in ('asset_manifest_sha256', 'control_math_sha256'):
    assert infos[key] == contract[key]
for key in ('rng_state', 'env_rng_state'):
    assert isinstance(infos[key], torch.Tensor) and infos[key].dtype == torch.uint8 and infos[key].ndim == 1 and infos[key].numel()
assert seal['task_actor_update'] == 14188
_validate_native_learning_state(contract, saved)
report = dict(status='actual_source029_parent_full_state_verified',
    checkpoint_sha256=seal['checkpoint_sha256'], contract_sha256=seal['contract_sha256'],
    source_commit='e5bd9c1367b875d7f229b9a1930647013d023372',
    verifier_module_sha256=hashlib.sha256((source / 'src/wheeled_tasks/chassis/integrated_lineage.py').read_bytes()).hexdigest(),
    consumed_updates=seal['learning_updates'], actor_consumed=seal['actor_updates_consumed'],
    recovery_actor=seal['task_actor_update'], stage_actor=seal['stage_actor_update'],
    learning_lineage_updates=seal['learning_lineage_updates'],
    actor_tensors=len(saved['actor_state_dict']), critic_tensors=len(saved['critic_state_dict']),
    Adam_states=len(saved['optimizer_state_dict']['state']),
    local_CPU_only=True, remote_writes=0, GPU_calls=0, PPO_updates=0, qualification=None)
output = root / 'source029_parent_verification001.json'
assert not output.exists()
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
