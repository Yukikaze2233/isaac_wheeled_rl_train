"""One exact native speed43500 CPU verification; no rollout/update/Isaac/SSH."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

SOURCE = Path('/home/yukikaze/Documents/workspace/robot_rl/.worktrees/serial-dispatch-source028-20261009')
COMMIT = 'e5bd9c1367b875d7f229b9a1930647013d023372'
CONTRACT_SHA = '941fb9e1f7a8ced3aa83fca76e82b7cf3064b594602d1e2900e56e0df0cc6b64'
PARENT_SHA = 'e04c2f7a20c462b156bd6e21c4eaf2d14bec5e2a2a7956e327c26e383d603967'
FILES = ('model.pt', 'model.contract.json', 'model.checkpoint.json', 'contract.json',
         'progress.json', 'completion.json', 'agent_config.json')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--download-receipt', required=True, type=Path)
    parser.add_argument('--checkpoint-sha256', required=True,
                        help='Real immutable remote/download pin, not a predicted SHA')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{64}', args.checkpoint_sha256)
    assert not args.output.exists()
    directory = args.directory.resolve(strict=True)
    assert not any(part.endswith('.pending') for part in directory.parts)
    assert subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    subprocess.run(['git', '-C', str(SOURCE), 'diff', '--exit-code', COMMIT, '--',
                    'src', 'scripts', 'contracts', 'model'], check=True, capture_output=True)
    pins = json.loads(args.download_receipt.read_text())['files']
    assert set(pins) == set(FILES)
    for name in FILES:
        path = directory / name
        assert path.resolve(strict=True) == Path(pins[name]['local_path']).resolve(strict=True)
        assert path.stat().st_size == pins[name]['size'] and sha(path) == pins[name]['sha256']
    assert pins['model.pt']['sha256'] == args.checkpoint_sha256
    assert pins['model.contract.json']['sha256'] == pins['contract.json']['sha256'] == CONTRACT_SHA
    assert (directory / 'model.contract.json').read_bytes() == (SOURCE / 'contracts/v6_task_campaign_speed_worker_v26.json').read_bytes()
    completion = json.loads((directory / 'completion.json').read_text())
    assert completion['status'] == 'checkpoint_sealed' and completion['successful_updates'] == 43500
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'scripts')]
    import torch
    import export_chassis_candidate as exporter
    from wheeled_tasks.chassis.integrated_lineage import _validate_native_learning_state
    from wheeled_tasks.chassis.full_curriculum import checkpoint_update_count
    _, saved, updates, _, contract, _, agent, _, _ = exporter.authenticate(directory / 'model.pt')
    assert updates == checkpoint_update_count(directory / 'model.pt') == 43500
    seal = json.loads((directory / 'model.checkpoint.json').read_text())
    infos = saved['infos']
    expected = dict(successful_updates_total=43500, actor_updates_consumed=43320,
                    task_actor_update=2097, stage_actor_update=29653,
                    learning_lineage_updates=29753, critic_warmup_remaining=0)
    assert all(infos.get(k) == value for k, value in expected.items())
    for name, key in (('learning_updates', 'successful_updates_total'),
                      ('actor_updates_consumed', 'actor_updates_consumed'),
                      ('task_actor_update', 'task_actor_update'),
                      ('stage_actor_update', 'stage_actor_update'),
                      ('learning_lineage_updates', 'learning_lineage_updates')):
        assert seal[name] == infos[key]
    assert seal['task_campaign']['domain'] == infos['task_campaign']['domain'] == contract['task_campaign']['domain'] == 'speed'
    lineage = dict(lineage_id='source028-joint-serial-v1',
                   plan_sha256='86311f7ab75644d6be44e79c2ccf427648e5cabec22e4e75ed60f64f472f7ffd',
                   origin_checkpoint_sha256=PARENT_SHA, domain='speed', joint_actor_updates=2097,
                   domain_actor_updates=2097, domain_complete=False)
    assert infos['integrated_lineage'] == seal['integrated_lineage'] == lineage
    for key in ('rng_state', 'env_rng_state'):
        value = infos[key]
        assert isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1 and value.numel()
    _validate_native_learning_state(contract, saved)
    assert (len(saved['actor_state_dict']), len(saved['critic_state_dict']),
            len(saved['optimizer_state_dict']['state'])) == (9, 8, 17)
    assert all(sha(directory / name) == pins[name]['sha256'] for name in FILES)
    report = dict(schema='actual-speed43500-native-local-verification-v1',
                  status='strict_native_actor_critic_Adam_restore_CPU_verified', checkpoint_sha256=args.checkpoint_sha256,
                  contract_sha256=CONTRACT_SHA, source_commit=COMMIT, clocks=expected,
                  integrated_lineage=lineage, original_saved_agent_config_sha256=sha(directory / 'agent_config.json'),
                  actor_tensors=9, critic_tensors=8, Adam_states=17, saved_RNG_states_present=True,
                  CUDA_RNG_restore_executed=False, original_full_role_validation='Performed next by mean replay authenticate_candidate',
                  input_files_sha256={name: sha(directory / name) for name in FILES},
                  download_receipt_sha256=sha(args.download_receipt), verifier_sha256=sha(Path(__file__)),
                  verifier_module_sha256=sha(SOURCE / 'src/wheeled_tasks/chassis/integrated_lineage.py'),
                  Isaac_scenes=0, GPU_calls=0, PPO_updates=0, SSH_calls=0, qualification=None)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
