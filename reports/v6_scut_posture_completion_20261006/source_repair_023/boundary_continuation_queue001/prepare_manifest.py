"""Build a literal external observer manifest; never launches or signals."""
from pathlib import Path
import hashlib
import json
import subprocess

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
REMOTE = Path('/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358')
NEW_COMMIT = 'b0d2800a338fa11099b355ece48c0baaccbb1b02'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    live = json.loads((SOURCE / 'live015.json').read_text())
    staged = json.loads((SOURCE / 'terrain_boundary_staging001/remote_staging_receipt.json').read_text())
    command = list(live['job']['command'])
    old_source = REMOTE / 'source_6f07ec298e01'
    new_source = Path(staged['source_root'])
    command[command.index(str(old_source / 'scripts/run_task_campaign.py'))] = str(new_source / 'scripts/run_task_campaign.py')
    command[command.index('--contract') + 1] = str(new_source / 'contracts/v6_task_campaign_v22.json')
    pins = {p['kind']: {'pid': p['pid'], 'starttime': p['starttime_ticks']}
            for p in live['owners'] if p['kind'] in ('wrapper', 'supervisor')}
    manifest = dict(
        schema='task-campaign-boundary-continuation-v1',
        old_source_commit='6f07ec298e01d2686216787cfba34b394151f1aa',
        source_commit=NEW_COMMIT, run_root=str(REMOTE),
        old_source_root=str(old_source), new_source_root=str(new_source),
        old_contract_relative_path='contracts/v6_task_campaign_v21.json',
        new_contract_relative_path='contracts/v6_task_campaign_v22.json',
        old_contract_sha256='119c77c21da31238ac1af285d0e8df1c627816b0d9645fa099899a6496f11755',
        new_contract_sha256=staged['plan_sha256'],
        source_archive=staged['source_archive'], source_archive_sha256=staged['source_archive_sha256'],
        old_job_sha256='731960a9b0841d3ed1770d719d2f31c6501f577ff8f5bf461af5cf46aed3837a',
        old_source_revision_sha256='b17e79369bdf23c862d619281c70cf93f5e924d2bf738bfbb19e00ae63b5ffd2',
        old_source_revision_receipt=str(REMOTE / 'source_revision_023.json'),
        source_revision_receipt=str(REMOTE / 'source_revision_024.json'),
        queue_receipt=str(REMOTE / 'boundary_continuation_024.receipt.json'),
        status_path=str(REMOTE / 'boundary_continuation_024.status.json'),
        launch_log=str(REMOTE / 'boundary_continuation_024.launch.log'),
        runtime_script='/home/kaiser/robot-rl-sim60/bin/sim60-runtime.sh',
        runtime_script_sha256='c76cc88f7273a400430ded6251ebc32d984a36fed366733a27ed76e2c94dcff7',
        old_owners=pins, command=command,
    )
    code_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    code_files = {}
    for name in ('queue_task_campaign_boundary_continuation.py', 'queue_task_campaign_continuation.py'):
        current = REPO / 'scripts' / name
        frozen = subprocess.check_output(['git', 'show', f'{code_commit}:scripts/{name}'], cwd=REPO)
        if current.read_bytes() != frozen:
            raise ValueError(f'Observer code is not frozen at {code_commit}: {name}')
        code_files[name] = {'sha256': digest(current), 'size': current.stat().st_size}
    frozen_helper = subprocess.check_output(['git', 'show', f'{NEW_COMMIT}:scripts/queue_task_campaign_continuation.py'], cwd=REPO)
    if frozen_helper != (REPO / 'scripts/queue_task_campaign_continuation.py').read_bytes():
        raise ValueError('Native external queue helper differs from staged production source')
    (HERE / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    record = dict(schema='external-boundary-observer-code-v1', observer_code_commit=code_commit,
                  production_target_commit=NEW_COMMIT, files=code_files,
                  manifest_sha256=digest(HERE / 'manifest.json'),
                  manifest_semantic_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  production_source_replaced=False, PPO_updates=0, qualification=None)
    (HERE / 'code_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
