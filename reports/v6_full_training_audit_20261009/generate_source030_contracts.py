"""Generate static v27 artifacts only; no future checkpoint or admission claim."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/home/yukikaze/Documents/workspace/robot_rl/.worktrees/audit-logger-timing-20261009')
commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
assert commit == 'f627f678fae98d374c049d76a444430dda70306a'
sys.path.insert(0, str(root / 'src'))
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis.integrated_lineage import PHYSICAL_KEYS

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

old = {}
for version in (24, 25, 26):
    files = production.artifact_files(version)
    names = [files[k] for k in ('recipe', 'lineage', 'plan', 'manifest')]
    names += [f for domain in ('speed', 'terrain', 'jump', 'stepjump') for f in (
        production.worker_file(domain, version=version), production.design_file(domain, version=version))]
    old.update({name: digest(root / name) for name in names})
target_source = '/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_serial_dispatch030_20261009'
target_run = '/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/train'
result = production.write_bundle(root, source_root=target_source, run_root=target_run, version=27)
assert len(result['files']) == 12
assert all(digest(root / name) == sha for name, sha in old.items())
plan = json.loads((root / production.artifact_files(27)['plan']).read_text())
manifest = production.validate_plan(plan, root)
assert plan['total_ceiling'] == 121000
assert manifest['source_root'] == target_source
new_recipe = json.loads((root / production.artifact_files(27)['recipe']).read_text())
old_recipe = json.loads((root / production.artifact_files(26)['recipe']).read_text())
assert new_recipe['domains']['speed'] == old_recipe['domains']['speed']
effective_checks = {}
for domain in ('speed', 'terrain', 'jump', 'stepjump'):
    worker = json.loads((root / production.worker_file(domain, version=27)).read_text())
    assert production.validate_static_worker(worker, root) == domain
    assert (worker['actor_dim'], worker['critic_dim'], worker['action_dim']) == (35, 81, 6)
    assert (worker['policy_dt'], worker['physics_dt']) == (.02, .005)
    effective = production.static_training_config(worker, root)
    assert effective.get('recovery_phase_audit', False) == (domain != 'speed')
    effective_checks[domain] = dict(recovery_phase_audit=effective.get('recovery_phase_audit', False))
    if domain == 'speed':
        old_worker = json.loads((root / production.worker_file(domain, version=26)).read_text())
        # Named identity/recipe/design pins change; physical fields and the
        # entire speed teaching recipe remain the frozen v26 source.
        assert {key: worker.get(key) for key in PHYSICAL_KEYS} == {key: old_worker.get(key) for key in PHYSICAL_KEYS}
    else:
        inventory = production.production_source_files(worker, root=root)
        assert {'src/wheeled_tasks/chassis/step_climb.py', 'src/wheeled_tasks/chassis/state_snapshot.py',
            'src/wheeled_tasks/chassis/integrated_completed_speed_revision.py',
            'scripts/chassis_remote_job.py', production.worker_file('speed', version=26)} <= set(inventory)
report = dict(schema='source030-v27-static-contract-generation-v1', status='static_canonical_contracts_generated',
    source_code_commit=commit, target_source=target_source, target_run=target_run,
    files=[dict(path=name, sha256=digest(root / name)) for name in result['files']], effective_checks=effective_checks,
    preserved_old_files=old, remote_writes=0, PPO_updates=0, actual_admission=None,
    future_checkpoint=None, freeze_ready=False, qualification=None)
output = Path(__file__).resolve().parent / 'source030_contract_generation001.json'
assert not output.exists()
output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(dict(status=report['status'], artifacts=len(result['files']), old_preserved=len(old))))
