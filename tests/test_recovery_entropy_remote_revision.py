"""Exercise the real archive/command admission before an entropy migration."""
import io
import json
from pathlib import Path
import tarfile

import pytest

import chassis_remote_job as remote
from wheeled_tasks.chassis import recovery_entropy_revision as recipe
from wheeled_tasks.chassis import task_campaign as campaign

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('defect', [None, 'missing_header', 'wrong_commit',
                                    'num_envs', 'old_sha', 'extracted_bytes', 'unknown_schema'])
def test_v21_real_wrapper_archive_and_command_admission(tmp_path, monkeypatch, defect):
    old, new, run = (tmp_path / name for name in ('old', 'new', 'run'))
    run.mkdir()
    names = [recipe.SOURCE_PLAN_FILE, recipe.PLAN_FILE, 'scripts/run_task_campaign.py']
    for root in (old, new):
        for name in names:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes() if name.endswith('.json') else b'# never executed\n')
    command = ['/env/python', str(old / names[2]), '--contract', str(old / recipe.SOURCE_PLAN_FILE),
               '--num-envs', '4096', '--run-dir', str(run / 'train')]
    previous = dict(command=command)
    (run / 'job.json').write_text(json.dumps(previous))
    changed = list(command)
    changed[1], changed[3] = str(new / names[2]), str(new / recipe.PLAN_FILE)
    archive, commit = tmp_path / 'source.tar.gz', 'a' * 40
    header = {} if defect == 'missing_header' else {'comment': 'b' * 40 if defect == 'wrong_commit' else commit}
    with tarfile.open(archive, 'w:gz', format=tarfile.PAX_FORMAT, pax_headers=header) as bundle:
        for name in names:
            data = (new / name).read_bytes()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    receipt = dict(schema='task-plan-revision-v21', previous_job_sha256=remote.sha(run / 'job.json'),
                   command=changed, source_commit=commit, old_source_root=str(old), new_source_root=str(new),
                   new_contract_relative_path=recipe.PLAN_FILE,
                   old_contract_sha256=remote.sha(old / recipe.SOURCE_PLAN_FILE),
                   new_contract_sha256=remote.sha(new / recipe.PLAN_FILE),
                   source_archive=str(archive), source_archive_sha256=remote.sha(archive))
    if defect == 'num_envs':
        changed[changed.index('--num-envs') + 1] = '20000'
    elif defect == 'old_sha':
        receipt['old_contract_sha256'] = '0' * 64
    elif defect == 'extracted_bytes':
        (new / names[2]).write_text('# changed after archive\n')
    elif defect == 'unknown_schema':
        receipt['schema'] = 'task-plan-revision-v22'
    path = tmp_path / 'revision.json'
    path.write_text(json.dumps(receipt))
    calls = []

    def checked_migration(source, target, directory):
        # Real migration constraints have separate canonical/ledger tests. This
        # fixture isolates the wrapper and proves it actually invokes that check.
        assert source == json.loads((ROOT / recipe.SOURCE_PLAN_FILE).read_text())
        assert target == json.loads((ROOT / recipe.PLAN_FILE).read_text())
        calls.append(directory)

    monkeypatch.setattr(campaign, 'validate_plan_migration', checked_migration)
    if defect is None:
        assert remote.validate_source_revision(run, changed, previous, path) == receipt
        assert calls == [run / 'train']
    else:
        with pytest.raises(ValueError):
            remote.validate_source_revision(run, changed, previous, path)
        assert calls == []
