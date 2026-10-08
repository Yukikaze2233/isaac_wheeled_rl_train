"""The named repair rejects unknown, partial or already-active startup failures."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import repair_known_native_cli_startup as repair

@pytest.fixture
def q(tmp_path, monkeypatch):
    old = tmp_path / 'boundary_integrated_028.native'; old.mkdir()
    files = {'invocation.intent.json': {}, 'source_selection.json': {},
             'receipt.json': dict(status='failed', exit_code=1, PPO_updates=0,
                 report_sha256=None, manifest_sha256=repair.MANIFEST_SHA,
                 command=['python', '--validate-scene', '1']),
             'stdout.log': '', 'stderr.log': 'before simulator CLI refusal'}
    for name, value in files.items():
        (old/name).write_text(json.dumps(value) if isinstance(value, dict) else value)
    monkeypatch.setattr(repair, 'FAILED_PINS', {name: repair.sha(old/name) for name in files})
    return SimpleNamespace(target_revision=26, config_sha256=repair.MANIFEST_SHA,
        run_root=tmp_path, intent=tmp_path/'launch.intent.json',
        queue_receipt=tmp_path/'launch.receipt.json', owners=lambda: [])

def test_only_authentic_failed_cli_admission_is_eligible(q):
    assert repair.authenticate_failure(q)['PPO_updates'] == 0

@pytest.mark.parametrize('fault', ['bytes', 'scene', 'launch', 'receipt', 'owner', 'manifest', 'revision'])
def test_unknown_or_existing_work_cannot_be_repaired(q, fault):
    if fault == 'bytes': (q.run_root/'boundary_integrated_028.native/stderr.log').write_text('other fault')
    if fault == 'scene': (q.run_root/'boundary_integrated_028.native/scene').mkdir()
    if fault == 'launch': q.intent.write_text('{}')
    if fault == 'receipt': q.queue_receipt.write_text('{}')
    if fault == 'owner': q.owners = lambda: [{'role':'worker'}]
    if fault == 'manifest': q.config_sha256 = '0'*64
    if fault == 'revision': q.target_revision = 25
    with pytest.raises(ValueError): repair.authenticate_failure(q)

@pytest.mark.parametrize('field,value', [('status','passed'), ('exit_code',0), ('PPO_updates',1),
    ('report_sha256','0'*64), ('command',['python','--validate-scene','1','--learning-rate-scale','1.0'])])
def test_even_pinned_failure_cannot_hide_simulation_or_another_fault(q, monkeypatch, field, value):
    p=q.run_root/'boundary_integrated_028.native/receipt.json'
    data=json.loads(p.read_text());data[field]=value;p.write_text(json.dumps(data))
    monkeypatch.setitem(repair.FAILED_PINS, 'receipt.json', repair.sha(p))
    with pytest.raises(ValueError): repair.authenticate_failure(q)
