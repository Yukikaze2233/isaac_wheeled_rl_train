"""CPU-only structural validation; never execute the emitted /proc inspector."""
import ast
from pathlib import Path

PATH=Path(__file__).with_name('verify_first_PPO_020.py')
TEXT=PATH.read_text()
TREE=ast.parse(TEXT)
REMOTE=next(ast.literal_eval(row.value) for row in TREE.body if isinstance(row,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in row.targets))


def test_both_programs_compile_only():
    compile(TREE,str(PATH),'exec')
    compile(ast.parse(REMOTE),str(PATH)+':REMOTE','exec')


def test_no_connection_or_training_commands():
    calls=[n for n in ast.walk(ast.parse(REMOTE)) if isinstance(n,ast.Call)]
    names={getattr(n.func,'attr',getattr(n.func,'id',None)) for n in calls}
    assert not {'system','Popen','run','kill','killpg','signal','launch','new_session','learn'}&names
    assert 'subprocess' not in REMOTE
    assert "pathlib.Path('/proc')" in REMOTE


def test_old_019_parent_and_new_020_admission_unique_attempt():
    assert "parent.parent.parent.name!='attempt_003'" in REMOTE
    assert "directory.parent.name=='attempt_004'" in REMOTE
    assert 'source_revision_020.json' in REMOTE and 'source_repair_launch_020.json' in REMOTE
    assert 'predecessor_020.json' in REMOTE and 'source019-first' not in REMOTE
    assert "origin=before['consumed_updates']" in REMOTE and '29679' not in REMOTE and '2758' not in REMOTE


def test_all_snapshots_adapter_and_full_learning_state():
    assert "pinned=startup['source_sha256']" in REMOTE
    assert "src/wheeled_tasks/chassis/root_kinematics.py" in REMOTE
    assert "sha(directory/'source'/name)!=digest" in REMOTE
    assert "for name,digest in pinned.items()" in REMOTE
    assert "('rng_state','env_rng_state')" in REMOTE and 'cuda_rng_states' in REMOTE
    assert "('actor_state_dict','critic_state_dict','optimizer_state_dict')" in REMOTE
    assert "post-reset bitwise identity claim" in REMOTE


def test_all_admission_evidence_is_reused():
    settings=next(row for row in TREE.body if isinstance(row,ast.FunctionDef) and row.name=='settings_for')
    code=ast.unparse(settings)
    for key in ('remote_30_roles','native_domains','semantic_review','paired_summary','integration_review',
                'integration_native_review','baseline_native_jump','baseline_native_stepjump'):
        assert key in code
    assert 'engineering_limitations' in REMOTE
    assert "qualification=None" in REMOTE
    assert "max(0,endpoint-task_origin)" in REMOTE
    assert "unchanged_no" not in REMOTE
