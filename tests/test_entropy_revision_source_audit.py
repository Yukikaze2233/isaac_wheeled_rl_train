"""Fail-closed source pinning for v21; no framework, scene, contract or state writes."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from wheeled_tasks.chassis import recovery_entropy_revision as recipe


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/tmp/scut-script-frozen-022-pbifeg_2')


def entry(root=ROOT):
    spec = importlib.util.spec_from_file_location('source_audit_' + root.name, root/'scripts/train_chassis.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(root=ROOT):
    return json.loads((root/recipe.WORKER_FILE).read_text())


def test_v21_inherits_exact_v20_pins_and_adds_all_four_admission_contracts():
    old = json.loads((ROOT/recipe.SOURCE_WORKER_FILE).read_text())
    inherited = entry(BASE).scut_curriculum_source_files(old, root=BASE)
    sources = entry().scut_curriculum_source_files(worker(), root=ROOT)
    assert sources == list(dict.fromkeys(inherited + [
        'src/wheeled_tasks/chassis/recovery_entropy_revision.py',
        recipe.SOURCE_PLAN_FILE, recipe.SOURCE_WORKER_FILE, recipe.SOURCE_DESIGN_FILE,
        recipe.PLAN_FILE, recipe.WORKER_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE]))
    assert all((ROOT/name).is_file() for name in sources)
    assert worker()['task_campaign']['revision_migration'] == old['task_campaign']['revision_migration']


def test_absent_entropy_binding_keeps_actual_v20_path_and_never_calls_entropy_recipe(monkeypatch):
    old = json.loads((ROOT/recipe.SOURCE_WORKER_FILE).read_text())
    expected = entry(BASE).scut_curriculum_source_files(old, root=BASE)

    def unexpected(*args, **kwargs):
        raise AssertionError('Old source audit called the inactive entropy revision')

    monkeypatch.setattr(recipe, 'canonical_worker', unexpected)
    assert entry().scut_curriculum_source_files(old, root=ROOT) == expected
    assert entry().scut_curriculum_source_files({}) == []


@pytest.mark.parametrize('case', ['null_binding', 'unknown_binding', 'bool_binding', 'other_domain',
                                  'missing_entropy', 'bool_entropy', 'different_entropy',
                                  'other_mechanical_binding', 'changed_std', 'changed_reward', 'zero_warmup'])
def test_bad_opt_in_cannot_fall_back_to_the_valid_mechanical_binding(case):
    config = worker()
    metadata = config['task_campaign']
    if case == 'null_binding':
        metadata['entropy_revision_migration'] = None
    elif case == 'unknown_binding':
        metadata['entropy_revision_migration']['schema'] = 'unreviewed'
    elif case == 'bool_binding':
        metadata['entropy_revision_migration']['entropy_coef'] = False
    elif case == 'other_domain':
        metadata['domain'] = 'flat'
    elif case == 'missing_entropy':
        del config['entropy_coef']
    elif case == 'bool_entropy':
        config['entropy_coef'] = False
    elif case == 'different_entropy':
        config['entropy_coef'] = .005
    elif case == 'other_mechanical_binding':
        metadata['revision_migration']['schema'] = 'task-worker-revision-v19'
    elif case == 'changed_std':
        config['initial_noise_std'] *= .5
    elif case == 'changed_reward':
        config['unreviewed_reward'] = 1.
    elif case == 'zero_warmup':
        config['resume_critic_warmup_updates'] = 0
    with pytest.raises(ValueError):
        entry().scut_curriculum_source_files(config, root=ROOT)


@pytest.mark.parametrize('file', [recipe.SOURCE_WORKER_FILE, recipe.SOURCE_PLAN_FILE,
                                 recipe.SOURCE_DESIGN_FILE, recipe.WORKER_FILE,
                                 recipe.PLAN_FILE, recipe.DESIGN_FILE, recipe.MANIFEST_FILE])
def test_tampered_bound_predecessor_or_frozen_admission_file_is_rejected(tmp_path, file):
    # Copy contract JSONs and native source only to the owned pytest temp root.
    # The candidate and original predecessor remain byte-for-byte unchanged.
    shutil.copytree(ROOT/'contracts', tmp_path/'contracts')
    shutil.copytree(ROOT/'src', tmp_path/'src', ignore=shutil.ignore_patterns('__pycache__'))
    path = tmp_path/file
    value = json.loads(path.read_text())
    value['unreviewed_tampering'] = True
    path.write_text(json.dumps(value) + '\n')
    with pytest.raises(ValueError):
        entry().scut_curriculum_source_files(worker(), root=tmp_path)
