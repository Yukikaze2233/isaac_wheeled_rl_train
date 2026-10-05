"""Every new v17 teaching dependency is bound before an Isaac app starts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v17_train_entry", ROOT / "scripts/train_chassis.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def worker(domain):
    return json.loads((ROOT / f"contracts/v6_task_campaign_{domain}_worker_v17.json").read_text())


@pytest.mark.parametrize("domain,module", [("recovery", "recovery_prefix_curriculum"),
    ("terrain", "scut_terrain_revision"), ("jump", "scut_jump_revision"), ("stepjump", "scut_jump_revision")])
def test_v17_audits_recipe_native_teaching_modules_and_bound_predecessor(domain, module):
    cfg = worker(domain)
    files = entry.scut_curriculum_source_files(cfg)
    assert len(files) == len(set(files))
    assert all((ROOT / path).is_file() for path in files)
    assert f"src/wheeled_tasks/chassis/{module}.py" in files
    assert "src/wheeled_tasks/chassis/scut_base_tracking.py" in files
    assert "src/wheeled_tasks/chassis/scut_curriculum_revision.py" in files
    assert cfg["task_campaign"]["revision_migration"]["source_contract_file"] in files
    if domain != "recovery":
        assert cfg["nonstandard_start"]["profiles_file"] in files
        assert "src/wheeled_algo/recovery_rollout.py" in files
    else:
        assert "src/wheeled_tasks/chassis/recovery_curriculum.py" in files
    cfg["evaluation_exact_cases"] = True
    if "nonstandard_start" in cfg:
        cfg["nonstandard_start"]["enabled"] = False
    assert entry.scut_curriculum_source_files(cfg) == files


def test_old_workers_do_not_implicitly_enable_new_recipe_audit():
    cfg = json.loads((ROOT / "contracts/v6_task_campaign_recovery_worker_v8.json").read_text())
    assert entry.scut_curriculum_source_files(cfg) == []


@pytest.mark.parametrize("field,value", [("source_contract_sha256", "0" * 64),
    ("origin_actor_update", 2293)])
def test_changed_revision_binding_refuses_launch(field, value):
    cfg = worker("recovery")
    cfg["task_campaign"]["revision_migration"][field] = value
    with pytest.raises(ValueError, match="exact revision binding"):
        entry.scut_curriculum_source_files(cfg)


def test_new_jump_birth_audit_does_not_invent_a_release_stop_teacher():
    cfg = worker("jump")
    assert "release_stop" not in cfg
    assert entry.nonstandard_start_source_files(cfg, require_release_stop=False)
    with pytest.raises(ValueError, match="Release-stop settings"):
        entry.nonstandard_start_source_files(cfg)
