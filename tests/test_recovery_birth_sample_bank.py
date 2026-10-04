"""CPU integrity checks for the data-only, observed recovery birth candidate."""
from copy import deepcopy
import builtins
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import recovery_birth_sample_bank as bank_module


@pytest.fixture(scope="module")
def bank():
    return bank_module.build_bank()


def resign(bank):
    """A tamperer can recompute checksums; source equality must still reject it."""
    for sample in bank["samples"]:
        sample["sample_sha256"] = bank_module.content_digest(sample, "sample_sha256")
    bank["content_sha256"] = bank_module.content_digest(bank, "content_sha256")


def test_eight_samples_match_observed_source_and_joint_units(bank):
    assert bank_module.validate_bank(bank, expected=bank)["source_authenticated"]
    assert len(bank["samples"]) == 8
    assert {s["source"]["lane"] for s in bank["samples"]} == {1, 2}
    assert {s["source"]["event"] for s in bank["samples"]} == set(bank_module.EVENT_LEVELS)
    for sample in bank["samples"]:
        assert sample["root_com_velocity_world6"] != [0.] * 6
        assert len(sample["joints"]) == 18
        assert sample["joints"]["L_spring_slide"]["velocity_unit"] == "m/s"
        assert sample["joints"]["L_joint3"]["velocity_unit"] == "rad/s"
        assert sample["software_state_provenance"]["restore"] is False
        assert sample["source_conditions"]["actual_per_body_mass_inertia_COM"]["status"] == "unknown_not_exported"
        assert sample["source_conditions"]["actual_complete_material_parameters_per_lane"]["status"] == "unknown_not_exported"


def test_stdlib_validator_imports_no_torch_or_isaac(bank, tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank, allow_nan=False))
    code = "import sys,json;sys.path.insert(0,'scripts');import recovery_birth_sample_bank as b;b.validate_bank(json.load(open(sys.argv[1])));assert 'torch' not in sys.modules;assert not any(n.startswith(('isaac','omni')) for n in sys.modules)"
    subprocess.run([sys.executable, "-c", code, str(path)], cwd=ROOT, check=True)


def test_every_exported_kinematic_component_equals_actual_snapshot(bank):
    import torch
    source = torch.load(bank_module.DEFAULT_SOURCE / "event_snapshots.pt", map_location="cpu", weights_only=True)
    observed_link_com_differences = []
    for sample in bank["samples"]:
        state = source["snapshots"][sample["source"]["snapshot_index"]]
        lane = sample["source"]["lane"]
        pose = state["physics.root_link_pose_w"][lane].clone()
        pose[:3] -= state["env.origins"][lane]
        assert sample["root_link_pose_local_xyzw"] == pose.tolist()
        velocity = torch.cat((state["physics.root_com_lin_vel_w"][lane], state["physics.root_com_ang_vel_w"][lane]))
        assert sample["root_com_velocity_world6"] == velocity.tolist()
        for index, name in enumerate(sample["joint_order"]):
            assert sample["joints"][name]["position"] == float(state["physics.joint_pos"][lane, index])
            assert sample["joints"][name]["velocity"] == float(state["physics.joint_vel"][lane, index])
        observed_link_com_differences.append(float((state["physics.root_link_lin_vel_w"][lane] - velocity[:3]).norm()))
    assert min(observed_link_com_differences) > 1e-4  # Wrong link-frame export would materially fail this test.


def test_consumer_requires_external_bank_file_sha(bank, tmp_path):
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank, allow_nan=False))
    pin = bank_module.digest(path.read_bytes())
    assert bank_module.load_bank(path, expected_sha256=pin, expected=bank) == bank
    with pytest.raises(ValueError, match="Externally pinned bank file SHA"):
        bank_module.load_bank(path, expected_sha256="0" * 64)


@pytest.mark.parametrize("filename", list(bank_module.PINNED_SOURCE_SHA256))
def test_source_sha_tamper_rejected_before_deserialization(filename, tmp_path, monkeypatch):
    for name in bank_module.PINNED_SOURCE_SHA256:
        raw = (bank_module.DEFAULT_SOURCE / name).read_bytes()
        (tmp_path / name).write_bytes(raw + b" " if name == filename else raw)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith(("isaac", "omni")):
            raise AssertionError("Source authentication must precede deserialization or simulation imports")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match="Authenticated source SHA differs"):
        bank_module.build_bank(source_dir=tmp_path)


@pytest.mark.parametrize("change,match", [
    (lambda s: s["joint_order"].__setitem__(0, s["joint_order"][1]), "Joint mapping"),
    (lambda s: s["joints"].pop("L_joint1"), "Named q18/dq18"),
    (lambda s: s["joints"]["L_spring_slide"].__setitem__("type", "continuous"), "Joint type"),
    (lambda s: s["joints"]["L_spring_slide"].__setitem__("velocity_unit", "rad/s"), "Joint units"),
    (lambda s: s["frames"].__setitem__("root_velocity", "root_link_body_velocity"), "Velocity/root frame"),
    (lambda s: s["root_link_pose_local_xyzw"].__setitem__(6, 0.), "Non-unit root quaternion"),
    (lambda s: s["root_com_velocity_world6"].__setitem__(0, float("nan")), "Nonfinite"),
    (lambda s: s["joints"]["L_joint1"].__setitem__("velocity", float("inf")), "Nonfinite"),
    (lambda s: s["software_state_provenance"].__setitem__("restore", True), "never be restored"),
    (lambda s: s["source_quality"].__setitem__("new_episode_completion_credit", True), "new episode credit"),
    (lambda s: s.__setitem__("qualification", True), "must not claim"),
])
def test_invalid_sample_rejected(bank, change, match):
    bad = deepcopy(bank)
    change(bad["samples"][0])
    # No checksum fabrication here: structural constraints must reject first.
    with pytest.raises(ValueError, match=match):
        bank_module.validate_bank(bad)


@pytest.mark.parametrize("target", ["source_artifacts", "bindings"])
def test_asset_or_source_sha_tamper_rejected(bank, target):
    bad = deepcopy(bank)
    key = "report.json" if target == "source_artifacts" else "asset_manifest_sha256"
    bad[target][key] = "0" * 64
    with pytest.raises(ValueError, match="SHA differs"):
        bank_module.validate_bank(bad)


@pytest.mark.parametrize("field", ["snapshot_index", "event_record_sha256", "root_com_velocity_world6", "joint_position"])
def test_resigned_source_event_or_state_tamper_rejected_by_authentication(bank, field):
    bad = deepcopy(bank)
    sample = bad["samples"][0]
    if field == "snapshot_index":
        sample["source"][field] += 1
    elif field == "event_record_sha256":
        sample["source"][field] = "0" * 64
    elif field == "root_com_velocity_world6":
        sample[field][0] += .1
    else:
        sample["joints"]["L_joint1"]["position"] += .1
    resign(bad)
    with pytest.raises(ValueError, match="authenticated source extraction"):
        bank_module.validate_bank(bad, expected=bank)


def test_source_event_name_tamper_and_duplicate_sample_rejected(bank):
    bad = deepcopy(bank)
    bad["samples"][0]["source"]["event"] = "followup_success"
    with pytest.raises(ValueError, match="Source event differs"):
        bank_module.validate_bank(bad)
    bad = deepcopy(bank)
    bad["samples"][1] = deepcopy(bad["samples"][0])
    resign(bad)
    with pytest.raises(ValueError, match="duplicated/missing"):
        bank_module.validate_bank(bad)


def test_builder_reproducible_and_cli_validates_existing_bank(bank, tmp_path, capsys):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    assert bank_module.main(["--validate", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source_authenticated"] and not result["dynamic_seed_replay_validated"]
    assert bank_module.build_bank() == bank
