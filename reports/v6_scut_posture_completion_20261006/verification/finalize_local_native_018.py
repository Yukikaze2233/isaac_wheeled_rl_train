#!/usr/bin/env python3
"""Validate and preserve an unchanged frozen v18 local engineering report."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import platform
import tarfile

COMMIT = "4ba339299c203cf2aba77183ec81f2f017dc9bf9"
ARCHIVE = "f93a7ef910a51bb151de13401e01947c075688f5479181c8ff45c753de935715"
ROOT = Path("/tmp/scut-v18-native-4ba-local")
OUT = Path(__file__).resolve().parent
ORIGINS = {"jump": 5552, "stepjump": 0, "speed": 5063}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=ORIGINS)
    args = parser.parse_args()
    domain = args.domain
    identity = json.loads((ROOT / "launch_identity.json").read_text())
    assert identity["source_commit"] == COMMIT and identity["archive_sha256"] == ARCHIVE
    run = identity["domains"][domain]
    assert run["returncode"] == 0 and run["status"] == "engineering_probe_completed", run
    source = Path(identity["clean_root"])
    current = {name: digest(source / name) for name in identity["source_before"]}
    assert current == identity["source_before"], "Frozen source bytes changed"
    frozen = json.loads((OUT / "clean_30_roles_018.json").read_text())
    assert frozen["source_commit"] == COMMIT and frozen["all_selected_source_bytes_match_archive"]
    assert current == frozen["exactsource_files_before"]
    path = ROOT / domain / "report.json"
    raw = path.read_bytes()
    report = json.loads(raw)
    expected = dict(schema="scut-curriculum-native-probe-v18", status="engineering_probe_completed",
        domain=domain, num_envs=4096, policy_steps=600, physics_steps=2400,
        elapsed_physics_s=12., formal_updates_consumed=0, optimizer_steps=0,
        qualification=None, acceptance="not_assessed", run_role="train",
        engineering_clock_reset_before_birth=True, actor_tensors_unchanged=True,
        checkpoint_unchanged=True, source_files_unchanged=True, target_contract_unchanged=True,
        source_files_changed=[], randomization_or_delay_overrides=[],
        forced_engineering_actor_update=ORIGINS[domain])
    assert all(report.get(k) == v for k, v in expected.items()), {
        k: (report.get(k), v) for k, v in expected.items() if report.get(k) != v}
    assert report["target_contract_sha256"] == digest(source / f"contracts/v6_task_campaign_{domain}_worker_v18.json")
    assert report["startup"]["gpu_dynamics_enabled"] is True
    assert report["startup"]["runtime_preflight"]["status"] == "runtime_configuration_passed"
    assert report["summary"]["counts"]["all_policy_intervals"] == 4096 * 600
    assert report["summary"]["coverage"]["functional_qualification"] is False
    for name, value in report["source_files_sha256"].items():
        assert digest(source / name) == value, name
    metadata = json.loads((OUT.parent / "deploy/source_candidate_018.json").read_text())
    assert metadata["source_commit"] == COMMIT and metadata["archive_sha256"] == ARCHIVE
    archive_path = Path(metadata["archive_local"])
    assert digest(archive_path) == ARCHIVE
    with tarfile.open(archive_path) as archive:
        assert archive.pax_headers["comment"] == COMMIT
        for name, value in report["source_files_sha256"].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == value, name
    timing = report["control_timing"]
    assert timing["physics_steps"] == 2400 and timing["policy_steps"] == 600
    assert timing["pd_updates"] == timing["feedback_refreshes"] == 4
    assert timing["target_decodes"] == 1
    assert timing["physics_dt"] == .005 and timing["policy_dt"] == .02
    proof = report["ordinary_posture_native_proof"]
    initial, final = proof["initial"], proof["final"]
    assert initial["native_root_quaternion_readback_passed"] is True
    assert proof["actual_native_velocity_delta_observed"] is True
    assert proof["actual_root_rotation_observed"] is True
    assert initial["birth_selected_count"] > 0 and initial["birth_abs_angle_max_rad"] > 0
    assert initial["push_clock_std_s"] > 0 and initial["push_speed_cap_m_s"] > 0
    assert final["push_events"] > 0 and final["push_velocity_delta_max_m_s"] > 0
    for observed in (initial, final):
        assert observed["geometry_bound"] is True and observed["relative_actor_updates"] == 0
        assert observed["birth_angle_cap_rad"] == .03 and observed["push_speed_cap_m_s"] == .05
        assert 5. <= observed["push_clock_min_s"] <= observed["push_clock_max_s"] <= 10.
        assert 0. < observed["birth_abs_angle_max_rad"] <= .03
    checkpoint = Path(report["checkpoint"]["checkpoint"])
    assert digest(checkpoint) == identity["checkpoint_sha256"] == report["checkpoint"]["checkpoint_sha256"]
    assert digest(Path(report["checkpoint"]["checkpoint_contract"])) == report["checkpoint"]["checkpoint_contract_sha256"]
    destination = OUT / f"native_{domain}_018_final.report.json.gz"
    packed = gzip.compress(raw, mtime=0)
    if destination.exists():
        assert destination.read_bytes() == packed, "Refusing to replace a different native receipt"
    else:
        destination.write_bytes(packed)
    assert gzip.decompress(destination.read_bytes()) == raw
    receipt = dict(status="native_engineering_completed_no_PPO_no_qualification", domain=domain,
        source_commit=COMMIT, archive_sha256=ARCHIVE, clean_root=str(source),
        report=str(destination), report_sha256=digest(destination), raw_report_sha256=digest(path),
        command=run["command"], returncode=run["returncode"], source_files_unchanged=True,
        execution_host=platform.node(), gpu_prelaunch=run.get("gpu_prelaunch", identity["gpu_prelaunch"]),
        validator_sha256=digest(Path(__file__)),
        exactsource_file_count=len(current), checkpoint_sha256=digest(checkpoint),
        policy_steps=600, physics_steps=2400, elapsed_physics_s=12., num_envs=4096,
        formal_updates_consumed=0, optimizer_steps=0, qualification=None,
        wall_seconds=report["wall_seconds"], ordinary_posture_native_proof=proof,
        control_timing=timing, structural_coverage=report["summary"]["coverage"],
        native_counts=report["summary"]["counts"],
        limitations=["Root quaternion and push velocity use actual native readback assertions; raw pose arrays are not stored.",
            "This report does not certify joint q/qd readback or task success.",
            "One bounded deterministic old-flat-actor engineering probe; no optimizer, budget consumption or qualification."])
    (OUT / f"native_{domain}_018_final.identity.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in ("status", "domain", "wall_seconds", "source_commit", "report")}, indent=2))
    print(json.dumps(dict(ordinary_posture_native_proof=proof, control_timing=timing,
        coverage=receipt["structural_coverage"]), indent=2))


if __name__ == "__main__":
    main()
