#!/usr/bin/env python3
"""CPU-only evidence audit; never restores state or imports Isaac/Kit."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
NATIVE = OUT.parent / "native_prefix_002"
FROZEN = "b4ce80964d47c0127e0b3d9929ad35a620bbf204"
CORE = (
    "src/wheeled_tasks/chassis/recovery_curriculum.py",
    "src/wheeled_tasks/chassis/recovery_training.py",
    "src/wheeled_tasks/chassis/recovery.py",
    "src/wheeled_algo/recovery_rollout.py",
    "src/wheeled_tasks/chassis/env.py",
)
REQUIRED = {
    "physics.root_link_pose_w": (4, 7),
    "physics.root_com_lin_vel_w": (4, 3),
    "physics.root_com_ang_vel_w": (4, 3),
    "physics.joint_pos": (4, 18),
    "physics.joint_vel": (4, 18),
    "env.origins": (4, 3),
}
EVENTS = (
    "phase_PREPARE", "blend_entry_pre_effort", "blend_first_applied_post_physics",
    "blend_last_mixed_post_physics", "first_pure_rl_post_physics", "motion_unlock", "followup_success",
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def assert_state(state):
    for name, shape in REQUIRED.items():
        value = state[name]
        assert isinstance(value, torch.Tensor) and tuple(value.shape) == shape, name
        assert torch.isfinite(value).all(), name
    norm = state["physics.root_link_pose_w"][:, 3:].norm(dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=2e-4, rtol=0.)


def main():
    source = {}
    for name in CORE:
        raw = subprocess.check_output(["git", "show", f"{FROZEN}:{name}"], cwd=ROOT)
        current = (ROOT / name).read_bytes()
        assert current == raw, f"Audit requires frozen/current identity: {name}"
        source[name] = {"revision": FROZEN, "sha256": sha(raw)}
    files = {name: {"sha256": sha((NATIVE / name).read_bytes()), "bytes": (NATIVE / name).stat().st_size}
             for name in ("report.json", "events.jsonl", "event_snapshots.pt", "effective_contract.json")}
    report, config = read(NATIVE / "report.json"), read(NATIVE / "effective_contract.json")
    payload = torch.load(NATIVE / "event_snapshots.pt", map_location="cpu", weights_only=False)
    assert payload["schema"] == "v6-prefix-event-snapshots-v1"
    assert payload["qualification"] is False and payload["physx_internal_state_restore_claim"] is False
    events = [json.loads(line) for line in (NATIVE / "events.jsonl").read_text().splitlines()]
    snapshots = payload["snapshots"]
    assert len(snapshots) == len(events) == 68
    rows = []
    for event in events:
        if event["lane"] not in (1, 2) or event["event"] not in EVENTS:
            continue
        state, lane = snapshots[event["snapshot_index"]], event["lane"]
        assert_state(state)
        root = state["physics.root_link_pose_w"][lane].clone()
        root[:3] -= state["env.origins"][lane]
        gravity = state["physics.projected_gravity_b"][lane]
        velocity = state["physics.root_com_lin_vel_w"][lane]
        omega = state["physics.root_com_ang_vel_w"][lane]
        dq = state["physics.joint_vel"][lane]
        rows.append({**event, "source_event_only_no_lesson_credit": True,
            "root_pose_local_xyzw": root.tolist(), "root_com_velocity_world6": torch.cat((velocity, omega)).tolist(),
            "dq18_max_abs": float(dq.abs().max()), "dq18_norm": float(dq.norm()),
            "planar_com_speed_m_s": float(velocity[:2].norm()), "gyro_norm_rad_s": float(omega.norm()),
            "tilt_deg": math.degrees(math.acos(float((-gravity[2]).clamp(-1., 1.)))),
            "closure_gap_max_m": float(state["closure_gap_m"][lane].max()),
            "root_link_vs_com_velocity_difference_m_s": float((state["physics.root_link_lin_vel_w"][lane] - velocity).norm()),
            "native_phase": int(state["fsm.phase"][lane]),
            "control_phase_used": int(state["control.phase_used"][lane]) if "control.phase_used" in state else None,
            "release_finished": bool(state["recovery.release_finished"][lane]),
            "learning_mask": bool(state["recovery.learning_mask"][lane]),
            "motion_released": bool(state["fsm.motion_released"][lane]),
            "true_stable_peak_s": float(state["recovery.true_stable_peak"][lane]),
            "true_success": bool(state["recovery.true_success"][lane]),
            "snapshot_keys": len(state),
        })
    assert len(rows) == 14
    joints = report["native_joint_names"]
    assert len(joints) == len(set(joints)) == 18
    worker_path = ROOT / "contracts/v6_task_campaign_recovery_worker_v8.json"
    worker = read(worker_path)
    table_path = ROOT / worker["recovery_training"]["profiles_file"]
    table = read(table_path)
    assert sha(table_path.read_bytes()) == worker["recovery_training"]["profiles_sha256"]
    assert set(joints) == set(table["tree_joint_names"])
    sys.path.insert(0, str(ROOT / "src"))
    from wheeled_tasks.chassis.recovery_curriculum import RecoveryResetCurriculum
    env = NS(num_envs=4, device="cpu", nominal=torch.zeros(4, 18), robot=NS(joint_names=joints),
             scene_groups=report["case_names"], kinds=["flat"] * 4,
             skills=NS(specs=worker["skill_specs"]), cfg={"asset_manifest_sha256": worker["asset_manifest_sha256"]},
             generator=torch.Generator().manual_seed(290929))
    settings = worker["recovery_training"]["pose_curriculum"]
    base = RecoveryResetCurriculum(env, table, settings)
    # Existing sampler is a reset-only discrete bank, and preserves non-reset lanes.
    before = base.profile_ids.clone()
    sample = base.reset(torch.tensor([1, 2]), actor_update=400)
    assert sample.shape == (2,) and torch.equal(base.profile_ids[[0, 3]], before[[0, 3]])
    endpoint = next(e for e in events if e["lane"] == 2 and e["event"] == "first_pure_rl_post_physics")
    state = snapshots[endpoint["snapshot_index"]]
    rejects = []
    for field, value in (("root_linear_velocity_m_s", state["physics.root_com_lin_vel_w"][2].tolist()),
                         ("full_joint_velocities", dict(zip(joints, state["physics.joint_vel"][2].tolist())))):
        candidate = deepcopy(table)
        candidate["profiles"]["upright"][field] = value
        try:
            RecoveryResetCurriculum(env, candidate, settings)
        except ValueError as exc:
            assert "zero initial velocities" in str(exc), str(exc)
            rejects.append({"field": field, "result": "rejected_as_expected", "message": str(exc)})
        else:
            raise AssertionError(f"Nonzero native endpoint accepted by V8 bank: {field}")
    all_keys = set().union(*(s.keys() for s in snapshots))
    missing_continue = [key for key in (
        "env.episode_length_buf", "env.command_clock", "env.height_clock", "env.push_clock", "env.raw_height",
        "recovery.ever_blended", "recovery.handover_count", "recovery.wheel_balance_active", "fsm.enabled",
        "v5.current_motor_bounds", "v5.requested_motor_effort", "physics.previous_joint_vel", "physics.joint_acc",
    ) if key not in all_keys]
    launch_path = ROOT / "reports/v6_task_campaign_20261001/deploy/launch.json"
    launch = read(launch_path)
    api_path = ROOT.parent / ".deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py"
    api_text = api_path.read_text()
    api_methods = ("write_root_link_pose_to_sim_index", "write_root_com_velocity_to_sim_index",
                   "write_joint_state_to_sim_index", "write_joint_velocity_to_sim_index", "reset")
    api_lines = {name: next(i for i, line in enumerate(api_text.splitlines(), 1)
                           if line.lstrip().startswith(f"def {name}(")) for name in api_methods}
    result = {"schema": "v6-recovery-dynamic-reset-state-read-only-audit-v1", "qualification": False,
        "physical_restore_executed": False, "new_gpu_or_ssh_actions": False, "training_core_or_contract_modified": False,
        "current_launch": {k: launch[k] for k in ("source_commit", "plan_revision", "active_domain", "active_attempt", "scheduled_actor_endpoints")},
        "launch_sha256": sha(launch_path.read_bytes()), "frozen_core": source, "native_prefix_002": files,
        "installed_physx_articulation_api": {"path": str(api_path), "sha256": sha(api_path.read_bytes()),
            "method_definition_lines": api_lines, "inspection_only_not_imported_or_executed": True,
            "state_setters_are_kinematic_only": True},
        "asset_manifest_sha256": worker["asset_manifest_sha256"], "native_joint_names": joints,
        "source_worker_sha256": sha(worker_path.read_bytes()), "source_profiles_sha256": sha(table_path.read_bytes()),
        "snapshot_count": len(snapshots), "event_count": len(events), "required_kinematic_fields": {k: list(v) for k, v in REQUIRED.items()},
        "recorded_missing_fields": report["summary"]["snapshot_field_missing"],
        "additional_software_continuation_fields_not_exported": missing_continue,
        "physx_contact_solver_warmstart_exported": False,
        "signal_delay_present": snapshots[endpoint["snapshot_index"]]["signal_delay.present"],
        "event_phase_note": "Reference-entry events are before torque; applied events are after physics/update but before contact/sensor-buffer append. Cache and consumed observation are previous/current policy-call evidence, not a coherent full engine continuation.",
        "cpu_checks": {"frozen_core_identity": "passed", "14_selected_physical_states_finite_shape_quaternion": "passed",
            "native_joint_order_set_matches_asset": "passed", "old_sampler_reset_only_preserves_other_lanes": "passed",
            "old_sampler_rejects_native_nonzero_velocity": rejects}, "selected_success_source_events": rows,
        "limits": ["Only two successful stand-followup first cohorts, one seed, full-role delay disabled.",
            "Source success is evidence provenance; newly seeded lessons must reset dwell/success/elapsed and re-earn their own result.",
            "Raw 200Hz frames export root-link linear velocity and body gyro, not the COM-world6 tuple expected by the restore API; arbitrary BLEND slots require added native snapshots or justified frame conversion.",
            "No pose/velocity interpolation was performed and no dynamic validation is claimed."]}
    (OUT / "evidence.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "read_only_cpu_audit_completed", "selected_states": len(rows), "output": str(OUT / "evidence.json")}))


if __name__ == "__main__":
    main()
