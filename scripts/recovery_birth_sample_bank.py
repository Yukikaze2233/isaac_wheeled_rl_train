#!/usr/bin/env python3
"""Build/validate data-only recovery births; never imports Kit or writes a simulator.

The builder authenticates native_prefix_002 before CPU torch deserialization.
The stdlib validator is reusable without torch. Source-aware validation compares
against a newly extracted authenticated bank, rather than trusting self-checksums.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "reports/v6_locomotion_quality_repair_20261005/recovery_audit/native_prefix_002"
DEFAULT_OUTPUT = ROOT / "reports/v6_locomotion_quality_repair_20261005/recovery_audit/state_seed_audit/recovery_birth_sample_bank_candidate_v1.json"
PINNED_SOURCE_SHA256 = {
    "report.json": "a7f152ce7108322785ee091e80dcd5a691a214aafbe060875885cbc6e25c3e32",
    "events.jsonl": "b46a88a1aec516103761e7bcb7086154a29ad1bfb46068f98ffcd335ebe69696",
    "event_snapshots.pt": "8875321f2ef9cf177e19fe40a9e625a97e53f9cb24e83b7bb5127f4d221e0aa9",
    "effective_contract.json": "fd9c9dd996fa16a4a768d4ee193111deb04c5f8235824c5775b5be102a2c8b0e",
}
PINNED_ASSET_SHA256 = "875fa71e89d4d669af0a5471b2334f878d302e06f1b354fc245b95cb5d6f29ba"
EVENT_LEVELS = {
    "motion_unlock": "easy",
    "first_pure_rl_post_physics": "medium",
    "blend_last_mixed_post_physics": "medium",
    "blend_entry_pre_effort": "harder",
}
LANE_CASES = {1: "recovery_right_side_90_stand", 2: "recovery_prone_belly_legs_above_stand"}
FRAMES = {
    "root_pose": "root_link_position_relative_to_source_environment_origin_orientation_world_xyzw",
    "root_velocity": "root_center_of_mass_world_linear_xyz_then_world_angular_xyz",
    "joint_state": "named_full_tree_DOF_in_source_native_joint_order",
    "translation_only_relocation": True,
}
FALSE_FLAGS = ("dynamic_seed_replay_validated", "exact_snapshot_resume", "qualification")
CONTROL_FIELDS = (
    "physics_dt", "policy_dt", "v5_control", "command_slew", "policy_action_order", "actor_dim",
    "actor_frame_dim", "critic_dim", "action_dim", "history_length", "actor_layout", "critic_layout",
    "command_transport", "usb_transport", "signal_delay", "actuator_response", "recovery_training",
    "control_math_sha256", "control_math_source",
)
MECHANICAL_FILES = ("model_spec.json", "mechanical_limits.json", "gas_spring_binding.json", "fit_10mpa.json",
                    "constraints.json", "coordinate_mapping.json", "robot.usda")
SAMPLE_KEYS = {
    "schema", "sample_id", "difficulty_label", *FALSE_FLAGS, "terrain", "frames", "root_link_pose_local_xyzw",
    "root_com_velocity_world6", "joint_order", "joints", "bindings", "source", "source_conditions",
    "source_quality", "software_state_provenance", "sample_sha256",
}
BANK_KEYS = {
    "schema", "candidate_only", "automatic_training_read", *FALSE_FLAGS, "source_artifacts", "bindings",
    "source_contract_projections", "sample_count", "samples", "limits", "content_sha256",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_digest(value, checksum_key):
    return digest(canonical({k: v for k, v in value.items() if k != checksum_key}))


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "Nonfinite candidate value")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def vector(value, count, name):
    require(isinstance(value, list) and len(value) == count
            and all(type(v) in (int, float) and math.isfinite(v) for v in value), f"Invalid finite {name} shape")


def sha_value(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), f"Invalid SHA: {name}")


def validate_sample(sample, *, joint_types, bindings):
    require(isinstance(sample, dict) and set(sample) == SAMPLE_KEYS, "RecoveryBirthSample-v1 fields differ")
    require(sample["schema"] == "RecoveryBirthSample-v1", "Birth sample schema differs")
    finite_tree(sample)
    require(all(sample[name] is False for name in FALSE_FLAGS), "Unvalidated candidate must not claim validation/qualification")
    require(sample["frames"] == FRAMES, "Velocity/root frame differs from root COM world6 contract")
    require(sample["terrain"] == {"family": "flat", "required_platform_height_m": 0.0}, "Birth terrain differs")
    vector(sample["root_link_pose_local_xyzw"], 7, "root pose")
    norm2 = sum(v * v for v in sample["root_link_pose_local_xyzw"][3:])
    require(abs(norm2 - 1.) < 4e-4, "Non-unit root quaternion")
    vector(sample["root_com_velocity_world6"], 6, "root COM world velocity")
    names = sample["joint_order"]
    require(isinstance(names, list) and len(names) == len(set(names)) == 18
            and set(names) == set(joint_types), "Joint mapping differs from authenticated asset")
    require(set(sample["joints"]) == set(names), "Named q18/dq18 mapping differs")
    for name in names:
        joint = sample["joints"][name]
        require(set(joint) == {"type", "position", "velocity", "position_unit", "velocity_unit"}, "Joint fields differ")
        kind = joint_types[name]
        require(kind in ("continuous", "revolute", "prismatic") and joint["type"] == kind, f"Joint type differs: {name}")
        units = ("m", "m/s") if kind == "prismatic" else ("rad", "rad/s")
        require((joint["position_unit"], joint["velocity_unit"]) == units, f"Joint units differ: {name}")
        require(all(type(joint[k]) in (int, float) and math.isfinite(joint[k]) for k in ("position", "velocity")), "Nonfinite joint state")
    require(sample["bindings"] == bindings, "Sample asset/control/mechanics SHA differs")
    source = sample["source"]
    require(set(source) == {"run", "lane", "case", "birth_id", "event", "snapshot_index", "physical_tick", "policy_tick",
                            "reference_tick", "sim_time_s", "event_record_sha256"}, "Source event fields differ")
    lane, event = source["lane"], source["event"]
    require(type(lane) is int and lane in LANE_CASES and source["case"] == LANE_CASES[lane], "Source lane/case differs")
    require(source["run"] == "native_prefix_002" and source["birth_id"] == 1 and event in EVENT_LEVELS, "Source event differs")
    require(sample["sample_id"] == f"native_prefix_002_lane{lane}_{event}" and sample["difficulty_label"] == EVENT_LEVELS[event], "Source event/sample identity differs")
    for name in ("snapshot_index", "physical_tick", "policy_tick", "reference_tick"):
        require(type(source[name]) is int and source[name] >= 0, f"Invalid source {name}")
    require(math.isclose(source["sim_time_s"], source["physical_tick"] * .005, abs_tol=1e-9), "Source physical clock differs")
    sha_value(source["event_record_sha256"], "event record")
    require(sample["software_state_provenance"]["restore"] is False, "Source software state must never be restored")
    require(sample["source_quality"]["source_cohort_partition"] == "success"
            and sample["source_quality"]["new_episode_completion_credit"] is False, "Source result must not become new episode credit")
    sha_value(sample["sample_sha256"], "sample")
    require(content_digest(sample, "sample_sha256") == sample["sample_sha256"], "Sample checksum differs")


def validate_bank(bank, *, expected=None):
    """Validate schema and checksums; expected adds authenticated source equality.

    A checksum in the candidate itself is not an authenticity claim. Consumers
    must pin the bank file SHA, or supply an independently rebuilt expected bank.
    """
    require(isinstance(bank, dict) and set(bank) == BANK_KEYS and bank["schema"] == "RecoveryBirthSampleBank-v1", "Birth bank fields/schema differ")
    finite_tree(bank)
    require(bank["candidate_only"] is True and bank["automatic_training_read"] is False
            and all(bank[name] is False for name in FALSE_FLAGS), "Data-only bank safety flags differ")
    require(bank["source_artifacts"] == PINNED_SOURCE_SHA256, "Authenticated source SHA differs")
    bindings = bank["bindings"]
    require(bindings["asset_manifest_sha256"] == PINNED_ASSET_SHA256, "Asset manifest SHA differs")
    for name, value in bindings.items():
        sha_value(value, name)
    projections = bank["source_contract_projections"]
    require(digest(canonical(projections["control_contract"])) == bindings["control_contract_sha256"], "Control contract projection SHA differs")
    require(digest(canonical(projections["mechanics_contract"])) == bindings["mechanics_contract_sha256"], "Mechanics contract projection SHA differs")
    types = projections["joint_types"]
    require(bank["sample_count"] == len(bank["samples"]) == 8, "Candidate must contain exactly eight observed samples")
    identities = set()
    for sample in bank["samples"]:
        validate_sample(sample, joint_types=types, bindings=bindings)
        identities.add((sample["source"]["lane"], sample["source"]["event"]))
    require(identities == {(lane, event) for lane in LANE_CASES for event in EVENT_LEVELS}, "Source events duplicated/missing")
    require(content_digest(bank, "content_sha256") == bank["content_sha256"], "Bank checksum differs")
    if expected is not None:
        require(bank == expected, "Candidate differs from authenticated source extraction (state/event/provenance)")
    return {"status": "passed", "samples": 8, "source_authenticated": expected is not None,
            "qualification": False, "dynamic_seed_replay_validated": False, "exact_snapshot_resume": False}


def authenticate_sources(source_dir):
    raw = {}
    for name, expected in PINNED_SOURCE_SHA256.items():
        raw[name] = (Path(source_dir) / name).read_bytes()
        require(digest(raw[name]) == expected, f"Authenticated source SHA differs: {name}")
    return raw


def load_bank(path, *, expected_sha256, expected=None):
    """Require an externally pinned file SHA before a consumer accepts data."""
    raw = Path(path).read_bytes()
    sha_value(expected_sha256, "externally pinned bank")
    require(digest(raw) == expected_sha256, "Externally pinned bank file SHA differs")
    bank = read_json(path)
    validate_bank(bank, expected=expected)
    return bank


def build_bank(repo_root=ROOT, source_dir=DEFAULT_SOURCE):
    raw = authenticate_sources(source_dir)  # Must precede any torch deserialization.
    report, cfg = json.loads(raw["report.json"]), json.loads(raw["effective_contract.json"])
    require(report["status"] == "engineering_probe_completed" and report["run_role"] == "full"
            and report["qualification"] is False and report["checkpoint_unchanged"] and report["actor_tensors_unchanged"], "Source execution/authentication differs")
    require(report["contract_sha256"] == PINNED_SOURCE_SHA256["effective_contract.json"], "Source effective contract SHA differs")
    require(cfg["asset_manifest_sha256"] == PINNED_ASSET_SHA256, "Source asset SHA differs")
    asset_dir = Path(repo_root) / cfg["asset_directory"]
    manifest_raw = (asset_dir / "manifest.json").read_bytes()
    require(digest(manifest_raw) == PINNED_ASSET_SHA256, "Actual asset manifest SHA differs")
    manifest = json.loads(manifest_raw)
    mechanical_files = {name: manifest["files_sha256"][name] for name in MECHANICAL_FILES}
    for name, expected in mechanical_files.items():
        require(digest((asset_dir / name).read_bytes()) == expected, f"Asset mechanical file SHA differs: {name}")
    spec = read_json(asset_dir / "model_spec.json")
    joint_types = {row["name"]: row["type"] for row in spec["joints"]}
    names = report["native_joint_names"]
    require(len(names) == len(set(names)) == 18 and set(names) == set(manifest["tree_joint_names"]) == set(joint_types), "Native joint mapping differs from asset")
    control = {"schema": "source-control-contract-projection-v1", "fields": {key: deepcopy(cfg[key]) for key in CONTROL_FIELDS}}
    mechanics = {"schema": "source-mechanics-contract-projection-v1", "asset_manifest_sha256": PINNED_ASSET_SHA256,
        "asset_directory": cfg["asset_directory"], "files_sha256": mechanical_files,
        "physics_dt": cfg["physics_dt"], "position_iterations": cfg["position_iterations"],
        "velocity_iterations": cfg["velocity_iterations"], "configured_dynamics_randomization": cfg["dynamics_randomization"],
        "configured_contact_domain": cfg["contact_domain"]}
    bindings = {"asset_manifest_sha256": PINNED_ASSET_SHA256, "model_spec_sha256": mechanical_files["model_spec.json"],
        "source_effective_contract_sha256": PINNED_SOURCE_SHA256["effective_contract.json"],
        "source_worker_contract_sha256": report["source_worker_sha256"], "control_math_sha256": cfg["control_math_sha256"],
        "control_contract_sha256": digest(canonical(control)), "mechanics_contract_sha256": digest(canonical(mechanics))}
    import io
    import torch
    payload = torch.load(io.BytesIO(raw["event_snapshots.pt"]), map_location="cpu", weights_only=True)
    require(payload["schema"] == "v6-prefix-event-snapshots-v1" and payload["qualification"] is False
            and payload["physx_internal_state_restore_claim"] is False, "Snapshot source scope differs")
    events = [json.loads(line) for line in raw["events.jsonl"].splitlines()]
    snapshots = payload["snapshots"]
    require(len(events) == len(snapshots) == 68, "Source event/snapshot coverage differs")
    cohorts = {row["lane"]: row for row in report["summary"]["rows"]}
    startup = report["startup"]
    samples = []
    for lane in LANE_CASES:
        require(cohorts[lane]["partition"] == "success" and startup["terrain_families"][lane] == "flat", "Source cohort is not successful flat case")
        for event_name, level in EVENT_LEVELS.items():
            matches = [event for event in events if event["lane"] == lane and event["event"] == event_name]
            require(len(matches) == 1, "Source event missing/ambiguous")
            event = matches[0]
            require(event["case"] == LANE_CASES[lane] and cohorts[lane]["events"][event_name] == event["sim_time_s"], "Source event/cohort differs")
            state = snapshots[event["snapshot_index"]]
            def tensor(name, shape):
                value = state[name]
                require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and tuple(value.shape) == shape
                        and bool(torch.isfinite(value).all()), f"Nonfinite/shape source tensor: {name}")
                return value[lane]
            pose = tensor("physics.root_link_pose_w", (4, 7)).clone()
            origin = tensor("env.origins", (4, 3))
            pose[:3] -= origin
            velocity = torch.cat((tensor("physics.root_com_lin_vel_w", (4, 3)), tensor("physics.root_com_ang_vel_w", (4, 3))))
            q, dq = tensor("physics.joint_pos", (4, 18)), tensor("physics.joint_vel", (4, 18))
            named = {}
            for index, name in enumerate(names):
                kind = joint_types[name]
                pos_unit, vel_unit = ("m", "m/s") if kind == "prismatic" else ("rad", "rad/s")
                named[name] = {"type": kind, "position": float(q[index]), "velocity": float(dq[index]),
                               "position_unit": pos_unit, "velocity_unit": vel_unit}
            gravity = tensor("physics.projected_gravity_b", (4, 3))
            sample = {"schema": "RecoveryBirthSample-v1", "sample_id": f"native_prefix_002_lane{lane}_{event_name}",
                "difficulty_label": level, **dict.fromkeys(FALSE_FLAGS, False), "terrain": {"family": "flat", "required_platform_height_m": 0.0},
                "frames": deepcopy(FRAMES), "root_link_pose_local_xyzw": pose.tolist(), "root_com_velocity_world6": velocity.tolist(),
                "joint_order": names.copy(), "joints": named, "bindings": bindings.copy(),
                "source": {"run": "native_prefix_002", **{key: event[key] for key in ("lane", "case", "birth_id", "event", "snapshot_index",
                    "physical_tick", "policy_tick", "reference_tick", "sim_time_s")}, "event_record_sha256": digest(canonical(event))},
                "source_conditions": {"scope": "source_full_role_observation_not_new_episode_parameters", "seed": report["seed"],
                    "source_origin_world_m": origin.tolist(), "source_signal_delay_enabled": report["signal_delay_enabled"],
                    "source_command_transport_module_enabled": report["command_transport_enabled"],
                    "source_command_transport_lane_enabled": bool(state["command_transport.enabled"][lane]),
                    "observed_total_mass_kg": startup["mass_kg_each"][lane],
                    "actual_per_body_mass_inertia_COM": {"status": "unknown_not_exported", "sampled_properties_sha256": startup["dynamics_randomization"]["sampled_properties_sha256"]},
                    "reported_dynamics_metrics": deepcopy(startup["dynamics_randomization"]["metrics"]),
                    "reported_contact_domain_summary": deepcopy(startup["contact_domain"]),
                    "actual_complete_material_parameters_per_lane": {"status": "unknown_not_exported"},
                    "observed_motor_strength6": tensor("env.motor_strength", (4, 6)).tolist(),
                    "observed_spring_strength1": tensor("env.spring_strength", (4, 1)).tolist(),
                    "new_episode_dynamics_and_materials": "unspecified_must_sample_or_bind_explicitly"},
                "source_quality": {"source_cohort_partition": "success", "source_fixed_birth_completion": "2_of_4_one_seed_not_population_qualification",
                    "source_actor_pt_sha256": report["checkpoint"]["checkpoint_sha256"], "source_followup": "stand_only", "new_episode_completion_credit": False,
                    "closure_gap_max_m": float(state["closure_gap_m"][lane].max()), "planar_com_speed_m_s": float(velocity[:2].norm()),
                    "gyro_norm_rad_s": float(velocity[3:].norm()), "tilt_deg": math.degrees(math.acos(float((-gravity[2]).clamp(-1., 1.))))},
                "software_state_provenance": {"restore": False, "native_phase": int(state["fsm.phase"][lane]),
                    "phase_ticks": int(state["fsm.phase_ticks"][lane]), "ready_ticks": int(state["fsm.ready_ticks"][lane]),
                    "stable_ticks": int(state["fsm.stable_ticks"][lane]), "motion_released": bool(state["fsm.motion_released"][lane]),
                    "release_finished": bool(state["recovery.release_finished"][lane]), "elapsed_s": float(state["recovery.elapsed"][lane]),
                    "true_stable_peak_s": float(state["recovery.true_stable_peak"][lane]), "true_success": bool(state["recovery.true_success"][lane]),
                    "learning_mask": bool(state["recovery.learning_mask"][lane]),
                    "recommended_new_birth": "native_SELECT_with_new_zero_dwell_success_elapsed_history_and_transport_response_reset"},
            }
            sample["sample_sha256"] = content_digest(sample, "sample_sha256")
            samples.append(sample)
    bank = {"schema": "RecoveryBirthSampleBank-v1", "candidate_only": True, "automatic_training_read": False,
        **dict.fromkeys(FALSE_FLAGS, False), "source_artifacts": PINNED_SOURCE_SHA256.copy(), "bindings": bindings,
        "source_contract_projections": {"control_contract": control, "mechanics_contract": mechanics, "joint_types": joint_types},
        "sample_count": 8, "samples": samples, "limits": [
            "Eight observed states only; no pose/velocity interpolation or invented samples.",
            "Difficulty labels describe candidate lessons, not dynamically validated ordering.",
            "Cold dynamic seed replay requires a future native birth interface and re-earned gates; exact PhysX contact/solver warmstart is absent.",
            "Source software completion/dwell/phase is provenance only; never restore it to claim a performed script.",
            "Source full-role delay-disabled flat/stand results do not qualify other birth poses, motion, terrain, material/dynamics or delay conditions.",
            "Per-body mass/inertia/COM and complete materials are not exported; aggregate startup observations do not define a new episode.",
            "Bank is stored in reports; no contract or training default points to it.",
        ]}
    bank["content_sha256"] = content_digest(bank, "content_sha256")
    validate_bank(bank)
    return bank


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate", type=Path, help="Validate an existing candidate against authenticated source extraction")
    args = parser.parse_args(argv)
    expected = build_bank(args.repo_root, args.source_dir)
    if args.validate:
        result = validate_bank(read_json(args.validate), expected=expected)
        result["bank_file_sha256"] = digest(args.validate.read_bytes())
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(expected, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        result = validate_bank(read_json(args.output), expected=expected)
        result.update(output=str(args.output), bank_file_sha256=digest(args.output.read_bytes()))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
