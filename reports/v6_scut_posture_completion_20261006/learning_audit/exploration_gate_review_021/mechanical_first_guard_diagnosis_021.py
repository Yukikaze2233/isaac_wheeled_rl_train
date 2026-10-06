"""Read native terminal evidence; sweep sealed geometry on CPU, never physics."""
from pathlib import Path
from hashlib import sha256
import gzip
import importlib.util
import json
import math
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = Path("/tmp/scut-recovery-frozen-021-qv1ukmbr")
PAIR = HERE / "REC3076_pair_001"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def gz_rows(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream]


def main():
    sys.path.insert(0, str(SOURCE / "src"))
    spec = importlib.util.spec_from_file_location("sealed_profile_cpu_only", SOURCE / "scripts/build_v6_recovery_profiles.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = module.RecoveryProfileBuilder()
    profiles = json.loads((SOURCE / "contracts/v6_recovery_profiles_v2.json").read_text())
    names = {1708: "step200_wheels_split", 2216: "double_step_first_riser_split"}
    evidence, paths = {}, [Path(__file__)]
    for arm in ("mean", "stochastic"):
        selected = {row["lane"]: row for row in gz_rows(PAIR / arm / "first_birth_rows.jsonl.gz") if row["lane"] in names}
        events = [row for row in gz_rows(PAIR / arm / "events.jsonl.gz") if row["lane"] in names and row["birth_id"] == 1]
        structural = json.loads((PAIR / arm / "teaching_summary.json").read_text())["native_formal_probe_structural_audit"]
        for row in selected.values():
            assert row["termination_reasons"] == ["closure_gap"] and not row["whole_interval_actor_valid"]
            assert row["recovery_phase"] == 5 and row["recovery_route"] == 4
            assert row["recovery_handover_count"] == 0 and not row["recovery_ever_blended"]
            assert row["first_handover_reference_tick"] == row["first_motion_release_reference_tick"] == -1
        evidence[arm] = {"terminal_rows": selected, "terminal_events": events,
                         "full_window_guard_count": structural["counts"]["closure_guard_terminal_lanes"],
                         "full_window_max_pin_gap_m": structural["max_pin_gap_m"],
                         "full_window_max_pin_gap_lane_tick": None}
        paths += [PAIR / arm / name for name in ("first_birth_rows.jsonl.gz", "events.jsonl.gz", "teaching_summary.json")]
    assert evidence["mean"]["terminal_rows"] == evidence["stochastic"]["terminal_rows"]
    assert evidence["mean"]["terminal_events"] == evidence["stochastic"]["terminal_events"]

    from wheeled_tasks.chassis.mechanism_observer import ClosedChainHeightModel
    from wheeled_tasks.chassis.step_trigger import estimate_height_if_grounded
    from wheeled_tasks.chassis.recovery import RecoveryFSM
    lookup = ClosedChainHeightModel(json.loads((SOURCE / "contracts/v6_height_lookup_v1.json").read_text()), "cpu")
    birth = profiles["profiles"][names[1708]]
    twin = profiles["profiles"][names[2216]]
    for key in ("root_position_m", "root_quaternion_xyzw", "control6_rad", "full_joint_positions", "initial_release_s"):
        assert birth[key] == twin[key]
    q = torch.tensor([birth["control6_rad"]], dtype=torch.float32)
    points, valid = lookup.wheel_positions(q)
    gravity = torch.tensor([[0., 0., -1.]])
    height, height_valid = estimate_height_if_grounded(points, gravity, lookup.wheel_radius_m,
        grounded=valid, sensor_valid=valid, wheel_axle_b=lookup.wheel_axles_b)
    prepare = profiles["prepare_reference"]["pd_control6_rad"]
    fsm = RecoveryFSM(1, "cpu", {}, prepare, profiles["targets"]["fold"]["control6_rad"],
        profiles["targets"]["thrust"]["control6_rad"], prepare, profiles["axis_signs"])
    fsm.reset(torch.tensor([0]), q, True)
    fsm.update(q, torch.zeros_like(q), gravity, torch.zeros((1, 3)), height.min(-1).values,
        torch.tensor([False]), torch.tensor([False]))

    # Counterfactual only: the real chassis moves. These samples keep the birth
    # root transform fixed and solve all passive coordinates at each reference.
    origin = np.asarray(birth["root_position_m"])
    start = np.asarray(birth["control6_rad"])
    sweeps = {}
    for label, goal in {"prepare_command": prepare, "fold": profiles["targets"]["fold"]["control6_rad"]}.items():
        full_q, samples = dict(birth["full_joint_positions"]), []
        for fraction in np.linspace(0., 1., 21):
            control = (1 - fraction) * start + fraction * np.asarray(goal)
            full_q = builder.solve(dict(zip(builder.controls, control)), full_q)
            geometry = builder.validate_q(full_q)
            minimum = None
            for row in builder.collision_bounds(full_q, np.eye(3)):
                candidates = [(float(origin[2] + row["lower"][2]), "ground")]
                z = builder.halfspace_min_z(row["points"], -origin[0])
                if z is not None:
                    candidates.append((float(origin[2] + z - .2), "first_tread"))
                clearance, face = min(candidates)
                candidate = {"fraction": float(fraction), "body": row["body"],
                    "clearance_m": clearance, "surface": face, **geometry}
                if minimum is None or clearance < minimum["clearance_m"]:
                    minimum = candidate
            samples.append(minimum)
        sweeps[label] = {"samples": samples, "worst": min(samples, key=lambda x: x["clearance_m"]),
                        "first_negative": next((x for x in samples if x["clearance_m"] < 0), None)}
    paths += [SOURCE / name for name in ("scripts/build_v6_recovery_profiles.py", "scripts/build_v6_height_lookup.py",
        "contracts/v6_recovery_profiles_v2.json", "contracts/v6_height_lookup_v1.json",
        "contracts/v6_task_campaign_recovery_worker_v19.json", "src/wheeled_tasks/chassis/recovery.py",
        "src/wheeled_tasks/chassis/recovery_training.py", "src/wheeled_tasks/chassis/mechanism_observer.py",
        "src/wheeled_tasks/chassis/env.py", "model/纯底盘_v6_232mm_stop105/urdf/model_spec.json",
        "model/纯底盘_v6_232mm_stop105/urdf/tools/build_v6_closedchain.py")]
    output = {"schema": "source021-shared-script-first-closure-guard-CPU-diagnosis-v1",
        "source_commit": "d0dce6181df83b03fe166abcbadef3c81de2e57e",
        "evidence": evidence, "terminal_records_exactly_equal_between_arms": True,
        "birth_geometry_validation": builder.validate_q(birth["full_joint_positions"]),
        "shared_geometry_birth": {key: birth[key] for key in ("root_position_m", "control6_rad", "initial_release_s", "geometry_validation")},
        "ideal_unmoved_birth_SENSOR_counterfactual": {"wheel_grounded_height_m": height.tolist(),
            "height_difference_m": float((height[:, 0] - height[:, 1]).abs()),
            "valid": height_valid.tolist(), "route": int(fsm.route[0]), "phase": int(fsm.phase[0]),
            "note": "not the native first sensor sample after30ms free release; no true contact inferred"},
        "fixed_birth_transform_geometric_sweeps": sweeps,
        "sweep_max_closure_residual_m": builder.max_gap_m,
        "whole_window_max_location_unavailable": True,
        "limitations": ["No native per-lane q/qref/torque/contact/pin-gap trace or first-gap argmax was captured.",
            "Counterfactual negative clearance is a collision risk, not a reconstruction of the actual moving chassis or proof of sole cause.",
            "Static closure solves do not test inertia, solver/contact dynamics, tracking, actual support or self collision.",
            "Two shared first terminals precede actor ownership; entropy cannot fix this script-owned failure.",
            "Global730.632mm cannot be assigned to either identified lane or either policy tick.",
            "The 3mm guard remains unchanged; no training/deployment/mechanical qualification granted."],
        "next_native_diagnostic": {"scope": "source-bound independent0PPO local only after separate approval; formal Kaiser continues",
            "retain": ["same own-REC3076 parent and full native RNG/birth/properties", "4096 population and same lane IDs", "solver64/32", "dt.005", "all original gas/gravity/PD/delay/contact properties", "3mm guard"],
            "capture5ms": ["two lanes reference and actual full joint q/dq", "all pin errors and argmax before reset", "script phase/route/qref/release/reroute", "requested/clamped/applied motor and gas torque", "root pose/velocity/gyro", "wheel/shell contact body/terrain identity, normals/impulses and separation", "mass/COM/friction/gain/strength/delay randomization per lane"],
            "window": "initial write through first1.3s or first terminal; retain pre-reset terminal tensors; then later5guards and global argmax separate",
            "judge": "reproduce baseline first guards, locate earliest gap and contact/saturation/reference change, then one source-bounded script/birth change and repeat; never lower guard"},
        "minimal_proposals_not_implemented": [
            "Containment: a new explicit contract may temporarily remove only the two proven-bad original-chain split profiles from PPO reset sampling and retain them in exact diagnostic/evaluation; do not silently grant their skills or claim repair.",
            "Physics repair depends on native trace: preserve high-side support and use a collision-clear terrain-edge recovery path before symmetric fold/prepare; require actual support/asymmetry evidence before the near-upright route. Merely slowing references or adding a near-upright gate is not proven sufficient.",
            "If the same safe script/contact trajectory still opens pins, diagnose closure/contact solver and collision representation separately; entropy and looser3mm bounds are not remedies."],
        "files_sha256": {str(path): digest(path) for path in paths},
        "CPU_only": True, "CUDA_initialized": torch.cuda.is_initialized(), "PPO_updates": 0,
        "SSH": False, "physics_simulation_run": False, "production_modified": False, "qualification": None}
    assert not output["CUDA_initialized"]
    (HERE / "mechanical_first_guard_diagnosis_021.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"CPU_only": True, "shared_guard_lanes": list(names),
                      "quasi_static_sweeps": {k: v["worst"] for k, v in sweeps.items()},
                      "json_sha256": digest(HERE / "mechanical_first_guard_diagnosis_021.json")}, indent=2))


if __name__ == "__main__":
    main()
