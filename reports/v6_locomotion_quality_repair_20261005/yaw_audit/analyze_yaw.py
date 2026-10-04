#!/usr/bin/env python3
"""Read-only inputs, CPU-only frozen reward counterfactuals; no Isaac or PPO.

Run with the existing isaacsim60-venv Python. Output is diagnostic evidence,
not observed runtime reward payments or a model qualification.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn.functional as F
import yaml

SOURCE = "b4ce80964d47c0127e0b3d9929ad35a620bbf204"
SCUT_SOURCE = "b8ff79f3df855faf9dc92f4a282bd80c42649466"
REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
OUT = Path(__file__).resolve().parent
PROBE = REPO / "reports/v6_locomotion_quality_repair_20261005/probe/native_003"
SCUT = REPO.parent / "scut-wheeled-legged-rl"
torch.set_num_threads(1)
torch.set_default_device("cpu")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen(repo: Path, source: str, name: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), "show", f"{source}:{name}"])


def read(path: Path):
    return json.loads(path.read_text())


def stats(value: torch.Tensor):
    return {"mean": float(value.mean()), "std_population": float(value.std(correction=0)),
            "abs_mean": float(value.abs().mean()), "min": float(value.min()), "max": float(value.max())}


def wheel_axes():
    """Nominal asset FK establishes signs only; not a dynamic no-slip claim."""
    directory = REPO / WORKER["asset_directory"]
    manifest = read(directory / "manifest.json")
    assert sha((directory / "manifest.json").read_bytes()) == WORKER["asset_manifest_sha256"]
    joints = {j.find("child").attrib["link"]: j for j in ET.parse(directory / "robot.urdf").getroot().findall("joint")}
    rotations, axes = {"base_link": np.eye(3)}, {}

    def rotation(link):
        if link in rotations:
            return rotations[link]
        joint = joints[link]
        parent = rotation(joint.find("parent").attrib["link"])
        rx, ry, rz = map(float, joint.find("origin").attrib.get("rpy", "0 0 0").split())
        sx, cx, sy, cy, sz, cz = math.sin(rx), math.cos(rx), math.sin(ry), math.cos(ry), math.sin(rz), math.cos(rz)
        origin = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                           [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])
        axis = np.array(list(map(float, joint.find("axis").attrib["xyz"].split())))
        axis /= np.linalg.norm(axis)
        name = joint.attrib["name"]
        axes[name] = parent @ origin @ axis
        q = manifest["nominal_joint_pos"].get(name, 0.)
        cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        local = np.eye(3) + math.sin(q)*cross + (1.-math.cos(q))*(cross @ cross)
        rotations[link] = parent @ origin @ local
        return rotations[link]

    for name in ("L_link3", "R_link3"):
        rotation(name)
    selected = {name: axes[name].tolist() for name in ("L_joint3", "R_joint3")}
    assert selected["L_joint3"][1] > .99 and selected["R_joint3"][1] < -.99
    return {"nominal_axes_base_xyz": selected, "manifest_sha256": WORKER["asset_manifest_sha256"],
            "urdf_sha256": sha((directory / "robot.urdf").read_bytes()),
            "scope": "nominal tree FK in manifest Xforward/Yleft/Zup, not runtime contact kinematics"}


def exact_native_functions():
    names = {"reward_terms", "dense_error_terms", "apply_dense_tracking", "apply_manual_tracking"}
    data = frozen(REPO, SOURCE, "src/wheeled_tasks/chassis/rewards.py")
    module = ast.parse(data)
    definitions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in definitions} == names
    namespace = {"torch": torch, "F": F, "math": math,
                 "V5Control": SimpleNamespace(LEGS=(0, 1, 3, 4), WHEELS=(2, 5))}
    exec(compile(ast.Module(definitions, []), f"{SOURCE}:rewards.py", "exec"), namespace)
    return namespace, sha(data), {n.name: n.lineno for n in definitions}


NATIVE, REWARD_SHA, FUNCTION_LINES = exact_native_functions()
WORKER_PATH = REPO / "contracts/v6_task_campaign_speed_worker_v14.json"
WORKER = read(WORKER_PATH)
DENSE = WORKER["dense_tracking"]
DT = WORKER["policy_dt"]


def native_counterfactual(yaw, velocity=None, gravity=None, commands=None, support=None):
    """Yaw terms are exact. Unknown leg states are zero placeholders, never totaled."""
    yaw = torch.as_tensor(yaw, dtype=torch.float64, device="cpu").flatten()
    count = yaw.numel()
    velocity = torch.zeros(count, 3) if velocity is None else velocity
    gravity = torch.tensor([[0., 0., -1.]]).repeat(count, 1) if gravity is None else gravity
    commands = torch.tensor([[0., 0., .305]]).repeat(count, 1) if commands is None else commands
    support = torch.ones(count) if support is None else support
    omega = torch.zeros(count, 3, dtype=torch.float64)
    omega[:, 2] = yaw
    zero6, ordinary = torch.zeros(count, 6), torch.ones(count, dtype=torch.bool)
    reference = SimpleNamespace(jumping=torch.zeros(count, dtype=torch.bool),
                                terrain_mode=torch.zeros(count), elapsed=torch.ones(count) * 10.)
    terms = NATIVE["reward_terms"](velocity, omega, gravity, commands[:, 2], commands,
        zero6, zero6, zero6, zero6, zero6, zero6, torch.zeros(count, 2, 3),
        support, ordinary, torch.zeros(count, dtype=torch.bool))
    base = {k: terms[k].clone() for k in ("track_yaw", "yaw_square")}
    NATIVE["apply_manual_tracking"](terms, velocity, omega, gravity, commands, support,
                                    ordinary, reference, WORKER["reference_reward"])
    manual = {k: terms[k].clone() for k in ("track_yaw", "yaw_square", "yaw_precision")}
    NATIVE["apply_dense_tracking"](terms, velocity, omega, gravity, commands[:, 2], commands,
                                   support, ordinary, reference, DENSE)
    return base, manual, terms


SCUT_BASE = "source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py"
SCUT_DATA = frozen(SCUT, SCUT_SOURCE, SCUT_BASE)
SCUT_TREE = ast.parse(SCUT_DATA)
SCUT_FN = next(n for n in ast.walk(SCUT_TREE) if isinstance(n, ast.FunctionDef) and n.name == "_get_rewards")
SCUT_NAMES = {"ang_vel_err", "rew_track_ang_vel_z", "rew_track_ang_vel_z_square",
              "rew_stand_still", "rew_stand_still_lin_vel"}
SCUT_NODES = [n for n in SCUT_FN.body if (
    isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in SCUT_NAMES for t in n.targets)
) or (isinstance(n, ast.If) and any(isinstance(s, ast.Assign) and any(
    isinstance(t, ast.Name) and t.id == "track_yaw_rate_err" for t in s.targets) for s in n.body))]
assert len(SCUT_NODES) == 8  # Includes the two original gate multiplications.
SCUT_CODE = compile(ast.Module(SCUT_NODES, []), f"{SCUT_SOURCE}:{SCUT_BASE}", "exec")
SCUT_RUNS = {
    "flat_and_rotation": "pretrained/26_infantry/flat_and_rotation/2026-07-29_23-20-03/params/env.yaml",
    "rough_dash": "pretrained/26_infantry/rough_dash/2026-07-19_19-15-07/params/env.yaml",
}


def scut_terms(yaw: torch.Tensor, config):
    count = yaw.numel()
    omega = torch.zeros(count, 3, dtype=torch.float64)
    omega[:, 2] = yaw
    cfg = SimpleNamespace(**{k: float(config[k]) for k in (
        "ang_vel_err_constraint", "ang_vel_z_sigma", "ang_vel_z_square_sigma")})
    robot = SimpleNamespace(data=SimpleNamespace(root_ang_vel_b=omega, root_lin_vel_b=torch.zeros(count, 3)))
    enabled = config["stand_still_deadzone_enabled"] == "true"
    mask = torch.full((count,), enabled, dtype=torch.bool)
    namespace = {"torch": torch, "self": SimpleNamespace(cfg=cfg, robot=robot),
                 "tracking_command": torch.zeros(count, 3), "stand_still_lin_mask": mask,
                 "stand_still_yaw_mask": mask, "vel_reward_gate": torch.ones(count)}
    exec(SCUT_CODE, namespace)
    weights = config["rewards"]
    positive = float(weights["track_ang_vel_z"]) * namespace["rew_track_ang_vel_z"]
    square = float(weights["track_ang_vel_z_square"]) * namespace["rew_track_ang_vel_z_square"]
    # Normalized gate=1 comparison; not a rollout of the SCUT policy/plant.
    return positive, square


def main():
    report = read(PROBE / "report.json")
    effective = read(PROBE / "effective_contract.json")
    assert sha(frozen(REPO, SOURCE, "src/wheeled_tasks/chassis/env.py")) == report["source_sha256"]["src/wheeled_tasks/chassis/env.py"]
    assert sha((REPO / "scripts/probe_locomotion_quality.py").read_bytes()) == report["source_sha256"]["scripts/probe_locomotion_quality.py"]
    assert sha((REPO / "src/wheeled_tasks/chassis/rewards.py").read_bytes()) == REWARD_SHA
    assert sha(WORKER_PATH.read_bytes()) == sha(frozen(REPO, SOURCE, str(WORKER_PATH.relative_to(REPO))))
    assert effective["dense_tracking"]["yaw"] == DENSE["yaw"]
    assert DT == .02 and WORKER["physics_dt"] == .005
    assert not report["runtime_config"]["signal_delay"]["enabled"]
    seal_path = REPO / "models/v14_speed_update24000_diagnostic_20261005/model.checkpoint.json"
    seal = read(seal_path)
    meta = report["models"]["v14"]["metadata"]
    assert seal["checkpoint_sha256"] == meta["source_checkpoint_sha256"]
    assert seal["learning_updates"] == 24000
    assert meta["checkpoint_infos"]["task_actor_update"] == 3685
    assert meta["checkpoint_infos"]["actor_updates_consumed"] == 23880

    configs, scut_sources = {}, {}
    for name, filename in SCUT_RUNS.items():
        data = frozen(SCUT, SCUT_SOURCE, filename)
        configs[name] = yaml.load(data.decode(), Loader=yaml.BaseLoader)
        cfg = configs[name]
        scut_sources[name] = {"path": filename, "sha256": sha(data), "weights": {
            k: cfg["rewards"].get(k) for k in ("track_ang_vel_z", "track_ang_vel_z_square", "stand_still", "stand_still_lin_vel")},
            "deadzone_enabled": cfg["stand_still_deadzone_enabled"],
            "deadzone_threshold": cfg["stand_still_deadzone_threshold"],
            "gate_config": {k: cfg.get(k, "absent; base getter defaults false") for k in (
                "vel_upright_gate_enabled", "vel_orientation_x_gate_enabled", "vel_orientation_y_gate_enabled",
                "vel_height_gate_enabled", "vel_height_gate_mode", "vel_height_gate_full_error", "vel_height_gate_zero_error")}}

    errors = torch.tensor([0., .01, .02, .04, .088, -.088, .15, .2, .5], dtype=torch.float64)
    base, manual, terms = native_counterfactual(errors)
    active_cost = -(terms["dense_yaw_broad"] + terms["dense_yaw_precision"])
    assert float(active_cost[0]) == 0. and float(active_cost[4]) == float(active_cost[5])
    assert torch.all(active_cost[[0, 1, 2, 3, 4, 6, 7, 8]].diff() > 0.)
    assert "yaw_precision" not in terms and "yaw_square" not in terms
    assert torch.equal(terms["track_yaw"], torch.ones_like(errors))
    _, _, unsupported = native_counterfactual(errors, support=torch.zeros(len(errors)),
        gravity=torch.tensor([[.3, .1, -.948683298]]).repeat(len(errors), 1))
    assert torch.equal(terms["dense_yaw_broad"], unsupported["dense_yaw_broad"])
    assert torch.equal(terms["dense_yaw_precision"], unsupported["dense_yaw_precision"])
    tables = []
    for i, e in enumerate(errors):
        row = {"yaw_error_rad_s": float(e), "v14_broad_density": float(terms["dense_yaw_broad"][i]),
               "v14_precision_density": float(terms["dense_yaw_precision"][i]),
               "v14_loss_vs_zero_per_s": float(active_cost[i]),
               "v14_loss_vs_zero_per_50hz_reward": float(active_cost[i] * DT),
               "legacy_base_loss_vs_zero_per_s_inactive": float(1 - base["track_yaw"][i] - base["yaw_square"][i]),
               "legacy_manual_yaw_precision_cost_inactive": float(-manual["yaw_precision"][i])}
        for name, cfg in configs.items():
            positive, square = scut_terms(errors, cfg)
            row[f"scut_{name}_gate1_loss_vs_zero_per_s"] = float(positive[0] - positive[i] - square[i])
        tables.append(row)

    windows, inputs, exact_stands = [], {}, {}
    for model, cases in report["cases"].items():
        for case, info in cases.items():
            path = PROBE / Path(info["frames_file"]).name
            data = path.read_bytes()
            frames = [json.loads(line) for line in data.splitlines() if line]
            inputs[path.name] = {"sha256": sha(data), "frames": len(frames)}
            for phase in ("stand", "steady", "stop"):
                if phase != "stand" and model != "v14":
                    continue
                rows = [f for f in frames if f["phase"] == phase]
                if not rows:
                    continue
                yaw = torch.tensor([f["omega_rad_s"][2] for f in rows], dtype=torch.float64)
                commands = torch.tensor([f["command"] for f in rows], dtype=torch.float64)
                velocity = torch.tensor([f["velocity_m_s"] for f in rows], dtype=torch.float64)
                gravity = torch.tensor([f["projected_gravity_b"] for f in rows], dtype=torch.float64)
                _, _, actual = native_counterfactual(yaw, velocity, gravity, commands)
                zero_yaw = commands[:, 1]
                _, _, corrected = native_counterfactual(zero_yaw, velocity, gravity, commands)
                assert torch.equal(actual["stand_still_lin_vel"], corrected["stand_still_lin_vel"])
                assert torch.equal(actual["dense_stationary_broad"], corrected["dense_stationary_broad"])
                assert torch.equal(actual["dense_stationary_precision"], corrected["dense_stationary_precision"])
                loss = -(actual["dense_yaw_broad"] + actual["dense_yaw_precision"])
                full_expected = {"stand": 2600, "steady": 800, "stop": 2000}[phase]
                window = {"model": model, "case": case, "phase": phase, "samples_200hz": len(rows),
                    "duration_s": len(rows) * .005, "full_expected_window": len(rows) == full_expected,
                    "case_status": info["status"], "terminal_reasons": info["terminal_reasons"],
                    "command_yaw_rad_s": stats(commands[:, 1]), "yaw_truth_rad_s": stats(yaw),
                    "yaw_error_rad_s": stats(yaw - commands[:, 1]),
                    "v14_yaw_loss_density_mean": float(loss.mean()),
                    "v14_yaw_loss_per_policy_reward_mean": float(loss.mean() * DT)}
                if phase == "stand":
                    wheels = torch.tensor([f["wheel_speed_rad_s"] for f in rows], dtype=torch.float64)
                    actions = torch.tensor([f["actions_control_order"] for f in rows], dtype=torch.float64)
                    requested = torch.tensor([f["wheel_requested_effort_nm"] for f in rows], dtype=torch.float64)
                    applied = torch.tensor([f["wheel_applied_effort_nm"] for f in rows], dtype=torch.float64)
                    saturation = torch.tensor([f["wheel_saturated"] for f in rows], dtype=torch.float64)
                    window.update({"mean_velocity_link_body_m_s": velocity.mean(0).tolist(),
                        "mean_wheel_speed_control_axes_rad_s": wheels.mean(0).tolist(),
                        "mean_action_control_order": actions.mean(0).tolist(),
                        "mean_requested_wheel_targets_rad_s": (actions[:, [2, 5]] * 10.).mean(0).tolist(),
                        "mean_wheel_speed_axis_sum_rad_s": float(wheels.sum(-1).mean()),
                        "mean_requested_wheel_effort_nm": requested.mean(0).tolist(),
                        "mean_applied_wheel_effort_nm": applied.mean(0).tolist(),
                        "wheel_saturation_fraction": saturation.mean(0).tolist(),
                        "v14_L1_density_using_link_velocity_proxy": float(actual["stand_still_lin_vel"].mean()),
                        "v14_stationary_broad_density_using_link_velocity_proxy_support1": float(actual["dense_stationary_broad"].mean()),
                        "v14_stationary_precision_density_using_link_velocity_proxy_support1": float(actual["dense_stationary_precision"].mean()),
                        "manual_positive_yaw_gate_mean": float(torch.exp(-gravity[:, :2].square().sum(-1) /
                            WORKER["reference_reward"]["upright_denominator"]).mean()),
                        "stand_case_displacement_m": info["phases"][phase]["displacement_from_phase_anchor_m"],
                        "mean_yaw_rps": float(yaw.mean() / (2. * math.pi)),
                        "mean_yaw_deg_s": float(yaw.mean() * 180. / math.pi)})
                    exact_stands[model] = window
                if phase != "stand":
                    windows.append(window)

    assert all(not v for v in report["cases"]["v14"]["stand_305mm"]["actual_modules"]["command_transport_enabled"])
    assert set(exact_stands) == {"flat", "old_speed", "v14"}
    assert all(w["full_expected_window"] and w["command_yaw_rad_s"]["abs_mean"] == 0.
               for w in exact_stands.values())
    result = {
        "schema": "v14-zero-yaw-frozen-cpu-audit-v1", "status": "cpu_checks_passed_diagnostic_only",
        "qualification": None, "optimizer_steps": 0, "physics_steps_started": 0,
        "source": {"commit": SOURCE, "reward_sha256": REWARD_SHA, "function_lines": FUNCTION_LINES,
            "worker_path": str(WORKER_PATH.relative_to(REPO)), "worker_file_sha256": sha(WORKER_PATH.read_bytes()),
            "native_report_sha256": sha((PROBE / "report.json").read_bytes()),
            "probe_capture_source_sha256": report["source_sha256"]["scripts/probe_locomotion_quality.py"],
            "capture_source_not_in_training_b4ce809": True,
            "native_reward_module_not_hash_logged_by_probe": True,
            "frozen_env_hash_matches_probe": True, "current_reward_file_matches_frozen": True,
            "raw_frames": inputs},
        "model_clock": {"pt_sha256": seal["checkpoint_sha256"], "learning_updates": 24000,
            "learning_lineage_updates": meta["checkpoint_infos"]["learning_lineage_updates"],
            "global_actor_updates_consumed": 23880, "task_actor_update": 3685,
            "v14_origin_task_actor": 3563, "v14_new_actor_updates": 122,
            "current_live_snapshot": False},
        "measurement_scope": {"probe_started_utc": report["started_at_utc"], "probe_finished_utc": report["finished_at_utc"],
            "frame_dt_s": .005, "reward_policy_dt_s": DT, "num_envs": report["num_envs"],
            "gyro": "root_com_ang_vel_b actual truth, rad/s; not scaled observation or delayed feedback",
            "linear_velocity": "root_link_lin_vel_b capture; runtime reward uses root_com_lin_vel_b",
            "delay": report["delay_scope"], "signal_delay_enabled": False,
            "command_transport_instance_present": True, "actual_command_transport_enabled": [False] * 7,
            "reward_reconstruction": "exact yaw functions re-evaluated at raw physics states; not a logged policy reward stream",
            "linear_stationary_counterfactual": "link-velocity proxy with support=1; not exact runtime COM payment",
            "effective_common_probe_reward_extras": {"stand_still_lin_vel": "absent", "locomotion_posture": "absent"},
            "effective_common_probe_yaw_matches_v14_training": True,
            "complete_stand_frames_each": 2600, "complete_stand_duration_s_each": 13.0},
        "active_yaw": {"dense_settings": DENSE["yaw"], "gate": "ones; independent of support/upright/quiet command blend",
            "constant_track_yaw_peak_density": 1., "removed_after_manual": ["yaw_square", "yaw_precision"],
            "stationary_L1_direct_yaw_dependency": False, "stationary_dense_direct_yaw_dependency": False,
            "heading_angle_anchor": False, "gyro_policy_observation_scale": .5},
        "wheel_coordinate_check": wheel_axes(),
        "synthetic_counterfactuals": tables, "stand_windows": exact_stands,
        "v14_motion_phase_windows": windows,
        "scut": {"commit": SCUT_SOURCE, "base_source": SCUT_BASE, "base_sha256": sha(SCUT_DATA),
            "assignment_lines": [n.lineno for n in SCUT_NODES], "runs": scut_sources,
            "comparison_gate": "1; state-machine scale=1 ordinary-ground comparison, not SCUT rollout",
            "posture_v_terms": "velocity-conditioned projected-gravity square; not gyro angular-velocity penalty"},
        "checks": ["frozen env SHA matches native003", "current reward file equals frozen reward SHA",
            "sealed update24000/task3685/origin3563 clock consistency", "probe dense yaw matches worker",
            "zero cost at zero yaw", "sign symmetric cost", "increasing error increases cost",
            "manual yaw terms replaced exactly once", "yaw unchanged by support/lean gates",
            "yaw-only changes do not change stationary linear costs", "full 13s stand windows",
            "actual probe communication model disabled", "SCUT saved run weights loaded from frozen primary source"],
        "limitations": ["No new Isaac, GPU, training, hardware or remote work executed.",
            "Physics-state counterfactuals do not establish the sampled PPO advantage or learning guarantee.",
            "Reward term values were not emitted in native003; no total-runtime-reward/payment claim.",
            "Only seven motion/stand cases at one seed/height and a deterministic mean actor were compared.",
            "No angular-rate command cases exist in native003; no positive/reverse rotation qualification.",
            "Wheel action/effort asymmetry is observed, not proven caused by asymmetric friction.",
            "Boundary-truncated steady/stop windows are not full-duration stopping successes.",
            "SCUT saved YAML proves configured weights, not convergence time, forgetting immunity or identical mechanics."]}
    output = OUT / "evidence.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    stand = exact_stands["v14"]
    print(json.dumps({"status": result["status"], "output": str(output),
        "new_actor_updates_at_probe": 122, "stand_yaw_rad_s": stand["yaw_truth_rad_s"]["mean"],
        "stand_yaw_cost_per_s": stand["v14_yaw_loss_density_mean"],
        "stand_yaw_cost_per_policy_reward": stand["v14_yaw_loss_per_policy_reward_mean"],
        "physics_or_optimizer_work_started": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
