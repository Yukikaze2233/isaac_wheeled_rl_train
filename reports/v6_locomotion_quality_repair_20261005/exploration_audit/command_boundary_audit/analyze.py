#!/usr/bin/env python3
"""CPU-only static command design audit; never start physics or read a remote event."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import torch
import yaml

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def yaml_fragment(path, first, last):
    # Only pinned data fragments; removing tuple annotations does not execute YAML objects.
    lines = path.read_text().splitlines()
    return yaml.safe_load(textwrap.dedent("\n".join(lines[first-1:last])).replace("!!python/tuple", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE.parents[3])
    parser.add_argument("--scut-root", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "audit.json")
    args = parser.parse_args(); root = args.root.resolve(); scut = args.scut_root or root.parent / "scut-wheeled-legged-rl"
    manifest = read(HERE / "inputs.json")
    require(not args.output.exists(), "Refusing to overwrite audit output")
    for group, base in (("local_files_sha256", root), ("scut_files_sha256", scut)):
        for name, expected in manifest[group].items():
            require(sha((base / name).read_bytes()) == expected, "Pinned input changed: " + name)
    for name in manifest["frozen_core_files"]:
        blob = subprocess.check_output(["git", "show", manifest["core_revision"] + ":" + name], cwd=root)
        require(sha(blob) == manifest["local_files_sha256"][name], "Actual local core differs from b4 blob: " + name)
    for name, expected in manifest["scut_files_sha256"].items():
        blob = subprocess.check_output(["git", "show", manifest["scut_source_revision"] + ":" + name], cwd=scut)
        require(sha(blob) == expected, "Actual selected SCUT file differs from b8 blob: " + name)
    sys.path.insert(0, str(root / "src"))
    motion = importlib.import_module("wheeled_tasks.chassis.motion_limits")
    difficulty = importlib.import_module("wheeled_tasks.chassis.locomotion_difficulty")
    skills = importlib.import_module("wheeled_tasks.chassis.skill_commands")
    contract = read(root / manifest["contract_file"])
    require(contract["motion_limits"] == dict(wheel_radius_m=.06, track_width_m=.4373,
        max_wheel_speed_rad_s=90., lateral_acceleration_m_s2=2.06), "Audited command envelope changed")
    require(contract["v5_control"]["leg_kp"] == 160. and contract["v5_control"]["leg_kd"] == 2.5
        and contract["v5_control"]["wheel_action_clip"] * contract["v5_control"]["wheel_velocity_scale"] == 90., "Audited target/PD premise changed")
    limits = contract["motion_limits"]; r, track, cap = limits["wheel_radius_m"], limits["track_width_m"], limits["max_wheel_speed_rad_s"]
    original = torch.tensor([[5., 0., .305], [0., 3*math.tau, .305], [5., 3*math.tau, .305], [.5, 3*math.tau, .305]], dtype=torch.float64, device="cpu")
    projected = motion.project_commands(original, limits)
    geometry = []
    for label, raw, after in zip(("pure_5m_s", "pure_3rev_s", "combined_extremes", "instantaneous_spin_translate_extreme"), original, projected):
        wheels = motion.wheel_speeds(float(raw[0]), float(raw[1]), r, track)
        motion.validate_command(after.tolist(), limits)
        geometry.append(dict(case=label, original=raw.tolist(), projected=after.tolist(),
            original_nominal_wheel_speed_rad_s=list(wheels), original_wheel_speed_within_target_cap=max(map(abs, wheels)) <= cap,
            original_lateral_acceleration_m_s2=abs(float(raw[0] * raw[1]))))
    # Exercise the existing spin command update only. This synthetic CPU command state is NOT a physical sample.
    env = SimpleNamespace(episode_length_buf=torch.zeros(1, dtype=torch.long), policy_dt=.02,
        commands=torch.zeros((1, 3), dtype=torch.float64), command_target=torch.zeros((1, 2), dtype=torch.float64))
    probe = SimpleNamespace(env=env, batches=[(torch.tensor([0]), contract["skill_specs"]["spin_translate_3rps"])],
        reference_target=torch.tensor([[.5, 0.]], dtype=torch.float64),
        command_base=torch.tensor([[0., 3*math.tau, .305]], dtype=torch.float64),
        yaw=lambda: torch.tensor([0.], dtype=torch.float64), apply_command_sequence=lambda: None)
    skills.SkillCommands.update(probe)
    spin = env.command_target[0].tolist()
    require(spin == [.5, 3*math.tau], "Native spin update premise changed")
    scut_configs = {}
    for role, entry in manifest["scut_config_fragments"].items():
        path = scut / entry["file"]
        scut_configs[role] = {key: yaml_fragment(path, *bounds) for key, bounds in entry["line_ranges"].items()}
        require(scut_configs[role]["ordinary"]["commands"]["ranges"]["lin_vel_x"] == [-2.7, 2.7], "SCUT ordinary domain changed")
    snapshot = read(root / manifest["snapshot_file"])
    require(snapshot["source_commit"] == manifest["core_revision"], "Training snapshot source mismatch")
    ended_groups = snapshot["baseline_interval"]["groups"]
    failures = {name: {key: row[key] for key in ("frames", "episodes", "failures", "timeouts", "boundary_truncations", "height_command_mean_m", "height_mae_m")}
                for name, row in ended_groups.items() if name.startswith(("release_stop", "stand_"))}
    for row in failures.values():
        require(row["episodes"] > 0 and row["failures"] + row["timeouts"] == row["episodes"], "Incomplete failure denominator")
        row["failure_fraction"] = row["failures"] / row["episodes"]
        row["boundary_subset_of_timeouts"] = True
    native = read(root / manifest["engineering_population_file"])
    population = native["startup"]["scene_group_counts"]
    require(sum(population.values()) == 4096 and set(population) == {x["name"] for x in contract["scene_groups"]}, "Population reference differs from current contract")
    height_population = sum(n for name, n in population.items() if "__height_train" in name)
    kin = dict(target_rolling_speed_m_s=r*cap, wheel_speed_at_5m_s_rad_s=5/r,
        wheel_target_headroom_at_5m_s_rad_s=cap-5/r, nominal_PD_positive_request_at_5m_s_Nm=.6*(cap-5/r),
        pure_yaw_target_ceiling_rad_s=2*r*cap/track, pure_yaw_target_ceiling_rev_s=2*r*cap/track/math.tau,
        minimum_ramp_to_5m_s_seconds=5/1.5, minimum_ramp_to_3rev_s_seconds=3*math.tau/4,
        pure_horizontal_pitch_threshold_for_projected_5m_s_at_target_ceiling_deg=math.degrees(math.acos(math.sqrt(5/(r*cap)))))
    result = dict(schema="frozen-command-domain-design-audit-v1", status="pinned_CPU_analysis_passed", qualification=False,
        root_cause_verified=False, candidate_implemented_or_deployed=False, source_revision=manifest["core_revision"],
        input_manifest_sha256=sha((HERE/"inputs.json").read_bytes()), analysis_script_sha256=sha(Path(__file__).read_bytes()),
        geometry=geometry, nominal_kinematic_calculations=kin,
        difficulty_envelopes={str(clock): difficulty.difficulty_envelope(contract["locomotion_difficulty"], clock) for clock in (3563, 3683, 3803, 3923, 4185, snapshot["progress"]["task_actor_update"])},
        native_spin_CPU_counterexample=dict(command_target=spin, lateral_product_m_s2=abs(spin[0]*spin[1]),
            nominal_wheel_speed_rad_s=list(motion.wheel_speeds(*spin, r, track)),
            maximum_body_coordinate_vx_rate_for_constant_yaw_m_s2=.5*3*math.tau,
            scope="Only native SkillCommands.update exercised on synthetic CPU command state; no physical observation"),
        height_design=dict(sampling=contract["skill_specs"]["forward_5__height_train"]["height_sampling"],
            transition_seconds=4., analytic_max_height_reference_rate_at_430mm_m_s=.125*1.5/4,
            configured_endpoint_probability=.2, configured_outer_10percent_band_probability=.2+.8*(.4/1.2),
            height_train_population_reference=height_population, height_train_fraction_reference=height_population/4096,
            startup_population_source="Existing engineering 4096 population, not a new current-training state read",
            behavior_pool_counts=native["startup"]["behavior_pool_counts"]),
        scut_saved_configs=scut_configs,
        observed_failure_scope=dict(snapshot_file=manifest["snapshot_file"], snapshot_queried_at_utc=snapshot["queried_at_utc"],
            native_event=snapshot["event"], native_source_invocation=snapshot["source_invocation"],
            event_steps=snapshot["baseline_interval"]["event_steps_inclusive"], task_actor_updates=snapshot["baseline_interval"]["task_actor_updates_inclusive"],
            interpretation="Existing 90 actor-update native-event difference; exploration/reset/DR/delay included; no causal matched comparison",
            groups=failures, complete_counter_groups=snapshot["baseline_interval"]["complete_counter_groups"],
            complete_all_height_train_failure_distribution=None, missing_distribution="Saved compact snapshots retain only three release-stop height pairs; no per-height bins or all high-speed height-train groups"),
        limitations=["Target envelope is nominal no-slip kinematics, not a hard state-speed bound or dynamic qualification",
            "5m/s PD headroom is nominal before motor strength/degradation/transport/response; training can reduce available effort",
            "Projection checks run at sampling; native spin reference rotates later and bypasses longitudinal slew, without repeating lateral projection",
            "SCUT saved July configs prove configured domains, while current b8 interpreter source does not certify exact historical SDK/runtime",
            "Static height limits and competence-independent 360-actor grade expansion cannot guarantee high-speed balance or retention",
            "No qualification or automatic no-forgetting guarantee follows from mixed population or absence of blocking gates"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output.resolve()), sha256=sha(args.output.read_bytes()), qualification=False)))


if __name__ == "__main__":
    main()
