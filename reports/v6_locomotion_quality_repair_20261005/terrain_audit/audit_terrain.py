#!/usr/bin/env python3
"""Read only the frozen v14 inputs and exercise their CPU-only validators."""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
REVISION = "b4ce80964d47c0127e0b3d9929ad35a620bbf204"
LOADED = {}


def frozen(path):
    return subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=ROOT)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = frozen(path)
    if (ROOT / path).read_bytes() != raw:
        raise ValueError(f"Frozen input differs from current readable binding: {path}")
    return json.loads(raw)


class FrozenSource(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load project modules from git, including the dirty local preflight file."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "wheeled_tasks" or fullname.startswith("wheeled_tasks."):
            base = "src/" + fullname.replace(".", "/")
            choices = [(base + "/__init__.py", True), (base + ".py", False)]
        elif fullname in ("run_task_campaign", "run_chassis_blocks", "train_chassis", "chassis_remote_job"):
            choices = [("scripts/" + fullname + ".py", False)]
        else:
            return None
        for name, package in choices:
            if name in TRACKED:
                return importlib.util.spec_from_loader(fullname, self, origin=str(ROOT / name), is_package=package)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = str(Path(module.__spec__.origin).relative_to(ROOT))
        raw = frozen(name)
        module.__file__ = str(ROOT / name)
        if module.__spec__.submodule_search_locations is not None:
            module.__path__ = [str((ROOT / name).parent)]
        LOADED[name] = {"frozen_sha256": sha(raw), "worktree_sha256": sha((ROOT / name).read_bytes()),
                        "loaded_from": "git_show_frozen_source"}
        exec(compile(raw, f"git:{REVISION}/{name}", "exec"), module.__dict__)


def diff(before, after, prefix=""):
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(before.keys() | after.keys()):
            path = prefix + "/" + key
            if key not in before:
                rows.append({"path": path, "change": "added", "after": after[key]})
            elif key not in after:
                rows.append({"path": path, "change": "removed", "before": before[key]})
            else:
                rows.extend(diff(before[key], after[key], path))
        return rows
    return [] if before == after else [{"path": prefix, "change": "changed", "before": before, "after": after}]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def phase_probes(config):
    import torch
    from wheeled_tasks.chassis import rewards
    from wheeled_tasks.chassis.references import CommandReference
    from wheeled_tasks.chassis.nonstandard_start import birth_probabilities
    from wheeled_tasks.chassis.step_climb import StepClimbTask
    from wheeled_tasks.chassis.task import Phase, choose_scene_groups

    def ordinary_terms(vx, yaw, support, *, accel=0., elapsed=0., drift=.1):
        commands = torch.tensor([[vx, yaw, .305]], dtype=torch.float64)
        velocity = torch.tensor([[vx + drift, .03, 0.]], dtype=torch.float64)
        omega = torch.tensor([[.1, -.1, yaw]], dtype=torch.float64)
        angle = .15
        gravity = torch.tensor([[math.sin(angle), .05, -math.cos(angle)]], dtype=torch.float64)
        height = torch.tensor([.316], dtype=torch.float64)
        supported = torch.tensor([support], dtype=torch.float64)
        ordinary = torch.ones(1, dtype=torch.bool)
        ref = SimpleNamespace(pitch=torch.tensor([math.atan(accel / 9.81)], dtype=torch.float64),
            acceleration=torch.tensor([accel], dtype=torch.float64),
            elapsed=torch.tensor([elapsed], dtype=torch.float64),
            terrain_mode=torch.zeros(1, dtype=torch.long), jumping=torch.zeros(1, dtype=torch.bool))
        zero6 = torch.zeros(1, 6, dtype=torch.float64)
        terms = rewards.reward_terms(velocity, omega, gravity, height, commands, zero6, zero6,
            zero6, zero6, zero6, zero6, torch.zeros(1, 2, 3, dtype=torch.float64), supported,
            ordinary, torch.zeros(1, dtype=torch.bool))
        terms.pop("lin_vel_square")
        terms.update(rewards.reference_motion_terms(velocity, gravity, commands, supported,
            ordinary, ref, config["reference_reward"], locomotion_posture=config["dense_tracking"]["locomotion_posture"]))
        rewards.apply_manual_tracking(terms, velocity, omega, gravity, commands, supported,
            ordinary, ref, config["reference_reward"])
        rewards.apply_dense_tracking(terms, velocity, omega, gravity, height, commands, supported,
            ordinary, ref, config["dense_tracking"])
        return {k: float(v.item()) for k, v in terms.items()}

    rows = []
    for name, vx, yaw in [("stand", 0., 0.), ("forward", 3., 0.), ("ordinary_yaw", 0., 1.), ("release_zero", 0., 0.)]:
        # env.py:1393-1397 restores ordinary support in LANDING/RECOVERY.
        values = {p.name: ordinary_terms(vx, yaw, 1.) for p in (Phase.GROUND, Phase.LANDING, Phase.RECOVERY)}
        assert values["GROUND"] == values["LANDING"] == values["RECOVERY"]
        for accel, elapsed in [(-1.5, 0.), (0., 4.), (1.5, 12.)]:
            assert ordinary_terms(vx, yaw, 1., accel=accel, elapsed=elapsed) == values["GROUND"]
        rows.append({"case": name, "checked_phases": list(values), "terms_identical": True,
                     "tested_density_per_second": sum(values["GROUND"].values()),
                     "stand_still_lin_vel": values["GROUND"]["stand_still_lin_vel"],
                     "pitch_floor": values["GROUND"]["locomotion_pitch_floor"],
                     "roll_floor": values["GROUND"]["locomotion_roll_floor"]})
    unsupported = ordinary_terms(3., 0., 0.)
    assert unsupported["locomotion_pitch_floor"] < 0.
    window = rewards.PassiveLandingWindow(3, "cpu", .02, config["landing_tracking"])
    ordinary_landing_max = 0.
    for i in range(100):
        phase = torch.tensor([Phase.GROUND, Phase.LANDING if i % 2 else Phase.RECOVERY, Phase.RECOVERY])
        weight = window.update(phase, torch.zeros(3, dtype=torch.bool))
        assert not weight.any()
        terms = rewards.passive_landing_costs(torch.tensor([.009, .011, .02]), torch.ones(3) * .1,
                                             weight, config["landing_tracking"])
        ordinary_landing_max = max(ordinary_landing_max, *(float(x.abs().max()) for x in terms.values()))
    explicit = rewards.PassiveLandingWindow(1, "cpu", .02, config["landing_tracking"])
    weights = []
    for i in range(100):
        weights.append(float(explicit.update(torch.tensor([Phase.LANDING if i % 2 == 0 else Phase.RECOVERY]),
                                            torch.ones(1, dtype=torch.bool)).item()))
    assert weights[0] == 1. and max(weights[31:]) == 0.
    costs = rewards.passive_landing_costs(torch.tensor([.011]), torch.tensor([.1]), torch.tensor([1.]), config["landing_tracking"])
    assert all(float(v.item()) <= 0. for v in costs.values())

    # Ordinary rolling/body climbs cannot enter the manual reference exemption.
    climb_names = [g["name"] for g in config["scene_groups"] if config["skill_specs"][g["name"]]["kind"] in ("body_climb", "rolling_up", "rolling_stairs")]
    manual_mapping = {"step_up": 1., "stairs": 1., "step_down": -1., "stairs_down": -1.}
    assert all(manual_mapping.get(config["skill_specs"][name]["kind"], 0.) == 0. for name in climb_names)
    reference = CommandReference(1, "cpu", config["command_reference"])
    modes = []
    for elapsed in [-.01, 0., .5, 3.99, 4.01, 5.49, 5.5, 100.]:
        reference.update(torch.tensor([.305]), torch.zeros(1), torch.ones(1), torch.tensor([elapsed]),
                         torch.tensor([5.]), torch.zeros(1, dtype=torch.bool), torch.zeros(1), torch.tensor([.06]))
        modes.append({"elapsed_s": elapsed, "terrain_mode": float(reference.terrain_mode.item()), "phase": int(reference.phase.item())})
    assert modes[-1]["terrain_mode"] == modes[-2]["terrain_mode"] == 0.
    names = ["body_climb_20"]
    climb = StepClimbTask(names, config["skill_specs"], "cpu", .02)
    positions = torch.tensor([[[-.4, 0., .06], [-.5, 0., .06]]])
    before = float(climb.height_tracking_scale(positions).item())
    positions[:, :, 0] = -.05
    edge = float(climb.height_tracking_scale(positions).item())
    positions[:, :, 0] = .4
    after = float(climb.height_tracking_scale(positions).item())
    positions[:, :, 0] = -.05
    climb.index[:] = climb.length
    completed = float(climb.height_tracking_scale(positions).item())
    assert before == after == completed == 1. and edge == 0.
    test_terms = {"dense_height_broad": torch.tensor([-1.]), "track_height": torch.tensor([1.]),
                  "locomotion_pitch_floor": torch.tensor([-2.]), "locomotion_roll_floor": torch.tensor([-3.]),
                  "motor_torque": torch.tensor([-4.]), "dense_velocity_broad": torch.tensor([-5.])}
    rewards.apply_climb_height_tracking(test_terms, torch.tensor([0.]))
    assert test_terms["dense_height_broad"].item() == 0.
    assert test_terms["track_height"].item() == 1.
    assert all(test_terms[k].item() == v for k, v in [("locomotion_pitch_floor", -2.), ("locomotion_roll_floor", -3.), ("motor_torque", -4.), ("dense_velocity_broad", -5.)])
    climb.reset(torch.tensor([0]))
    positions[:, :, 0] = .4
    positions[:, :, 2] = .26
    completed_events = []
    for _ in range(80):
        completed_events.append(bool(climb.observe(positions, torch.ones(1, 2, dtype=torch.bool),
                                                 torch.ones(1, dtype=torch.bool)).item()))
    assert sum(completed_events) == 1 and bool(climb.success.item())
    climb.reset(torch.tensor([0]))
    for _ in range(4):
        climb.observe(positions, torch.zeros(1, 2, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    assert bool(climb.flight_violation.item()) and not bool(climb.success.item())
    scenes = choose_scene_groups(config["scene_groups"], 4096)
    counts = {g["name"]: sum(x[0] == g["name"] for x in scenes) for g in config["scene_groups"]}
    assert sum(counts.values()) == 4096 and min(counts.values()) > 0
    return {"status": "cpu_checks_passed", "ordinary_phase_cases": rows,
            "unsupported_pitch_floor_retained": True, "ordinary_passive_landing_max_abs_density": ordinary_landing_max,
            "explicit_passive_contact_flicker_does_not_renew_window": True,
            "manual_reference_is_finite": modes, "ordinary_climb_manual_reference_mode": 0,
            "ordinary_climb_skills": climb_names,
            "climb_height_scale": {"before_riser": before, "on_riser": edge, "both_wheels_past_riser": after, "completed_route": completed},
            "climb_gate_preserves_pitch_roll_effort_and_velocity_costs": True,
            "climb_repeated_top_residence_platform_events": sum(completed_events),
            "climb_unsupported_0p08_seconds_invalidates_success": True,
            "nonstandard_probabilities_at_actor_endpoint_2186": birth_probabilities(config["nonstandard_start"], 2186),
            "population_scene_group_counts": counts,
            "qualification": False, "native_simulation_exercised": False,
            "scope": "function-level CPU algebra and frozen env call-site review; not a native rollout or exhaustive incentive proof"}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run with CUDA_VISIBLE_DEVICES='' for this CPU audit")
    OUT.mkdir(parents=True, exist_ok=True)
    sys.meta_path.insert(0, FrozenSource())
    import torch
    from wheeled_tasks.chassis import locomotion_quality_revision as revision
    from wheeled_tasks.chassis import task_campaign as campaign
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.evaluation import fixed_suite_contract, select_evaluation_cases
    from wheeled_tasks.chassis.preflight import ROLES, design_preflight
    import run_task_campaign as runner

    plan = read(revision.PLAN_FILE)
    source_plan = read(revision.SOURCE_PLAN_FILE)
    speed = read(revision.worker_file("speed"))
    terrain = read(revision.worker_file("terrain"))
    old_terrain = read(revision.SOURCES["terrain"][0])
    migration = campaign.validate_plan_migration(source_plan, plan, plan["revision_migration"]["run_directory"])
    worker_migration = campaign.validate_worker_revision(old_terrain, terrain)
    for domain in ("speed", "terrain"):
        assert read(revision.design_file(domain)) == revision.revised_design(domain)
    receipts = runner.validate_contracts(plan)
    assert len(receipts) == 30
    materialized = []
    for domain, config in (("speed", speed), ("terrain", terrain)):
        row = next(x for x in plan["domains"] if x["name"] == domain)
        names = sorted(set(row["required_case_names"]) | set(row.get("monitor_case_names", [])))
        control = load_chassis_control(ROOT / config["control_math_source"])
        for role in ROLES:
            effective = deepcopy(config)
            seed, count = plan["seed"], plan["target_num_envs"]
            if role != "train":
                seed = plan["evaluation_seeds"][int(role == "confirmation")]
                effective["evaluation"]["seed"] = seed
                effective["evaluation"] = select_evaluation_cases(effective["evaluation"], names)
                effective = fixed_suite_contract(effective)
                count = len(names) * effective["evaluation"]["episodes_per_case"]
            result = design_preflight(effective, control, ROOT, stage=row["kind"], role=role,
                                      seed=seed, num_envs=count, block_index=0)
            materialized.append({"domain": domain, "role": role, "seed": seed, "num_envs": count,
                "preflight": result, "command_transport_enabled": effective["command_transport"]["enabled"],
                "legacy_usb_transport_enabled": effective["usb_transport"]["enabled"],
                "signal_delay_enabled": effective["signal_delay"]["enabled"],
                "actuator_response_enabled": effective["actuator_response"]["enabled"],
                "nonstandard_start_enabled": effective["nonstandard_start"]["enabled"],
                "release_stop_enabled": effective["release_stop"]["enabled"],
                "quality_rewards": {k: effective["dense_tracking"][k] for k in ("stand_still_lin_vel", "locomotion_posture")}})
    inherited = ["asset_directory", "asset_manifest_sha256", "control_math_source", "control_math_sha256",
        "physics_dt", "policy_dt", "position_iterations", "velocity_iterations", "history_length",
        "actor_frame_dim", "actor_dim", "critic_dim", "action_dim", "actor_layout", "critic_layout",
        "policy_action_order", "policy_architecture", "actor_observation_source", "actor_reserved_frame30",
        "v5_control", "motion_limits", "command_slew", "usb_transport", "command_transport", "signal_delay",
        "actuator_response", "actuator_response_evidence", "control_model_status", "physics_prior_scope",
        "dense_tracking", "precision_tracking", "reference_reward", "landing_tracking", "termination_event_cost",
        "success_event_bonus", "closure_gap_termination_m", "fall_confirmation_seconds", "contact_domain",
        "hardware_deployment_ready", "working_margin_weight", "knee_working_margin_rad",
        "track_height_during_recovery", "height_reference", "signal_perturbations", "dynamics_randomization",
        "monitor_applied_effort", "zero_critic_wheel_positions"]
    equality = [{"field": key, "equal_to_speed_v14": terrain[key] == speed[key],
                 "terrain_value": terrain[key]} for key in inherited]
    assert all(row["equal_to_speed_v14"] for row in equality)
    assert terrain["evaluation"] == old_terrain["evaluation"]
    assert all(terrain["skill_specs"][k] == v for k, v in old_terrain["skill_specs"].items())
    assert len(terrain["scene_groups"]) == len(old_terrain["scene_groups"]) + 6
    row = next(x for x in plan["domains"] if x["name"] == "terrain")
    endpoint = plan["domain_completion"]["schedule"]["end_actor_updates"]["terrain"]
    origin = terrain["task_campaign"]["revision_migration"]["origin_actor_update"]
    assert (origin, endpoint, row["parent_domain"]) == (1186, 2186, "flat")
    audit = {"frozen_source_revision": REVISION, "scope": "read_only_static_and_cpu",
        "campaign_sha256": sha(frozen(revision.PLAN_FILE)), "core_equal_to_speed_v14": equality,
        "terrain_v6_to_v14_diff": diff(old_terrain, terrain),
        "terrain_vs_speed_difference_paths": [x["path"] for x in diff(speed, terrain)],
        "terrain_sampling_pools": row["sampling_pools"], "terrain_behavior_pool_fractions": terrain["behavior_pool_fractions"],
        "old_skill_specs_retained_exactly": True, "old_evaluation_retained_exactly": True,
        "old_scene_group_count": len(old_terrain["scene_groups"]), "new_scene_group_count": len(terrain["scene_groups"]),
        "automatic_endpoint": {"clock": "domain_actor_updates", "origin": origin, "endpoint": endpoint,
            "actor_updates_added": endpoint-origin, "critic_warmup_updates_charged_separately": terrain["resume_critic_warmup_updates"],
            "parent_domain": row["parent_domain"], "own_completed_terrain_learning_state_resumed": True,
            "advance_requires_evaluation": plan["domain_completion"]["schedule"]["advance_requires_evaluation"],
            "outcome": "unassessed_research_candidate", "qualification": False},
        "deliberate_speed_only_difference": {"terrain_locomotion_difficulty": terrain.get("locomotion_difficulty"),
            "speed_locomotion_difficulty": speed.get("locomotion_difficulty"),
            "terrain_release_translate_range": terrain["skill_specs"]["release_stop_translate"]["release_stop_magnitude_ranges"],
            "speed_release_translate_range": speed["skill_specs"]["release_stop_translate"]["release_stop_magnitude_ranges"]}}
    save("static_diff.json", audit)
    save("cpu_preflight.json", {"status": "passed", "frozen_source_revision": REVISION,
        "campaign_role_count": len(receipts), "plan_migration": migration,
        "terrain_worker_migration": {"status": "passed", "validator_return": worker_migration},
        "independent_design_reconstruction": "exact_match_both_workers", "materialized_roles": materialized,
        "cuda_initialized": torch.cuda.is_initialized(), "isaac_imported": any(k.startswith(("isaacsim", "isaaclab", "omni.")) for k in sys.modules),
        "qualification": False, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    save("phase_cpu_checks.json", phase_probes(terrain))
    reviewed = {}
    for name in ("src/wheeled_tasks/chassis/env.py", "src/wheeled_tasks/chassis/scut_observation.py",
                 "src/wheeled_tasks/chassis/signal_delay.py", "src/wheeled_tasks/chassis/command_transport.py",
                 "src/wheeled_tasks/chassis/skill_commands.py"):
        raw = frozen(name)
        reviewed[name] = {"frozen_sha256": sha(raw), "worktree_sha256": sha((ROOT / name).read_bytes())}
        assert raw == (ROOT / name).read_bytes(), name
    save("source_receipt.json", {"frozen_source_revision": REVISION, "loaded_modules": LOADED,
        "reviewed_not_imported_source": reviewed,
        "all_project_imports_from_frozen_git": True, "dirty_local_preflight_not_loaded": True,
        "cuda_initialized": torch.cuda.is_initialized(), "ssh_calls": 0, "simulation_calls": 0,
        "training_contracts_modified": False})
    print(json.dumps({"status": "passed", "roles": 30, "terrain_actor_origin": origin,
                      "terrain_actor_endpoint": endpoint, "cuda_initialized": torch.cuda.is_initialized(), "qualification": False}))


TRACKED = set(subprocess.check_output(["git", "ls-tree", "-r", "--name-only", REVISION], cwd=ROOT).decode().splitlines())
if __name__ == "__main__":
    main()
