"""Read-only CPU oracle for native recovery reference/outcome clock ordering.

Uses actual RecoveryFSM and RecoveryTraining.outcome from --source. Constant
valid sensor and true-physics inputs isolate timing from dynamics, geometry,
policy, delayed feedback and contact estimation. Does not import native env.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(torch, RecoveryFSM, RecoveryPhase, RecoveryTraining, settings, dt, start,
             actual_first_sample_bad=False):
    cfg = {k: settings[k] for k in RecoveryFSM.DEFAULTS if k in settings}
    cfg["dt"] = .005
    q, dq = torch.zeros(1, 6), torch.zeros(1, 6)
    fsm = RecoveryFSM(1, "cpu", cfg, *[torch.zeros(6) for _ in range(4)], [1., 1., -1., -1.])
    fsm.reset([0], q, torch.ones(1, dtype=torch.bool))
    if start == "initial_rl":
        fsm.phase[:] = RecoveryPhase.RL
    elif start.startswith("blend_remaining_"):
        remaining = int(start.rsplit("_", 1)[1])
        fsm.phase[:] = RecoveryPhase.BLEND
        fsm.phase_ticks[:] = fsm._ticks(cfg["blend_seconds"]) - remaining

    recovery = RecoveryTraining.__new__(RecoveryTraining)
    recovery.env = SimpleNamespace(policy_dt=.02, dt=dt)
    recovery.settings = settings
    recovery.fsm = fsm
    recovery.enabled = torch.ones(1, dtype=torch.bool)
    recovery.release_finished = torch.ones(1, dtype=torch.bool)
    recovery.required_platform_height = torch.zeros(1)
    recovery.followup_command = torch.tensor([[0., 0., .305]])
    for name in ("elapsed", "true_stable_seconds", "true_stable_peak", "followup_seconds"):
        setattr(recovery, name, torch.zeros(1))
    for name in ("premature_release", "motion_release_observed", "true_success",
                 "learning_mask", "policy_motion_hold"):
        setattr(recovery, name, torch.zeros(1, dtype=torch.bool))
    gravity, gyro = torch.tensor([[0., 0., -1.]]), torch.zeros(1, 3)
    height, support = torch.tensor([.305]), torch.ones(1, dtype=torch.bool)
    contact, undesired, velocity = torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), torch.zeros(1, 3)
    stride, decimation = round(.005 / dt), round(.02 / dt)
    first_fsm_rl = first_fsm_hold = first_sensor_release = None
    first_actual_eligible = first_owned_policy = first_success = None
    release_sample = None
    trace = []
    with torch.no_grad():
        for policy in range(1, 151):
            recovery.begin_policy()
            for substep in range(decimation):
                sample_t = (policy - 1) * .02 + substep * dt
                if substep % stride == 0:
                    fsm.update(q, dq, gravity, gyro, height, support, support)
                    if bool(fsm.pure_rl[0]) and first_fsm_rl is None:
                        first_fsm_rl = round(sample_t, 6)
                    if bool(fsm.holding_rl[0]) and first_fsm_hold is None:
                        first_fsm_hold = round(sample_t, 6)
                    if bool(fsm.motion_released[0]) and first_sensor_release is None:
                        first_sensor_release = round(sample_t, 6)
                # Same ordering and interval ownership predicate as torque
                # (recovery_training.py): reference precedes torque and physics.
                recovery.elapsed += dt
                recovery.learning_mask &= recovery.release_finished & (fsm.pure_rl | fsm.failed)
            actual_t = round(policy * .02, 6)
            if bool(recovery.pure_rl[0]) and first_actual_eligible is None:
                first_actual_eligible = actual_t
            if bool(recovery.learning_mask[0]) and first_owned_policy is None:
                first_owned_policy = actual_t
            first_release = bool(fsm.motion_released[0]) and not bool(recovery.motion_release_observed[0])
            peak_before, hold_before = float(recovery.true_stable_peak[0]), float(recovery.true_stable_seconds[0])
            current_contact = contact.clone()
            if actual_first_sample_bad and policy == 1:
                current_contact[:] = False
            recovery.outcome(current_contact, undesired, gravity, gyro, height, velocity)
            if first_release:
                release_sample = dict(policy_index=policy, actual_time_s=actual_t,
                    sensor_release_time_s=first_sensor_release,
                    sensor_stable_ticks=int(fsm.stable_peak_ticks[0]),
                    actual_hold_s_before=hold_before, actual_peak_s_before=peak_before,
                    actual_hold_s_after=float(recovery.true_stable_seconds[0]),
                    actual_peak_s_after=float(recovery.true_stable_peak[0]),
                    qualified_before=peak_before >= 1. - 1e-5,
                    premature_release=bool(recovery.premature_release[0]),
                    followup_s=float(recovery.followup_seconds[0]))
            if bool(recovery.true_success[0]) and first_success is None:
                first_success = actual_t
            if policy <= 2 or first_release or first_success == actual_t:
                trace.append(dict(policy_index=policy, time_s=actual_t,
                    fsm_phase=int(fsm.phase[0]), sensor_ticks=int(fsm.stable_ticks[0]),
                    sensor_peak_ticks=int(fsm.stable_peak_ticks[0]),
                    actual_hold_s=float(recovery.true_stable_seconds[0]),
                    actual_peak_s=float(recovery.true_stable_peak[0]),
                    followup_s=float(recovery.followup_seconds[0]),
                    owned_policy=bool(recovery.learning_mask[0]),
                    premature=bool(recovery.premature_release[0]),
                    success=bool(recovery.true_success[0])))
            if first_success is not None or bool(recovery.premature_release[0]):
                break
    result = dict(start=start, physics_dt=dt, reference_dt=.005, policy_dt=.02,
        actual_first_sample_bad=actual_first_sample_bad,
        first_fsm_rl_s=first_fsm_rl, first_fsm_hold_s=first_fsm_hold,
        first_actual_eligible_s=first_actual_eligible, first_wholly_actor_owned_policy_s=first_owned_policy,
        release_sample=release_sample, first_true_success_s=first_success, trace=trace)
    if actual_first_sample_bad:
        assert release_sample["premature_release"] and first_success is None, result
        assert .97 < release_sample["actual_peak_s_after"] < .99, result
    else:
        assert release_sample is not None and not release_sample["premature_release"], result
        assert release_sample["sensor_stable_ticks"] == 200, result
        assert release_sample["actual_peak_s_after"] >= 1. - 1e-5, result
        assert first_success is not None, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.exists():
        raise ValueError("Refusing to overwrite clock receipt")
    sys.path.insert(0, str(source / "src"))
    import torch
    from wheeled_tasks.chassis.recovery import RecoveryFSM, RecoveryPhase
    from wheeled_tasks.chassis.recovery_training import RecoveryTraining
    assert Path(sys.modules[RecoveryFSM.__module__].__file__).resolve().is_relative_to(source)
    assert Path(sys.modules[RecoveryTraining.__module__].__file__).resolve().is_relative_to(source)
    torch.set_num_threads(1)
    files = [source / "src/wheeled_tasks/chassis" / f for f in ("recovery.py", "recovery_training.py", "env.py")]
    files.append(source / "contracts/v6_task_campaign_recovery_worker_v18.json")
    before = {str(p.relative_to(source)): digest(p) for p in files}
    settings = json.loads(files[-1].read_text())["recovery_training"]
    cases = [run_case(torch, RecoveryFSM, RecoveryPhase, RecoveryTraining, settings, dt, start)
        for dt in (.001, .005)
        for start in ["initial_rl", *[f"blend_remaining_{n}" for n in range(1, 5)], "full_select_prepare_blend"]]
    cases += [run_case(torch, RecoveryFSM, RecoveryPhase, RecoveryTraining, settings, dt, "initial_rl", True)
        for dt in (.001, .005)]
    assert all(digest(source / f) == h for f, h in before.items())
    result = dict(schema="source019-read-only-recovery-dual-clock-CPU-oracle-v1", status="passed",
        source_root=str(source), source_sha256=before, source_files_unchanged=True,
        reader_sha256=digest(Path(__file__)), cpu_threads=1, cases=len(cases),
        stable_cases=12, negative_actual_contact_controls=2, results=cases,
        conclusion="No systematic premature release from two clock cadences under constant truly stable input at any of four reference handover phases. First-release actual sample is credited to hold before premature check. Deliberately removing first actual stable sample in initially-RL fixture leaves .98s and correctly rejects release.",
        limitations=["Pure CPU constant-input clock oracle, not simulation/geometry/contact observer/policy evaluation.",
            "Uses actual FSM and outcome implementations; script targets are zero vectors to isolate cadence.",
            "Reference runs before torque/physics; per-policy outcome follows four reference samples. Interval ownership predicate mirrors native code, but no physical torque computation is certified.",
            "No evidence here identifies which gate fails in actual training; current native receipts omit gate tags."])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], cases=len(cases), output=str(args.output.resolve()), sha256=digest(args.output))))


if __name__ == "__main__":
    main()
