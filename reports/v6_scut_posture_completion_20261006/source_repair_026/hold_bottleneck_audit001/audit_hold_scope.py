"""Read-only source023 hold audit; no simulator, training, or remote mutation."""
from __future__ import annotations

import ast
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch


REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FROZEN = "6f07ec298e01d2686216787cfba34b394151f1aa"
BASE = REPO / "reports/v6_scut_posture_completion_20261006/source_repair_026"
WINDOW = BASE / "learning_status002/latest100_actor5001_5100.arithmetic001.json"
SUMMARY = BASE / "learning_status002/latest100_actor5001_5100.summary001.json"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def functions(raw: bytes):
    return {node.name: node for node in ast.walk(ast.parse(raw))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    identities = {}
    frozen_files = {}
    for name in ("recovery.py", "rewards.py", "recovery_training.py", "env.py"):
        rel = "src/wheeled_tasks/chassis/" + name
        raw = subprocess.check_output(["git", "show", FROZEN + ":" + rel], cwd=REPO)
        local = (REPO / rel).read_bytes()
        frozen_files[name] = raw
        identities[rel] = {"frozen_sha256": sha(raw), "local_sha256": sha(local),
                           "byte_equal": raw == local}
    # Production hold code is unchanged, although v24 adds other integration hooks.
    for name in ("recovery.py", "rewards.py"):
        assert identities["src/wheeled_tasks/chassis/" + name]["byte_equal"]
    selected_ast = {}
    for name, selected in (("recovery_training.py", ("begin_policy", "torque", "outcome")),
                           ("env.py", ("_recovery_metrics",))):
        old = functions(frozen_files[name])
        new = functions((REPO / "src/wheeled_tasks/chassis" / name).read_bytes())
        selected_ast[name] = {
            fn: ast.dump(old[fn], include_attributes=False) ==
                ast.dump(new[fn], include_attributes=False) for fn in selected}
        if name == "recovery_training.py":
            assert all(selected_ast[name].values()), selected_ast[name]
        else:
            # v24 adds a component clock elsewhere in this function. Authenticate
            # only the unchanged hold gate/cost logging branches for this audit.
            for attribute in ("recovery_takeover_settings", "recovery_hold_cost_diagnostics"):
                def branch(nodes):
                    return next(node for node in ast.walk(nodes["_recovery_metrics"])
                                if isinstance(node, ast.If) and attribute in ast.dump(node.test))
                key = attribute + "_branch"
                selected_ast[name][key] = (ast.dump(branch(old), include_attributes=False)
                                          == ast.dump(branch(new), include_attributes=False))
                assert selected_ast[name][key], key

    # Execute only the actual frozen native CPU density formula, with autograd.
    native_node = functions(frozen_files["rewards.py"])["dense_error_terms"]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[native_node], type_ignores=[]),
                 "frozen-source023:dense_error_terms", "exec"), namespace)
    settings = {"broad_scale": 1., "broad_weight": .25,
                "fine_scale": .4, "fine_weight": .1}
    rows = []
    for speed in (0., .05, .1, .2, .25, .3, .5):
        e = torch.tensor(speed / .25, dtype=torch.float64, requires_grad=True)
        broad, fine = namespace["dense_error_terms"](e, settings)
        total = broad + fine
        derivative, = torch.autograd.grad(total, e)
        analytic = (-.25 * float(e.detach()) / math.sqrt(1. + float(e.detach()) ** 2)
                    - 2. * .1 * float(e.detach()) / .4 ** 2
                    * math.exp(-(float(e.detach()) / .4) ** 2))
        assert math.isclose(float(derivative), analytic, rel_tol=1e-12, abs_tol=1e-12)
        assert (float(derivative) == 0.) if speed == 0. else (float(derivative) < 0.)
        rows.append({"wheel_rim_speed_m_s": speed, "gate_relative_error": float(e.detach()),
                     "broad_reward_density_per_s": float(broad.detach()),
                     "fine_reward_density_per_s": float(fine.detach()),
                     "total_reward_density_per_s": float(total.detach()),
                     "derivative_wrt_dimensionless_error": float(derivative),
                     "step_payment_at_20ms": .02 * float(total.detach())})
    native_probe = {"scope": "Actual frozen dense_error_terms CPU float64/autograd only; no Env, PPO, rollout, or performance qualification",
                    "source_function_ast_sha256": sha(ast.dump(native_node, include_attributes=False).encode()),
                    "formula": "-.25*(sqrt(1+e^2)-1)+.1*(exp(-(e/.4)^2)-1)",
                    "error": "e=max(abs(wheel dq))*.06/.25", "rows": rows,
                    "all_nonzero_error_derivatives_negative": True}
    window = json.loads(WINDOW.read_text())
    gates = window["hold_gate_means"]
    costs = window["hold_cost_means"]
    report = {
        "schema": "source023-native-hold-scope-audit-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_commit": FROZEN, "sources": identities, "selected_function_ast_equal": selected_ast,
        "window": {"actor_in_attempt": [5001, 5100], "REC_actor": [11426, 11525],
                   "event_steps": [38630, 38729], "arithmetic_file": str(WINDOW.relative_to(REPO)),
                   "arithmetic_sha256": sha(WINDOW.read_bytes()),
                   "summary_file": str(SUMMARY.relative_to(REPO)), "summary_sha256": sha(SUMMARY.read_bytes()),
                   "scope": "Existing complete100 stochastic training/DR window; no new event parse or outcome inference"},
        "hold_gate_means": gates, "hold_cost_means": costs,
        "scope_definitions": {
            "gate_fractions": "Per strategy frame: sum(holding_rl & gate_pass)/max(sum(holding_rl),1); reported window is mean of rollout scalar means. Holding population changes with phase.",
            "support_and_body_clear": "Encoder/IMU observer estimates, not true wheel contacts or collision-free evidence",
            "stable_peak_and_break": "Per-frame mean over all4096 lanes, followed by rollout/window means; lifetime-since-reset states, not window increments or means over ended episodes",
            "hold_cost_mask": "recovery_enabled & whole_interval_actor_owned & begin_policy_motion_hold; includes first-release interval and actor-caused terminal, does not require1s stable",
            "hold_cost_state": "Physical-endstate dq/omega; differs from observer FSM feedback and end-frame holding mask",
            "logger_identity_limit": "24-frame rollout averaging is supported by saved RSL logger snapshot; current external runtime logger byte identity was not independently pinned by this window raw",
        },
        "FSM_release": {
            "clock": "5ms feedback ticks; consecutive200 ticks, reset if any predicate fails",
            "gates": {"valid": "Finite sensor values and valid gravity vector", "support": "Observer estimated support",
                      "body_clear": "Observer estimated body clear", "tilt": "<8deg",
                      "gyro": "Full norm<.75rad/s", "estimated_height": ".27<h<.36m",
                      "wheel_speed": "max(abs(dq_wheel))*.06<.25m/s", "leg_speed": "max(abs(dq_leg))<2rad/s"},
            "scope": "RL owns after blend;1s holds followup motion commands, not initial actor handoff"},
        "physical_true_success": {
            "before_release": "Pure RL + true dual contacts + no undesired contact + target platform + tilt<12deg + horizontal gyro norm<.75 + actual height error<=.05m + planar speed<.25 for1s",
            "after_release": "Same physical stability plus abs(vx-goal)<=.12 and abs(yaw_rate-goal)<=.2 for1s; standing also planar speed<.10; prior1s truth evidence and no failure required",
            "premature_release": "First observed FSM release without prior1s physical stability is a distinct failure",
            "scope": "Separate physical truth from observer release; existing marginal scalars cannot locate mismatches"},
        "marginal_nonpass_fraction": {
            "support": 1. - gates["/recovery/hold_gate/support_pass_fraction"],
            "body_clear": 1. - gates["/recovery/hold_gate/body_clear_pass_fraction"],
            "wheel_speed": 1. - gates["/recovery/hold_gate/wheel_speed_pass_fraction"]},
        "native_density_probe": native_probe,
        "findings": [
            "Hold wheel/leg/gyro densities are logged with nonzero negative means; existing code pays costs before the1s release gate. Missing payment is not supported.",
            "Support/body-clear and wheel-speed have the largest instantaneous marginal nonpasses in holding_rl; they are priorities for joint trace, not causal attribution.",
            ".274s and31.84 are all-lane/window averages and cannot be described as per-ended-episode peaks/breaks or pureL2 outcomes.",
            "Fine wheel precision is near saturation at its release threshold, but broad cost retains error discrimination; derivative sign is not a PPO learning guarantee or proof weight is sufficient.",
            "The mature lesson probabilities stayed.20/.30/.50 in the compared windows. Existing mixed completion change cannot be attributed to a between-window lesson-share switch.",
        ],
        "minimal_next_evidence": "On one unchanged-model holding trajectory, align5ms gate conjunction and dwell reset with estimated support/body_clear, true contacts, dq/gyro, and20ms actor ownership/begin-policy hold-cost mask. No contract change based only on marginal means.",
        "training_mutations": 0, "PPO_updates": 0, "new_gates": 0,
        "reward_or_threshold_changes": 0, "qualification": None,
    }
    target = OUT / "report.json"
    with target.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"report": str(target), "sha256": sha(target.read_bytes()),
                      "native_probe_rows": len(rows), "source_AST_checks": selected_ast}, ensure_ascii=False))


if __name__ == "__main__":
    main()
