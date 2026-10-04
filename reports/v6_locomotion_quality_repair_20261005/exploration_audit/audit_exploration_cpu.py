"""Read immutable checkpoints on CPU; never collect a rollout or update a model.

The script writes only its small audit JSON. Native actor/critic strict loads and
an Adam state load are checked without constructing a simulator or PPO runner.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
import subprocess

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SCUT = ROOT.parent / "scut-wheeled-legged-rl"
POLICY_ORDER = ("L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3")
SOURCES = (
    ("old_flat", "models/v6_model_delivery_20261003T2235/recommended_flat/model_final.pt",
     "models/v6_model_delivery_20261003T2235/recommended_flat/model_final.contract.json",
     "models/v6_model_delivery_20261003T2235/recommended_flat/model_final.checkpoint.json",
     "models/v6_model_delivery_20261003T2235/recommended_flat/agent_config.json",
     "7e3644f595ade15f8d43315f9e5e61e4db528fd8fac8df71bcc0441e0868891f"),
    ("old_speed_sealed", "models/v6_speed_candidate_19837_20261005/model.pt",
     "models/v6_speed_candidate_19837_20261005/contract.json",
     "models/v6_speed_candidate_19837_20261005/source.checkpoint.json",
     "models/v6_speed_candidate_19837_20261005/agent_config.json",
     "123841b609439939e9c9c8962c0171330892276446a692c936c8f33375ce11da"),
    ("v14_immutable_update24000", "reports/v6_locomotion_quality_repair_20261005/probe/v14_update24000_input/model.pt",
     "reports/v6_locomotion_quality_repair_20261005/probe/v14_update24000_input/model.contract.json",
     "reports/v6_locomotion_quality_repair_20261005/probe/v14_update24000_input/model.checkpoint.json",
     "models/v6_speed_candidate_19837_20261005/agent_config.json",
     "430c1c395e93c886899dfeffd46b7d19a958f6f743198d30eb15ba9a9fb3c3e0"),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tensor_tree_audit(value):
    """All saved actor/critic/Adam tensors, including scalar steps, must be finite."""
    tensors = []
    def visit(x):
        if isinstance(x, torch.Tensor):
            require(x.device.type == "cpu", "Checkpoint contains non-CPU mapped tensor")
            require(bool(torch.isfinite(x).all()), "Checkpoint contains nonfinite tensor")
            tensors.append(x)
        elif isinstance(x, dict):
            for y in x.values():
                visit(y)
        elif isinstance(x, (tuple, list)):
            for y in x:
                visit(y)
    visit(value)
    return {"finite": True, "tensor_count": len(tensors), "elements": sum(x.numel() for x in tensors)}


def equal_tree(a, b):
    if isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and torch.equal(a, b)
    if isinstance(b, dict):
        return isinstance(a, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in b)
    if isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def native_cpu_load(saved, config, contract):
    obs = TensorDict({"policy": torch.zeros(2, contract["actor_dim"]),
                      "critic": torch.zeros(2, contract["critic_dim"])}, batch_size=[2])
    args = deepcopy(config["actor"])
    args.pop("class_name")
    actor = MLPModel(obs, config["obs_groups"], "actor", contract["action_dim"], **args)
    args = deepcopy(config["critic"])
    args.pop("class_name")
    critic = MLPModel(obs, config["obs_groups"], "critic", 1, **args)
    initial_std = actor.distribution.std_param.detach().clone()
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    critic.load_state_dict(saved["critic_state_dict"], strict=True)
    optimizer = torch.optim.Adam(chain(actor.parameters(), critic.parameters()), lr=config["algorithm"]["learning_rate"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    checks = {"actor_exact": equal_tree(actor.state_dict(), saved["actor_state_dict"]),
              "critic_exact": equal_tree(critic.state_dict(), saved["critic_state_dict"]),
              "optimizer_exact": equal_tree(optimizer.state_dict(), saved["optimizer_state_dict"])}
    require(all(checks.values()), "CPU strict load did not preserve full learning tensors")
    named = list(actor.named_parameters())
    require(named[0][0] == "distribution.std_param", "Unexpected native actor parameter order")
    param_id = saved["optimizer_state_dict"]["param_groups"][0]["params"][0]
    std_adam = saved["optimizer_state_dict"]["state"][param_id]
    with torch.inference_mode():
        mean = actor(obs)
        sampled = actor(obs, stochastic_output=True)
        mean_again = actor(obs)
        require(torch.equal(mean, mean_again), "Stochastic sampling altered deterministic actor output")
        require(bool(torch.isfinite(mean).all() & torch.isfinite(sampled).all()), "Native actor output not finite")
    return {**checks, "initializer_std": initial_std.tolist(),
            "strict_load_overwrites_initializer": not torch.equal(initial_std, actor.distribution.std_param),
            "runtime_distribution_type": type(actor.distribution).__name__,
            "std_type": actor.distribution.std_type, "std_range": actor.distribution.std_range,
            "deterministic_output_is_mean": True,
            "std_adam_step": float(std_adam["step"]),
            "optimizer_steps_executed": 0, "rollouts_collected": 0}


def summarize(label, pt_name, contract_name, seal_name, config_name, expected_sha):
    pt, contract_path, seal_path, config_path = (ROOT / s for s in (pt_name, contract_name, seal_name, config_name))
    before = sha(pt)
    require(before == expected_sha, f"Pinned PT hash changed: {label}")
    contract, seal, config = read(contract_path), read(seal_path), read(config_path)
    saved = torch.load(pt, map_location="cpu", weights_only=True)
    infos = saved["infos"]
    require(seal["sealed"] is True and seal["checkpoint_sha256"] == before, f"Invalid native seal: {label}")
    require(seal["contract_sha256"] == infos["contract_sha256"] == sha(contract_path), f"Contract identity mismatch: {label}")
    require(seal["learning_updates"] == infos["successful_updates_total"], f"Learning clock mismatch: {label}")
    require(tuple(contract["policy_action_order"]) == POLICY_ORDER, "Policy axis order changed")
    require((contract["actor_dim"], contract["critic_dim"], contract["action_dim"]) == (35, 81, 6), "Actor ABI changed")
    std = saved["actor_state_dict"]["distribution.std_param"].double()
    require(std.shape == (6,) and bool(torch.isfinite(std).all()) and bool((std > 0).all()), "Invalid scalar std tensor")
    tensors = {k: tensor_tree_audit(saved[k]) for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict")}
    native = native_cpu_load(saved, config, contract)
    actual = std.clamp(*native["std_range"])
    settings = contract["v5_control"]
    scales = [settings["leg_position_scale"]] * 4 + [settings["wheel_velocity_scale"]] * 2
    bounds = [settings["action_clip"]] * 4 + [settings.get("wheel_action_clip", settings["action_clip"])] * 2
    result = {"label": label, "checkpoint": str(pt), "checkpoint_sha256": before,
              "contract": str(contract_path), "contract_sha256": sha(contract_path),
              "native_seal": str(seal_path), "native_seal_sha256": sha(seal_path),
              "agent_config": str(config_path), "agent_config_sha256": sha(config_path),
              "agent_config_scope": ("old_speed_agent_config_for_cpu_factory_only; v14 actual live PPO config must be read from native receipts"
                                     if label.startswith("v14") else "downloaded native agent_config"),
              "clocks": {k: infos.get(k) for k in ("successful_updates_total", "learning_lineage_updates", "task_actor_update", "stage_actor_update", "actor_updates_consumed")},
              "identity": {k: infos.get(k) for k in ("asset_manifest_sha256", "control_math_sha256")},
              "timing": {k: contract[k] for k in ("physics_dt", "policy_dt")},
              "policy_action_order": list(POLICY_ORDER), "raw_std": std.tolist(), "actual_std": actual.tolist(),
              "mean_actual_std": float(actual.mean()),
              "differential_gaussian_entropy_nats": float((actual.log() + .5 * math.log(2 * math.pi * math.e)).sum()),
              "entropy_coefficient": config["algorithm"]["entropy_coef"],
              "unsaturated_reference_std": [float(s * t) for s, t in zip(actual, scales)],
              "reference_units": ["rad"] * 4 + ["rad/s"] * 2,
              "action_scale": scales, "controller_raw_action_bounds": bounds,
              "runner_action_clip": config["clip_actions"],
              "controller_target_transform": "legs nominal + .25*action then nearest-angle wrap; wheels 10*action; controller clamps legs at +/-3, wheels at +/-9",
              "tensors": tensors, "native_cpu_strict_load": native,
              "recovery_rollout_requested": bool(contract.get("recovery_training", {}).get("enabled") or contract.get("nonstandard_start", {}).get("enabled")),
              "transfer_noise_floor": contract.get("transfer_noise_floor"),
              "contract_initial_noise_std": contract.get("initial_noise_std"),
              "actor_parent_metadata": infos.get("task_campaign"),
              "checkpoint_unchanged_after_cpu_audit": sha(pt) == before}
    require(result["checkpoint_unchanged_after_cpu_audit"], "Audit mutated input checkpoint")
    return result


def scut_audit():
    base = SCUT / "pretrained/26_infantry"
    rows = []
    for label, relative in (("scut_flat17500", "flat_and_rotation/2026-07-29_23-20-03/model_17500.pt"),
                            ("scut_rough_dash0", "rough_dash/2026-07-19_19-15-07/model_0.pt"),
                            ("scut_rough_dash3500", "rough_dash/2026-07-19_19-15-07/model_3500.pt")):
        path = base / relative
        before = sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        std = saved["model_state_dict"]["std"].double()
        require(std.shape == (6,) and bool((std > 0).all()), "SCUT scalar std invalid")
        finite = tensor_tree_audit(saved["model_state_dict"])
        rows.append({"label": label, "checkpoint": str(path), "checkpoint_sha256": before,
                     "checkpoint_iter": saved["iter"], "key": "model_state_dict.std", "std": std.tolist(),
                     "mean_std": float(std.mean()), "finite": finite,
                     "unsaturated_reference_std": [float(x * (.5 if i < 4 else 10.)) for i, x in enumerate(std)],
                     "reference_units": ["rad"] * 4 + ["rad/s"] * 2,
                     "checkpoint_unchanged": sha(path) == before,
                     "training_clock_warning": "Checkpoint iteration is its stored run-local iter; model_0 is not proof of random initialization"})
    return {"source_commit": subprocess.check_output(["git", "-C", str(SCUT), "rev-parse", "HEAD"], text=True).strip(),
            "checkpoints": rows,
            "saved_rough_dash_agent": {"file": str(base / "rough_dash/2026-07-19_19-15-07/params/agent.yaml"),
                 "sha256": sha(base / "rough_dash/2026-07-19_19-15-07/params/agent.yaml"),
                 "policy_class": "ActorCritic", "noise_std_type": "scalar", "state_dependent_std": False,
                 "init_noise_std": 1.0, "entropy_coef": .005, "clip_actions": None,
                 "launch_checkpoint": "flat_direct/2026-07-14_22-59-05/model_5500.pt", "resume_training": False},
            "saved_rough_dash_env": {"file": str(base / "rough_dash/2026-07-19_19-15-07/params/env.yaml"),
                 "sha256": sha(base / "rough_dash/2026-07-19_19-15-07/params/env.yaml"),
                 "leg_action_scale_rad": .5, "wheel_velocity_action_scale_rad_s": 10.0,
                 "max_wheel_vel_rad_s": 100.0, "use_action_low_pass_filter": False},
            "runtime_bounds_warning": "SCUT configured scalar state-independent policy is established by saved YAML and std key; its exact historical installed stock ActorCritic implementation is not bundled here. Do not borrow this project's RSL5 clamp range as proof of their bounds."}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(97163)
    rows = [summarize(*entry) for entry in SOURCES]
    flat, speed, v14 = rows
    std_step_accounting = []
    for earlier, later in ((flat, speed), (speed, v14)):
        actor_delta = (later["clocks"]["task_actor_update"] - earlier["clocks"]["task_actor_update"]
                       if earlier is speed else later["clocks"]["task_actor_update"])
        adam_delta = later["native_cpu_strict_load"]["std_adam_step"] - earlier["native_cpu_strict_load"]["std_adam_step"]
        require(adam_delta == actor_delta * 20, "Saved std Adam step count differs from actor minibatch exposure")
        std_step_accounting.append({"from": earlier["label"], "to": later["label"],
            "actor_updates_delta": actor_delta, "std_adam_step_delta": adam_delta,
            "epochs_times_minibatches": 20, "exactly_matches": True,
            "limit": "Count and authenticated full-restore receipts are consistent with continuity; no historical intermediate PT time series was replayed"})
    native_path = ROOT / "reports/v6_locomotion_quality_repair_20261005/deploy/first_PPO_verification_014.json"
    native = read(native_path)
    receipt = native["evidence"]["resume_verification"]
    require(receipt["parent_checkpoint_sha256"] == speed["checkpoint_sha256"], "v14 resume parent mismatch")
    require(all(receipt[k] is True for k in ("actor_exact", "critic_exact", "optimizer_exact")), "Native restore receipt incomplete")
    sources = [ROOT / s for s in ("scripts/train_chassis.py", "src/wheeled_tasks/agents/v40_ppo_cfg.py", "src/wheeled_tasks/chassis/policy_transfer.py",
        "src/wheeled_tasks/chassis/v5_control.py", "src/wheeled_tasks/chassis/scut_observation.py",
        "src/wheeled_algo/recovery_rollout.py", "src/wheeled_algo/chassis_export.py")]
    frozen_matches = {}
    for path in sources:
        relative = path.relative_to(ROOT).as_posix()
        frozen_bytes = subprocess.check_output(["git", "-C", str(ROOT), "show", f"b4ce80964d47c0127e0b3d9929ad35a620bbf204:{relative}"])
        frozen_matches[relative] = sha(path) == hashlib.sha256(frozen_bytes).hexdigest()
    require(all(frozen_matches.values()), "Audited training/control/mask/export source differs from v14 b4 freeze")
    from rsl_rl.modules import distribution
    from rsl_rl.models import mlp_model
    from rsl_rl.algorithms import ppo
    from rsl_rl.utils import logger
    from rsl_rl.runners import on_policy_runner
    sources += [Path(x.__file__) for x in (distribution, mlp_model, ppo, logger, on_policy_runner)]
    native_distribution = GaussianDistribution(6)
    require(native_distribution.std_range == [1e-6, 1e6], "RSL5 default clamp range changed")
    predecessor_receipts = []
    for name in ("reports/v6_task_campaign_20261001/deploy/v9_resume_verification_010.json",
                 "reports/v6_mixed_locomotion_20261004/deploy/mixed_resume_verification_011.json"):
        path, data = ROOT / name, read(ROOT / name)
        require(all(data["exact_initial_restore"].values()), "Historical source restore receipt failed")
        predecessor_receipts.append({"file": str(path), "sha256": sha(path),
            "parent_sha256": data.get("parent_flat_sha256", data.get("parent_sha256")),
            "exact_initial_restore": data["exact_initial_restore"],
            "scope": "historical native receipt; original initial-state tensors not all locally retained for independent recheck"})
    mean_probe_path = ROOT / "reports/v6_locomotion_quality_repair_20261005/probe/native_003/report.json"
    mean_probe = read(mean_probe_path)
    for model_name, model in zip(("flat", "old_speed", "v14"), rows):
        metadata = mean_probe["models"][model_name]["metadata"]
        require(metadata["source_checkpoint_sha256"] == model["checkpoint_sha256"], "Mean probe checkpoint differs from audited PT")
        require(metadata["contract_sha256"] == model["contract_sha256"], "Mean probe source contract differs from audited PT")
    require(mean_probe["scope"] == "short_matched_deterministic_diagnostics", "Mean probe is not deterministic")
    mean_stand = {name: {phase: {k: value[k] for k in ("samples_200hz", "duration_s", "planar_speed_mean_m_s", "vx_actual_mean_m_s")}
                        for phase, value in cases["stand_305mm"]["phases"].items()}
                  for name, cases in mean_probe["cases"].items()}
    result = {"schema": "v6-immutable-exploration-CPU-audit-v1", "status": "CPU_audit_complete",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "qualification": False,
        "scope": "Three local immutable learning checkpoints and local SCUT public checkpoints; no simulator, SSH, GPU, PPO update or live config mutation",
        "models": rows, "scut": scut_audit(),
        "ratio_speed_over_flat_per_axis": [b / a for a, b in zip(flat["actual_std"], speed["actual_std"])],
        "ratio_v14_over_speed_per_axis": [b / a for a, b in zip(speed["actual_std"], v14["actual_std"])],
        "std_adam_step_accounting": std_step_accounting,
        "native_v14_restore_receipt": {"file": str(native_path), "sha256": sha(native_path),
            "receipt": receipt, "invocation": native["evidence"]["invocation"]["command"]},
        "historical_parent_receipts": predecessor_receipts,
        "matched_mean_probe": {"file": str(mean_probe_path), "sha256": sha(mean_probe_path),
            "scope": mean_probe["scope"], "status": mean_probe["status"],
            "stand_305mm_phases": mean_stand,
            "scope_warning": "This is an existing deterministic short native replay, not a new stochastic-versus-mean ablation"},
        "source_files_sha256": {str(p): sha(p) for p in sources},
        "v14_frozen_source_commit": "b4ce80964d47c0127e0b3d9929ad35a620bbf204",
        "current_core_sources_equal_frozen_v14": frozen_matches,
        "findings": {
            "nonfinite_actor_critic_adam": False,
            "v14_std_reset_to_initializer_evidence": False,
            "native_std_clamp_active_on_these_saved_parameters": False,
            "std_type": "scalar means 6 independently learned positive scalar-space parameters, not one shared scalar",
            "std_is_state_independent": True,
            "native_logger": "Policy/mean_std is mean(output_std); cached distribution is from the last minibatch recompute, prior to that optimizer step, so tiny differences from a saved post-update raw std need not be a bug",
            "native_entropy": "loss subtracts .005*mean(sum-axis Gaussian differential entropy) on the compact valid-RL population; entropy encourages exploration and is not a task completion reward",
            "native_mask": "whole interval actor ownership selects actor surrogate, critic loss and entropy using the same compact storage; excluded script/zero-actuation-hold/BLEND transitions are not unmasked entropy-only samples. A pure-RL policy_motion_hold with active actor ownership can remain eligible",
            "explicit_std_floor": "only --transfer path can apply optional transfer_noise_floor; authenticated v14 invocation is --resume --resume-task-change and no floor is set",
            "comparison_limit": "stochastic training behavior includes exploration, randomization and resets; deterministic ONNX/native mean replay excludes exploration. Their velocity statistics cannot be used as matched behavior comparisons",
            "unproven": "Large std can worsen sampled actions, but neither std growth nor training-vs-mean metric differences prove the cause of mean-policy drift or forgetting; no physical sample-vs-mean ablation was run here"}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "exploration_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "models": len(rows), "scut_models": len(result["scut"]["checkpoints"]),
        "optimizer_steps": 0, "report": str(OUT / "exploration_audit.json")}, indent=2))


if __name__ == "__main__":
    main()
