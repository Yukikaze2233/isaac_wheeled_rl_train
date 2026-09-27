"""Paired diagnostics preserve learning identity and isolate the declared treatment."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.diagnostic_curriculum import (
    diagnostic_contract, height_scan_contract, height_response_fit,
    validate_diagnostic_resume, PolicyUpdateProbe,
    DiagnosticRewardAccumulator, REWARD_TERMS,
)
from wheeled_tasks.chassis.episode_metrics import EpisodeMetrics
from wheeled_tasks.chassis.full_curriculum import resolve_plan, stage_contract

ROOT = Path(__file__).resolve().parents[1]


def configurations():
    load = lambda name: json.loads((ROOT / name).read_text())
    curriculum = resolve_plan(load("contracts/v6_scut35_budget_retry_v1.json"), load)
    source = stage_contract(load(curriculum["base_contract"]), curriculum, curriculum["stages"][0], 16384)
    return source, load("contracts/v6_p0_diagnostic_v1.json")


def test_arms_only_differ_in_two_widths_and_metadata():
    source, plan = configurations()
    a = diagnostic_contract(source, plan, "A_control", 16384)
    b = diagnostic_contract(source, plan, "B_wide", 16384)
    assert {k for k in a if a[k] != b[k]} == {"precision_tracking", "diagnostic_study"}
    assert {k for k in a["precision_tracking"] if a["precision_tracking"][k] != b["precision_tracking"][k]} == {
        "stationary_width_m_s", "height_width_m"}
    for c in (a, b):
        assert validate_diagnostic_resume(source, c)["actor_critic_optimizer"] == "preserved"
        assert c["reference_reward"]["stationary_width_m_s"] == .1
        assert c["evaluation"]["mode"] == "monitor" and "regression_recovery" not in c["evaluation"]
        assert c["evaluation"]["cases"] == source["evaluation"]["cases"]
        assert len(c["scene_groups"]) == 8 and sum(g["fraction"] for g in c["scene_groups"]) == 1
        assert {s["command"][2] for s in c["skill_specs"].values()} == {.26, .28, .305, .32}
        assert all(s["command"][1] == 0 for s in c["skill_specs"].values())
        for key in ("command_transport", "contact_domain", "dynamics_randomization"):
            assert c[key]["enabled_fraction"] == 0
        for key in ("physics_dt", "policy_dt", "v5_control", "actor_layout", "asset_manifest_sha256"):
            assert c[key] == source[key]
    changed = deepcopy(b)
    changed["reference_reward"]["stationary_width_m_s"] = .15
    with pytest.raises(ValueError, match="undeclared"):
        validate_diagnostic_resume(source, changed)


def test_scan_is_separate_from_the_fixed_regression_protocol():
    source, plan = configurations()
    train = diagnostic_contract(source, plan, "A_control", 16384)
    scan = height_scan_contract(train)
    assert scan["evaluation"]["protocol_id"] != train["evaluation"]["protocol_id"]
    assert scan["evaluation"]["warmup_seconds"] == 5
    assert len(scan["evaluation"]["cases"]) == 8
    candidate = {"cases": {c["name"]: {**c, "frames": 100, "full_horizon_episodes": 4,
        "requested_episodes": 4, "height_actual_mean_m": .2 * c["command"][2] + .27}
        for c in scan["evaluation"]["cases"]}}
    fit = height_response_fit(candidate, "stand")
    assert fit["slope"] == pytest.approx(.2) and fit["intercept_m"] == pytest.approx(.27)
    assert fit["fit_rmse_m"] < 1e-12
    for case in candidate["cases"].values():
        case["full_horizon_episodes"] = 0
    assert height_response_fit(candidate, "stand")["status"] == "insufficient_complete_episodes"


def test_height_bias_and_planar_speed_do_not_cancel():
    metrics = EpisodeMetrics(["stand", "stand"], "cpu", .02)
    data = {"episode_ticks": torch.tensor([2, 2]), "position": torch.zeros(2, 3),
        "velocity": torch.tensor([[.1, 0., 0.], [-.1, 0., 0.]]), "omega": torch.zeros(2, 3),
        "commands": torch.tensor([[0., 0., .32], [0., 0., .32]]), "height": torch.tensor([.30, .34]),
        "planar_speed_world": torch.tensor([.2, .4]), "reward": torch.zeros(2),
        "motor_effort": torch.zeros(2, 6), "gravity": torch.tensor([[0., 0., -1.]]).repeat(2, 1),
        "gap": torch.zeros(2), "done": torch.zeros(2, dtype=torch.bool),
        "terminated": torch.zeros(2, dtype=torch.bool), "success": torch.zeros(2, dtype=torch.bool),
        "reasons": {"boundary": torch.zeros(2, dtype=torch.bool)}}
    metrics.observe(data)
    result = metrics.report()["groups"]["stand"]
    assert result["height_actual_mean_m"] == pytest.approx(.32)
    assert result["height_mae_m"] == pytest.approx(.02)
    assert result["height_bias_m"] == pytest.approx(0., abs=2e-8)
    assert result["vx_actual_mean_m_s"] == 0
    assert result["planar_speed_mean_cm_s"] == pytest.approx(30.)
    assert result["planar_speed_reference"] == "world_horizontal_root_link"


def test_policy_probe_uses_identical_observations_without_consuming_rng():
    class Distribution:
        def update(self, value):
            self.normal = torch.distributions.Normal(value, torch.full_like(value, .2))

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Linear(35, 6)
            self.distribution = Distribution()

        def get_latent(self, observations):
            return observations

        @property
        def output_distribution_params(self):
            return self.distribution.normal.loc, self.distribution.normal.scale

        @property
        def output_mean(self):
            return self.distribution.normal.mean

        @property
        def output_std(self):
            return self.distribution.normal.stddev

        def get_kl_divergence(self, before, after):
            return torch.distributions.kl_divergence(
                torch.distributions.Normal(*before), torch.distributions.Normal(*after)).sum(-1)

    actor = Actor()
    storage = SimpleNamespace(observations=torch.randn(3, 4, 35))
    probe = PolicyUpdateProbe(actor, storage, [(torch.tensor([0, 1]), {"group_name": "stand"}),
                                              (torch.tensor([2, 3]), {"group_name": "forward"})])
    rng = torch.get_rng_state().clone()
    probe.capture()
    storage.observations.zero_()
    with torch.no_grad():
        actor.mlp.bias.add_(.1)
    metrics = probe.metrics()
    assert torch.equal(rng, torch.get_rng_state())
    assert float(metrics["DiagnosticUpdate/kl_rollout_probe"]) == pytest.approx(.75, abs=1e-5)


def test_rollout_diagnostics_keep_activation_denominator_and_drain_without_aliasing():
    batches = [(torch.tensor([0, 1]), {"group_name": "stand"})]
    with torch.inference_mode():
        accumulator = DiagnosticRewardAccumulator(2, "cpu", batches, {"height_precision": 1.})
        components = {name: torch.tensor([-1., 0.]) for name in REWARD_TERMS}
        accumulator.observe(components, {"height_precision": torch.tensor([True, False])})
        components = {name: torch.tensor([0., 0.]) for name in REWARD_TERMS}
        accumulator.observe(components, {"height_precision": torch.tensor([True, True])})
    result = accumulator.drain(.02)
    assert float(result["/diagnostic/stand/reward/height_precision"]) == pytest.approx(-.005)
    assert float(result["/diagnostic/stand/saturation/height_precision"]) == pytest.approx(1 / 3)
    assert float(result["/diagnostic/stand/activation/height_precision"]) == pytest.approx(.75)
    assert accumulator.drain(.02) == {}


def test_study_runs_both_arms_from_same_source_without_capability_gates(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("diagnostic_study_test", ROOT / "scripts/run_chassis_diagnostic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.signal, "signal", lambda *_: None)
    source, plan = configurations()
    parent_dir = tmp_path / "source"
    parent_dir.mkdir()
    parent = parent_dir / "model_final.pt"
    parent.write_bytes(b"actor critic Adam750")
    source_path = parent_dir / "contract.json"
    source_path.write_text(json.dumps(source))
    (parent_dir / "completion.json").write_text(json.dumps({"status": "completed", "parent_updates": 700, "successful_updates": 50}))
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"retained100")
    plan.update(source_checkpoint_sha256=module.digest(parent), source_contract_sha256=module.digest(source_path),
                baseline_checkpoint_sha256=module.digest(baseline))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    args = SimpleNamespace(contract=path, resume=parent, baseline_checkpoint=baseline, transfer=None,
        run_dir=tmp_path / "study", num_envs=64, stage="diagnostic", device="cpu", max_runtime_seconds=1000.)
    study = module.DiagnosticStudy(args)
    calls, evaluations = [], []

    def child(command, output, training=False):
        assert training and command[command.index("--resume") + 1] == str(parent)
        assert "--resume-diagnostic" in command and command[command.index("--updates") + 1] == "100"
        calls.append(command)
        output.mkdir(parents=True)
        (output / "model_final.pt").write_bytes(b"trained arm")
        return {"successful_updates": 100, "parent_updates": 0, "actor_updates_in_block": 90,
                "learning_rate": 7.5e-6, "checkpoint_sha256": "mock", "diagnostic_origin": {"source_formal_updates": 750}}

    def evaluate(contract, checkpoints, directory, seed, labels, step, *, scan):
        evaluations.append((labels, seed, step, scan))

    monkeypatch.setattr(study, "run_child", child)
    monkeypatch.setattr(study, "evaluate", evaluate)
    assert study.run() == 0
    assert len(calls) == 2 and len(evaluations) == 12
    assert study.report["successful_updates"] == 200 and study.report["status"] == "diagnostic_completed"
    selection = json.loads((args.run_dir / "artifact_selection.json").read_text())
    assert selection["deployment_checkpoint"] is None and not selection["automatic_formal_promotion"]
    child_dir = args.run_dir / "B_wide/train"
    (child_dir / "progress.json").write_text(json.dumps({"pid": 12345, "successful_updates": 3,
        "parent_updates": 0, "training_transitions": 3 * 64 * 24, "phase": "training"}))
    study.completed_updates = 100
    study.phase = "B_wide/training"
    assert study.publish(child_dir)
    progress = json.loads((args.run_dir / "progress.json").read_text())
    assert progress["successful_updates"] == 103 and progress["source_formal_updates"] == 750
    assert progress["worker_pid"] == 12345 and progress["arm_updates"] == 3


def test_completed_diagnostic_units_are_sealed_before_the_whole_study_finishes(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from chassis_batch_export import export_ready
    paths = [("train/A_control/train/completion.json", "completed"),
             ("train/A_control/train/checkpoints/update_00000050/completion.json", "checkpoint_sealed"),
             ("train/baseline_scan/seed_190619/evaluation.json", "evaluated")]
    for name, status in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": status}))
    index = export_ready(tmp_path)
    assert not index["run_finished"]
    assert {b["kind"] for b in index["batches"]} == {"training_block", "checkpoint_snapshot", "evaluation"}
