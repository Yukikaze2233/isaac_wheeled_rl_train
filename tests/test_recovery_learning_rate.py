"""Adaptive PPO rollback preserves its LR meaning and its consumption ledger."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from rsl_rl.algorithms import PPO

from test_new_asset_blocks import Campaign, argument, seal

ROOT = Path(__file__).resolve().parents[1]


def worker():
    spec = importlib.util.spec_from_file_location(
        "recovery_worker", ROOT / "scripts/train_chassis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_worker_preflight(monkeypatch, module, contract, arguments):
    import wheeled_tasks.chassis.control_contract as control
    import wheeled_tasks.chassis.preflight as preflight

    # Exercise the real CLI validation without loading Isaac or physical assets.
    frozen = {**contract, "control_math_source": "unused.json", "asset_directory": "unused",
              "stages": [{"name": "jump"}]}
    with monkeypatch.context() as patch:
        patch.setattr(module, "preflight", lambda *_: (frozen, {}))
        patch.setattr(control, "load_chassis_control", lambda *_: {})
        patch.setattr(preflight, "design_preflight", lambda *_a, **_kw: {})
        patch.setattr(module.sys, "argv", [str(ROOT / "scripts/train_chassis.py"),
            *arguments, "--preflight-only"])
        return module.main()


def test_adaptive_rollback_ratio_above_one_reaches_real_worker_entry(
        tmp_path, monkeypatch):
    class AdaptiveCampaign(Campaign):
        def execute(self, command, log_path, training_directory=None):
            if training_directory is not None:
                assert run_worker_preflight(
                    monkeypatch, worker(), self.blocks.contract, command[3:]) == 0
            code = super().execute(command, log_path, training_directory)
            if training_directory is not None:
                path = training_directory / "completion.json"
                completion = json.loads(path.read_text())
                # Three adaptive blocks finish above the initial contract LR.
                completion["learning_rate"] = (
                    8e-4 if len(self.calls) <= 3 else 4e-4)
                path.write_text(json.dumps(completion))
            return code

    run = AdaptiveCampaign(tmp_path, monkeypatch,
        [{"stand", "jump_03"}, set(), set(), {"stand", "jump_03", "jump_06"}],
        updates=8)
    code, report = run.run()
    assert code == 0 and report["status"] == "stage_accepted"
    retry = run.calls[3]
    assert float(argument(retry, "--learning-rate-scale")) == 4.
    assert argument(retry, "--resume").endswith("block_000/model_final.pt")
    assert argument(retry, "--consumed-updates") == "6"
    assert argument(retry, "--actor-updates-consumed") == "6"
    assert argument(retry, "--stage-actor-update") == "2"
    assert report["successful_updates"] == report["actor_updates_consumed"] == 8
    attempt, = report["recovery_attempts"]
    assert attempt["previous_learning_rate"] == 8e-4
    assert attempt["learning_rate"] == 4e-4
    assert attempt["learning_rate_factor"] == .5
    assert run.blocks.contract["learning_rate"] == 1e-4


@pytest.mark.parametrize("scale", ["0", "-1", "nan", "inf", "-inf"])
def test_worker_entry_rejects_nonpositive_or_nonfinite_recovery_ratio(
        monkeypatch, capsys, scale):
    with pytest.raises(SystemExit) as error:
        run_worker_preflight(monkeypatch, worker(), {},
            ["--stage", "jump", f"--learning-rate-scale={scale}"])
    assert error.value.code == 2
    assert "finite and positive" in capsys.readouterr().err


def test_rsl_restore_and_explicit_recovery_preserve_adam_moments():
    module = worker()
    original = PPO(torch.nn.Linear(2, 1), torch.nn.Linear(2, 1), None,
                   learning_rate=1e-4)
    sum(p.square().sum() for p in original.actor.parameters()).backward()
    original.optimizer.step()
    observed_rate = 0.0008779149519890263
    original.optimizer.param_groups[0]["lr"] = observed_rate
    checkpoint = deepcopy(original.save())

    restored = PPO(torch.nn.Linear(2, 1), torch.nn.Linear(2, 1), None,
                   learning_rate=1e-4)
    restored.load(checkpoint, load_cfg=None, strict=True)
    # This exercises the installed RSL implementation, not a fake load method.
    assert restored.learning_rate == observed_rate
    assert module.configure_learning_rate(restored, 1e-4) == observed_rate
    state = deepcopy(restored.optimizer.state_dict()["state"])
    actor = deepcopy(restored.actor.state_dict())

    rate = module.configure_learning_rate(
        restored, 1e-4, observed_rate / 1e-4 * .5)
    assert rate == pytest.approx(observed_rate * .5)
    assert restored.learning_rate == rate
    assert all(group["lr"] == rate for group in restored.optimizer.param_groups)
    for index, values in state.items():
        for key, value in values.items():
            assert torch.equal(value, restored.optimizer.state_dict()["state"][index][key])
    assert all(torch.equal(value, restored.actor.state_dict()[key])
               for key, value in actor.items())


def test_recovery_rate_overflow_is_rejected_before_changing_optimizer():
    module = worker()
    algorithm = PPO(torch.nn.Linear(2, 1), torch.nn.Linear(2, 1), None,
                    learning_rate=1e-4)
    with pytest.raises(ValueError, match="Optimizer learning rate"):
        module.configure_learning_rate(algorithm, 1e308, 1e308)
    assert algorithm.learning_rate == 1e-4
    assert algorithm.optimizer.param_groups[0]["lr"] == 1e-4


@pytest.mark.parametrize("patience,block_count,status,expected", [
    ("missing", 3, "failed", True),
    (3, 3, "failed", True),
    ("missing", 2, "failed", False),
    (None, 3, "failed", False),
    ("missing", 3, "stopped", False),
])
def test_interrupted_rollback_uses_main_loop_patience_without_refunding_budget(
        tmp_path, monkeypatch, patience, block_count, status, expected):
    run = Campaign(tmp_path, monkeypatch, [], updates=1000)
    run.args.run_dir.mkdir()
    retained = tmp_path / "retained/model_final.pt"
    seal(retained, run.path, consumed=338, course=218)
    original_checkpoint = retained.read_bytes()
    settings = deepcopy(run.blocks.evaluation_settings)
    if patience == "missing":
        settings.pop("regression_patience")
    else:
        settings["regression_patience"] = patience
    rate = 0.0003901844231062339
    run.blocks.report.update(successful_updates=378, actor_updates_consumed=378,
        resumed_updates=378, retention_learning_checkpoint=str(retained),
        learning_rate_scale=rate / run.blocks.contract["learning_rate"])
    gate = {"lost_parent_passes": ["stand"]}
    previous = {"status": status, "blocks": [
        {"successful_updates_total": 378 - 40 * (block_count - index - 1),
         "rollback_trigger_cases": ["stand"], "capability_gate": gate}
        for index in range(block_count)]}

    restored = run.blocks.pending_regression_recovery(settings, previous)

    assert restored == (retained if expected else None)
    assert retained.read_bytes() == original_checkpoint
    assert run.blocks.report["successful_updates"] == 378
    assert run.blocks.report["actor_updates_consumed"] == 378
    if expected:
        attempt, = run.blocks.report["recovery_attempts"]
        assert attempt["consumed_updates"] == 378
        assert attempt["remaining_updates"] == 622
        assert attempt["previous_learning_rate"] == pytest.approx(rate)
        assert attempt["learning_rate"] == pytest.approx(rate * .5)
        assert run.blocks.report["completed_interrupted_recovery"] is True
        assert (run.args.run_dir / "recovery_history.json").exists()
        # Retrying the interrupted transaction cannot consume a second retry.
        assert run.blocks.pending_regression_recovery(settings, previous) is None
        assert len(run.blocks.report["recovery_attempts"]) == 1
    else:
        assert not run.blocks.report.get("recovery_attempts")
        assert "completed_interrupted_recovery" not in run.blocks.report
        assert not (run.args.run_dir / "recovery_history.json").exists()
