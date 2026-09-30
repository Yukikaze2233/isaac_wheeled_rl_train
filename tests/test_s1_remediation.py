"""Reward forks must preserve physical semantics, learning origin and spent credit."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis import remediation as repair
from wheeled_tasks.chassis.policy_transfer import validate_reward_resume

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_chassis_remediation import RemediationStudy, validate_contracts


def read(name):
    return json.loads((ROOT / name).read_text())


@pytest.fixture
def source():
    return read("reports/v6_s1_remediation_20260930/source/contract.json")


@pytest.fixture
def plan():
    return read("contracts/v6_s1_remediation_v1.json")


@pytest.mark.parametrize("arm", repair.ARMS)
def test_resume_allows_only_declared_named_change(source, arm):
    target = read(f"contracts/v6_s1_{arm}_v1.json")
    assert validate_reward_resume(source, target)["kind"] == repair.REVISION
    for key, value in (("physics_dt", .005), ("policy_dt", .01), ("actor_dim", 36),
                       ("learning_rate_schedule", "adaptive"), ("resume_critic_warmup_updates", 0)):
        changed = deepcopy(target)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_reward_resume(source, changed)
    for key in ("evaluation", "dense_tracking", "usb_transport"):
        changed = deepcopy(target)
        changed[key]["unknown_change"] = True
        with pytest.raises(ValueError):
            validate_reward_resume(source, changed)


def test_both_arms_all_roles_use_independent_design(plan):
    assert len(validate_contracts(plan)) == 10


def test_worker_rejects_wrong_source_or_consumption(monkeypatch):
    config = read("contracts/v6_s1_B_stationary_v1.json")
    args = SimpleNamespace(resume_reward_change=True, resume="pinned.pt", consumed_updates=658,
        actor_updates_consumed=648, stage_actor_update=338, updates=40, learning_rate_scale=1.,
        num_envs=20000, seed=290927, block_index=0)
    monkeypatch.setattr(repair, "digest", lambda _: repair.CHECKPOINT_SHA)
    repair.validate_worker(config, args)
    for key, value in (("consumed_updates", 618), ("actor_updates_consumed", 658),
                       ("stage_actor_update", 618), ("seed", 290928), ("learning_rate_scale", .5)):
        changed = deepcopy(args)
        setattr(changed, key, value)
        with pytest.raises(ValueError):
            repair.validate_worker(config, changed)
    monkeypatch.setattr(repair, "digest", lambda _: "different")
    with pytest.raises(ValueError):
        repair.validate_worker(config, args)


def study(tmp_path, plan):
    args = SimpleNamespace(contract=ROOT / "contracts/v6_s1_remediation_v1.json",
        run_dir=tmp_path / "run", max_runtime_seconds=100, transfer=None, stage="S1_remediation",
        num_envs=20000, device="cpu")
    job = RemediationStudy(args)
    job.root.mkdir()
    job.claim = tmp_path / "persistent-claim.json"
    job.claimed = True
    return job


def test_forks_share_origin_but_never_refund_spent_updates(tmp_path, plan):
    job = study(tmp_path, plan)
    commands = [job.train_command(recipe) for recipe in plan["stages"]]
    def value(command, flag):
        return command[command.index(flag) + 1]
    assert [value(c, "--resume") for c in commands] == [plan["source_checkpoint"]] * 2
    assert [value(c, "--consumed-updates") for c in commands] == ["618", "658"]
    assert [value(c, "--actor-updates-consumed") for c in commands] == ["618", "648"]
    job.account("A_fixed", {"successful_updates": 40, "actor_updates_in_block": 30})
    job.account("B_stationary", {"status": "failed", "successful_updates": 13, "actor_updates_in_block": 3})
    assert job.report["successful_updates"] == 671
    assert job.report["actor_updates_consumed"] == 651
    assert json.loads(job.claim.read_text())["consumed_updates"] == 671
    with pytest.raises(ValueError):
        job.account("B_stationary", {"successful_updates": 0})
    with pytest.raises(FileExistsError):
        job.claim.open("x")
    job.account("B_stationary", {"successful_updates": 40, "actor_updates_in_block": 30})
    assert job.report["successful_updates"] == 698
    assert job.report["actor_updates_consumed"] == 678


def test_failed_worker_still_charges_final_updates(tmp_path, plan, monkeypatch):
    job = study(tmp_path, plan)
    output = job.root / "A_fixed/train"
    def fail(*args, **kwargs):
        output.mkdir()
        (output / "completion.json").write_text(json.dumps({"status": "failed", "successful_updates": 12,
            "actor_updates_in_block": 2}))
        return 1
    monkeypatch.setattr(job, "execute", fail)
    with pytest.raises(RuntimeError):
        job.run_child([], output, training=True)
    assert job.report["successful_updates"] == 630


def test_missing_required_evaluation_fails_closed():
    with pytest.raises(ValueError, match="invalid_evaluation_coverage"):
        repair.validate_evaluation({"status": "evaluated", "candidates": [{"cases": {}, "unfinished_episodes": 0}]}, ["stand"])


def test_learning_audit_preserves_inputs_and_rng_tracks_real_dwell(monkeypatch):
    from wheeled_tasks.chassis import learning_audit
    class Probe:
        def __init__(self, *args): pass
        def capture(self): pass
        def metrics(self): return {}
    monkeypatch.setattr(learning_audit, "PolicyUpdateProbe", Probe)
    commands = torch.tensor([[0., 0., .305], [0., 0., .23]])
    batches = [(torch.arange(2), {"group_name": "stand", "kind": "stand"})]
    env = SimpleNamespace(num_envs=2, device="cpu", policy_dt=1., commands=commands,
        skills=SimpleNamespace(batches=batches), cfg={"new_asset_curriculum": {
            "height_sampling_schedule": {"fixed_heights_m": [.23, .305]}}})
    storage = SimpleNamespace(advantages=torch.tensor([[[1.], [-1.]]]),
        returns=torch.ones(1, 2, 1), values=torch.zeros(1, 2, 1))
    audit = learning_audit.LearningAudit(env, SimpleNamespace(actor=None, storage=storage))
    data = {"commands": commands.clone(), "reward": torch.ones(2), "planar_speed_world": torch.zeros(2),
        "height": commands[:, 2].clone(), "reasons": {"fall": torch.zeros(2)}, "done": torch.zeros(2, dtype=torch.bool)}
    rng = torch.get_rng_state().clone()
    components = {"dense_stationary_broad": torch.tensor([-.1, -.2])}
    with torch.inference_mode():
        for i in range(15):
            data["done"][1] = i == 7
            audit.observe(data, components)
    assert audit.hold.tolist() == [15., 7.]
    assert torch.equal(commands, data["commands"])
    audit.capture()
    metrics = audit.drain()
    assert torch.equal(rng, torch.get_rng_state())
    assert metrics["LearningAudit/stand/height_305mm/continuous_zero_15s"] == pytest.approx(1/15)
    assert metrics["LearningAudit/stand/height_230mm/continuous_zero_10s"] == 0
    assert metrics["LearningAudit/stand/advantage_positive_fraction"] == .5
    assert components["dense_stationary_broad"].tolist() == pytest.approx([-.1, -.2])
    assert audit.sums.eq(0).all() and audit.steps == 0


def test_native_reward_change_is_stationary_only(source):
    from calibrate_v6_rewards import calibration_suite
    a, raw_a = calibration_suite(source["dense_tracking"], stationary_anchor=False)
    target = read("contracts/v6_s1_B_stationary_v1.json")
    b, raw_b = calibration_suite(target["dense_tracking"], stationary_anchor=False)
    assert a["all_checks_passed"] and b["all_checks_passed"]
    assert "stationary_anchor" not in raw_a["stand_correct"][1]
    for name in ("move_correct", "move_half_speed", "move_overspeed"):
        torch.testing.assert_close(raw_a[name][0], raw_b[name][0])
    for name in ("stand_drift", "stand_side_drift"):
        assert b["trajectories"][name]["discounted_return"] < a["trajectories"][name]["discounted_return"]


def test_native_ppo_fixed_lr_critic_warmup_restore_and_noninvasive_audit():
    from test_recovery_rollout_mask import algorithm, collect
    from train_chassis import configure_learning_rate
    from wheeled_tasks.chassis.learning_audit import LearningAudit
    plain, observed = algorithm(), algorithm()
    commands = torch.tensor([[0., 0., .305], [0., 0., .23]])
    env = SimpleNamespace(num_envs=2, device="cpu", policy_dt=.02, commands=commands,
        skills=SimpleNamespace(batches=[(torch.arange(2), {"group_name": "stand", "kind": "stand"})]),
        cfg={"new_asset_curriculum": {"height_sampling_schedule": {"fixed_heights_m": [.23, .305]}}})
    audit = LearningAudit(env, observed)
    initial = deepcopy(observed.actor.state_dict())
    for iteration in range(3):
        for alg in (plain, observed):
            alg.schedule = "fixed"
            configure_learning_rate(alg, 1e-4, 1.)
            alg.actor.requires_grad_(iteration > 0)
            obs = collect(alg, torch.ones(6, 2, dtype=torch.bool))
            alg.compute_returns(obs)
        data = {"commands": commands, "reward": torch.ones(2), "planar_speed_world": torch.zeros(2),
            "height": commands[:, 2], "reasons": {"fall": torch.zeros(2)}, "done": torch.zeros(2, dtype=torch.bool)}
        with torch.inference_mode():
            for _ in range(6):
                audit.observe(data, {"tracking": torch.ones(2)})
        audit.capture()
        for alg in (plain, observed):
            torch.manual_seed(765)
            alg.update()
            assert alg.learning_rate == 1e-4
            assert all(group["lr"] == 1e-4 for group in alg.optimizer.param_groups)
        values = audit.drain()
        assert all(torch.equal(v, observed.actor.state_dict()[k]) for k, v in plain.actor.state_dict().items())
        if iteration == 0:
            assert all(torch.equal(v, observed.actor.state_dict()[k]) for k, v in initial.items())
            assert abs(values["DiagnosticUpdate/kl_rollout_probe"]) < 1e-7
        # Resume Adam/model into a fresh worker, then perform another native PPO update.
        if iteration == 1:
            saved = [deepcopy(alg.save()) for alg in (plain, observed)]
            plain, observed = algorithm(), algorithm()
            for alg, state in zip((plain, observed), saved):
                alg.load(state, load_cfg=None, strict=True)
            audit = LearningAudit(env, observed)
    assert any(not torch.equal(v, observed.actor.state_dict()[k]) for k, v in initial.items())
