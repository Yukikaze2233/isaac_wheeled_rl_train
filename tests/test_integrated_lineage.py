"""Real native CPU model/Adam seals; admission never executes learning or account writes."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis import integrated_lineage as joint
from wheeled_tasks.chassis.task_campaign import CAMPAIGN_ID, DOMAINS


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def fixture(tmp_path):
    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir()
    run.mkdir()
    physical = dict(asset_manifest_sha256="a" * 64, control_math_sha256="b" * 64,
                    physics_dt=.005, policy_dt=.02, actor_dim=35, critic_dim=81,
                    action_dim=6, history_length=1,
                    policy_architecture=dict(actor_hidden_dims=[256, 128, 64],
                                             critic_hidden_dims=[256, 128, 64], activation="elu",
                                             scope="unchanged_actor35_critic81_six_actions"),
                    position_iterations=64, velocity_iterations=32, closure_gap_termination_m=.003)
    contracts = {}
    rows = []
    for domain in DOMAINS:
        worker = dict(physical, target_num_envs=4096, learning_rate=1e-4,
                      learning_rate_schedule="fixed", resume_critic_warmup_updates=10,
                      task_campaign=dict(campaign_id=CAMPAIGN_ID, domain=domain,
                                         updates=20000, clock="domain_actor_updates"))
        path = source / "contracts" / (domain + ".json")
        write(path, worker)
        contracts[domain] = path
        rows.append(dict(name=domain, updates=20000, contract=str(path.relative_to(source)), sha256=joint.digest(path)))
    campaign = dict(campaign_id=CAMPAIGN_ID, total_ceiling=121000, historical_consumed_updates=1000,
                    historical_actor_consumed_updates=980, domains=rows,
                    rollback_refunds_consumed_updates=False, automatic_promotion=False,
                    domain_completion=dict(schedule=dict(end_actor_updates={d: 15000 for d in DOMAINS})))
    campaign_path = source / "campaign.json"
    write(campaign_path, campaign)
    manifest = dict(schema=joint.SCHEMA, lineage_id="source026-joint-v1",
                    source_root=str(source), run_root=str(run),
                    campaign_plan=dict(file="campaign.json", sha256=joint.digest(campaign_path)),
                    recovery_entry=dict(contract_file="contracts/recovery.json",
                                        contract_sha256=joint.digest(contracts["recovery"]), actor_endpoint=15000),
                    domains=[dict(name=d, parent_domain=p, contract_file=f"contracts/{d}.json",
                                  contract_sha256=joint.digest(contracts[d]), critic_warmup_updates=10)
                             for d, p in joint.CHAIN])
    path = source / "joint_plan.json"
    write(path, manifest)
    binding, _ = joint.load_plan(path, joint.digest(path), source, run)
    result = SimpleNamespace(source=source, run=run, contracts=contracts, manifest=manifest,
                             campaign=campaign, path=path, binding=binding)
    result.attempts = {d: dict(domain=d, consumed_updates=n, actor_updates=n - 10)
                       for d, n in (("flat", 10000), ("recovery", 15000), ("speed", 5000),
                                    ("terrain", 2000), ("jump", 1000))}
    result.rec = checkpoint(result, "recovery", count=26000, task=15000, lineage=25000, final=True)
    result.rec_record = dict(status="domain_schedule_finished_unassessed", candidate_checkpoint=result.rec["path"],
                             candidate_checkpoint_sha256=result.rec["sha256"],
                             schedule_completion=dict(reason="iteration_endpoint", end_actor_update=15000,
                                                      actual_actor_update=15000))
    result.entry = joint.admit_entry(binding, result.rec, result.rec_record)
    return result


def checkpoint(f, domain, *, count, task, lineage, final=False, meta=None, warmup=0, suffix="001",
               actor_consumed=None):
    path = f.run / domain / ("attempt_" + suffix) / "train" / ("model_final.pt" if final else "model.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = json.loads(f.contracts[domain].read_text())
    contract_sha = joint.digest(f.contracts[domain])
    algorithm = joint._native_learning_template(contract)
    # Populate native Adam's lazy saved states directly; no optimization is performed.
    for group in algorithm.optimizer.param_groups:
        for parameter in group["params"]:
            algorithm.optimizer.state[parameter] = dict(step=torch.tensor(4.),
                                                        exp_avg=torch.zeros_like(parameter),
                                                        exp_avg_sq=torch.zeros_like(parameter))
    saved = algorithm.save()
    infos = dict(contract_sha256=contract_sha, successful_updates_total=count, learning_lineage_updates=lineage,
                 stage_actor_update=lineage - 10, task_actor_update=task,
                 actor_updates_consumed=count - 100 if actor_consumed is None else actor_consumed,
                 critic_warmup_remaining=warmup, task_campaign=contract["task_campaign"],
                 asset_manifest_sha256=contract["asset_manifest_sha256"], control_math_sha256=contract["control_math_sha256"],
                 rng_state=torch.get_rng_state(), env_rng_state=torch.Generator().get_state())
    if meta is not None:
        infos["integrated_lineage"] = deepcopy(meta)
    saved["infos"] = infos
    torch.save(saved, path)
    path.with_suffix(".contract.json").write_bytes(f.contracts[domain].read_bytes())
    seal = dict(schema="chassis-learning-checkpoint-v1", sealed=True, checkpoint_sha256=joint.digest(path),
                contract_sha256=contract_sha, learning_updates=count, learning_lineage_updates=lineage,
                stage_actor_update=infos["stage_actor_update"], task_actor_update=task,
                actor_updates_consumed=infos["actor_updates_consumed"], task_campaign=contract["task_campaign"])
    if meta is not None:
        seal["integrated_lineage"] = deepcopy(meta)
    write(path.with_suffix(".checkpoint.json"), seal)
    completion = dict(status="completed", parent_updates=count - 10, successful_updates=10, task_actor_update=task)
    if meta is not None:
        completion["integrated_lineage"] = deepcopy(meta)
    if final:
        write(path.parent / "completion.json", completion)
    return dict(path=str(path), sha256=joint.digest(path))


def metadata(f, domain, *, joint_actor=2, domain_actor=2, complete=False):
    return dict(lineage_id=f.manifest["lineage_id"], plan_sha256=f.binding.sha256,
                origin_checkpoint_sha256=f.rec["sha256"], domain=domain,
                joint_actor_updates=joint_actor, domain_actor_updates=domain_actor, domain_complete=complete)


def edit_seal(reference, mutate):
    path = Path(reference["path"]).with_suffix(".checkpoint.json")
    value = json.loads(path.read_text())
    mutate(value)
    write(path, value)


def test_first_speed_explicitly_rebases_rec_and_preserves_old_branch_charges(fixture):
    f = fixture
    selection = joint.select_learning_source(f.binding, "speed", f.entry)
    assert selection["checkpoint"] == f.rec
    assert selection["mode"] == "rebase_from_normal_REC"
    assert selection["fresh_joint_actor_updates"] == selection["fresh_domain_actor_updates"] == 0
    assert selection["learning_lineage_updates"] == 25000
    before = deepcopy(f.attempts)
    allowance = joint.remaining_budget(f.binding, selection, f.attempts)
    assert allowance["consumed_updates"] == 34000
    assert allowance["historical_and_joint_domain_consumed_updates"] == 5000
    assert allowance["maximum_actor_updates"] == 14990
    assert allowance["charged_critic_updates"] == 10
    assert allowance["requested_total_updates"] == 15000
    assert allowance["fresh_domain_actor_updates"] == 0
    assert allowance["rollback_refunds_consumed_updates"] is False
    assert f.attempts == before


def test_fault_resume_latest_joint_own_and_partial_critic_only(fixture):
    f = fixture
    first = checkpoint(f, "speed", count=34004, task=0, lineage=25004,
                       meta=metadata(f, "speed", joint_actor=0, domain_actor=0), warmup=6)
    second = checkpoint(f, "speed", count=34012, task=2, lineage=25012,
                        meta=metadata(f, "speed"), warmup=0, suffix="002")
    previous = joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[first])
    assert previous["critic_warmup_remaining"] == 6
    assert previous["fresh_joint_actor_updates"] == 0
    selection = joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[second, first],
                                             previous_selection=previous)
    assert selection["checkpoint"] == second
    assert selection["fresh_joint_actor_updates"] == selection["fresh_domain_actor_updates"] == 2
    f.attempts["joint_speed"] = dict(domain="speed", consumed_updates=12, actor_updates=2)
    allowance = joint.remaining_budget(f.binding, selection, f.attempts, requested_actor_updates=7)
    assert allowance["charged_critic_updates"] == 0
    assert allowance["requested_total_updates"] == 7
    assert allowance["domain_remaining_charged_updates"] == 14988


@pytest.mark.parametrize("domain,parent_domain,joint_actor", [("terrain", "speed", 100), ("jump", "terrain", 200),
                                                            ("stepjump", "jump", 300)])
def test_inherit_completed_joint_parent_into_fresh_domain(fixture, domain, parent_domain, joint_actor):
    f = fixture
    meta = metadata(f, parent_domain, joint_actor=joint_actor, domain_actor=100, complete=True)
    parent = checkpoint(f, parent_domain, count=34000, task=100, lineage=25000 + joint_actor + 30,
                        meta=meta, final=True)
    record = dict(status="joint_domain_finished_unassessed", candidate_checkpoint=parent["path"],
                  candidate_checkpoint_sha256=parent["sha256"])
    selected = joint.select_learning_source(f.binding, domain, f.entry, parent=parent, parent_record=record)
    assert selected["checkpoint"] == parent
    assert selected["fresh_joint_actor_updates"] == joint_actor
    assert selected["fresh_domain_actor_updates"] == 0
    assert selected["critic_warmup_remaining"] == 10


@pytest.mark.parametrize("change", ["not_sealed", "rec_not_endpoint", "record_not_final", "completion_running",
                                    "seal_clock_mismatch", "source_hash", "root_escape"])
def test_rec_entry_rejects_unfinished_unsealed_or_unbound_sources(fixture, tmp_path, change):
    f = fixture
    ref, record = deepcopy(f.rec), deepcopy(f.rec_record)
    if change == "not_sealed":
        edit_seal(ref, lambda s: s.update(sealed=False))
    elif change == "rec_not_endpoint":
        record["schedule_completion"]["actual_actor_update"] = 14999
    elif change == "record_not_final":
        record["candidate_checkpoint"] = str(f.run / "recovery" / "other.pt")
    elif change == "completion_running":
        path = Path(ref["path"]).parent / "completion.json"
        value = json.loads(path.read_text())
        value["status"] = "running"
        write(path, value)
    elif change == "seal_clock_mismatch":
        edit_seal(ref, lambda s: s.update(task_actor_update=14999))
    elif change == "source_hash":
        ref["sha256"] = "0" * 64
    else:
        outside = tmp_path / "outside.pt"
        outside.write_bytes(Path(ref["path"]).read_bytes())
        ref["path"] = str(outside)
    with pytest.raises(ValueError):
        joint.admit_entry(f.binding, ref, record)


@pytest.mark.parametrize("bad", ["wrong_lineage", "wrong_origin", "wrong_plan", "clock_mismatch", "seal_meta_mismatch",
                                 "critic_extra", "legacy_own"])
def test_resume_rejects_cross_lineage_or_inconsistent_clocks(fixture, bad):
    f = fixture
    meta = metadata(f, "speed")
    if bad == "wrong_lineage":
        meta["lineage_id"] = "other-lineage"
    elif bad == "wrong_origin":
        meta["origin_checkpoint_sha256"] = "d" * 64
    elif bad == "wrong_plan":
        meta["plan_sha256"] = "d" * 64
    elif bad == "clock_mismatch":
        meta["domain_actor_updates"] = 3
    ref = checkpoint(f, "speed", count=34012, task=2, lineage=25012,
                     meta=None if bad == "legacy_own" else meta, warmup=11 if bad == "critic_extra" else 0)
    if bad == "seal_meta_mismatch":
        edit_seal(ref, lambda s: s["integrated_lineage"].update(joint_actor_updates=3))
    with pytest.raises(ValueError):
        joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[ref])


def test_latest_checkpoint_cannot_regress_or_replace_same_clock(fixture):
    f = fixture
    older = checkpoint(f, "speed", count=34011, task=1, lineage=25011,
                       meta=metadata(f, "speed", joint_actor=1, domain_actor=1))
    newer = checkpoint(f, "speed", count=34012, task=2, lineage=25012,
                       meta=metadata(f, "speed"), suffix="002")
    previous = joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[newer])
    with pytest.raises(ValueError, match="regressed"):
        joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[older], previous_selection=previous)
    replacement = checkpoint(f, "speed", count=34012, task=2, lineage=25012,
                             meta=metadata(f, "speed"), warmup=1, suffix="003")
    with pytest.raises(ValueError, match="fork"):
        joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[newer, replacement])


def test_child_rejects_uncompleted_joint_parent(fixture):
    f = fixture
    parent = checkpoint(f, "speed", count=34012, task=2, lineage=25012,
                        meta=metadata(f, "speed"), final=True)
    record = dict(status="joint_domain_finished_unassessed", candidate_checkpoint=parent["path"],
                  candidate_checkpoint_sha256=parent["sha256"])
    with pytest.raises(ValueError, match="completion"):
        joint.select_learning_source(f.binding, "terrain", f.entry, parent=parent, parent_record=record)


def test_completed_joint_own_is_not_reopened_with_unused_cap(fixture):
    f = fixture
    own = checkpoint(f, "speed", count=34012, task=2, lineage=25012,
                     meta=metadata(f, "speed", complete=True), final=True)
    selected = joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[own])
    f.attempts["joint_speed"] = dict(domain="speed", consumed_updates=12, actor_updates=2)
    allowance = joint.remaining_budget(f.binding, selected, f.attempts)
    assert allowance["domain_remaining_charged_updates"] == 14988
    assert not allowance["admitted"]
    assert allowance["requested_total_updates"] == 0


def test_remaining_budget_rejects_refund_overrun_and_forged_selection(fixture):
    f = fixture
    selected = joint.select_learning_source(f.binding, "speed", f.entry)
    with pytest.raises(ValueError, match="exceeds"):
        joint.remaining_budget(f.binding, selected, f.attempts, requested_actor_updates=14991)
    too_much = deepcopy(f.attempts)
    too_much["speed"]["consumed_updates"] = 20001
    with pytest.raises(ValueError, match="budget"):
        joint.remaining_budget(f.binding, selected, too_much)
    with pytest.raises(ValueError, match="precedes"):
        joint.remaining_budget(f.binding, selected, {})
    forged = deepcopy(selected)
    forged["critic_warmup_remaining"] = 0
    with pytest.raises(ValueError, match="authenticated"):
        joint.remaining_budget(f.binding, forged, f.attempts)
    exhausted = deepcopy(f.attempts)
    exhausted["speed"]["consumed_updates"] = 19995
    allowance = joint.remaining_budget(f.binding, selected, exhausted)
    assert not allowance["admitted"]
    assert allowance["requested_total_updates"] == allowance["charged_critic_updates"] == 0


@pytest.mark.parametrize("field,value", [("total_ceiling", 121001), ("historical_consumed_updates", 999),
                                      ("rollback_refunds_consumed_updates", True)])
def test_frozen_campaign_caps_and_no_refund_cannot_change(fixture, field, value):
    f = fixture
    campaign = deepcopy(f.campaign)
    campaign[field] = value
    write(f.source / "campaign.json", campaign)
    manifest = deepcopy(f.manifest)
    manifest["campaign_plan"]["sha256"] = joint.digest(f.source / "campaign.json")
    write(f.path, manifest)
    with pytest.raises(ValueError, match="caps"):
        joint.load_plan(f.path, joint.digest(f.path), f.source, f.run)


def test_plan_and_contract_bytes_are_rechecked_at_each_admission(fixture):
    f = fixture
    f.contracts["speed"].write_text(f.contracts["speed"].read_text() + " ")
    with pytest.raises(ValueError, match="SHA"):
        joint.select_learning_source(f.binding, "speed", f.entry)


@pytest.mark.parametrize("solver,value", [("position_iterations", 48), ("velocity_iterations", 16),
                                       ("closure_gap_termination_m", .01)])
def test_resigned_worker_cannot_change_native_physical_parameters(fixture, solver, value):
    f = fixture
    worker = json.loads(f.contracts["speed"].read_text())
    worker[solver] = value
    write(f.contracts["speed"], worker)
    manifest = deepcopy(f.manifest)
    manifest["domains"][0]["contract_sha256"] = joint.digest(f.contracts["speed"])
    write(f.path, manifest)
    with pytest.raises(ValueError, match="physical/policy identity"):
        joint.load_plan(f.path, joint.digest(f.path), f.source, f.run)


def test_real_full_state_rejects_missing_adam_and_rng(fixture):
    f = fixture
    path = Path(f.rec["path"])
    saved = torch.load(path, weights_only=True)
    del saved["optimizer_state_dict"]["state"][0]
    torch.save(saved, path)
    ref = dict(path=str(path), sha256=joint.digest(path))
    edit_seal(ref, lambda s: s.update(checkpoint_sha256=ref["sha256"]))
    record = deepcopy(f.rec_record)
    record["candidate_checkpoint_sha256"] = ref["sha256"]
    with pytest.raises(ValueError, match="Adam"):
        joint.admit_entry(f.binding, ref, record)
    saved["infos"].pop("rng_state")
    torch.save(saved, path)
    ref["sha256"] = joint.digest(path)
    edit_seal(ref, lambda s: s.update(checkpoint_sha256=ref["sha256"]))
    record["candidate_checkpoint_sha256"] = ref["sha256"]
    with pytest.raises(ValueError, match="RNG"):
        joint.admit_entry(f.binding, ref, record)


def resign_rec(f, saved):
    path = Path(f.rec["path"])
    torch.save(saved, path)
    reference = dict(path=str(path), sha256=joint.digest(path))
    edit_seal(reference, lambda s: s.update(checkpoint_sha256=reference["sha256"]))
    record = deepcopy(f.rec_record)
    record["candidate_checkpoint_sha256"] = reference["sha256"]
    return reference, record


@pytest.mark.parametrize("bad", ["arbitrary_same_count_keys", "wrong_native_shape", "wrong_native_dtype",
                                 "wrong_adam_shape", "negative_adam_variance"])
def test_resigned_real_checkpoint_must_match_native_model_and_adam(fixture, bad):
    f = fixture
    saved = torch.load(f.rec["path"], weights_only=True)
    if bad == "arbitrary_same_count_keys":
        saved["actor_state_dict"] = {"distribution.std_param": torch.ones(6),
                                     **{f"actor.{i}": torch.ones(2) for i in range(8)}}
        saved["critic_state_dict"] = {f"critic.{i}": torch.ones(2) for i in range(8)}
        for state in saved["optimizer_state_dict"]["state"].values():
            state.update(exp_avg=torch.zeros(2), exp_avg_sq=torch.zeros(2))
    elif bad == "wrong_native_shape":
        saved["actor_state_dict"]["mlp.0.weight"] = torch.zeros(256, 34)
    elif bad == "wrong_native_dtype":
        saved["critic_state_dict"]["mlp.0.weight"] = saved["critic_state_dict"]["mlp.0.weight"].double()
    elif bad == "wrong_adam_shape":
        saved["optimizer_state_dict"]["state"][1]["exp_avg"] = torch.zeros(256, 34)
    else:
        saved["optimizer_state_dict"]["state"][1]["exp_avg_sq"].fill_(-1)
    reference, record = resign_rec(f, saved)
    with pytest.raises(ValueError, match="native|Adam"):
        joint.admit_entry(f.binding, reference, record)


def test_native_template_uses_real_architecture_without_learning_or_rng_changes(fixture, monkeypatch):
    from rsl_rl.algorithms import PPO

    saved = torch.load(fixture.rec["path"], weights_only=True)
    assert saved["actor_state_dict"]["mlp.0.weight"].shape == (256, 35)
    assert saved["actor_state_dict"]["mlp.2.weight"].shape == (128, 256)
    assert saved["actor_state_dict"]["mlp.4.weight"].shape == (64, 128)
    assert saved["actor_state_dict"]["mlp.6.weight"].shape == (6, 64)
    assert saved["critic_state_dict"]["mlp.0.weight"].shape == (256, 81)
    assert saved["critic_state_dict"]["mlp.6.weight"].shape == (1, 64)

    def forbidden(*args, **kwargs):
        pytest.fail("Read-only admission attempted learning or rollout collection")

    monkeypatch.setattr(PPO, "update", forbidden)
    monkeypatch.setattr(PPO, "act", forbidden)
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden)
    before = torch.get_rng_state().clone()
    joint.admit_entry(fixture.binding, fixture.rec, fixture.rec_record)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("joint_actor,lineage,actor_consumed", [
    (100000, 25012, 33912),  # Fresh clock cannot borrow the historical charged total.
    (5000, 25012, 33912),    # Actor-consumption headroom alone is not model learning.
    (101, 25111, 26000),    # Model-learning headroom alone is not actor consumption.
    (0, 24999, 33912),      # The model cannot precede its sealed REC origin.
    (0, 25012, 25899),      # Neither can its actor-consumption counter.
])
def test_fresh_joint_clock_is_bounded_by_both_actual_origin_increments(fixture, joint_actor, lineage, actor_consumed):
    f = fixture
    task = min(2, joint_actor)
    reference = checkpoint(f, "speed", count=34012, task=task, lineage=lineage,
                           actor_consumed=actor_consumed,
                           meta=metadata(f, "speed", joint_actor=joint_actor, domain_actor=task))
    with pytest.raises(ValueError, match="sealed REC origin"):
        joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[reference])


def test_entry_admission_does_not_cache_or_trust_origin_seal_clocks(fixture):
    f = fixture
    own = checkpoint(f, "speed", count=34012, task=2, lineage=25012, meta=metadata(f, "speed"))
    edit_seal(f.rec, lambda s: s.update(learning_lineage_updates=24000))
    with pytest.raises(ValueError, match="seal and full-state clocks"):
        joint.select_learning_source(f.binding, "speed", f.entry, own_candidates=[own])
