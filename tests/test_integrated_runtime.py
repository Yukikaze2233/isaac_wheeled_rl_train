"""Pinned local joint runtime candidates; real CPU seals, no rollout or learning."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_integrated_lineage import fixture as original_lineage_fixture, checkpoint, metadata, edit_seal, write
from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis.integrated_sampling import load_sources
from wheeled_tasks.chassis.ordinary_posture_curriculum import OrdinaryPostureCurriculum
from wheeled_tasks.chassis.recovery_curriculum import pose_probabilities
from wheeled_tasks.chassis.recovery_prefix_curriculum import _pose_settings, release_seconds, validate_prefix_curriculum
from wheeled_tasks.chassis.terrain_runout import validate_runout
from wheeled_tasks.chassis.skill_commands import sample_planar_command

ROOT = Path(__file__).resolve().parents[1]
SCUT = ROOT / "references" / "scut_curriculum_b8ff79f3"


@pytest.fixture(scope="module")
def sources():
    return load_sources(ROOT)


@pytest.fixture
def lineage_fixture(original_lineage_fixture, sources):
    """Bind real frozen source worker contracts to the real native CPU model seals."""
    f = original_lineage_fixture
    for domain, path in f.contracts.items():
        file = runtime.sampling.SOURCE_BINDINGS[domain][0]
        path.write_bytes((ROOT / file).read_bytes())
    for row in f.campaign["domains"]:
        row["sha256"] = lineage.digest(f.contracts[row["name"]])
    write(f.source / "campaign.json", f.campaign)
    f.manifest["campaign_plan"]["sha256"] = lineage.digest(f.source / "campaign.json")
    f.manifest["recovery_entry"]["contract_sha256"] = lineage.digest(f.contracts["recovery"])
    for row in f.manifest["domains"]:
        row["contract_sha256"] = lineage.digest(f.contracts[row["name"]])
    write(f.path, f.manifest)
    f.binding, _ = lineage.load_plan(f.path, lineage.digest(f.path), f.source, f.run)
    f.rec = checkpoint(f, "recovery", count=26000, task=15000, lineage=25000, final=True)
    f.rec_record.update(candidate_checkpoint=f.rec["path"], candidate_checkpoint_sha256=f.rec["sha256"])
    f.entry = lineage.admit_entry(f.binding, f.rec, f.rec_record)
    return f


def candidate(f, sources, domain, selection):
    return runtime.apply(sources[domain], ROOT, SCUT, binding=f.binding, selection=selection)


def seal_clocks(reference, clocks):
    """Use real fixture checkpoint bytes; only saved metadata is added, with a new seal."""
    path = Path(reference["path"])
    saved = torch.load(path, weights_only=True)
    saved["infos"]["integrated_runtime_clocks"] = deepcopy(clocks)
    torch.save(saved, path)
    result = dict(path=str(path), sha256=lineage.digest(path))
    edit_seal(result, lambda s: s.update(checkpoint_sha256=result["sha256"], integrated_runtime_clocks=deepcopy(clocks)))
    return result


def terminal(f, current, delta, *, suffix="001"):
    domain = current["domain"]
    state = runtime.advance_clock_document(current, delta)
    joint_actor, domain_actor = state["joint_actor_updates"], state["domain_actor_updates"]
    stage_count = runtime.STAGES.index(domain) + 1
    learning = 25000 + joint_actor + 10 * stage_count
    reference = checkpoint(f, domain, count=34000 + joint_actor + 10 * stage_count,
                           task=domain_actor, lineage=learning, final=True, suffix=suffix,
                           meta=metadata(f, domain, joint_actor=joint_actor, domain_actor=domain_actor, complete=True))
    reference = seal_clocks(reference, state)
    record = dict(status="joint_domain_finished_unassessed", candidate_checkpoint=reference["path"],
                  candidate_checkpoint_sha256=reference["sha256"])
    return reference, record


def first(f, sources):
    selected = lineage.select_learning_source(f.binding, "speed", f.entry)
    return candidate(f, sources, "speed", selected)


def test_new_speed_mature_recovery_and_zero_new_difficulty(lineage_fixture, sources):
    f = lineage_fixture
    before = deepcopy(sources["speed"])
    config = first(f, sources)
    audit = config["integrated_runtime_candidate"]
    assert sources["speed"] == before
    assert config["recovery_training"]["profiles_file"] == "contracts/v6_recovery_profiles_v2.json"
    expected_rec = deepcopy(sources["recovery"]["recovery_training"])
    prefix = config["recovery_training"]["prefix_curriculum"]
    expected_rec["prefix_curriculum"] = deepcopy(prefix)
    assert config["recovery_training"] == expected_rec
    assert prefix["origin_actor_update"] == 0
    validate_prefix_curriculum(prefix)
    for update in (0, 10, 10000):
        assert pose_probabilities(_pose_settings(prefix), update) == pytest.approx((.2, .3, .5))
        assert release_seconds(prefix, update) == .2
    assert config["recovery_training"]["max_script_seconds"] == 8.
    assert config["recovery_training"]["blend_seconds"] == .2
    assert config["recovery_training"]["stable_seconds"] == 1.
    assert config["recovery_hold_cost"] == sources["recovery"]["recovery_hold_cost"]
    assert config["entropy_coef"] == 0.
    assert config["locomotion_difficulty"]["origin_actor_update"] == 0
    assert audit["component_runtime"]["speed"]["nonstandard_start"]["origin_actor_update"] == 0
    env = SimpleNamespace(cfg=config, task_actor_update=0)
    assert runtime.component_actor_updates(env, "speed") == 0
    assert runtime.component_actor_updates(env, "recovery") == 15000
    assert runtime.row_difficulty_envelope(env, "forward_5")["vx_max_m_s"] == 2.
    posture, clock = runtime.row_settings(env, "stand_305mm", "ordinary_posture_curriculum")
    assert posture == sources["recovery"]["ordinary_posture_curriculum"] and clock == 15000
    curriculum = OrdinaryPostureCurriculum(1, "cpu", posture, torch.tensor([True]))
    assert curriculum.amplitudes(clock)[1:] == pytest.approx((.15, .25))


def test_actual_joint_terminals_carry_all_rehearsal_clocks_forward(lineage_fixture, sources):
    f = lineage_fixture
    config = first(f, sources)
    for domain, learned in (("terrain", 360), ("jump", 240), ("stepjump", 80)):
        source_state = config["integrated_runtime_candidate"]["clocks_at_source"]
        reference, record = terminal(f, source_state, learned)
        selection = lineage.select_learning_source(f.binding, domain, f.entry, parent=reference, parent_record=record)
        config = candidate(f, sources, domain, selection)
        audit = config["integrated_runtime_candidate"]
        clocks = audit["clocks_at_source"]
        assert clocks["domain_actor_updates"] == clocks["component_actor_updates"][domain] == 0
        assert clocks["component_actor_updates"]["speed"] == clocks["joint_actor_updates"]
        assert clocks["component_actor_updates"]["recovery"] == 15000 + clocks["joint_actor_updates"]
        assert audit["learning_source"]["checkpoint"] == reference
        assert config["terrain_route_runout"] == sources["terrain"]["terrain_route_runout"]
        assert config["release_stop"] == sources["speed"]["release_stop"]
        assert config["episode_seconds"] == 32.
        assert config["evaluation"] == sources[domain]["evaluation"]
        assert config["stages"] == sources[domain]["stages"]
        assert config["task_campaign"] == sources[domain]["task_campaign"]
        assert len(config["scut_reuse_candidate"]["mapped_groups"]) == 32
        saved_source = runtime.reuse.read_sources(SCUT)["rough_dash"]
        override = saved_source["terrain_command_overrides"]["low_speed_stair_for_rm"]
        bindings = audit["scut_terrain_bindings"]
        assert set(bindings) == set(runtime.SMALL_STAIR_NAMES)
        for name in runtime.SMALL_STAIR_NAMES:
            spec = config["skill_specs"][name]
            assert bindings[name] == dict(override="low_speed_stair_for_rm", vx_interval=0)
            assert spec["planar_command_sampling"]["vx_range_m_s"] == override["lin_vel_x"][0]
            assert spec["planar_command_sampling"]["yaw_range_rad_s"] == override["ang_vel_z_non_heading"]
            assert spec["height_sampling"]["range_m"] == [
                max(override["height_range"][0][0], sources[domain]["height_range_m"][0]),
                min(override["height_range"][0][1], sources[domain]["height_range_m"][1])]
            assert spec.get("request_events") == sources["terrain"]["skill_specs"][name].get("request_events")
            native_geometry = next(row for row in sources["terrain"]["scene_groups"] if row["name"] == name)
            actual_geometry = next(row for row in config["scene_groups"] if row["name"] == name)
            assert {k: v for k, v in actual_geometry.items() if k != "fraction"} == {
                k: v for k, v in native_geometry.items() if k != "fraction"}
        assert audit["runtime_ready"] is False and audit["qualification"] is None
        assert audit["PPO_updates"] == 0
        if domain in ("jump", "stepjump"):
            assert config["jump_landing"] == sources["jump"]["jump_landing"]
            assert config["jump_cycle_reset_on_accepted"] is True
            # Optional native routing may be installed locally while deployment is pending.
            observed_runout = validate_runout(config)
            assert observed_runout is None or observed_runout == sources["terrain"]["terrain_route_runout"]
            assert "terrain_runout" in audit["pending_native_call_sites"]
            env = SimpleNamespace(cfg=config, task_actor_update=0)
            terrain_group = next(name for name in config["skill_specs"] if name.startswith("small_stairs_"))
            assert runtime.row_difficulty_envelope(env, terrain_group) is None
    assert clocks["component_actor_updates"] == {"recovery": 15680, "speed": 680, "terrain": 320,
                                                "jump": 80, "stepjump": 0}
    env = SimpleNamespace(cfg=config, task_actor_update=7)
    assert runtime.component_actor_updates(env, "speed") == 687
    assert runtime.component_actor_updates(env, "jump") == 87
    spec = config["skill_specs"]["small_stairs_up_005mm"]
    generator = torch.Generator().manual_seed(262628)
    commands = torch.tensor(spec["command"]).expand(1000, -1).clone()
    sample_planar_command(commands, spec, lambda n: torch.rand(n, generator=generator))
    assert (commands[:, 0] >= .5).all() and (commands[:, 0] <= 1.5).all()
    assert (commands[:, 0] > 1.).any() and (commands[:, 1].abs() > .05).any()
    assert (commands[:, 1].abs() <= .1).all()


def test_same_joint_resume_restores_component_state_without_restarting(lineage_fixture, sources):
    f = lineage_fixture
    initial = first(f, sources)["integrated_runtime_candidate"]["clocks_at_source"]
    clocks = runtime.advance_clock_document(initial, 12)
    reference = checkpoint(f, "speed", count=34022, task=12, lineage=25022,
                           meta=metadata(f, "speed", joint_actor=12, domain_actor=12))
    reference = seal_clocks(reference, clocks)
    selected = lineage.select_learning_source(f.binding, "speed", f.entry, own_candidates=[reference])
    config = candidate(f, sources, "speed", selected)
    assert config["integrated_runtime_candidate"]["clocks_at_source"] == clocks
    env = SimpleNamespace(cfg=config, task_actor_update=13)
    assert runtime.component_actor_updates(env, "speed") == 13
    assert runtime.component_actor_updates(env, "recovery") == 15013
    env.task_actor_update = 11
    with pytest.raises(ValueError, match="regressed"):
        runtime.component_actor_updates(env, "speed")


@pytest.mark.parametrize("bad", ["missing", "seal_mismatch", "joint_clock", "component_start", "component_value", "old_plan"])
def test_rejects_unsealed_invented_or_legacy_component_progress(lineage_fixture, sources, bad):
    f = lineage_fixture
    initial = first(f, sources)["integrated_runtime_candidate"]["clocks_at_source"]
    clocks = runtime.advance_clock_document(initial, 2)
    reference = checkpoint(f, "speed", count=34012, task=2, lineage=25012,
                           meta=metadata(f, "speed", complete=True), final=True)
    if bad != "missing":
        if bad == "joint_clock":
            clocks["joint_actor_updates"] = 5000
        elif bad == "component_start":
            clocks["component_start_joint_actor_updates"]["speed"] = 3563
        elif bad == "component_value":
            clocks["component_actor_updates"]["speed"] = 5000
        elif bad == "old_plan":
            clocks["plan_sha256"] = "f" * 64
        reference = seal_clocks(reference, clocks)
        if bad == "seal_mismatch":
            edit_seal(reference, lambda s: s["integrated_runtime_clocks"].update(joint_actor_updates=3))
    record = dict(status="joint_domain_finished_unassessed", candidate_checkpoint=reference["path"],
                  candidate_checkpoint_sha256=reference["sha256"])
    selection = lineage.select_learning_source(f.binding, "terrain", f.entry, parent=reference, parent_record=record)
    with pytest.raises(ValueError, match="component|Component"):
        candidate(f, sources, "terrain", selection)


@pytest.mark.parametrize("role", ["fixed_eval", "eval", "playback", "native_review", "preflight", "export", "qualification"])
def test_non_training_role_is_noop_without_sources_clocks_or_reference(role, sources):
    assert runtime.apply(sources["jump"], "/missing", "/missing", role=role) == sources["jump"]


def test_source_sha_references_and_special_samplers_remain_bound(lineage_fixture, sources):
    config = first(lineage_fixture, sources)
    audit = config["integrated_runtime_candidate"]
    assert audit["production_integration_audit"]["sha256"] == runtime.AUDIT[1]
    for pin in audit["referenced_files"].values():
        assert lineage.digest(ROOT / pin["file"]) == pin["sha256"]
    assert set(audit["scut_source_bindings"]) == {"rough_dash", "rough_rotation_stair"}
    for files in audit["scut_source_bindings"].values():
        for pin in files.values():
            assert lineage.digest(SCUT / pin["file"]) == pin["sha256"]
    for name, spec in config["skill_specs"].items():
        provenance = config["integrated_sampling_candidate"]["row_provenance"][name]
        if name not in config["scut_reuse_candidate"]["mapped_groups"]:
            assert spec == sources[provenance["component"]]["skill_specs"][provenance["source_name"]]


def test_old_native_default_clock_and_runtime_fields_are_unchanged(sources):
    env = SimpleNamespace(cfg=sources["speed"], task_actor_update=4)
    assert runtime.component_actor_updates(env, "any_unused_name") == 4
    assert runtime.row_settings(env, "unused", "ordinary_posture_curriculum") == (
        sources["speed"]["ordinary_posture_curriculum"], 4)


def test_edited_native_source_and_forged_selection_rejected(lineage_fixture, sources):
    f = lineage_fixture
    selection = lineage.select_learning_source(f.binding, "speed", f.entry)
    changed = deepcopy(sources["speed"])
    changed["physics_dt"] = .001
    with pytest.raises(ValueError, match="exact frozen"):
        candidate(f, {**sources, "speed": changed}, "speed", selection)
    forged = deepcopy(selection)
    forged["fresh_joint_actor_updates"] = 5000
    with pytest.raises(ValueError, match="actual source clocks"):
        candidate(f, sources, "speed", forged)


def test_selected_checkpoint_must_have_same_native_physical_identity(original_lineage_fixture, sources):
    f = original_lineage_fixture
    selection = lineage.select_learning_source(f.binding, "speed", f.entry)
    with pytest.raises(ValueError, match="physical/policy identity"):
        candidate(f, sources, "speed", selection)
