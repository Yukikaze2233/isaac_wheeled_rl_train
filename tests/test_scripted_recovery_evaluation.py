"""Explicit CPU fixtures, not approved V7 quotas or native recovery evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from support_scut_flat_scripted_recovery import config, rig, tick, ready
from support_flat_evaluation_runtime import stock_state
from support_flat_evaluation_setup import make_env
from support_flat_policy_evaluation import protocol as flat_protocol
from wheeled_tasks.chassis.flat_evaluation_runtime import BoundEvaluationInputs, canonical, sha, runtime_source_files
from wheeled_tasks.chassis.flat_evaluation_setup import FlatEvaluationSetup
from wheeled_tasks.chassis.flat_policy_evaluation import validate_protocol
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_tasks.chassis.training_profile import profile_identity
from wheeled_tasks.chassis.scripted_recovery_evaluation import (
    SCHEMA, WINDOW_SCHEMA, ROLE, EVENTS, RecoveryMeasurements, scene_groups, validate_binding)

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    p = flat_protocol()
    p.update(schema=SCHEMA, role=ROLE, mode="fixed_measurement", handoff_source="continuous_native_script",
        cases=[dict(case_id="recover", base_request=[.5, -.2, .305], profile="left_side_90", chain_type="fallen_full_chain"),
               dict(case_id="ordinary", base_request=[-.5, .2, .3], profile=None, chain_type="ordinary_reference")])
    return p


def window_protocol():
    from wheeled_tasks.chassis.flat_episode_selection import RECOVERY_WINDOW_SETTINGS
    return {**protocol(), 'schema': WINDOW_SCHEMA, 'horizon_seconds': 28.,
        'recovery_episode_window': deepcopy(RECOVERY_WINDOW_SETTINGS)}


def fixture_inputs(tmp_path):
    c, p = config(), protocol()
    contract, protocol_path, overlay, checkpoint = [tmp_path / name for name in (
        "contract.json", "protocol.json", "overlay.json", "cpu_actor.pt")]
    contract.write_bytes(canonical(c)); protocol_path.write_bytes(canonical(p))
    overlay.write_bytes(canonical(dict(num_envs=2, lane_case_ids=["recover", "ordinary"], record_diagnostics=True)))
    hashes = {name: sha((ROOT / name).read_bytes()) for name in runtime_source_files(c)}
    infos = {**profile_identity(c, hashes), "profile_source_sha256": hashes,
        "contract_sha256": sha(contract.read_bytes()), "asset_manifest_sha256": c["asset_manifest_sha256"],
        "control_math_sha256": c["control_math_sha256"], "stage": "foundation", "stage_actor_update": 3500,
        "global_actor_update_offset": 0, "global_actor_update": 3500, "actor_updates_consumed": 3521}
    torch.save(dict(infos=infos, actor_state_dict=deepcopy(stock_state())), checkpoint)
    return dict(root=ROOT, checkpoint=checkpoint, contract=contract, protocol=protocol_path,
        manifest=ROOT / c["asset_directory"] / "manifest.json", control=ROOT / c["control_math_source"], overlay=overlay)


def test_final_motion_recovery_projection_uses_protocol_quotas_and_original_controller():
    parent = json.loads((ROOT / 'configs/v7_formal_20261012/final_motion_review.json').read_bytes())
    original = deepcopy(parent)
    p = window_protocol()
    lanes = ['recover'] * 2 + ['ordinary'] * 2
    effective = validate_binding(p, parent, lanes, ROOT)
    assert parent == original
    assert effective['scene_groups'] == scene_groups(p, lanes)
    assert effective['recovery_training'] == parent['recovery_training']
    assert effective['formal_recovery_episode_window'] == parent['formal_recovery_episode_window']
    assert effective['formal_foundation_nominal_birth'] is True
    assert effective['formal_terrain']['type'] == 'plane'
    from wheeled_tasks.chassis.training_profile import validate_physx_solver_selection
    for change in ({'physx_solver_type': 0}, {'physx_external_forces_every_iteration': True},
                   {'scut_reward_mapping': {**effective['scut_reward_mapping'], 'undesired_link_names': []}}):
        with pytest.raises(ValueError):
            validate_physx_solver_selection({**effective, **change})
    for key in ('formal_small_rough_mix', 'formal_scripted_recovery_mix',
                'formal_flat_rehearsal', 'formal_flat_rehearsal_collider',
                'formal_dynamic_nonsteady_birth', 'formal_critic_adaptation'):
        assert key not in effective


def test_bound_protocol_preserves_contract_checkpoint_source_and_reaches_native_input_admission(tmp_path):
    paths = fixture_inputs(tmp_path)
    before = paths["contract"].read_bytes()
    bound = BoundEvaluationInputs.capture(**paths)
    assert bound.run_role == ROLE and bound.is_scripted_recovery and not bound.is_rough
    assert bound.authenticate()["actor_clocks"]["global_actor_update"] == 3500
    c = bound.evaluation_config()
    expected = deepcopy(bound.contract); expected.update(record_diagnostics=True, scene_groups=scene_groups(protocol(), ("recover", "ordinary")))
    assert c == expected and paths["contract"].read_bytes() == before
    setup = bound.build_setup()
    receipt = design_preflight(c, bound.control, ROOT, stage="foundation", role=ROLE,
        seed=17, num_envs=2, evaluation_setup=setup)
    assert receipt["recovery_training"]["status"] == "authenticated"
    assert receipt["runtime_expected"]["modules"]["recovery_training"]
    with pytest.raises(ValueError, match="evaluation protocol"):
        design_preflight(c, bound.control, ROOT, stage="foundation", role="source_flat_evaluation", seed=17, num_envs=2)
    with pytest.raises(ValueError, match="evaluation protocol"):
        design_preflight(c, bound.control, ROOT, stage="foundation", role=ROLE, seed=17, num_envs=2)
    spec = importlib.util.spec_from_file_location("recovery_eval_entry", ROOT / "scripts/evaluate_flat_policy.py")
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    actual, assets, identity = entry.native_inputs(NS(**paths))  # CPU-only, stops before Kit/owner/launch.
    assert actual.run_role == ROLE and assets and identity["parent_checkpoint_sha256"] == sha(paths["checkpoint"].read_bytes())
    infos = bound.load_checkpoint()["infos"]
    infos["profile_source_sha256"]["src/wheeled_tasks/chassis/env.py"] = "0" * 64
    infos.update(profile_identity(bound.contract, infos["profile_source_sha256"]))
    torch.save(dict(infos=infos, actor_state_dict=deepcopy(stock_state())), paths["checkpoint"])
    with pytest.raises(ValueError, match="source|runtime"):
        BoundEvaluationInputs.capture(**paths).authenticate()


def test_profile_and_chain_type_are_explicit_and_zero_torque_uses_bound_release():
    c, p = config(), protocol()
    for profile, chain in (("upright_release", "upright_zero_torque"), ("crouched_release", "crouched_zero_torque")):
        p["cases"][0].update(profile=profile, chain_type=chain)
        assert validate_binding(p, c, ("recover", "ordinary"), ROOT)["scene_groups"][0]["recovery_pose"] == profile
    p["cases"][0].update(profile="upright", chain_type="upright_zero_torque")
    with pytest.raises(ValueError, match="chain type"):
        validate_protocol(p)
    with pytest.raises(ValueError, match="protocol and role"):
        FlatEvaluationSetup(protocol(), ("recover", "ordinary"), 3500, 3500)
    with pytest.raises(ValueError, match="case order"):
        scene_groups(protocol(), ("ordinary", "recover"))


def test_extended_protocol_requires_the_matching_physical_window_and_preserves_legacy_binding():
    from wheeled_tasks.chassis.flat_episode_selection import RECOVERY_WINDOW_KEY, RECOVERY_WINDOW_SETTINGS
    c, p = config(), window_protocol()
    validate_protocol(p)
    with pytest.raises(ValueError, match='same window'):
        validate_binding(p, c, ('recover', 'ordinary'), ROOT)
    c[RECOVERY_WINDOW_KEY] = deepcopy(RECOVERY_WINDOW_SETTINGS)
    assert RECOVERY_WINDOW_KEY in validate_binding(p, c, ('recover', 'ordinary'), ROOT)
    with pytest.raises(ValueError, match='same window'):
        validate_binding(protocol(), c, ('recover', 'ordinary'), ROOT)
    for changed in ({'horizon_seconds': 20.}, {'recovery_episode_window': {}}, {'schema': 'unknown'}):
        with pytest.raises(ValueError):
            validate_protocol(dict(p, **changed))


def test_extended_native_loop_bound_covers_prelude_plus_the_full_actor_window(tmp_path):
    from wheeled_tasks.chassis.flat_episode_selection import RECOVERY_WINDOW_KEY, RECOVERY_WINDOW_SETTINGS
    from wheeled_tasks.chassis.flat_evaluation_loop import FlatEvaluationLoop
    paths = fixture_inputs(tmp_path)
    c = json.loads(paths['contract'].read_bytes())
    c[RECOVERY_WINDOW_KEY] = deepcopy(RECOVERY_WINDOW_SETTINGS)
    paths['contract'].write_bytes(canonical(c))
    paths['protocol'].write_bytes(canonical(window_protocol()))
    saved = torch.load(paths['checkpoint'], weights_only=False)
    hashes = {name: sha((ROOT / name).read_bytes()) for name in runtime_source_files(c)}
    saved['infos'].update(profile_identity(c, hashes), profile_source_sha256=hashes,
        contract_sha256=sha(paths['contract'].read_bytes()))
    torch.save(saved, paths['checkpoint'])
    bound = BoundEvaluationInputs.capture(**paths)
    bound.authenticate()
    loop = FlatEvaluationLoop(bound, device='cpu', limit_check=lambda: None)
    assert loop.max_steps == 1400 * bound.protocol['episodes_per_lane']


def test_extended_measurement_keeps_a_short_open_chain_censored(rig):
    from test_recovery_episode_window import extended
    extended(rig)
    meter = RecoveryMeasurements(window_protocol(), ('recover', 'ordinary'), rig.recovery.source_identity())
    extras = produced(rig)
    meter.accept(meter.prepare(extras, [0, 1], policy_interval_id=1))
    record = {'lane': 0, 'episode_id': 0, 'outcome': 'censored'}
    meter.close_records([record])
    window = record['scripted_recovery']['terminal_measurement']['episode_window']
    assert window['required_post_handoff_seconds'] == 20.
    assert window['first_actor_start_s'] == -1.
    assert not window['full_actor_window_observed']
    report = meter.report()
    assert report['schema'] == WINDOW_SCHEMA
    assert report['cases']['recover']['full_post_handoff_window_episodes'] == 0


def test_recovery_contact_overlay_retains_the_explicit_cases_and_surface():
    from wheeled_tasks.chassis.contact_generation_distance import UPPER_LINK_VERSION
    from wheeled_tasks.chassis.flat_evaluation_runtime import validate_overlay

    p = protocol()
    overlay = dict(num_envs=2, lane_case_ids=["recover", "ordinary"], record_diagnostics=True,
                   contact_generation_distance=UPPER_LINK_VERSION)
    assert validate_overlay(p, overlay) == overlay
    for change in ({"contact_generation_distance": "unknown"},
                   {"evaluation_surface": "plane_regression_v1"}):
        with pytest.raises(ValueError):
            validate_overlay(p, dict(overlay, **change))


def test_actual_native_lane_permutation_binds_commands_profiles_and_unchanged_clocks_rng():
    p = protocol(); setup = FlatEvaluationSetup(p, ("recover", "ordinary"), 3500, 3500, run_role=ROLE)
    e = make_env(setup)
    e.cfg = {**validate_binding(p, config(), setup.lane_case_ids, ROOT), "record_diagnostics": True, "seed": 17}
    e.clone_indices, e.scene_groups = [1, 0], ["ordinary", "recover"]
    e.recovery = NS(source_flat_handoff=True, pose_curriculum=None, profile_names=[None, "left_side_90"], enabled=torch.tensor([False, True]))
    before = e.generator.get_state().clone()
    setup.apply_before_first_reset(e)
    assert torch.equal(before, e.generator.get_state())
    assert (e.stage_actor_update, e.global_actor_update) == (3500, 3500)
    torch.testing.assert_close(e.flat_episode.commands.fixed_request_command3,
        torch.tensor([[-.5, 0., .2], [.5, 0., -.2]]))
    assert e.startup_report["flat_evaluation_setup"]["actual_lane_case_ids"] == ["ordinary", "recover"]


def observer(rig):
    rig.episode_length_buf.zero_()  # The CPU reset excerpt omits the earlier counter clear.
    return RecoveryMeasurements(protocol(), ("recover", "ordinary"), rig.recovery.source_identity())


def produced(rig):
    _, _, _, extras = tick(rig)
    return extras


def test_direct_takeover_reports_real_handover_without_requiring_a_blend(rig):
    e = rig
    meter = observer(e)
    extras = produced(e)
    meter.accept(meter.prepare(extras, [0, 1], policy_interval_id=1))
    row = deepcopy(meter.active[0])
    row["ever_completed"]["rl_entered"] = True
    row["last_diagnostics"].update(handover_count=1, ever_blended=False,
        followup_s=3.7, true_success=True, rl_stable_s=.02)
    identity = deepcopy(meter.source_identity)
    identity["takeover_version"] = "cpu_direct_fixture"
    identity["effective_recovery_settings"].update(blend_seconds=0.,
        direct_takeover={"version": "cpu_direct_fixture"})
    measurement = meter.terminal_measurement(row, identity)
    assert measurement["takeover_mode"] == "direct"
    assert measurement["single_handover_observed"]
    assert not measurement["blend_observed"]
    assert not measurement["legacy_blend_and_hold_classifier_applicable"]
    assert measurement["endpoint_continuous_followup_s"] == 3.7
    assert measurement["endpoint_producer_followup_complete"]
    assert not measurement["qualification"]
    assert meter.terminal_measurement(row, meter.source_identity)["legacy_blend_and_hold_classifier_applicable"]
    identity["effective_recovery_settings"]["blend_seconds"] = .2
    with pytest.raises(ValueError, match="inconsistent producer identity"):
        meter.terminal_measurement(row, identity)


def test_followup_observation_counts_predicate_losses_and_command_coincidence(rig):
    e, meter = rig, observer(rig)
    ready(e)
    # Synthetic predicate/command endpoints exercise measurement only. They do
    # not demonstrate physical stability or a causal effect of command changes.
    samples = ((.4, .5, True), (.42, .5, True), (0., .5, True),
               (.02, -.5, True), (0., .5, True), (.02, .5, True),
               (0., -.5, False), (0., -.5, True))
    for interval, (duration, vx, valid) in enumerate(samples, 1):
        extras = produced(e)
        data = extras["diagnostics"]
        data["recovery_followup_s"][0] = duration
        data["effective_command"][0] = torch.tensor([vx, 0., .305])
        data["measurement_valid_mask"][0] = valid
        if interval == 1:
            extras["recovery_transition"]["events"]["rl_entered"][0] = True
            extras["recovery_transition"]["occurrence_ids"]["rl_entered"][0] = 4
        pending = meter.prepare(extras, [0, 1], policy_interval_id=interval)
        assert meter.last_interval_id == interval - 1
        meter.accept(pending)
    rows = [dict(lane=0, episode_id=0, outcome="censored")]
    meter.close_records(rows)
    observed = rows[0]["scripted_recovery"]["followup_observation"]
    assert observed["observed_frames"] == 7
    assert observed["predicate_frames"] == 4
    assert observed["predicate_losses"] == 2
    assert observed["command_changes"] == 2
    assert observed["losses_with_same_interval_command_change"] == 1
    assert observed["longest_producer_followup_s"] == pytest.approx(.42)
    assert observed["predicate_fraction"] == pytest.approx(4 / 7)
    assert "previous_command" not in observed


def test_real_paid_finish_events_are_transactional_and_terminal_is_separate(rig):
    e, meter = rig, observer(rig)
    # Explicit CPU pulse fixture: exercise the production payload/diagnostics,
    # not a claim that injected phase values dynamically demonstrate recovery.
    extras = produced(e)
    pending = meter.prepare(extras, [0, 1], policy_interval_id=1)
    assert meter.active == {} and meter.last_interval_id == 0
    meter.accept(pending)
    ready(e)
    for interval, names in ((2, EVENTS[:2]), (3, EVENTS[2:])):
        extras = produced(e)
        for name in names:
            extras["recovery_transition"]["events"][name][0] = True
            extras["recovery_transition"]["occurrence_ids"][name][0] = interval * 4
        if interval == 3:
            extras["diagnostics"]["recovery_true_success"][0] = True
        meter.accept(meter.prepare(extras, [0, 1], policy_interval_id=interval))
        assert not extras["diagnostics"]["done"][0]
    extras = produced(e)  # Later predicate loss, no forced early-success done.
    meter.accept(meter.prepare(extras, [0, 1], policy_interval_id=4))
    rows = [dict(lane=0, episode_id=0, outcome="censored"), dict(lane=1, episode_id=0, outcome="censored")]
    meter.close_records(rows)
    recovery = rows[0]["scripted_recovery"]
    assert all(recovery["ever_completed"].values()) and recovery["followup_predicate_lost_after_completion"]
    assert not recovery["last_done"] and not recovery["terminal_after_handoff"]
    assert recovery["post_handoff_windows"]["rl_hold"]["frames"] == 3
    assert recovery["masked_frames"] == 1 and recovery["eligible_rl_frames"] == 3
    report = meter.report()
    assert report["cases"]["recover"]["censored_episodes"] == 1
    assert report["cases"]["ordinary"]["recorded_episodes"] == 0 and "Ordinary reference" in report["scope"]
    assert report["ppo_updates"] == report["actor_direct_loss_credit"] == 0


@pytest.mark.parametrize("field", ["done", "terminated", "timeouts", "policy_learning_mask", "returned_episode_ids", "policy_interval_id"])
def test_producer_paid_or_returned_interval_mismatch_rejected_without_committing(rig, field):
    meter = observer(rig); extras = produced(rig)
    payload = extras["recovery_transition"]
    if field == "policy_interval_id": payload[field] += 1
    elif field == "returned_episode_ids": payload[field][0] += 1
    else: payload[field][0] = ~payload[field][0]
    with pytest.raises(ValueError):
        meter.prepare(extras, [0, 1], policy_interval_id=1)
    assert not meter.active and meter.last_interval_id == 0


def test_actual_cpu_actor_loop_consumes_recovery_finish_and_closes_real_episode_boundaries(tmp_path, rig):
    from tensordict import TensorDict
    from wheeled_tasks.chassis.flat_evaluation_loop import FlatEvaluationLoop, reset_snapshot
    from test_scut_flat_reward_runtime import field
    bound = BoundEvaluationInputs.capture(**fixture_inputs(tmp_path)); bound.authenticate()
    e = rig
    e.cfg = {**bound.evaluation_config(), "seed": 17, "stage": "foundation", "num_envs": 2, "terrain_level": 0.}
    e.num_envs, e.physical_tick, e.decimation = 2, 0, 4
    e.clone_indices = [0, 1]
    e.generator, e.startup_report = torch.Generator().manual_seed(17), {}
    e.flat_episode.episode_id.fill_(-1); e.recovery.episode_ids.fill_(-1); e.episode_length_buf.zero_()
    bound.build_setup().apply_before_first_reset(e)
    original_reset = e.reset
    def reset(ids):
        original_reset(ids)
        e.episode_length_buf[ids] = 0  # Complete the omitted CPU reset-counter excerpt.
    e.reset = reset; e.reset(torch.arange(2))
    e.robot.data.root_com_lin_vel_w = field(e.robot.data.root_com_lin_vel_b.torch)
    e.robot.data.root_com_ang_vel_w = field(e.robot.data.root_com_ang_vel_b.torch)
    e.flat_evaluation_snapshot = lambda ids: reset_snapshot(e, ids)
    e.get_observations = lambda: TensorDict({"policy": torch.zeros(2, 35), "critic": torch.zeros(2, 78)}, batch_size=[2])
    def step(action):
        assert action.shape == (2, 6) and torch.isfinite(action).all()
        if e.tick == 2:
            e.scut_reward.joint_velocity[:, 0] = 501.  # Actual source numerical terminal, not recovery success.
        e.physical_tick += 4
        return tick(e)
    e.step = step
    written = []
    loop = FlatEvaluationLoop(bound, device="cpu", limit_check=lambda: None)
    result = loop.run(e, commit_records=lambda rows: written.extend(deepcopy(rows)))
    assert result["status"] == "attempt_quota_completed" and result["ppo_updates"] == 0
    assert result["episode_ledger"]["completed_attempts_per_lane"] == [1, 1]
    assert len(written) == 2 and written[0]["scripted_recovery"]["policy_ticks"] == 3
    assert written[0]["scripted_recovery"]["last_done"] and not any(written[0]["scripted_recovery"]["ever_completed"].values())
    assert result["scripted_recovery_measurements"]["cases"]["recover"]["closed_episodes"] == 1
    assert (e.stage_actor_update, e.global_actor_update) == (3500, 3500)


def test_actual_paid_command_latch_survives_release_inside_interval(rig):
    e = rig
    e.episode_length_buf.zero_()
    e.flat_episode.commands.set_fixed_requests(torch.arange(2), torch.tensor([[.5, 0., -.2], [-.5, 0., .2]]),
        torch.tensor([.305, .3]), evaluation_only=True)
    e.reset(torch.arange(2))
    ready(e)
    e.recovery.true_stable_peak[0] = 1.  # Fixture has completed the existing hold predicate.
    e.flat_episode.begin_policy(); e.recovery.begin_policy()
    assert e.recovery.policy_command_override_active.tolist() == [True, False]
    e.recovery.fsm.motion_released[0] = True  # Release occurs inside the physical interval.
    for _ in range(4): e.flat_episode.advance_physics(.005)
    e.tick += 1; e.episode_length_buf += 1
    _, _, _, extras = e.finish()
    d = extras["diagnostics"]
    assert d["recovery_motion_released"][0] and d["recovery_policy_command_override_active"][0]
    assert d["command_override_active"][0]
    assert torch.equal(d["command_override_active"], d["source_flat_episode"]["paid_override_active"] |
                       d["recovery_policy_command_override_active"])
    # The following policy interval has released ownership, independent of the
    # numerical similarity between effective and requested commands.
    e.recovery.begin_policy()
    assert not e.recovery.policy_command_override_active.any()


def test_action_history_records_real_payments_across_a_partial_handoff(rig):
    e, meter = rig, observer(rig)
    for interval, action in enumerate((-1., 1., 0., 0., 0., 0.), start=1):
        e.scut_reward.begin_policy(torch.full((2, 6), action))
        if interval == 2:
            ready(e)
        extras = produced(e)
        if interval == 2:
            # Explicit CPU fixture: RL owns the end pose, but the earlier
            # scripted substep makes this entire policy interval ineligible.
            extras["recovery_transition"]["events"]["rl_entered"][0] = True
            extras["recovery_transition"]["occurrence_ids"]["rl_entered"][0] = 7
            for container in (extras, extras["diagnostics"], extras["recovery_transition"]):
                container["policy_learning_mask"][0] = False
        before = deepcopy(meter.active)
        rng = torch.get_rng_state().clone()
        pending = meter.prepare(extras, [0, 1], policy_interval_id=interval)
        assert meter.active == before
        assert torch.equal(rng, torch.get_rng_state())
        meter.accept(pending)
        assert 1 not in meter.active  # Ordinary lanes are not recovery samples.

    samples = meter.active[0]["first_actor_action_history"]
    assert [s["episode_policy_tick"] for s in samples] == [3, 4, 5]
    assert [s["prior_interval_actor_owned"] for s in samples] == [False, True, True]
    assert [s["prior_two_intervals_actor_owned"] for s in samples] == [False, False, True]
    assert samples[0]["raw_actor_actions"] == [0.] * 6
    assert samples[0]["raw_previous_actor_actions"] == [1.] * 6
    assert samples[0]["raw_before_previous_actor_actions"] == [-1.] * 6
    assert samples[0]["paid_action_history_terms"] == pytest.approx({
        "action_rate": -.0012, "action_smoothness_leg": -.036,
        "action_smoothness_wheel": -.0036})
    assert samples[0]["terms_crossing_control_ownership"] == list(meter.ACTION_HISTORY_PAYMENTS)
    assert samples[1]["terms_crossing_control_ownership"] == list(meter.ACTION_HISTORY_PAYMENTS[1:])
    assert samples[2]["terms_crossing_control_ownership"] == []
    records = [dict(lane=0, episode_id=0, outcome="censored")]
    meter.close_records(records)
    saved = records[0]["scripted_recovery"]
    assert saved["first_actor_action_history"] == samples
    assert "previous_actor_owned_intervals" not in saved


def test_action_history_keeps_actor_owned_terminal_and_clears_at_episode_boundary(rig):
    e, meter = rig, observer(rig)
    meter.accept(meter.prepare(produced(e), [0, 1], policy_interval_id=1))
    ready(e)
    e.scut_reward.joint_velocity[0, 0] = 501.
    extras = produced(e)
    extras["recovery_transition"]["events"]["rl_entered"][0] = True
    extras["recovery_transition"]["occurrence_ids"]["rl_entered"][0] = 8
    meter.accept(meter.prepare(extras, [0, 1], policy_interval_id=2))
    samples = meter.active[0]["first_actor_action_history"]
    assert len(samples) == 1 and samples[0]["done"]
    assert set(samples[0]["paid_action_history_terms"].values()) == {0.}
    meter.close_records([dict(lane=0, episode_id=0, outcome="terminated")])
    # The production reset in the fixture omits its preceding counter clear.
    e.episode_length_buf[0] = 0
    meter.accept(meter.prepare(produced(e), [0, 1], policy_interval_id=3))
    assert meter.active[0]["episode_id"] == 1
    assert meter.active[0]["first_actor_action_history"] == []
    assert meter.active[0]["previous_actor_owned_intervals"] == [False]
    assert len(meter.active[0]["controller_phase_history"]) == 1
    assert meter.active[0]["controller_phase_history"][0]["first"]["episode_policy_tick"] == 1


def test_phase_history_keeps_pre_reset_motion_and_phase_boundaries_without_mutation(rig):
    from wheeled_tasks.chassis.recovery import RecoveryPhase
    e, meter = rig, observer(rig)
    phases = (RecoveryPhase.SIDE, RecoveryPhase.SIDE, RecoveryPhase.CAPTURE, RecoveryPhase.CAPTURE)
    for interval, phase in enumerate(phases, 1):
        if interval == 4:
            e.scut_reward.joint_velocity[0, 0] = 501.  # Real source numerical reset in the CPU fixture.
        extras = produced(e)
        data = extras["diagnostics"]
        # Explicit phase/measurement fixtures test the observer, not physical
        # success of these commanded script phases.
        data["recovery_phase"][0] = extras["recovery_transition"]["phase"][0] = int(phase)
        data["motor_position"][0, 0] = 10. + interval
        data["recovery_leg_target_error_rad"][0] = (0.1, 0.6, 0.2, 0.3)[interval - 1]
        before = deepcopy(meter.active)
        pending = meter.prepare(extras, [0, 1], policy_interval_id=interval)
        assert meter.active == before
        data["motor_position"].zero_()  # Captured values must survive reused tensors/autoreset.
        meter.accept(pending)
        assert 1 not in meter.active
    records = [dict(lane=0, episode_id=0, outcome="terminated")]
    meter.close_records(records)
    history = records[0]["scripted_recovery"]["controller_phase_history"]
    assert len(history) == 2
    assert [row["observed_policy_frames"] for row in history] == [2, 2]
    assert [row["first"]["episode_policy_tick"] for row in history] == [1, 3]
    assert [row["last"]["episode_policy_tick"] for row in history] == [2, 4]
    assert [row["max_leg_target_error_rad"] for row in history] == pytest.approx([.6, .3])
    assert history[0]["first"]["motor_position"][0] == 11.
    assert history[-1]["last"]["motor_position"][0] == 14.
    assert history[-1]["last"]["done"] and history[-1]["last"]["terminated"]
    assert "20ms endpoints" in meter.report()["controller_phase_history_scope"]
