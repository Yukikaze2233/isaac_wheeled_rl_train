"""Exercise the native reward block without requiring an Isaac GPU process."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.rewards import (
    DENSE_TRACKING_V1, PassiveLandingWindow, dense_error_terms,
    passive_landing_costs, validate_landing_tracking,
)
from wheeled_tasks.chassis.task import Phase, PhaseTracker


LEGACY = {"height_weight": 2., "velocity_weight": 1., "impact_weight": .05}
FINITE = {**LEGACY, "passive_semantics": "finite_task_cost_v1", "window_seconds": .6}


@pytest.fixture
def native_landing():
    path = Path(__file__).resolve().parents[1] / "src/wheeled_tasks/chassis/env.py"
    block, = [node for node in ast.walk(ast.parse(path.read_text()))
              if isinstance(node, ast.If) and ast.unparse(node.test) == "self.cfg.get('landing_tracking')"]
    function = ast.parse("def reward(self, height, vertical_velocity):\n"
                         "    components = {}\n    return components\n").body[0]
    function.body.insert(1, block)
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch, "Phase": Phase, "__package__": "wheeled_tasks.chassis"}
    exec(compile(module, "<native landing reward>", "exec"), namespace)
    return namespace["reward"]


def rig(settings, count=2, *, landing_task=False):
    return SimpleNamespace(cfg={"landing_tracking": deepcopy(settings)},
        phase=PhaseTracker(count, "cpu", .02), full_tasks=FullTaskSemantics,
        commands=torch.tensor([[0., 0., .23]]).repeat(count, 1),
        mode=torch.zeros(count, dtype=torch.long),
        body_mass=torch.ones(count, 1), contact_peak=torch.full((count,), 9.81),
        skills=SimpleNamespace(landing=torch.full((count,), landing_task)),
        passive_landing=PassiveLandingWindow(count, "cpu", .02, settings)
            if settings.get("passive_semantics") == "finite_task_cost_v1" else None,
        landing_reward_mask=torch.zeros(count))


def test_recovery_reward_inversion_removed_without_changing_physical_phase(native_landing):
    errors = torch.tensor([.009, .011])
    old, new = rig(LEGACY), rig(FINITE)
    old.phase.phase.fill_(Phase.FLIGHT)
    new.phase.phase.fill_(Phase.FLIGHT)
    contact = torch.ones(2, 2, dtype=torch.bool)
    request = torch.zeros(2, dtype=torch.bool)
    stable = errors < .01
    for _ in range(100):
        for env in (old, new):
            env.phase.update(contact, request, stable)
            terms = native_landing(env, env.commands[:, 2] + errors, torch.zeros(2))
            env.total = sum(terms.values()) + sum(dense_error_terms(errors, DENSE_TRACKING_V1["height"]))
        for key in vars(old.phase):
            if isinstance(getattr(old.phase, key), torch.Tensor):
                assert torch.equal(getattr(old.phase, key), getattr(new.phase, key))
    assert old.phase.phase.tolist() == [Phase.GROUND, Phase.RECOVERY]
    assert old.total[1] - old.total[0] > 2.6
    assert new.total[1] < new.total[0]
    assert new.landing_reward_mask.eq(0).all()


def test_ordinary_static_and_active_jump_never_receive_passive_shaping(native_landing):
    env = rig(FINITE, 3, landing_task=True)
    env.skills.landing[0] = False
    env.mode[1] = 4
    env.phase.phase.fill_(Phase.LANDING)
    terms = native_landing(env, env.commands[:, 2] + .03, torch.ones(3))
    for key in ("landing_height_tracking", "landing_velocity_tracking"):
        assert terms[key][:2].eq(0).all()
        assert terms[key][2] < 0
    assert env.landing_reward_mask.tolist() == [0., 0., 1.]


def test_genuine_landing_cost_is_dense_zero_centered_and_phase_independent():
    weight = torch.ones(3)
    terms = passive_landing_costs(torch.tensor([0., .01, .04]), torch.tensor([0., .1, .5]), weight, FINITE)
    for cost in terms.values():
        assert cost[0] == 0
        assert cost[2] < cost[1] < 0
    windows = [PassiveLandingWindow(1, "cpu", .02, FINITE) for _ in range(2)]
    for window in windows:
        window.update(torch.tensor([Phase.LANDING]), torch.tensor([True]))
    for tick in range(40):
        first = windows[0].update(torch.tensor([Phase.GROUND]), torch.tensor([True]))
        # Prolonged recovery and repeated FLIGHT/LANDING cannot renew payment.
        phase = [Phase.RECOVERY, Phase.FLIGHT, Phase.LANDING][tick % 3]
        second = windows[1].update(torch.tensor([phase]), torch.tensor([True]))
        assert torch.equal(first, second)
        assert float(second) == pytest.approx(max(0., 1. - (tick + 1) * .02 / .6), abs=1e-6)
    windows[1].reset(torch.tensor([0]))
    assert windows[1].update(torch.tensor([Phase.LANDING]), torch.tensor([True])).item() == 1.


def test_legacy_terms_and_all_impact_costs_unchanged(native_landing):
    old, new = rig(LEGACY, 5, landing_task=True), rig(FINITE, 5, landing_task=True)
    for env in (old, new):
        env.phase.phase[:] = torch.arange(5)
        env.phase.contact_time[:] = torch.arange(5) * .1
        env.mode[4] = 4
        env.contact_peak.fill_(9.81 * 5)
    h, v = old.commands[:, 2] + .025, torch.full((5,), .2)
    legacy, finite = native_landing(old, h, v), native_landing(new, h, v)
    landing = ((old.phase.phase == Phase.LANDING) | (old.phase.phase == Phase.RECOVERY)).float()
    h_ref, v_ref = FullTaskSemantics.landing_reference(old.commands[:, 2], old.phase.contact_time)
    assert torch.equal(legacy["landing_height_tracking"], landing * (old.mode != 4) * 2 * torch.exp(-((h-h_ref)/.035).square()))
    assert torch.equal(legacy["landing_velocity_tracking"], landing * (old.mode != 4) * torch.exp(-((v-v_ref)/.35).square()))
    assert torch.equal(legacy["landing_impact"], finite["landing_impact"])
    assert finite["landing_impact"][4] < 0


def test_reward_observation_preserves_inputs_and_rng(native_landing):
    env = rig(FINITE, landing_task=True)
    env.phase.phase.fill_(Phase.LANDING)
    before = {key: value.clone() for key, value in vars(env.phase).items() if isinstance(value, torch.Tensor)}
    inputs = [env.commands, env.mode, env.skills.landing, env.contact_peak,
              env.commands[:, 2].clone(), torch.zeros(2)]
    clones = [value.clone() for value in inputs]
    rng = torch.get_rng_state().clone()
    for _ in range(50):
        native_landing(env, inputs[-2], inputs[-1])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, original) for value, original in zip(inputs, clones))
    assert all(torch.equal(getattr(env.phase, key), value) for key, value in before.items())


@pytest.mark.parametrize("change", [{"passive_semantics": "typo"}, {"window_seconds": 0},
    {"window_seconds": float("nan")}, {"window_seconds": True}, {"height_weight": -1}])
def test_reject_invalid_repair_settings(change):
    with pytest.raises(ValueError):
        validate_landing_tracking({**FINITE, **change})


def test_phase_and_paid_window_are_logged_without_mutation(monkeypatch):
    from wheeled_tasks.chassis import learning_audit
    class Probe:
        def __init__(self, *args): pass
        def metrics(self): return {}
    monkeypatch.setattr(learning_audit, "PolicyUpdateProbe", Probe)
    commands = torch.tensor([[0., 0., .305], [0., 0., .305]])
    env = SimpleNamespace(num_envs=2, device="cpu", policy_dt=.02, commands=commands,
        skills=SimpleNamespace(batches=[(torch.arange(2), {"group_name": "stand", "kind": "stand"})]),
        cfg={"new_asset_curriculum": {"height_sampling_schedule": {"fixed_heights_m": [.305]}}})
    audit = learning_audit.LearningAudit(env, SimpleNamespace(actor=None, storage=None))
    data = dict(commands=commands, reward=torch.zeros(2), planar_speed_world=torch.zeros(2),
        height=commands[:, 2], reasons={"fall": torch.zeros(2)}, done=torch.zeros(2, dtype=torch.bool),
        physical_phase=torch.tensor([Phase.GROUND, Phase.RECOVERY]), physical_phase_time_s=torch.tensor([1., 2.]),
        landing_reward_mask=torch.tensor([0., .5]), landing_window_elapsed_s=torch.tensor([0., .3]),
        landing_task_eligible=torch.tensor([False, True]), velocity=torch.tensor([[.1, -.1, 0.], [-.1, .2, 0.]]))
    before = {key: value.clone() for key, value in data.items() if isinstance(value, torch.Tensor)}
    rng = torch.get_rng_state().clone()
    with torch.inference_mode():
        audit.observe(data, {"landing_height_tracking": torch.tensor([0., -.2])})
    result = audit.drain()
    assert result["LearningAudit/stand/phase/recovery_fraction"] == .5
    assert result["LearningAudit/stand/phase/recovery_time_s"] == 2.
    assert result["LearningAudit/stand/landing/paid_mask_fraction"] == .5
    assert result["LearningAudit/stand/landing/window_weight"] == .25
    assert result["LearningAudit/stand/velocity/signed_forward_m_s"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(data[key], value) for key, value in before.items())
