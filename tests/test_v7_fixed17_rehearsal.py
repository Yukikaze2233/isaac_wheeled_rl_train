"""V7 fixed rehearsal preserves reviewed requests through real command resampling."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.fixed_flat_rehearsal import (
    CASE_FILE, FIXED17_VERSION, FIXED17_CASE_FILE, FIXED17_CASE_SHA, FLAT_REFRESH_VERSION,
    load_fixed_flat_rehearsal, reorder_fixed_flat_rehearsal,
    install_fixed_flat_rehearsal, rehearsal_summary,
)
from wheeled_tasks.chassis.rough_commands import ScutRoughCommands
from wheeled_tasks.chassis.flat_command_sampling import ScutFlatCommands


ROOT = Path(__file__).resolve().parents[1]


def candidate():
    return {"reward_profile": "scut0730_rough_v1", "formal_flat_rehearsal": {
        "version": FIXED17_VERSION, "fraction": .25,
        "cases_file": FIXED17_CASE_FILE, "cases_sha256": FIXED17_CASE_SHA,
        "allocation": "equal_case_lanes", "reward_profile": "scut0730_rough_v1"}}


def flat_refresh():
    config = candidate()
    config.update(formal_flat_refresh=FLAT_REFRESH_VERSION, reward_profile="scut0719_flat_v1",
        formal_terrain={"type": "plane", "source": "scut0719", "env_spacing_m": 4.},
        curriculum_stage="foundation", enabled_stages=["foundation"])
    config["formal_flat_rehearsal"].update(fraction=1., reward_profile="scut0719_flat_v1")
    return config


def test_pinned_case_objects_and_allocation_retain_existing_evaluation_inputs():
    original = json.loads((ROOT / CASE_FILE).read_bytes())["cases"]
    spin = json.loads((ROOT / "configs/v7_formal_20261012/evaluation/foundation_spin225_mean_v1.json").read_bytes())["cases"]
    raw = (ROOT / FIXED17_CASE_FILE).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXED17_CASE_SHA
    assert json.loads(raw)["cases"] == original + spin
    before = torch.get_rng_state().clone()
    spec = load_fixed_flat_rehearsal(candidate(), ROOT, 4096)
    assert torch.equal(before, torch.get_rng_state())
    assert spec["rehearsal_lanes"] == 1024 and spec["rough_lanes"] == 3072
    assert len(spec["lane_counts"]) == 17
    assert sorted(spec["lane_counts"].values()) == [60] * 13 + [61] * 4
    assert spec["case_ids"] == tuple(case["case_id"] for case in original + spin)
    assert spec["case_requests"] == tuple(tuple(case["base_request"]) for case in original + spin)
    assert "unified rough" in rehearsal_summary(spec)["reward"]


def test_all_fixed_requests_survive_native_reordering_reset_and_rough_resampling():
    count = 4096
    original = load_fixed_flat_rehearsal(candidate(), ROOT, count)
    order = sorted(range(count), key=lambda i: f"env_{i}")
    spec = reorder_fixed_flat_rehearsal(original, order)
    grid = torch.zeros(10, 14, 3)
    grid[:, :, 0] = ((torch.arange(10) + .5) * 9. - 45.)[:, None]
    grid[:, :, 1] = ((torch.arange(14) + .5) * 9. - 63.)[None, :]
    origins = grid[4, torch.arange(count) % 14].clone()
    for lane, origin in original["origins_by_env"].items():
        origins[lane] = torch.tensor(origin)
    origins = origins[order]
    owner = ScutRoughCommands(count, "cpu", terrain_origins=grid, env_origins=origins)
    install_fixed_flat_rehearsal(spec, owner)
    selected = torch.tensor([case >= 0 for case in spec["case_index"]])
    assert torch.equal(owner.fixed_request_mask, selected)
    assert torch.equal(owner._training_rehearsal_mask, selected)
    expected = torch.tensor([spec["case_requests"][case] for case in spec["case_index"] if case >= 0])
    generator = torch.Generator().manual_seed(721)
    ids = torch.arange(count)
    for iteration in (0, 500, 4500):
        owner.reset(ids, torch.zeros(count), torch.zeros(count, dtype=torch.bool), generator, iteration)
        for _ in range(3):
            owner.episode_physics_ticks.fill_(2000)
            owner.time_left.zero_()
            owner.begin_policy()
            for _ in range(4):
                owner.advance_physics()
            owner.advance(.02, torch.ones(count), iteration, generator, world_positions=origins)
            assert torch.equal(owner.command3[selected, 0], expected[:, 0])
            assert torch.equal(owner.command3[selected, 2], expected[:, 1])
            assert torch.equal(owner.height[selected], expected[:, 2])
            assert bool((owner.command3[selected, 1] == 0).all())
            assert bool((owner.special_mode_id[selected] == -1).all())
            assert not bool(owner.is_heading[selected].any())
        assert owner.height[~selected].unique().numel() > 100
    diagnostics = owner.diagnostics()
    assert torch.equal(diagnostics["fixed_training_rehearsal"], selected)
    assert not diagnostics["fixed_evaluation_request"].any()
    assert owner.metrics()["/source_command/fixed_training_rehearsal_fraction"] == .25


@pytest.mark.parametrize("key,value", [
    ("fraction", .5), ("fraction", True), ("cases_file", CASE_FILE),
    ("cases_sha256", "0" * 64), ("allocation", "source_command_distribution"),
    ("reward_profile", "scut0719_flat_v1"), ("source_iteration_offset", 4500),
])
def test_fixed17_rejects_unreviewed_pool_reward_or_source_sampler(key, value):
    config = deepcopy(candidate())
    config["formal_flat_rehearsal"][key] = value
    with pytest.raises(ValueError, match="fixed17"):
        load_fixed_flat_rehearsal(config, ROOT, 4096)


def test_fixed17_cannot_enter_flat_reward_path_or_skip_case_bytes(tmp_path):
    config = candidate()
    config["reward_profile"] = "scut0719_flat_v1"
    with pytest.raises(ValueError, match="fixed17"):
        load_fixed_flat_rehearsal(config, ROOT, 4096)
    path = tmp_path / FIXED17_CASE_FILE
    path.parent.mkdir(parents=True)
    path.write_bytes((ROOT / FIXED17_CASE_FILE).read_bytes() + b" ")
    with pytest.raises(ValueError, match="bytes changed"):
        load_fixed_flat_rehearsal(candidate(), tmp_path, 4096)
    with pytest.raises(ValueError, match="allocate all 17"):
        load_fixed_flat_rehearsal(candidate(), ROOT, 32)


@pytest.mark.parametrize("count", [17, 4096])
def test_all_flat_allocation_keeps_origins_and_has_no_rough_or_recovery_lanes(count):
    before = torch.get_rng_state().clone()
    spec = load_fixed_flat_rehearsal(flat_refresh(), ROOT, count,
                                    ordinary_lane_ids=list(reversed(range(count))))
    assert torch.equal(before, torch.get_rng_state())
    assert spec["origins_by_env"] is None
    assert spec["rehearsal_lanes"] == spec["ordinary_lane_count"] == count
    assert spec["rough_lanes"] == spec["background_lanes"] == spec["recovery_lane_count"] == 0
    assert spec["ordinary_unrehearsed_lane_count"] == 0
    assert spec["actual_lane_fraction"] == 1.
    assert all(i >= 0 for i in spec["case_index"])
    assert sorted(spec["lane_counts"].values()) == ([1] * 17 if count == 17 else [240] + [241] * 16)
    summary = rehearsal_summary(spec)
    assert summary["flat_refresh"] == FLAT_REFRESH_VERSION
    assert summary["rough_lanes"] == 0
    assert summary["reward"] == "unchanged flat reward on every ordinary lane"
    assert summary["geometry"] == "existing infinite plane and unchanged environment origins"
    assert spec["background_group"] == "ordinary_unrehearsed"


def test_all_flat_requests_survive_real_flat_resampling_and_restart():
    count = 4096
    spec = load_fixed_flat_rehearsal(flat_refresh(), ROOT, count)
    order = sorted(range(count), key=lambda i: f"env_{i}")
    spec = reorder_fixed_flat_rehearsal(spec, order)
    owner = ScutFlatCommands(count, "cpu")
    install_fixed_flat_rehearsal(spec, owner)
    assert owner._training_rehearsal_mask.all()
    expected = torch.tensor([spec["case_requests"][case] for case in spec["case_index"]])
    generator = torch.Generator().manual_seed(721)
    ids, heading = torch.arange(count), torch.zeros(count)
    for iteration in (0, 500, 4500):
        owner.reset(ids, heading, torch.zeros(count, dtype=torch.bool), generator, iteration)
        for _ in range(3):
            owner.episode_physics_ticks.fill_(2000)
            owner.time_left.zero_()
            owner.begin_policy()
            for _ in range(4):
                owner.advance_physics()
            owner.advance(.02, heading, iteration, generator)
            assert torch.equal(owner.command3[:, 0], expected[:, 0])
            assert torch.equal(owner.command3[:, 2], expected[:, 1])
            assert torch.equal(owner.height, expected[:, 2])
            assert bool((owner.command3[:, 1] == 0).all())
            assert bool((owner.special_mode_id == -1).all())
            assert not bool(owner.is_heading.any() | owner.holding.any())
    assert not owner.diagnostics()["fixed_evaluation_request"].any()
    assert owner.metrics()["/source_command/fixed_training_rehearsal_fraction"] == 1.
    restored = ScutFlatCommands(count, "cpu")
    install_fixed_flat_rehearsal(spec, restored)
    other_generator = torch.Generator().manual_seed(18)
    restored.load_state_dict(owner.state_dict(generator), other_generator)
    assert torch.equal(restored.command3, owner.command3)
    assert restored._training_rehearsal_mask.all()
    assert torch.equal(other_generator.get_state(), generator.get_state())


@pytest.mark.parametrize("change", [
    {"formal_flat_refresh": "unknown"}, {"formal_flat_refresh": None},
    {"reward_profile": "scut0730_rough_v1"}, {"curriculum_stage": "terrain"},
    {"enabled_stages": ["foundation", "terrain"]},
    {"formal_terrain": {"type": "plane", "source": "scut0719", "env_spacing_m": 5.}},
    {"recovery_training": {"enabled": True}},
    {"recovery_training": {"enabled": False, "source_flat_handoff": "scut0719_scripted_handoff_v1"}},
    {"scene_groups": [{"name": "recovery", "recovery_pose": "prone"}]},
])
def test_all_flat_refresh_cannot_silently_enter_other_stage_or_recovery(change):
    config = flat_refresh()
    config.update(change)
    with pytest.raises(ValueError, match="fixed17 flat refresh"):
        load_fixed_flat_rehearsal(config, ROOT, 4096)


def test_all_flat_fraction_and_actual_lane_owner_are_strict():
    config = flat_refresh()
    config["formal_flat_rehearsal"]["fraction"] = .25
    with pytest.raises(ValueError, match="fixed17"):
        load_fixed_flat_rehearsal(config, ROOT, 4096)
    for ordinary in (list(range(16)), [0] * 17):
        with pytest.raises(ValueError, match="every actual lane"):
            load_fixed_flat_rehearsal(flat_refresh(), ROOT, 17, ordinary_lane_ids=ordinary)
    spec = load_fixed_flat_rehearsal(flat_refresh(), ROOT, 17)
    owner = ScutFlatCommands(17, "cpu")
    mask = torch.zeros(17, dtype=torch.bool)
    mask[0] = True
    with pytest.raises(ValueError, match="cannot include recovery"):
        install_fixed_flat_rehearsal(spec, owner, recovery_mask=mask)
    assert not owner.fixed_request_mask.any()
