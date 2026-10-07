"""Component logging validates clocks without expanding unused device vectors."""
import json
from pathlib import Path

import pytest
import torch

from wheeled_tasks.chassis.nonstandard_start import (
    LEVELS, NonstandardStart, PROFILE_FILE, nonstandard_start_settings,
)
from wheeled_tasks.chassis.ordinary_posture_curriculum import (
    OrdinaryPostureCurriculum, posture_settings,
)


ROOT = Path(__file__).resolve().parents[1]
JOINTS = json.loads((ROOT / PROFILE_FILE).read_text())["tree_joint_names"]


@pytest.fixture
def modules():
    birth = NonstandardStart(4, "cpu", .005, nonstandard_start_settings(100), ROOT, JOINTS)
    birth.configure_components(["speed", "speed", "terrain", None], {
        "speed": nonstandard_start_settings(100), "terrain": nonstandard_start_settings(0),
    })
    posture = OrdinaryPostureCurriculum(4, "cpu", posture_settings(100),
        torch.ones(4, dtype=torch.bool))
    posture.configure_components(["speed", "speed", "terrain", None], {
        "speed": posture_settings(100), "terrain": posture_settings(0),
    })
    return birth, posture


def forbid_expansion(*args, **kwargs):
    raise AssertionError("Logging must not expand component values to device lane vectors")


def test_birth_metrics_preserve_values_and_counter_clones_without_expansion(modules, monkeypatch):
    birth, _ = modules
    birth.selected[0] = True
    birth.remaining_steps[0] = 40
    birth.level_reset_counts[:] = torch.tensor([10, 20, 30])
    monkeypatch.setattr(birth, "_component_probabilities", forbid_expansion)
    metrics = birth.metrics({"speed": 1600, "terrain": 0})
    prefix = "/nonstandard_start/"
    expected = {
        prefix + "cancelled_reset_count": 0.,
        prefix + "holding_fraction": torch.tensor(.25),
        prefix + "capturing_fraction": torch.tensor(.25),
        prefix + "selected_fraction": torch.tensor(.25),
    }
    for component, relative, probabilities in (
            ("speed", 1500., (.65, .2, .15)), ("terrain", 0., (.85, .15, 0.))):
        base = prefix + f"component/{component}/"
        expected[base + "relative_actor_updates"] = relative
        expected.update({base + level + "_probability": probability
            for level, probability in zip(LEVELS, probabilities)})
    expected.update({prefix + level + "_reset_count": torch.tensor(count)
        for level, count in zip(LEVELS, (10, 20, 30))})
    assert metrics.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(metrics[key], value, rtol=0, atol=0)
        else:
            assert metrics[key] == value
    birth.level_reset_counts.add_(1)
    assert [metrics[prefix + level + "_reset_count"].item() for level in LEVELS] == [10, 20, 30]


def test_posture_observed_and_metrics_preserve_scalar_caps_without_expansion(modules, monkeypatch):
    _, posture = modules
    monkeypatch.setattr(posture, "amplitudes", forbid_expansion)
    clocks = {"speed": 3100, "terrain": 0}
    expected = {
        "eligible_count": 3, "birth_eligible_population": 3,
        "component/speed/relative_actor_updates": 3000,
        "component/speed/birth_angle_cap_rad": .15,
        "component/speed/push_speed_cap_m_s": .25,
        "component/terrain/relative_actor_updates": 0,
        "component/terrain/birth_angle_cap_rad": .03,
        "component/terrain/push_speed_cap_m_s": .05,
        "birth_selected_count": 0, "birth_rejected_count": 0,
        "birth_abs_angle_max_rad": 0., "push_clock_min_s": 0.,
        "push_clock_max_s": 0., "push_clock_std_s": 0.,
        "push_events": 0, "push_velocity_delta_max_m_s": 0., "geometry_bound": False,
    }
    assert posture.observed(clocks) == expected
    assert posture.metrics(clocks) == {"/ordinary_posture/" + key: float(value)
        for key, value in expected.items()}


@pytest.mark.parametrize("clocks", [0, {}, {"speed": 0},
    {"speed": 0, "terrain": 0, "recovery": 0}])
@pytest.mark.parametrize("index,method,expansion", [
    (0, "metrics", "_component_probabilities"), (1, "observed", "amplitudes"),
])
def test_logging_rejects_missing_or_extra_component_clocks(modules, monkeypatch, clocks, index, method, expansion):
    module = modules[index]
    monkeypatch.setattr(module, expansion, forbid_expansion)
    with pytest.raises(ValueError, match="clocks must match"):
        getattr(module, method)(clocks)


@pytest.mark.parametrize("invalid", [-1, 1., True, None, torch.tensor(1)])
@pytest.mark.parametrize("index,method,expansion", [
    (0, "metrics", "_component_probabilities"), (1, "observed", "amplitudes"),
])
def test_logging_rejects_non_integer_component_actor_clocks(modules, monkeypatch, invalid, index, method, expansion):
    module = modules[index]
    monkeypatch.setattr(module, expansion, forbid_expansion)
    with pytest.raises(ValueError, match="nonnegative integer"):
        getattr(module, method)({"speed": 0, "terrain": invalid})
