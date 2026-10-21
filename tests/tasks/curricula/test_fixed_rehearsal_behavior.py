"""Static fixed-case views preserve scene totals and use the real TB publisher."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tests.support.support_final_rough_recovery_plane import allocate
from wheeled_tasks.chassis.application.monitoring.episode_metrics import EpisodeMetrics
from wheeled_tasks.chassis.application.configuration.fixed_flat_rehearsal import reorder_fixed_flat_rehearsal


ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "src/wheeled_tasks/chassis/application/training/train.py"


def assignment():
    config = json.loads((ROOT / "configs/v7_formal_20261012/recovery_after10000_056.json").read_text())
    spec, scenes = allocate(config, 128)
    order = sorted(range(128), key=lambda i: f"env_{i}")
    return reorder_fixed_flat_rehearsal(spec, order), [scenes[i][0] for i in order]


def actual_collector(spec, groups):
    node = next(n for n in ast.walk(ast.parse(ENTRY.read_text())) if isinstance(n, ast.If)
        and any(isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "metrics"
                for t in child.targets) and isinstance(child.value, ast.Call)
                and ast.unparse(child.value.func) == "EpisodeMetrics" for child in n.body))
    scope = {"c": {"record_diagnostics": True, "policy_dt": .02,
                    "height_workspace": {"height_m": [.17, .4]}},
             "env": NS(scene_groups=groups, rehearsal_spec=spec), "args": NS(device="cpu")}
    exec(compile(ast.Module(body=[deepcopy(node)], type_ignores=[]), str(ENTRY), "exec"), scope)
    return scope["metrics"]


def sample(spec, tick=1):
    labels = spec["case_index"]
    count = len(labels)
    commands = torch.tensor([spec["case_requests"][i] if i >= 0 else (0., 0., .3) for i in labels])
    error = torch.tensor([.01 * (i + 1) if i >= 0 else 10. for i in labels])
    velocity, omega = torch.zeros(count, 3), torch.zeros(count, 3)
    velocity[:, 0] = commands[:, 0] + error
    omega[:, 2] = commands[:, 1] + 2 * error
    return {"velocity": velocity, "omega": omega, "commands": commands,
        "height": commands[:, 2] + error / 10, "position": torch.zeros(count, 3),
        "gravity": torch.tensor([[0., 0., -1.]]).repeat(count, 1),
        "episode_ticks": torch.full((count,), tick, dtype=torch.long),
        "motor_effort": torch.ones(count, 6), "reward": torch.full((count,), .05),
        "gap": torch.zeros(count), "done": torch.zeros(count, dtype=torch.bool),
        "terminated": torch.zeros(count, dtype=torch.bool), "success": torch.zeros(count, dtype=torch.bool),
        "reasons": {"boundary": torch.zeros(count, dtype=torch.bool),
                    "numerical": torch.zeros(count, dtype=torch.bool)}}


def test_native_case_views_match_errors_without_changing_scene_buffers_or_rng():
    spec, groups = assignment()
    legacy = EpisodeMetrics(groups, "cpu", .02, warmup_seconds=.02, height_range_m=[.17, .4])
    actual = EpisodeMetrics(groups, "cpu", .02, warmup_seconds=.02, height_range_m=[.17, .4],
                            fixed_rehearsal=spec)
    failed = next(i for i, label in enumerate(spec["case_index"]) if label == 0)
    rng = torch.get_rng_state()
    for tick in (1, 2, 3):
        data = sample(spec, tick)
        if tick == 2:
            data["measurement_valid_mask"] = torch.ones(128, dtype=torch.bool)
            data["measurement_valid_mask"][failed] = False
            data["velocity"][failed] = data["height"][failed] = float("nan")
            data["done"][failed] = data["terminated"][failed] = True
            data["reasons"]["numerical"][failed] = True
        if tick == 3:
            data["episode_ticks"][failed] = 1
        legacy.observe(data)
        actual.observe(data)
    original, report = legacy.report(), actual.report()
    extra = report.pop("fixed_rehearsal")
    assert report == original
    assert torch.equal(rng, torch.get_rng_state())
    for name in ("frames", "sums", "observed_frames", "episodes", "failures", "timeouts",
                 "torque_square", "torque_peak", "drift", "gap", "velocity_moments"):
        assert torch.equal(getattr(actual, name), getattr(legacy, name)), name
    assert list(extra["groups"]) == ["all", *spec["case_ids"]]
    assert "not_independent_evaluation" in extra["scope"]
    for index, name in enumerate(spec["case_ids"]):
        lanes = spec["case_index"].count(index)
        values = extra["groups"][name]
        assert values["frames"] == 2 * lanes - (2 if index == 0 else 0)
        assert values["observed_frames"] == 3 * lanes
        # High-spin command subtraction retains float32 rounding in the input.
        assert values["vx_mae_m_s"] == pytest.approx(.01 * (index + 1), abs=1e-6)
        assert values["yaw_mae_rad_s"] == pytest.approx(.02 * (index + 1), abs=1e-6)
        assert values["failures"] == (1 if index == 0 else 0)
    total = extra["groups"]["all"]
    assert total["frames"] == sum(extra["groups"][name]["frames"] for name in spec["case_ids"])
    assert total["frames"] < sum(group["frames"] for group in report["groups"].values())
    json.dumps(actual.report(), allow_nan=False)


def test_actual_entry_and_tensorboard_writer_publish_separate_fixed_case_behavior(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from torch.utils.tensorboard import SummaryWriter

    spec, groups = assignment()
    metrics = actual_collector(spec, groups)
    metrics.observe(sample(spec))
    node = next(n for n in ast.walk(ast.parse(ENTRY.read_text())) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "metrics is not None and report['successful_updates'] % 10 == 0")
    directory = tmp_path / "tensorboard"
    with SummaryWriter(str(directory)) as writer:
        scope = {"metrics": metrics, "report": {"successful_updates": 10}, "completed": 10010,
                 "runner": NS(logger=NS(writer=writer)), "args": NS(run_dir=tmp_path), "json": json}
        exec(compile(ast.Module(body=[deepcopy(node)], type_ignores=[]), str(ENTRY), "exec"), scope)
        writer.flush()
    saved = json.loads((tmp_path / "behavior_metrics.json").read_text())
    assert saved["groups"] == metrics.report()["groups"]
    assert saved["fixed_rehearsal"] == metrics.report()["fixed_rehearsal"]
    events = EventAccumulator(str(directory)).Reload()
    for name in ("all", *spec["case_ids"]):
        point, = events.Scalars(f"FixedFlatRehearsal/Behavior/{name}/vx_mae_m_s")
        assert point.step == 10009
        assert point.value == pytest.approx(saved["fixed_rehearsal"]["groups"][name]["vx_mae_m_s"])
    point, = events.Scalars("Behavior/source_flat/vx_mae_m_s")
    assert point.value == pytest.approx(saved["groups"]["source_flat"]["vx_mae_m_s"])


@pytest.mark.parametrize("spec", [None, {"settings": {"version": "v7_fixed15_plane_rehearsal_v1"}}])
def test_actual_entry_keeps_other_collectors_without_fixed17_views(spec):
    metrics = actual_collector(spec, ["ordinary", "ordinary"])
    assert "fixed_rehearsal" not in metrics.report()


@pytest.mark.parametrize("bad", ["version", "duplicate_name", "all_name", "missing_case",
                                 "wrong_length", "boolean_label", "out_of_range"])
def test_invalid_native_case_assignments_are_rejected_without_mutating_the_input(bad):
    spec, groups = assignment()
    spec = deepcopy(spec)
    spec["case_ids"] = list(spec["case_ids"])
    spec["case_index"] = list(spec["case_index"])
    if bad == "version": spec["settings"] = {"version": "unselected"}
    elif bad == "duplicate_name": spec["case_ids"][1] = spec["case_ids"][0]
    elif bad == "all_name": spec["case_ids"][0] = "all"
    elif bad == "missing_case": spec["case_index"] = [-1 if i == 0 else i for i in spec["case_index"]]
    elif bad == "wrong_length": spec["case_index"].pop()
    elif bad == "boolean_label": spec["case_index"][0] = True
    else: spec["case_index"][0] = 17
    before, rng = deepcopy(spec), torch.get_rng_state()
    with pytest.raises(ValueError, match="Fixed behavior"):
        EpisodeMetrics(groups, "cpu", .02, fixed_rehearsal=spec)
    assert spec == before and torch.equal(torch.get_rng_state(), rng)
