"""Compare the frozen publisher with batching, without booting Isaac or PPO."""
import ast
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from wheeled_tasks.chassis.state_snapshot import read_selected_physics, select_group_environments
from wheeled_tasks.chassis.training_timing import TrainingTiming


ROOT = Path(__file__).resolve().parents[1]
PARENT = "f99d3cbb556db873a299b8c9bc2cd68fcf34957a"


@lru_cache
def source(legacy=False):
    text = (subprocess.check_output(["git", "show", f"{PARENT}:scripts/train_chassis.py"],
            cwd=ROOT, text=True) if legacy else (ROOT / "scripts/train_chassis.py").read_text())
    return ast.parse(text)


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def publish_condition(node):
    return (isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)
            and ast.unparse(node.test.values[0]) == "args.publish_state")


def compile_function(node, namespace):
    node = deepcopy(node)
    node.body = [ast.Global(names=item.names) if isinstance(item, ast.Nonlocal) else item
                 for item in node.body]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 "<frozen-publisher-test>", "exec"), namespace)
    return namespace[node.name]


class Clock:
    def __init__(self):
        self.value = .25

    def monotonic(self):
        return self.value

    def time(self):
        return 1791500000. + self.value


def environment(group_count=3, *, noncontiguous=False, mixed_dtype=False):
    groups = [name for index in range(group_count) for name in (f"group_{group_count-index}",) * 2]
    n, bodies = len(groups), 3
    stride = 2 if noncontiguous else 1
    poses = torch.arange(n * bodies * 7 * stride, dtype=torch.float32).reshape(n, bodies, -1)[..., ::stride] / 17.
    origins = torch.arange(n * 3 * stride, dtype=torch.float64 if mixed_dtype else torch.float32).reshape(n, -1)[:, ::stride] / 19.
    commands = torch.arange(n * 3 * stride, dtype=torch.float16 if mixed_dtype else torch.float32).reshape(n, -1)[:, ::stride] / 23.
    # Division materializes the float tensors; restore strided views explicitly.
    if noncontiguous:
        def strided(value):
            storage = torch.empty(*value.shape[:-1], value.shape[-1] * 2, dtype=value.dtype)
            storage[..., ::2] = value
            return storage[..., ::2]
        poses, origins, commands = map(strided, (poses, origins, commands))
    steps = torch.arange(n * stride, dtype=torch.int64)[::stride] + 2**53
    if noncontiguous:
        storage = torch.empty(n * 2, dtype=torch.int64)
        storage[::2] = steps
        steps = storage[::2]
    return SimpleNamespace(scene_groups=groups, device="cpu", kinds=["stairs", "flat"] * group_count,
        origins=origins, commands=commands, episode_length_buf=steps, tick=411,
        surfaces=[[SimpleNamespace(x0=-4., x1=4., z0=index / 10., friction=.8)] for index in range(n)],
        robot=SimpleNamespace(body_names=["base", "left", "right"],
            data=SimpleNamespace(body_link_pose_w=SimpleNamespace(torch=poses))))


def publisher(env, directory, *, legacy=False, enabled=True, clock=None, timer=None):
    directory.mkdir()
    clock = clock or Clock()
    transition = (object(), object(), object(), {"log": {}})
    namespace = dict(env=env, args=SimpleNamespace(publish_state=enabled, run_dir=directory),
        identity=dict(contract_sha256="a" * 64, asset_manifest_sha256="b" * 64),
        c={"policy_dt": .02}, time=clock, json=json, last_publish=0., timing=timer,
        budget=SimpleNamespace(check=lambda: None), learning_audit=None, metrics=None,
        original_step=lambda actions: transition)
    compile_function(function(source(legacy), "timed_call"), namespace)
    step = function(source(legacy), "step")
    if legacy:
        body = next(node.body for node in step.body if publish_condition(node))
        template = ast.parse("def publish_state():\n    global last_publish\n").body[0]
        template.body += deepcopy(body)
        compile_function(template, namespace)
    else:
        cache = next(node for node in ast.walk(source()) if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "args.publish_state"
                     and any(isinstance(item, ast.ImportFrom) and item.module ==
                             "wheeled_tasks.chassis.state_snapshot" for item in node.body))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(cache)], type_ignores=[])),
                     "<publisher-cache>", "exec"), namespace)
        compile_function(function(source(), "publish_state"), namespace)
    compile_function(step, namespace)
    return namespace, transition, clock


@pytest.mark.parametrize("group_count", [3, 190])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("mixed_dtype", [False, True])
def test_frozen_publisher_json_is_exact_for_all_groups(tmp_path, group_count, noncontiguous, mixed_dtype):
    env = environment(group_count, noncontiguous=noncontiguous, mixed_dtype=mixed_dtype)
    inputs = [env.robot.data.body_link_pose_w.torch, env.origins, env.commands, env.episode_length_buf]
    before, rng = [value.clone() for value in inputs], torch.get_rng_state().clone()
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    old, _, _ = publisher(env, tmp_path / "old" / "same_run", legacy=True)
    new, transition, _ = publisher(env, tmp_path / "new" / "same_run")
    old["publish_state"]()
    assert new["step"](torch.zeros(2)) is transition
    raw = (new["args"].run_dir / "live_state.json").read_bytes()
    assert raw == (old["args"].run_dir / "live_state.json").read_bytes()
    state = json.loads(raw)
    assert [row["env_index"] for row in state["environments"]] == list(range(0, len(env.scene_groups), 2))
    assert len(state["environments"]) == group_count
    first = state["environments"][0]
    for key in ("body_link_pose_w", "commands", "episode_step"):
        assert state[key] == first[key]
    assert new["last_publish"] == old["last_publish"] == .25
    assert all(torch.equal(value, original) for value, original in zip(inputs, before))
    assert torch.equal(torch.get_rng_state(), rng)


def test_dtype_preservation_integer_boundaries_and_no_scalar_tensor_reads(monkeypatch):
    env = environment(mixed_dtype=True, noncontiguous=True)
    env.episode_length_buf[::2] = torch.tensor([2**53 + 1, 2**62 + 31, 2**63 - 1])
    selected, indices = select_group_environments(env.scene_groups, "cpu")
    transfers = []
    cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        transfers.append((tensor.dtype, tuple(tensor.shape)))
        return cpu(tensor, *args, **kwargs)

    def no_scalar(tensor):
        raise AssertionError("Snapshot must not read individual tensor scalars")

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    monkeypatch.setattr(torch.Tensor, "__int__", no_scalar)
    monkeypatch.setattr(torch.Tensor, "item", no_scalar)
    snapshot = read_selected_physics(indices, poses=env.robot.data.body_link_pose_w.torch,
        origins=env.origins, commands=env.commands, episode_steps=env.episode_length_buf)
    assert selected == (0, 2, 4)
    assert transfers == [(torch.float32, (3, 3, 7)), (torch.float64, (3, 3)),
                         (torch.float16, (3, 3)), (torch.int64, (3,))]
    assert snapshot.episode_steps == [2**53 + 1, 2**62 + 31, 2**63 - 1]
    assert all(type(value) is int for value in snapshot.episode_steps)


def test_selection_is_cached_and_live_values_are_reread(tmp_path):
    class Groups(list):
        calls = 0

        def index(self, value):
            self.calls += 1
            return super().index(value)

    env = environment()
    env.scene_groups = Groups(env.scene_groups)
    namespace, _, _ = publisher(env, tmp_path / "run")
    assert namespace["selected"] == (0, 2, 4) and env.scene_groups.calls == 3
    index_identity = id(namespace["selected_indices"])
    namespace["publish_state"]()
    env.commands[2, 0] = -12.5
    env.episode_length_buf[2] = 2**62 + 7
    namespace["publish_state"]()
    state = json.loads((tmp_path / "run/live_state.json").read_text())
    assert state["environments"][1]["commands"][0] == -12.5
    assert state["environments"][1]["episode_step"] == 2**62 + 7
    assert env.scene_groups.calls == 3 and id(namespace["selected_indices"]) == index_identity


def test_unchanged_throttle_disabled_path_and_atomic_replacement(tmp_path):
    namespace, _, clock = publisher(environment(), tmp_path / "run")
    reader = namespace["read_selected_physics"]
    reads = []
    namespace["read_selected_physics"] = lambda *a, **kw: (reads.append(True), reader(*a, **kw))[1]
    for wall, expected in ((.249, 0), (.25, 1), (.499, 1), (.50, 2)):
        clock.value = wall
        namespace["step"](None)
        assert len(reads) == expected
    assert not (tmp_path / "run/live_state.tmp").exists()
    disabled, _, _ = publisher(environment(), tmp_path / "disabled", enabled=False)
    assert "selected_indices" not in disabled
    disabled["step"](None)
    assert not list((tmp_path / "disabled").iterdir())


def test_json_failure_does_not_replace_snapshot_or_advance_publish_clock(tmp_path):
    for legacy in (False, True):
        namespace, _, _ = publisher(environment(), tmp_path / str(legacy), legacy=legacy)
        path = namespace["args"].run_dir / "live_state.json"
        path.write_text("previous snapshot")
        namespace["env"].commands[2, 1] = float("nan")
        with pytest.raises(ValueError, match="Out of range float"):
            namespace["publish_state"]()
        assert path.read_text() == "previous snapshot"
        assert namespace["last_publish"] == 0.


def test_native_timing_attributes_snapshot_and_file_io_exclusively(tmp_path, monkeypatch):
    clock = Clock()
    timer = TrainingTiming(clock=clock.monotonic)
    namespace, transition, _ = publisher(environment(), tmp_path / "run", clock=clock, timer=timer)
    reader, replace = namespace["read_selected_physics"], Path.replace

    def original_step(actions):
        clock.value += 1.
        return transition

    def snapshot(*args, **kwargs):
        clock.value += .4
        return reader(*args, **kwargs)

    def write_replace(path, target):
        result = replace(path, target)
        clock.value += .2
        return result

    namespace.update(original_step=original_step, read_selected_physics=snapshot)
    monkeypatch.setattr(Path, "replace", write_replace)
    clock.value += .2
    namespace["step"](None)
    result = timer.finish_iteration()
    assert result["Timing/env_step_seconds"] == pytest.approx(1.)
    assert result["Timing/state_publish_seconds"] == pytest.approx(.6)
    assert result["Timing/state_publish_calls"] == 1
    assert result["Timing/unattributed_wall_seconds"] == pytest.approx(.2)
    assert result["Timing/iteration_wall_seconds"] == pytest.approx(1.8)


def test_every_other_training_statement_is_unchanged():
    class RemovePublication(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == "publish_state":
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) == "args.publish_state":
                return None  # Publication source registration and one-time selection cache.
            if publish_condition(node):
                node.body = [ast.Pass()]
            return self.generic_visit(node)

    normalize = lambda tree: ast.dump(RemovePublication().visit(deepcopy(tree)), include_attributes=False)
    assert normalize(source()) == normalize(source(True))


def test_publication_helper_is_registered_in_runtime_sources_only_when_enabled():
    registration = next(node for node in ast.walk(source()) if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args.publish_state"
        and any(isinstance(item, ast.Expr) and ast.unparse(item.value).startswith("source_files.append(")
                for item in node.body))
    for enabled in (False, True):
        namespace = dict(args=SimpleNamespace(publish_state=enabled), source_files=["original.py"])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(registration)], type_ignores=[])),
                     "<snapshot-source-registration>", "exec"), namespace)
        assert namespace["source_files"] == (["original.py", "src/wheeled_tasks/chassis/state_snapshot.py"]
            if enabled else ["original.py"])
