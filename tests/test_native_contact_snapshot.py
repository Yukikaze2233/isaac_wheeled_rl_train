"""Selected native CSR rows and aliased normal/friction buffer ownership."""
import ast
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch


def methods():
    path = Path(__file__).resolve().parents[1] / "src/wheeled_tasks/chassis/env.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "ChassisEnv")
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef)
             and n.name in ("_compact_formal_contact_data", "_formal_native_contact_snapshot")]
    scope = dict(torch=torch, wp=NS(to_torch=lambda value: value))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def test_selected_contact_rows_preserve_native_mapping_and_survive_getter_aliasing():
    counts, starts = torch.zeros(4, 2, dtype=torch.int32), torch.zeros(4, 2, dtype=torch.int32)
    data = torch.arange(12, dtype=torch.float32)
    calls = []

    def normal(*, dt):
        assert dt == .005
        calls.append("normal")
        counts.copy_(torch.tensor([[1, 0], [0, 1], [0, 2], [1, 0]]))
        starts.copy_(torch.tensor([[8, 0], [0, 7], [0, 1], [5, 0]]))
        return data, data[:, None].expand(-1, 3), data[:, None].expand(-1, 3), data, counts, starts

    def friction(*, dt):
        calls.append("friction")
        counts.copy_(torch.tensor([[1, 0], [0, 0], [1, 0], [0, 1]]))
        starts.copy_(torch.tensor([[7, 0], [0, 0], [4, 0], [0, 10]]))
        return data[:, None].expand(-1, 3), data[:, None].expand(-1, 3), counts, starts

    view = NS(sensor_paths=["/env2/a", "/env2/b", "/env0/a", "/env0/b"],
              filter_paths=[["/floor", "/rough"] for _ in range(4)],
              sensor_count=4, filter_count=2, max_contact_data_count=12,
              get_contact_data=normal, get_friction_data=friction)
    env = NS(dt=.005, device="cpu", num_envs=3, robot=NS(body_names=["a", "b"]),
             contact_views=[(torch.tensor([2, 0]), view), (torch.tensor([1]), object())])
    for name, function in methods().items():
        if name.startswith("_formal") or name.startswith("_compact"):
            setattr(env, name, MethodType(function, env))
    result = env._formal_native_contact_snapshot(environment_ids=torch.tensor([0]))
    assert calls == ["normal", "friction"]
    assert len(result["groups"]) == 1
    group = result["groups"][0]
    assert group["environment_ids"].tolist() == [0]
    assert group["sensor_paths"] == ["/env0/a", "/env0/b"]
    assert group["selected_sensor_count"] == 2 and group["sensor_count"] == 4
    normal_data, friction_data = group["contact"], group["friction"]
    assert normal_data["status"] == friction_data["status"] == "ok"
    assert normal_data["source_sensor_indices"].tolist() == [2, 3]
    assert normal_data["counts"].tolist() == [[0, 2], [1, 0]]
    assert normal_data["buffer_indices"].tolist() == [1, 2, 5]
    assert normal_data["fields"]["normal_force_N"].tolist() == [1., 2., 5.]
    assert friction_data["buffer_indices"].tolist() == [4, 10]
    assert friction_data["fields"]["force_world_N"][:, 0].tolist() == [4., 10.]
    counts.fill_(99)
    data.fill_(99)
    assert normal_data["counts"].tolist() == [[0, 2], [1, 0]]
    assert normal_data["fields"]["normal_force_N"].tolist() == [1., 2., 5.]
    with pytest.raises(ValueError, match="unique actual"):
        env._formal_native_contact_snapshot(environment_ids=torch.tensor([0, 0]))
