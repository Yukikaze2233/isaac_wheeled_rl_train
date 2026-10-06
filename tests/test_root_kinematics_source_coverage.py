"""Both learning and fixed evaluation must record the loaded root adapter."""
import ast
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.mark.parametrize("entry,assignment", [("scripts/train_chassis.py", "source_files"), ("scripts/evaluate_chassis.py", "sources")])
def test_root_adapter_in_initial_runtime_source_snapshot(entry, assignment):
    tree=ast.parse((ROOT/entry).read_text())
    declarations=[n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id==assignment for t in n.targets) and isinstance(n.value, ast.List)]
    assert len(declarations)==1
    paths=[e.value for e in declarations[0].value.elts if isinstance(e,ast.Constant) and isinstance(e.value,str)]
    assert "src/wheeled_tasks/chassis/root_kinematics.py" in paths
    assert (ROOT/"src/wheeled_tasks/chassis/root_kinematics.py").is_file()


def test_dynamics_validation_model_writes_refresh_and_record_root_adapter():
    tree = ast.parse((ROOT / "scripts/validate_chassis_dynamics.py").read_text())
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and ast.unparse(node.target) == "(label, variant)"]
    assert len(loops) == 1
    # Each complete apply may alter COM principal axes; no read may precede the
    # refresh of the final native value. An apply exception leaves it invalid.
    assert [ast.unparse(node) for node in loops[0].body[:3]] == [
        "env.root_kinematics.invalidate_model()",
        "variant.apply(env.robot)",
        "env.root_kinematics.refresh_model()",
    ]
    source_snapshots = [value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                        for key, value in zip(node.keys, node.values)
                        if isinstance(key, ast.Constant) and key.value == "source_sha256"]
    assert len(source_snapshots) == 1 and isinstance(source_snapshots[0], ast.DictComp)
    paths = ast.literal_eval(source_snapshots[0].generators[0].iter)
    assert "src/wheeled_tasks/chassis/root_kinematics.py" in paths
