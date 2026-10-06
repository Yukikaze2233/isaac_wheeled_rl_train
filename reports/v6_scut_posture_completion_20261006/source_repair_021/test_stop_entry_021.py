"""Cheap CPU-only fixture: compile only entry(), never the emitted stop code."""
import ast
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
EXPECTED_SHA = "3cddee45f7c90eee1f6c240adb264ce87cd6c77e0b2c8ad57415b49cc4784045"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def emitted_tree(path):
    tree = ast.parse(path.read_text())
    literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                and isinstance(n.value, str) and "def entry(a):" in n.value]
    assert len(literals) == 1
    return ast.parse(literals[0]), literals[0]

def entry_node(tree):
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "entry"]
    assert len(nodes) == 1
    return nodes[0]

def run():
    emitter = ROOT / "emit_stop_previous_021.py"
    before = digest(emitter)
    assert before == EXPECTED_SHA
    tree, raw = emitted_tree(emitter)
    common, _ = emitted_tree(ROOT / "deploy_recovery_quality_021.py")
    node = entry_node(tree)
    assert ast.dump(node, include_attributes=False) == ast.dump(entry_node(common), include_attributes=False)
    namespace = {"pathlib": pathlib}
    # This one pure function is the only executed fragment of either artifact.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<entry-only>", "exec"), namespace)
    entry = namespace["entry"]
    fixture = json.loads((ROOT / "prestop_actual_020.json").read_text())
    expected = {3494082: "chassis_remote_job.py", 3494087: "run_task_campaign.py", 3494096: "train_chassis.py"}
    got = {owner["pid"]: entry(owner["command"]).name for owner in fixture["owners"]}
    assert got == expected
    old = pathlib.Path("/home/kaiser/robot-rl-sim60/experiments/v6-task-campaign-20261001T110045Z-282358/source_7f315f30741b/scripts")
    assert all(entry(owner["command"]).parent == old for owner in fixture["owners"])
    wrapper = next(x for x in fixture["owners"] if x["pid"] == 3494082)
    legacy_count = len([x for x in wrapper["command"] if x.endswith(".py")])
    assert legacy_count == 2
    assert next(x["command"] for x in fixture["owners"] if x["pid"] == 3494087) == fixture["job"]["command"]
    assert fixture["progress"]["pid"] == 3494096
    for argv in (["python"], ["python", "-B"], ["python", "-c", "later.py"], ["python", "-m", "module", "later.py"]):
        assert entry(argv) is None
    assert entry(["python", "foreign.py", "--", str(old / "run_task_campaign.py")]).name == "foreign.py"
    # Static verification only: no signal or full remote template execution.
    compile(raw, "<stop-syntax-only>", "exec")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == "os" and n.func.attr == "kill"]
    assert len(calls) == 1 and ast.unparse(calls[0]) == "os.kill(3494087, signal.SIGTERM)"
    marker = raw.index(".open('x')")
    file_sync = raw.index("f.flush();os.fsync(f.fileno())")
    dir_sync = raw.index("os.fsync(directory)")
    signal = raw.index("os.kill(3494087,signal.SIGTERM)")
    assert marker < file_sync < dir_sync < signal
    assert "assert not (r/'stop_requested_021.json').exists()" in raw
    assert "assert progress['pid']==3494096" in raw
    rejection = json.loads((ROOT / "stop_attempt_021.entry_parser_rejected.json").read_text())
    assert rejection["status"] == "rejected_before_marker_and_signal"
    assert rejection["PID_signalled"] is None and rejection["production_state_changed"] is False
    after = digest(emitter)
    assert after == before
    return {"status": "passed", "emitter_sha256_before": before, "emitter_sha256_after": after,
            "actual_fixture_sha256": digest(ROOT / "prestop_actual_020.json"),
            "actual_first_entries": got, "legacy_wrapper_script_count": legacy_count,
            "first_entry_wrapper_script_count": 1, "entry_AST_equals_deploy_COMMON": True,
            "actual_supervisor_matches_job": True, "progress_pid": fixture["progress"]["pid"],
            "fixture_actor": fixture["progress"]["task_actor_update"],
            "negative_cases": 5, "single_SIGTERM_after_file_and_directory_fsync": True,
            "remote_code_executed": False, "signals_sent": 0, "SSH": False, "GPU": False}

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
