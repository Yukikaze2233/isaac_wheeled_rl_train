"""Optional entropy wiring and native CPU probability/state invariants."""

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_recovery_rollout_mask import algorithm, collect
from wheeled_algo.recovery_rollout import attach_recovery_rollout_mask

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/train_chassis.py"


@pytest.fixture(scope="module")
def entry():
    spec = importlib.util.spec_from_file_location("train_chassis_entropy_test", ENTRY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree():
    return ast.parse(ENTRY.read_text())


def execute(nodes, context):
    module = ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(ENTRY), "exec"), context)


def entropy_override(entry, contract):
    # Execute the production config assignment, before the native runner is built.
    node = next(n for n in ast.walk(tree()) if isinstance(n, ast.If)
                and n.lineno > 500 and ast.unparse(n.test) == "'entropy_coef' in c")
    config = SimpleNamespace(algorithm=SimpleNamespace(entropy_coef=.005))
    execute([node], {"c": contract, "cfg": config,
                    "validate_entropy_coef": entry.validate_entropy_coef})
    return config.algorithm.entropy_coef


@pytest.mark.parametrize("value", [0, 0., .005, .0025, 1])
def test_finite_nonnegative_scalar(entry, value):
    assert entry.validate_entropy_coef(value) == float(value)


@pytest.mark.parametrize("value", [True, False, None, "0", [], {}, -.001,
                                   float("nan"), float("inf"), -float("inf"), 10 ** 1000])
def test_invalid_contract_rejected_before_asset_checks(entry, tmp_path, value):
    contract = tmp_path / "invalid.json"
    contract.write_text(json.dumps({"entropy_coef": value}))
    with pytest.raises(ValueError, match="entropy_coef"):
        entry.preflight(contract)


@pytest.mark.parametrize("contract,expected", [({}, .005), ({"entropy_coef": 0}, 0.),
                                               ({"entropy_coef": .0025}, .0025)])
def test_optional_assignment_preserves_default_and_explicit_zero(entry, contract, expected):
    cfg_tree = ast.parse((ROOT / "src/wheeled_tasks/agents/v40_ppo_cfg.py").read_text())
    defaults = [kw.value.value for node in ast.walk(cfg_tree) if isinstance(node, ast.Call)
                for kw in node.keywords if kw.arg == "entropy_coef"]
    assert defaults == [.005]
    original = deepcopy(contract)
    assert entropy_override(entry, contract) == expected
    assert contract == original


@pytest.mark.parametrize("optional", [{}, {"entropy_coef": 0.}, {"entropy_coef": .0025}])
def test_valid_preflight_preserves_optional_field_without_other_contract_changes(entry, tmp_path, optional):
    bundle = tmp_path / "asset"
    bundle.mkdir()
    manifest = {"model_kind": "v5_gas_spring_closedchain_research", "files_sha256": {}}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    settings = {"contract_id": "v5-gas-spring-commanded-research-v1",
                "task_source": "upper_level_commands", "stages": [{"updates": 1}],
                "total_updates": 1, "asset_directory": str(bundle),
                "asset_manifest_sha256": entry.digest(bundle / "manifest.json"), **optional}
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps(settings))
    assert entry.preflight(contract) == (settings, manifest)


@pytest.mark.parametrize("actual", [.006, float("nan"), None])
def test_runtime_readback_rejects_unexpected_native_coefficient(entry, actual):
    with pytest.raises(RuntimeError, match="entropy coefficient"):
        entry.verify_entropy_coef(SimpleNamespace(entropy_coef=actual), .005)


def test_same_contract_saved_entropy_is_inherited_without_silent_reset(entry):
    node = next(n for n in ast.walk(tree()) if isinstance(n, ast.If)
                and ast.unparse(n.test) == "'entropy_coef' in infos")
    context = {"infos": {"entropy_coef": 0.}, "migrating": False,
               "cfg": SimpleNamespace(algorithm=SimpleNamespace(entropy_coef=.005)),
               "validate_entropy_coef": entry.validate_entropy_coef}
    with pytest.raises(ValueError, match="Same-contract"):
        execute([node], context)
    context["cfg"].algorithm.entropy_coef = 0.
    execute([node], context)
    context["cfg"].algorithm.entropy_coef = .005
    context["migrating"] = True  # Named provenance validation is earlier in main().
    execute([node], context)
    context["infos"], context["migrating"] = {}, False
    execute([node], context)  # Legacy seals lack this field; do not invent one.


@pytest.mark.parametrize("actor_enabled", [False, True])
@pytest.mark.parametrize("drift", ["before", "after", None])
def test_actual_update_checks_both_sides_and_retains_completed_clocks(entry, drift, actor_enabled):
    function = next(n for n in ast.walk(tree()) if isinstance(n, ast.FunctionDef) and n.name == "update")
    start = next(i for i, n in enumerate(function.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "actor_enabled")
    end = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == "report['entropy_coef']")
    report = {"parent_updates": 10 if actor_enabled else 0, "successful_updates": 0,
              "actor_updates_in_block": 0}
    native = SimpleNamespace(entropy_coef=.005, learning_rate=1e-4,
                             optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]))
    env = SimpleNamespace(stage_actor_update=62, global_actor_update=162, task_actor_update=70)
    calls = []

    def update(*_args, **_kwargs):
        calls.append("native_optimizer")
        if drift == "after":
            native.entropy_coef = .006
        return {"value": 1.}

    context = {"report": report, "warmup_updates": 10,
               "runner": SimpleNamespace(alg=native),
               "cfg": SimpleNamespace(algorithm=SimpleNamespace(entropy_coef=.005)),
               "learning_schedule": "fixed", "policy_probe": None, "learning_audit": None,
               "timing": None, "original_update": update, "a": (), "kw": {},
               "timed_call": entry.timed_call, "verify_entropy_coef": entry.verify_entropy_coef,
               "c": {"task_campaign": {}, "learning_rate": 1e-4}, "env": env,
               "stage_actor_start": 62, "task_actor_start": 70,
               "args": SimpleNamespace(global_actor_update_offset=100, actor_updates_consumed=42)}
    context["c"]["task_campaign"] = {"domain": "recovery"}
    if drift == "before":
        native.entropy_coef = .006
    if drift is None:
        execute(function.body[start:end + 1], context)
        assert report["entropy_coef"] == .005 and report["successful_updates"] == 1
    else:
        with pytest.raises(RuntimeError, match="differs"):
            execute(function.body[start:end + 1], context)
    completed = int(drift != "before")
    actor_updates = completed * int(actor_enabled)
    assert len(calls) == report["successful_updates"] == completed
    assert report["actor_updates_in_block"] == actor_updates
    assert env.stage_actor_update == 62 + actor_updates
    assert env.task_actor_update == 70 + actor_updates
    assert env.global_actor_update == 162 + actor_updates
    if completed:
        assert report["actor_updates_consumed"] == 42 + actor_updates
        assert report["task_actor_update"] == 70 + actor_updates


@pytest.mark.parametrize("coefficient", [float("nan"), float("inf"), -float("inf"), None, "0.005"])
def test_invalid_native_entropy_completion_is_json_safe_preserves_error_and_never_seals(entry, tmp_path, coefficient):
    main = next(n for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == "main")
    outer = next(n for n in main.body if isinstance(n, ast.Try) and n.finalbody)
    saved = []
    native = SimpleNamespace(entropy_coef=coefficient,
                             optimizer=SimpleNamespace(param_groups=[{"lr": 1e-4}]))
    runner = SimpleNamespace(alg=native, logger=SimpleNamespace(stop_logging_writer=lambda: None),
                             save=lambda *_a, **_kw: saved.append("save"))
    original_error = "original post-update entropy invariant traceback"
    report = {"status": "failed", "error": original_error, "successful_updates": 1,
              "actor_updates_in_block": 1, "actor_updates_consumed": 43,
              "stage_actor_update": 63, "global_actor_update": 163, "task_actor_update": 71,
              "learning_lineage_updates": 10}
    execute(outer.finalbody, {"args": SimpleNamespace(run_dir=tmp_path), "report": report,
        "runner": runner, "profiler": None, "env": None, "timing": None, "launcher": None,
        "json": json, "datetime": entry.datetime, "timezone": entry.timezone,
        "validate_entropy_coef": entry.validate_entropy_coef})
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["status"] == "failed" and completion["error"] == original_error
    assert completion["entropy_coef"] is None
    assert completion["entropy_coef_status"] == "invalid_native_value"
    assert completion["entropy_coef_invalid_native_value"] == repr(coefficient)
    assert completion["successful_updates"] == completion["actor_updates_in_block"] == 1
    assert completion["final_learning_lineage_updates"] == 11
    assert completion["actor_updates_consumed"] == 43 and completion["task_actor_update"] == 71
    assert saved == [] and not (tmp_path / "model_final.pt").exists()


def test_receipts_bind_config_progress_checkpoint_and_resume(entry, tmp_path):
    main_tree = tree()
    statements = list(ast.walk(main_tree))
    config_write = next(n for n in statements if isinstance(n, ast.Expr)
                        and "'agent_config.json'" in ast.unparse(n))
    for contract in ({}, {"entropy_coef": 0.}, {"entropy_coef": .0025}):
        coefficient = entropy_override(entry, contract)
        runner_config = {"algorithm": {"entropy_coef": coefficient},
                         "actor": {"distribution_cfg": {"init_std": .2}}}
        execute([config_write], {"args": SimpleNamespace(run_dir=tmp_path),
                                "json": json, "runner_config": runner_config})
        assert json.loads((tmp_path / "agent_config.json").read_text()) == runner_config
    dicts = [n for n in statements if isinstance(n, ast.Dict)]
    keyed = lambda n: {k.value: v for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant)}
    progress = next(keyed(n) for n in dicts if "optimizer_phase" in keyed(n))
    assert ast.unparse(progress["entropy_coef"]) == "report['entropy_coef']"
    assert ast.unparse(progress["entropy_coef_source"]) == "report['entropy_coef_source']"
    save_fn = next(n for n in statements if isinstance(n, ast.FunctionDef) and n.name == "save")
    entropy_check = save_fn.body[0]
    assert "verify_entropy_coef(runner.alg, cfg.algorithm.entropy_coef)" in ast.unparse(entropy_check)
    save_dicts = [keyed(n) for n in ast.walk(save_fn) if isinstance(n, ast.Dict)]
    assert sum("entropy_coef" in n for n in save_dicts) == 2  # Native infos plus immutable seal.
    receipt = next(n for n in statements if isinstance(n, ast.Expr)
                   and ast.unparse(n).startswith("receipt.update("))
    execute([receipt], {"receipt": (restored := {}), "infos": {},
        "report": {"initial_learning_rate": 1e-4, "initial_entropy_coef": 0.,
                   "entropy_coef_source": "contract"}, "args": SimpleNamespace(learning_rate_scale=None)})
    assert restored["parent_entropy_coef"] is None
    assert restored["effective_entropy_coef"] == 0. and restored["entropy_coef_source"] == "contract"


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_native_only_entropy_gradient_changes_raw_probabilities_gae_and_restore_are_equal(entry, monkeypatch):
    # Use the installed native PPO/Adam/masked GAE, never a replacement algorithm.
    mask = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 1]], dtype=torch.bool)
    parent = algorithm()
    attach_recovery_rollout_mask(parent, num_envs=2)
    parent.compute_returns(collect(parent, mask))
    parent.update()  # Populate real Adam moments before the two exact restores.
    saved = deepcopy(parent.save())
    assert saved["optimizer_state_dict"]["state"]
    arms = []
    for contract in ({}, {"entropy_coef": 0.}):
        alg = algorithm()
        alg.load(deepcopy(saved), load_cfg=None, strict=True)
        assert_state_equal(saved, alg.save())
        alg.entropy_coef = entropy_override(entry, contract)
        entry.verify_entropy_coef(alg, alg.entropy_coef)
        alg.num_learning_epochs = alg.num_mini_batches = 1
        attach_recovery_rollout_mask(alg, num_envs=2)
        alg.compute_returns(collect(alg, mask))
        arms.append(alg)
    for name in ("actions", "actions_log_prob", "returns", "advantages", "values", "rewards"):
        assert torch.equal(getattr(arms[0].storage, name), getattr(arms[1].storage, name)), name
    for a, b in zip(arms[0].storage.distribution_params, arms[1].storage.distribution_params):
        assert torch.equal(a, b)
    results, gradients, counts = [], [], []
    # Isolate the analytical entropy gradient before clipping/Adam; no second step.
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *_a, **_kw: None)
    for alg in arms:
        calls = []
        monkeypatch.setattr(alg.optimizer, "step", lambda: calls.append("step"))
        torch.manual_seed(741)
        results.append(alg.update())
        gradients.append({"actor." + k: v.grad.detach().clone() for k, v in alg.actor.named_parameters()})
        gradients[-1].update({"critic." + k: v.grad.detach().clone() for k, v in alg.critic.named_parameters()})
        counts.append(len(calls))
        assert_state_equal(saved, alg.save())  # Sampling and loss do not reset saved Adam/std.
    assert results[0] == results[1] and counts == [1, 1]
    std_name = next(k for k in gradients[0] if k.endswith("distribution.std_param"))
    sigma = arms[0].actor.distribution.std_param.detach()
    torch.testing.assert_close(gradients[0][std_name] - gradients[1][std_name], -.005 / sigma,
                               atol=1e-6, rtol=1e-5)
    for name in gradients[0].keys() - {std_name}:
        assert torch.equal(gradients[0][name], gradients[1][name]), name
    assert not torch.cuda.is_initialized()
