"""Source/CPU regression only: never import or mock an Isaac environment runtime."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[2]
NEW_PYTHON = [
    "src/wheeled_tasks/direct/v40_serial/__init__.py",
    "src/wheeled_tasks/direct/v40_serial/env_cfg.py",
    "src/wheeled_tasks/direct/v40_serial/env.py",
    "src/wheeled_tasks/agents/v40_ppo_cfg.py",
    "src/wheeled_world/assets/v40.py",
    "scripts/train_v40.py", "scripts/play_v40.py", "scripts/check_v40_env.py",
]


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def tree(path):
    return ast.parse(source(path))


def load_cli():
    spec = importlib.util.spec_from_file_location("v40_train_cpu_test", ROOT / "scripts/train_v40.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", NEW_PYTHON)
def test_sources_parse_without_sim_import(path):
    tree(path)


@pytest.mark.parametrize("script", ["train_v40.py", "play_v40.py", "check_v40_env.py"])
def test_help_is_cpu_only(script):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"],
                          capture_output=True, text=True, timeout=20,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert proc.returncode == 0, proc.stderr
    assert "--preflight-only" in proc.stdout and "--research" in proc.stdout
    for node in tree("scripts/" + script).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            assert not name.startswith(("isaaclab", "isaacsim", "rsl_rl"))


@pytest.mark.parametrize("arguments", [["--num_envs", "3"], ["--max_steps", "41"], ["--apply"]])
def test_checker_rejects_unbounded_or_implicit_sim(arguments):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/check_v40_env.py"), *arguments],
                          capture_output=True, text=True, timeout=20)
    assert proc.returncode == 2
    assert "error:" in proc.stderr


def test_save_hook_covers_automatic_and_explicit_saves():
    cli = load_cli()
    calls = []

    class StockSaveStandIn:
        # CPU unit test of the save adapter only, not a substitute Isaac/RSL runtime.
        def save(self, path, infos=None):
            calls.append((path, infos))

    runner = StockSaveStandIn()
    manifest = {key: "bound-" + key for key in cli.METADATA_KEYS}
    cli.bind_checkpoint_metadata(runner, manifest)
    runner.save("automatic.pt")
    runner.save("final.pt", {"note": "final"})
    assert all(all(info[key] == manifest[key] for key in manifest) for _, info in calls)
    with pytest.raises(ValueError, match="cannot replace"):
        runner.save("bad.pt", {"contract_id": "unrelated"})
    with pytest.raises(ValueError):
        runner.save("nan.pt", {"metric": float("nan")})
    with pytest.raises(ValueError):
        runner.save("tuple.pt", {"metric": (1, 2)})
    with pytest.raises(ValueError):
        runner.save("key.pt", {1: "not a string key"})


def test_research_does_not_waive_false_collision_gate(monkeypatch):
    cli = load_cli()
    # Pure gate test: no Isaac module imported or faked.
    core = types.ModuleType("wheeled_tasks.v40.contract")
    core.load_contract = lambda path: {"contract_id": "test"}
    core.contract_digest = lambda contract: "a" * 64
    core.validate_asset = lambda contract, allow_research: {
        "manifest": {"collision_validation": {"passed": False}},
        "asset_manifest_sha256": "b" * 64,
    }
    monkeypatch.setitem(sys.modules, "wheeled_tasks.v40.contract", core)
    monkeypatch.setattr(cli.importlib.metadata, "version", lambda package: cli.TARGET_VERSIONS[package])
    args = types.SimpleNamespace(research=True, stage="stand", num_envs=1, contract=None)
    report, _, _ = cli.preflight(args)
    assert not report["ready"]
    assert "collision_validation.passed" in report["blockers"][0]
    assert report["simulation_started"] is False


@pytest.mark.parametrize('installed,expected,match', [
    ('5.1.0.0', '5.1.0', True), ('5.1.0', '5.1.0', True),
    ('5.1.1', '5.1.0', False), ('5.1.0.dev1', '5.1.0', False),
    ('2.7.0+cu128', '2.7.0+cu128', True), ('2.7.0+cu130', '2.7.0+cu128', False),
    ('2.7.0', '2.7.0+cu128', False), ('3.0.1', '3.0.1', True),
])
def test_runtime_release_and_cuda_build_matching(installed, expected, match):
    assert load_cli().runtime_version_matches(installed, expected) is match


def test_stock_ppo_configuration_is_explicit():
    text = source("src/wheeled_tasks/agents/v40_ppo_cfg.py")
    cls = next(node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.ClassDef))
    assignments = {node.targets[0].id: node.value for node in cls.body if isinstance(node, ast.Assign)}
    assert ast.literal_eval(assignments["num_steps_per_env"]) == 48
    assert ast.literal_eval(assignments["obs_groups"]) == {"policy": ["policy"], "critic": ["critic"]}
    policy = {kw.arg: ast.literal_eval(kw.value) for kw in assignments["policy"].keywords}
    assert policy["class_name"] == "ActorCritic"
    assert policy["actor_obs_normalization"] is False and policy["critic_obs_normalization"] is False
    assert policy["actor_hidden_dims"] == policy["critic_hidden_dims"] == [256, 128, 64]
    assert policy["activation"] == "elu"
    algorithm = {kw.arg: ast.literal_eval(kw.value) for kw in assignments["algorithm"].keywords}
    expected = {"class_name": "PPO", "learning_rate": 1e-4, "gamma": .99, "lam": .95,
                "num_learning_epochs": 5, "num_mini_batches": 4, "clip_param": .2,
                "entropy_coef": .005, "value_loss_coef": 4.0, "schedule": "adaptive"}
    assert all(algorithm[k] == v for k, v in expected.items())


def test_env_wiring_and_history_are_explicit_in_source():
    text = source("src/wheeled_tasks/direct/v40_serial/env.py")
    assert 'self.scene.articulations["robot"]' in text
    assert 'self.scene.sensors["contact"]' in text
    assert "clone_environments(" in text and "filter_collisions(" in text
    assert "dtype=torch.long" in text
    assert "self.history.update(obs25, tick=int(self.common_step_counter))" in text
    assert "self.history.reset(env_ids)" in text
    assert "self.scene.env_origins[env_ids]" in text
    assert "return terminated, time_out" in text
    assert "compute_torques(" in text and "set_joint_effort_target_index(" in text
    cfg = source("src/wheeled_tasks/direct/v40_serial/env_cfg.py")
    assert "observation_space = 125" in cfg and "state_space = 29" in cfg
    assert "is_finite_horizon = False" in cfg


def test_asset_has_no_second_pd_or_frame_rotation():
    text = source("src/wheeled_world/assets/v40.py")
    assert "UrdfFileCfg(" in text and "fix_base=False" in text
    assert "activate_contact_sensors=True" in text
    assert "stiffness=0.0" in text and "damping=0.0" in text
    assert "enabled_self_collisions=True" in text
    assert "quat_from_euler" not in text


def test_actual_contract_matches_runner_and_rejects_manifest_architecture_lie():
    cli = load_cli()
    contract = json.loads((ROOT / "contracts/own_v40_v1.json").read_text(encoding="utf-8"))
    cli.check_training_baseline(contract)
    contract["policy"]["activation"] = "tanh"
    with pytest.raises(ValueError, match="fixed V40 stock PPO baseline"):
        cli.check_training_baseline(contract)


def test_train_rejects_existing_run_and_ambiguous_load_mode():
    script = str(ROOT / "scripts/train_v40.py")
    existing = subprocess.run([sys.executable, script, "--preflight-only", "--research", "--run-dir", str(ROOT)],
                              capture_output=True, text=True, timeout=30)
    assert existing.returncode == 2
    report = json.loads(existing.stdout)
    assert any("run directory already exists" in message for message in report["blockers"])
    assert report["simulation_started"] is False
    ambiguous = subprocess.run([sys.executable, script, "--resume", "a.pt", "--finetune", "b.pt"],
                               capture_output=True, text=True, timeout=20)
    assert ambiguous.returncode == 2 and "not allowed with" in ambiguous.stderr


def isolated_tensor_method(name):
    """Execute only one source method with real core tensors, never an Isaac class.

    This is a CPU adapter regression, not a simulator mock or integration pass.
    """
    torch = pytest.importorskip("torch")
    load_cli()  # Add src without importing Isaac.
    from wheeled_tasks.v40 import core
    cls = next(node for node in tree("src/wheeled_tasks/direct/v40_serial/env.py").body
               if isinstance(node, ast.ClassDef) and node.name == "V40Env")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch, "compute_reward_terms": core.compute_reward_terms,
                 "build_observation": core.build_observation, "build_critic": core.build_critic}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "isolated_v40_tensor_adapter", "exec"), namespace)
    return torch, core, namespace[name]


@pytest.mark.parametrize("valid_rows", [[True, True], [True, False], [False, False]])
def test_reward_adapter_does_not_scale_core_rewards_twice(valid_rows):
    torch, core, get_rewards = isolated_tensor_method("_get_rewards")
    contract = core.load_contract()
    q = torch.tensor(contract["joints"]["nominal_positions"]).repeat(2, 1)
    zero3 = torch.zeros(2, 3)
    zero6 = torch.zeros(2, 6)
    height = torch.full((2,), .32)
    commands = torch.tensor([[0., 0., .32], [0., 0., .32]])
    validity = torch.tensor(valid_rows)
    data = types.SimpleNamespace(
        root_com_lin_vel_b=types.SimpleNamespace(torch=zero3),
        root_com_ang_vel_b=types.SimpleNamespace(torch=zero3),
        projected_gravity_b=types.SimpleNamespace(torch=torch.tensor([[0., 0., -1.], [0., 0., -1.]])),
    )
    adapter = types.SimpleNamespace(
        contract=contract, num_envs=2, device="cpu", robot=types.SimpleNamespace(data=data),
        _joint_state=lambda: (q, zero6), _base_height=lambda: height,
        _finite_state=validity, commands=commands.clone(), actions=zero6, previous_actions=zero6,
        torques=zero6, reset_terminated=~validity, extras={}, _episode_sums={},
        common_step_counter=1, _last_reward_tick=0,
        _command_ticks_left=torch.tensor([1, 2]), _commands_due=torch.zeros(2, dtype=torch.bool),
    )
    reward = get_rewards(adapter)
    # Actual default core stand reward: (2+1+2)*.01=.05, NOT .0005.
    expected = torch.where(validity, torch.full((2,), .05), torch.full((2,), -5.0))
    torch.testing.assert_close(reward, expected)
    torch.testing.assert_close(adapter.commands, commands)  # No resample before this transition's reward.
    assert adapter._command_ticks_left.tolist() == [0, 1]
    assert adapter._commands_due.tolist() == [True, False]
    for name, value in adapter._episode_sums.items():
        torch.testing.assert_close(adapter.extras["log"]["Reward/" + name], value.mean())


def test_observation_adapter_uses_real_history_critic_and_current_action():
    torch, core, get_observations = isolated_tensor_method("_get_observations")
    contract = core.load_contract()
    q = torch.tensor(contract["joints"]["nominal_positions"]).repeat(2, 1)
    height = torch.full((2,), .32)
    data = types.SimpleNamespace(
        root_com_ang_vel_b=types.SimpleNamespace(torch=torch.zeros(2, 3)),
        projected_gravity_b=types.SimpleNamespace(torch=torch.tensor([[0., 0., -1.], [0., 0., -1.]])),
        root_com_lin_vel_b=types.SimpleNamespace(torch=torch.tensor([[.1, .2, .3], [-.1, -.2, -.3]])),
    )
    adapter = types.SimpleNamespace(
        contract=contract, robot=types.SimpleNamespace(data=data), common_step_counter=4,
        _joint_state=lambda: (q, torch.zeros_like(q)), _base_height=lambda: height,
        _sample_commands=lambda ids: None, _commands_due=torch.zeros(2, dtype=torch.bool),
        commands=torch.tensor([[0., 0., .32], [0., 0., .32]]), actions=torch.full((2, 6), .2),
        history=core.HistoryStack(2, "cpu"),
    )
    first = get_observations(adapter)
    assert first["policy"].shape == (2, 125) and first["critic"].shape == (2, 29)
    torch.testing.assert_close(first["policy"][:, -6:], adapter.actions)
    torch.testing.assert_close(first["critic"][:, -4:-1], data.root_com_lin_vel_b.torch)
    torch.testing.assert_close(first["critic"][:, -1], height)
    first_snapshot = first["policy"].clone()
    adapter.actions.fill_(.4)
    adapter.common_step_counter += 1
    second = get_observations(adapter)
    torch.testing.assert_close(first["policy"], first_snapshot)  # Pending PPO transition is immutable.
    torch.testing.assert_close(second["policy"][:, -6:], adapter.actions)
    torch.testing.assert_close(get_observations(adapter)["policy"], second["policy"])
    adapter.history.reset(torch.tensor([0]))
    adapter.actions[0] = 0
    reset = get_observations(adapter)["policy"].reshape(2, 5, 25)
    torch.testing.assert_close(reset[0], reset[0, -1:].expand(5, 25))
    torch.testing.assert_close(reset[1].flatten(), second["policy"][1])


# Pure USD-interface doubles: no pxr or Isaac import, and no physics claims.
FAKE_RIGID_BODY_API = object()


class FakeStage:
    def __init__(self, count=2):
        self.edits = 0
        self.corrupt_readback = False
        self.prims = {}
        self.env_paths = [f"/World/envs/env_{i}" for i in range(count)]
        for env in self.env_paths:
            root = env + "/Robot"
            self.add(root, rigid=False)
            for body in ("base_link", "L_link1", "L_link2", "L_link3", "R_link1", "R_link2", "R_link3"):
                self.add(root + "/" + body, rigid=True)

    def add(self, path, rigid):
        self.prims[path] = FakePrim(self, path, rigid)
        return self.prims[path]

    def Traverse(self):
        return iter(self.prims.values())

    def GetPrimAtPath(self, path):
        return self.prims.get(str(path))


class FakePrim:
    def __init__(self, stage, path, rigid):
        self.stage, self.path, self.rigid = stage, path, rigid
        self.targets = []

    def GetName(self):
        return self.path.rsplit("/", 1)[-1]

    def GetPath(self):
        return self.path

    def IsValid(self):
        return True

    def HasAPI(self, api):
        return self.rigid if api is FAKE_RIGID_BODY_API else False


class FakeFilteredPairsAPI:
    def __init__(self, prim):
        self.prim = prim

    @classmethod
    def Apply(cls, prim):
        prim.stage.edits += 1
        return cls(prim)

    def GetFilteredPairsRel(self):
        return self

    def CreateFilteredPairsRel(self):
        return self

    def SetTargets(self, paths):
        self.prim.stage.edits += 1
        self.prim.targets = list(paths)
        if self.prim.stage.corrupt_readback:
            self.prim.targets = self.prim.targets[:-1]
        return True

    def GetTargets(self):
        return list(self.prim.targets)


def isolated_filter_function():
    load_cli()  # The extracted adapter imports the real stdlib shared validator.
    function = next(node for node in tree("src/wheeled_world/assets/v40.py").body
                    if isinstance(node, ast.FunctionDef) and node.name == "apply_approved_collision_filters")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "fake_stage_filter_adapter", "exec"), namespace)
    schema = types.SimpleNamespace(RigidBodyAPI=FAKE_RIGID_BODY_API, FilteredPairsAPI=FakeFilteredPairsAPI)
    return namespace[function.name], schema


def synthetic_reviewed_manifest():
    # New in-memory fixture only: NEVER edit/approve the actual on-disk manifest.
    manifest = json.loads((ROOT / "assets/urdf_v40/manifest.json").read_text(encoding="utf-8"))
    manifest["collision_validation"]["passed"] = True
    for pair in manifest["adjacent_collision_filter_pairs"]:
        pair["geometry_review_supported"] = True
        pair["policy_status"] = "nominal_proxy_filter_supported"
    return manifest


def test_filter_authoring_is_named_per_env_and_idempotent():
    apply_filters, schema = isolated_filter_function()
    stage = FakeStage(2)
    manifest = synthetic_reviewed_manifest()
    report = apply_filters(stage, stage.env_paths, manifest, usd_physics=schema)
    assert len(report["envs"]) == 2
    assert all(row["pair_count"] == 6 and row["relationship_target_count"] == 12 for row in report["envs"])
    for env in stage.env_paths:
        root = env + "/Robot/"
        for prim in stage.Traverse():
            if prim.path.startswith(root):
                assert all(target.startswith(root) for target in prim.targets)
        base = stage.GetPrimAtPath(root + "base_link")
        assert set(base.targets) == {root + "L_link1", root + "R_link1"}
    assert apply_filters(stage, stage.env_paths, manifest, usd_physics=schema) == report
    assert "PhysX effects unverified" in report["scope"]


@pytest.mark.parametrize("mutation", ["passed_only", "false_geometry", "pending_status", "unknown_pair", "duplicate_pair"])
def test_unapproved_filters_reject_before_any_stage_edits(mutation):
    apply_filters, schema = isolated_filter_function()
    stage = FakeStage(2)
    manifest = synthetic_reviewed_manifest()
    pairs = manifest["adjacent_collision_filter_pairs"]
    if mutation == "passed_only":
        # Reproduce the current pending review with a spoofed aggregate flag in memory.
        pairs[0]["geometry_review_supported"] = False
        pairs[0]["policy_status"] = "proposed_pending_material_review"
    elif mutation == "false_geometry":
        pairs[0]["geometry_review_supported"] = False
    elif mutation == "pending_status":
        pairs[0]["policy_status"] = "proposed_pending_material_review"
    elif mutation == "unknown_pair":
        pairs[0]["body2"] = "R_link3"
    else:
        pairs[-1] = dict(pairs[0])
    with pytest.raises(ValueError):
        apply_filters(stage, stage.env_paths, manifest, usd_physics=schema)
    assert stage.edits == 0


@pytest.mark.parametrize("mutation", ["missing_body", "duplicate_body", "cross_env_filter", "broad_root_filter"])
def test_all_clone_plans_are_validated_before_filter_edits(mutation):
    apply_filters, schema = isolated_filter_function()
    stage = FakeStage(2)
    root = stage.env_paths[1] + "/Robot"
    if mutation == "missing_body":
        del stage.prims[root + "/L_link1"]
    elif mutation == "duplicate_body":
        stage.add(root + "/nested/L_link1", rigid=True)
    elif mutation == "cross_env_filter":
        stage.prims[root + "/L_link1"].targets = [stage.env_paths[0] + "/Robot/base_link"]
    else:
        stage.prims[root].targets = ["/World/ground"]
    with pytest.raises(ValueError):
        apply_filters(stage, stage.env_paths, synthetic_reviewed_manifest(), usd_physics=schema)
    assert stage.edits == 0


def test_filter_readback_mismatch_is_a_failure_not_a_validation_flag():
    apply_filters, schema = isolated_filter_function()
    stage = FakeStage()
    stage.corrupt_readback = True
    with pytest.raises(RuntimeError, match="readback mismatch"):
        apply_filters(stage, stage.env_paths, synthetic_reviewed_manifest(), usd_physics=schema)


def test_base_visual_corner_clearance_uses_link_pose_and_env_ground_not_com():
    torch, _, clearance = isolated_tensor_method("_base_visual_clearance")
    _, _, base_height = isolated_tensor_method("_base_height")
    manifest = json.loads((ROOT / "assets/urdf_v40/manifest.json").read_text(encoding="utf-8"))
    lo, hi = manifest["base_visual_bounds_m"]
    corners = torch.tensor([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    s = 2 ** -.5
    origins = torch.tensor([[0., 0., 0.], [4., 0., 3.]])
    data = types.SimpleNamespace(
        root_link_pos_w=types.SimpleNamespace(torch=torch.tensor([[0., 0., .32], [4., 0., 3.28]])),
        root_link_quat_w=types.SimpleNamespace(torch=torch.tensor([[0., 0., 0., 1.], [0., s, 0., s]])),
        root_com_pos_w=types.SimpleNamespace(torch=torch.full((2, 3), 999.0)),
    )
    adapter = types.SimpleNamespace(robot=types.SimpleNamespace(data=data), _base_visual_corners=corners,
                                    scene=types.SimpleNamespace(env_origins=origins))
    adapter._base_height = types.MethodType(base_height, adapter)
    result = clearance(adapter)
    torch.testing.assert_close(result, torch.tensor([.32 + lo[2], .28 - hi[0]]), atol=2e-7, rtol=1e-5)
    assert result[0] > 0 and result[1] < 0  # Conservative box risk, not an exact mesh collision claim.
    env_text = source("src/wheeled_tasks/direct/v40_serial/env.py")
    assert "base_bounds_ground = clearance <= 0.0" in env_text
    assert "| base_bounds_ground" in env_text
    cfg_text = source("src/wheeled_tasks/direct/v40_serial/env_cfg.py")
    assert "replicate_physics=False" in cfg_text and "clone_in_fabric=False" in cfg_text


@pytest.fixture(scope="module")
def audited_research_filter_inputs():
    """Read actual hash-bound approval/raw records; never rewrite asset artifacts."""
    load_cli()
    from wheeled_tasks.v40.contract import audit_asset, load_contract
    audited = audit_asset(load_contract())
    return {key: audited[key] for key in ("manifest", "research_approval", "raw_manifest")}


def test_real_research_approval_filters_preserve_unrepaired_raw_flags(audited_research_filter_inputs):
    apply_filters, schema = isolated_filter_function()
    inputs = audited_research_filter_inputs
    snapshot = json.dumps(inputs, sort_keys=True)
    raw_pairs = {pair["joint"]: pair for pair in inputs["raw_manifest"]["adjacent_collision_filter_pairs"]}
    assert raw_pairs["L_joint1"]["geometry_review_supported"] is False
    assert raw_pairs["R_joint1"]["geometry_review_supported"] is False
    stage = FakeStage(2)
    report = apply_filters(stage, stage.env_paths, inputs["manifest"],
                           research_approval=inputs["research_approval"], raw_manifest=inputs["raw_manifest"],
                           usd_physics=schema)
    assert all(row["pair_count"] == 6 and row["relationship_target_count"] == 12 for row in report["envs"])
    assert json.dumps(inputs, sort_keys=True) == snapshot
    for env in stage.env_paths:
        root = env + "/Robot/"
        for prim in stage.Traverse():
            if prim.path.startswith(root):
                assert all(target.startswith(root) for target in prim.targets)
    env_text = source("src/wheeled_tasks/direct/v40_serial/env.py")
    assert 'self.research_approval = validated.get("research_approval")' in env_text
    assert 'self.raw_manifest = validated.get("raw_manifest")' in env_text
    assert "research_approval=self.research_approval, raw_manifest=self.raw_manifest" in env_text


@pytest.mark.parametrize("mutation", [
    "missing_approval", "missing_raw", "wrong_question", "wrong_scope", "changed_raw_flag",
    "relabelled_geometry", "incomplete_checks", "wrong_raw_hash", "unknown_pair", "hardware_approval",
])
def test_research_filter_rejects_incomplete_or_changed_evidence_before_edits(audited_research_filter_inputs, mutation):
    apply_filters, schema = isolated_filter_function()
    # Independent in-memory negative fixtures, never a change to actual user approval.
    inputs = json.loads(json.dumps(audited_research_filter_inputs))
    manifest, approval, raw = inputs["manifest"], inputs["research_approval"], inputs["raw_manifest"]
    if mutation == "missing_approval":
        approval = None
    elif mutation == "missing_raw":
        raw = None
    elif mutation == "wrong_question":
        approval["approval_source"]["question_id"] = "unrelated-question"
    elif mutation == "wrong_scope":
        approval["scope_id"] = "unrelated-model"
    elif mutation == "changed_raw_flag":
        raw["adjacent_collision_filter_pairs"][0]["geometry_review_supported"] = True
    elif mutation == "relabelled_geometry":
        manifest["adjacent_collision_filter_pairs"][0]["geometry_review_supported"] = True
        approval["adjacent_collision_filter_pairs"][0]["geometry_review_supported"] = True
    elif mutation == "incomplete_checks":
        manifest["collision_validation"]["details"]["checks"].pop("nonadjacent_positive_clearance")
    elif mutation == "wrong_raw_hash":
        approval["raw_manifest_sha256"] = "0" * 64
    elif mutation == "unknown_pair":
        manifest["adjacent_collision_filter_pairs"][0]["body2"] = "R_link3"
        approval["adjacent_collision_filter_pairs"][0]["body2"] = "R_link3"
    else:
        approval["hardware_deployment_approved"] = True
    stage = FakeStage(2)
    with pytest.raises(ValueError):
        apply_filters(stage, stage.env_paths, manifest, research_approval=approval, raw_manifest=raw, usd_physics=schema)
    assert stage.edits == 0


def test_raw_manifest_cannot_borrow_research_approval_to_relabel_material(audited_research_filter_inputs):
    apply_filters, schema = isolated_filter_function()
    inputs = audited_research_filter_inputs
    raw = json.loads(json.dumps(inputs["raw_manifest"]))
    raw["collision_validation"]["passed"] = True
    stage = FakeStage(1)
    with pytest.raises(ValueError):
        apply_filters(stage, stage.env_paths, raw, research_approval=inputs["research_approval"],
                      raw_manifest=inputs["raw_manifest"], usd_physics=schema)
    assert stage.edits == 0
