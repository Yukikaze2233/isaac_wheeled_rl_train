"""Round4 launch/progress gates on CPU; no simulator or independent training pilot."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo import v40_ground as ground
from wheeled_algo import v40_job as job
from wheeled_algo.v40_round4_launch import Round4Curriculum, checked_resume_state, validate_initialization_options
from wheeled_tasks.v40.contract import load_contract, validate_asset


@pytest.fixture
def cli():
    return runpy.run_path(str(ROOT / "scripts/train_v40.py"))


@pytest.fixture
def contract():
    return load_contract(ROOT / "contracts/own_v40_round4_full.json")


@pytest.fixture
def cached_ground(tmp_path, monkeypatch):
    path = tmp_path / "official-ground-fixture.usd"
    raw = b"ground identity test fixture, not a simulator asset"
    path.write_bytes(raw)
    monkeypatch.setattr(ground, "GROUND_ASSET_SHA256", hashlib.sha256(raw).hexdigest())
    return path


@pytest.fixture
def options(cached_ground):
    return SimpleNamespace(stage="locomotion", seed=42, research=True, num_envs=1024,
                           ground_usd=cached_ground, max_iterations=30000,
                           resume=None, finetune=None, warm_start=None, stage_transfer=None)


@pytest.fixture
def manifest(cli, contract, options):
    result = cli["make_manifest"](contract, validate_asset(contract, allow_research=True), options)
    result["runtime"] = {"checkpoint_format": "rsl_rl_5_split_mlp",
                         "versions": {"rsl-rl-lib": "5.5.1"},
                         "actor_class": "MLPModel", "critic_class": "MLPModel"}
    # CPU protocol fixture; actual PhysX readback belongs to the formal startup.
    result["domain_randomization_report"] = {"passed": True, "mapping_fixture": [0, 1]}
    return result


class CurriculumEnv:
    def __init__(self):
        self.calls = []
        self.steps = 0

    def set_training_iteration(self, iteration):
        self.calls.append(iteration)

    @property
    def training_curriculum_state(self):
        return {"iteration": self.calls[-1], "target_max_dv_m_s": min(1., self.calls[-1] / 10000)}

    def step(self, _action):
        self.steps += 1


def fresh_runner(env):
    obs = TensorDict({"policy": torch.zeros(2, 125), "critic": torch.zeros(2, 29)}, [2])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    # Read the existing official config's scalar instead of inventing a new std.
    tree = ast.parse((ROOT / "src/wheeled_tasks/agents/v40_ppo_cfg.py").read_text())
    init_std = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                    if isinstance(node, ast.keyword) and node.arg == "init_noise_std")
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64],
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": init_std})
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[256, 128, 64])
    runner = SimpleNamespace(env=env, alg=PPO(actor, critic, storage=None),
                             current_learning_iteration=0, device="cpu")
    runner.save = lambda path, infos=None: torch.save({**runner.alg.save(),
        "iter": runner.current_learning_iteration, "infos": infos}, path)
    return runner, init_std


@pytest.mark.parametrize("source", ["finetune", "warm_start", "stage_transfer"])
def test_full_profile_transfer_rejected_in_manifest_gate(cli, contract, options, source):
    setattr(options, source, Path("old-policy.pt"))
    with pytest.raises(ValueError, match="scratch"):
        cli["make_manifest"](contract, validate_asset(contract, allow_research=True), options)


def test_full_profile_requires_official_ground(contract, options):
    options.ground_usd = None
    with pytest.raises(ValueError, match="ground-usd"):
        validate_initialization_options(contract, options)


@pytest.mark.parametrize("field,value", [("num_envs", 256), ("max_iterations", 2), ("stage", "stand")])
def test_formal_profile_cannot_be_silently_reduced_to_a_pilot(contract, options, field, value):
    setattr(options, field, value)
    with pytest.raises(ValueError, match="1024|30000|locomotion"):
        validate_initialization_options(contract, options)


def test_fresh_stock_models_and_scratch_lineage(cli, manifest, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("scratch construction must not load any parent weights")
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(PPO, "load", forbidden)
    env = CurriculumEnv()
    Round4Curriculum(env, manifest)
    torch.manual_seed(42)
    runner, init_std = fresh_runner(env)
    cli["bind_checkpoint_metadata"](runner, manifest)
    torch.manual_seed(43)
    other, _ = fresh_runner(env)
    assert not torch.equal(runner.alg.actor.mlp[0].weight, other.alg.actor.mlp[0].weight)
    assert not torch.equal(runner.alg.critic.mlp[0].weight, other.alg.critic.mlp[0].weight)
    torch.testing.assert_close(runner.alg.actor.distribution.std_param, torch.full((6,), init_std), rtol=0, atol=0)
    assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state
    assert env.calls == [0]
    initialization = manifest["initialization"]
    assert initialization == manifest["source_provenance"]
    assert initialization["mode"] == "scratch" and initialization["seed"] == 42
    assert initialization["method"] == "random_initialization_from_scratch"
    assert "parent_checkpoint_sha256" not in initialization
    assert initialization["feature_flags"]["push_curriculum"] is True


def test_update_clock_advances_only_after_success_not_env_steps():
    env, manifest = CurriculumEnv(), {}
    course = Round4Curriculum(env, manifest, completed_updates=13)
    def update():
        assert env.calls[-1] == 13
        return {"loss": 1.}
    runner = SimpleNamespace(env=env, alg=SimpleNamespace(update=update))
    budget = job.TrainingBudget()
    with budget.bind(runner, on_update=course.on_update):
        for _ in range(100):
            runner.env.step(None)
        assert env.calls == [13]
        assert runner.alg.update() == {"loss": 1.}
    assert env.calls == [13, 14] and budget.completed_updates == 1
    assert manifest["training_curriculum"]["completed_updates"] == 14


def test_failed_update_does_not_advance_curriculum():
    env, manifest = CurriculumEnv(), {}
    course = Round4Curriculum(env, manifest)
    def fail():
        raise RuntimeError("incomplete PPO update")
    runner = SimpleNamespace(env=env, alg=SimpleNamespace(update=fail))
    budget = job.TrainingBudget()
    with pytest.raises(RuntimeError, match="incomplete"), budget.bind(runner, on_update=course.on_update):
        runner.alg.update()
    assert env.calls == [0] and budget.completed_updates == 0 and budget.update_failed


def test_callback_failure_is_not_a_planned_successful_stop():
    runner = SimpleNamespace(env=CurriculumEnv(), alg=SimpleNamespace(update=lambda: {}))
    budget = job.TrainingBudget()
    def fail(_count):
        raise job.PlannedStop("soft_stop")
    with pytest.raises(job.PlannedStop), budget.bind(runner, on_update=fail):
        runner.alg.update()
    assert budget.completed_updates == 1 and budget.update_failed


def test_snapshot_captures_real_push_amplitude_by_completed_updates(contract):
    from wheeled_tasks.v40.round4 import PushCurriculum
    push = PushCurriculum(contract, 2, "cpu")
    class Env:
        set_training_iteration = staticmethod(push.set_training_iteration)

        @property
        def training_curriculum_state(self):
            return push.state()
    manifest = {}
    course = Round4Curriculum(Env(), manifest)
    for count, amplitude in [(0, .1), (5000, .25), (10000, .5)]:
        course.on_update(count)
        snapshot = manifest["training_curriculum"]
        assert snapshot["timebase"] == "successful_ppo_updates"
        assert snapshot["completed_updates"] == snapshot["state"]["completed_ppo_updates"] == count
        assert snapshot["state"]["max_delta_v_m_s"] == amplitude


def test_ground_path_is_passed_through_environment_config(cli, options, monkeypatch):
    captured = {}
    cfg = SimpleNamespace(scene=SimpleNamespace(), sim=SimpleNamespace())
    def create_env(*, cfg):
        captured["cfg"] = cfg
        return SimpleNamespace()
    monkeypatch.setitem(sys.modules, "wheeled_tasks.direct.v40_serial.env_cfg", SimpleNamespace(V40EnvCfg=lambda: cfg))
    monkeypatch.setitem(sys.modules, "wheeled_tasks.direct.v40_serial.env", SimpleNamespace(V40Env=create_env))
    options.contract = ROOT / "contracts/own_v40_round4_full.json"
    options.usd_cache_dir = None
    options.device, options.headless = "cpu", True
    cli["make_env"](options)
    assert captured["cfg"].ground_usd_path == str(options.ground_usd.resolve())
    assert captured["cfg"].scene.num_envs == 1024


def test_run_job_forwards_callback_and_publishes_progress(cli, manifest, tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    for name in job.BASE_ARTIFACTS:
        (directory / name).write_text('{}')
    env = CurriculumEnv()
    course = Round4Curriculum(env, manifest)
    runner, _ = fresh_runner(env)
    # Only the job adapter is exercised; there is no standalone PPO training pilot.
    runner.alg.update = lambda: {"adapter_only": True}
    def learn(num_learning_iterations, init_at_random_ep_len):
        assert init_at_random_ep_len is False
        for index in range(num_learning_iterations):
            runner.env.step(None)
            runner.alg.update()
            runner.current_learning_iteration = index
    runner.learn = learn
    cli["bind_checkpoint_metadata"](runner, manifest, run_dir=directory)
    def export(checkpoint, source_manifest, output):
        from wheeled_algo.v40_export import export_checkpoint
        return export_checkpoint(checkpoint, source_manifest, output)
    receipt = job.run_training_job(runner, directory, job.TrainingBudget(), 2,
                                   exporter=export, on_update=course.on_update)
    assert receipt["completed_updates"] == 2
    saved = torch.load(directory / "model_final.pt", weights_only=True)
    assert saved["iter"] == 1  # Official RSL loop index is deliberately not our clock.
    assert saved["infos"]["training_curriculum"]["completed_updates"] == 2
    assert saved["infos"]["training_curriculum"]["state"]["target_max_dv_m_s"] == .0002
    snapshot = json.loads((directory / "run_manifest.json").read_text())
    assert snapshot["training_curriculum"] == saved["infos"]["training_curriculum"]
    sidecar = json.loads((directory / "policy.onnx.json").read_text())
    assert sidecar["run_manifest"]["training_curriculum"] == snapshot["training_curriculum"]


@pytest.fixture
def resume_checkpoint(cli, manifest, tmp_path):
    env = CurriculumEnv()
    course = Round4Curriculum(env, manifest)
    course.on_update(37)
    runner, _ = fresh_runner(env)
    runner.current_learning_iteration = 36
    checkpoint = tmp_path / "model_36.pt"
    cli["bind_checkpoint_metadata"](runner, manifest, run_dir=tmp_path)
    runner.save(checkpoint)
    return checkpoint


def test_same_full_contract_resume_uses_completed_update_offset(cli, manifest, resume_checkpoint):
    _, source = cli["checked_checkpoint"](resume_checkpoint, manifest)
    progress = checked_resume_state(resume_checkpoint, source, manifest)
    env = CurriculumEnv()
    course = Round4Curriculum(env, manifest, completed_updates=progress["completed_updates"])
    runner, _ = fresh_runner(env)
    cli["restore_checkpoint"](runner, resume_checkpoint, resume=True, expected_manifest=manifest)
    assert runner.current_learning_iteration == 37
    course.on_update(1)
    assert env.calls == [37, 38]


@pytest.mark.parametrize("tamper", ["origin", "infos_origin", "missing_progress", "negative_progress", "contract"])
def test_resume_rejects_old_or_unbound_initialization(cli, manifest, resume_checkpoint, tamper):
    path = resume_checkpoint.parent / "run_manifest.json"
    source = json.loads(path.read_text())
    saved = torch.load(resume_checkpoint, weights_only=True)
    if tamper == "origin":
        source["initialization"]["mode"] = "warm_start"
    elif tamper == "infos_origin":
        saved["infos"]["initialization"]["mode"] = "warm_start"
    elif tamper == "missing_progress":
        del saved["infos"]["training_curriculum"]
    elif tamper == "negative_progress":
        saved["infos"]["training_curriculum"]["completed_updates"] = -1
    else:
        source["contract_sha256"] = "0" * 64
    path.write_bytes(job.json_bytes(source))
    torch.save(saved, resume_checkpoint)
    runner, _ = fresh_runner(CurriculumEnv())
    with pytest.raises(ValueError):
        cli["restore_checkpoint"](runner, resume_checkpoint, resume=True, expected_manifest=manifest)
    assert runner.current_learning_iteration == 0 and not runner.alg.optimizer.state


def test_ground_sha_tamper_rejected(cached_ground):
    report = ground.verify_cached_ground(cached_ground)
    assert report["path"] == str(cached_ground.resolve()) and report["size"] == cached_ground.stat().st_size
    assert report["source_url"] == ground.GROUND_ASSET_URL
    cached_ground.write_bytes(b"arbitrary USD")
    with pytest.raises(ValueError, match="SHA256"):
        ground.verify_cached_ground(cached_ground)


def test_verified_ground_does_not_waive_usd_seed_gate(cli, options, monkeypatch):
    monkeypatch.setenv("V40_USD_SEED", "unbound-robot.usd")
    options.contract = None
    report, _, _ = cli["preflight"](options)
    assert not report["ready"]
    assert any("V40_USD_SEED" in blocker for blocker in report["blockers"])


def test_shared_ground_pin_matches_existing_replay_guard():
    tree = ast.parse((ROOT / "scripts/play_v40_onnx.py").read_text())
    values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in {"GROUND_ASSET_SHA256", "GROUND_ASSET_URL"}}
    assert values == {"GROUND_ASSET_SHA256": ground.GROUND_ASSET_SHA256, "GROUND_ASSET_URL": ground.GROUND_ASSET_URL}


def test_legacy_options_do_not_require_ground_or_scratch():
    validate_initialization_options({}, SimpleNamespace(warm_start=Path("round2.pt")))
