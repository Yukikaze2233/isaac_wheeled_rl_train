"""Native CPU restore/optimizer hooks and independent joint worker admission.

No Isaac scene, GPU rollout, account mutation or production capability credit.
"""
import ast
from copy import deepcopy
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_integrated_lineage import fixture as original_lineage_fixture
from test_integrated_runtime import lineage_fixture, sources, first
from test_recovery_rollout_mask import algorithm, collect
from train_chassis import digest, timed_call, write_progress, verify_entropy_coef
from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis import integrated_lineage as lineage
from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis.integrated_training import (
    critic_warmup_remaining, learning_metadata, make_design_hook, validate_design_hook)
from wheeled_tasks.chassis.policy_transfer import verify_learning_state_restore
from wheeled_tasks.chassis.terrain_runout import validate_runout

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'scripts/train_chassis.py'


def execute(nodes, context):
    module = ast.fix_missing_locations(ast.Module(body=deepcopy(nodes), type_ignores=[]))
    exec(compile(module, str(ENTRY), 'exec'), context)


def function(name):
    return next(n for n in ast.walk(ast.parse(ENTRY.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.fixture
def admitted(lineage_fixture, sources):
    f = lineage_fixture
    config = first(f, sources)
    selection = config['integrated_runtime_candidate']['learning_source']
    allowance = lineage.remaining_budget(f.binding, selection, f.attempts, requested_actor_updates=1)
    receipt = dict(selection=deepcopy(selection), allowance=allowance,
        component_clocks_at_source=deepcopy(config['integrated_runtime_candidate']['clocks_at_source']),
        actor_endpoint=15000, target_contract_sha256=lineage.digest(f.contracts['speed']))
    return f, config, receipt


def test_full_native_35_81_models_adam_and_cpu_env_rng_restored(admitted):
    f, config, receipt = admitted
    saved = torch.load(f.rec['path'], weights_only=True)
    fresh = lineage._native_learning_template(json.loads(f.contracts['recovery'].read_text()))
    fresh.load(saved, load_cfg=None, strict=True)
    assert verify_learning_state_restore(fresh, saved) == dict(actor_exact=True, critic_exact=True, optimizer_exact=True)
    before = torch.get_rng_state()
    generator = torch.Generator()
    try:
        # The exact production resume block, including native CPU and env RNG.
        resume = function('main')
        node = next(n for n in ast.walk(resume) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "'rng_state' in infos")
        execute([node], dict(infos=saved['infos'], torch=torch, env=SimpleNamespace(generator=generator)))
        assert torch.equal(torch.get_rng_state(), saved['infos']['rng_state'])
        assert torch.equal(generator.get_state(), saved['infos']['env_rng_state'])
    finally:
        torch.set_rng_state(before)
    assert critic_warmup_remaining(receipt, saved['infos']) == 10


@pytest.mark.parametrize('remaining', [0, 4, 10])
def test_own_fault_warmup_only_uses_saved_remainder(admitted, remaining):
    _, _, receipt = admitted
    receipt = deepcopy(receipt)
    receipt['selection'].update(mode='resume_joint_own', critic_warmup_remaining=remaining)
    assert critic_warmup_remaining(receipt, dict(critic_warmup_remaining=remaining)) == remaining
    with pytest.raises(ValueError, match='remainder'):
        critic_warmup_remaining(receipt, dict(critic_warmup_remaining=(remaining + 1) % 11))


@pytest.mark.parametrize('bad', [-1, 11, True, 0])
def test_cross_entry_rejects_extra_or_missing_critic_work(admitted, bad):
    _, _, receipt = admitted
    receipt = deepcopy(receipt)
    receipt['selection']['critic_warmup_remaining'] = bad
    with pytest.raises(ValueError):
        critic_warmup_remaining(receipt, {})


def update_context(receipt, tmp_path):
    native = algorithm()
    native.schedule = 'fixed'
    native.learning_rate = 1e-4
    native.entropy_coef = 0.
    for group in native.optimizer.param_groups:
        group['lr'] = 1e-4
    env = SimpleNamespace(stage_actor_update=62, global_actor_update=62, task_actor_update=0,
        generator=torch.Generator(), performance_curriculum=None)
    report = dict(parent_updates=0, successful_updates=0, attempted_updates=0, actor_updates_in_block=0,
        parent_training_transitions=0, learning_lineage_updates=25000, status='running', entropy_coef_source='frozen',
        stage_actor_update=62, global_actor_update=62, task_actor_update=0)
    contract = tmp_path / 'contract.json'
    contract.write_text('{}\n')
    return dict(report=report, warmup_updates=10, runner=SimpleNamespace(alg=native),
        cfg=SimpleNamespace(algorithm=SimpleNamespace(entropy_coef=0.), num_steps_per_env=6),
        learning_schedule='fixed', policy_probe=None, learning_audit=None, timing=None, original_update=native.update,
        a=(), kw={}, timed_call=timed_call, verify_entropy_coef=verify_entropy_coef,
        c=dict(task_campaign=dict(domain='speed'), integrated_production=dict(schema=production.WORKER_SCHEMA), learning_rate=1e-4),
        env=env, stage_actor_start=62, task_actor_start=0,
        args=SimpleNamespace(global_actor_update_offset=0, actor_updates_consumed=33000,
            num_envs=2, stage='speed', run_dir=tmp_path, contract=contract),
        integrated_receipt=receipt, learning_metadata=learning_metadata,
        json=json, torch=torch, os=os, Path=Path, digest=digest, write_progress=write_progress,
        datetime=__import__('datetime').datetime, timezone=__import__('datetime').timezone,
        recovery_rollout=None, confirmed_bins=[], identity=dict(contract_sha256=digest(contract)))


def committed_update(context):
    nodes = function('update').body
    start = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'actor_enabled')
    end = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == "report['entropy_coef']")
    execute(nodes[start:end + 1], context)


def actual_progress(context):
    nodes = function('update').body
    start = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'progress')
    end = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'write_progress')
    execute(nodes[start:end + 1], context)
    return json.loads((context['args'].run_dir / 'progress.json').read_text())


def test_actual_native_cpu_optimizer_keeps_critic_clock_and_persists_one_actor_everywhere(admitted, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _, _, receipt = admitted
    context = update_context(receipt, tmp_path)
    native = context['runner'].alg
    initial = deepcopy(native.actor.state_dict())
    for index in range(11):
        native.actor.requires_grad_(index >= 10)
        obs = collect(native, torch.ones(6, 2, dtype=torch.bool))
        native.compute_returns(obs)
        committed_update(context)
        progress = actual_progress(context)
        meta = progress['integrated_lineage']
        assert meta['joint_actor_updates'] == meta['domain_actor_updates'] == int(index >= 10)
        assert progress['integrated_runtime_clocks']['component_actor_updates'] == dict(recovery=15000 + int(index >= 10), speed=int(index >= 10))
        if index < 10:
            assert all(torch.equal(value, native.actor.state_dict()[name]) for name, value in initial.items())
    assert any(not torch.equal(value, native.actor.state_dict()[name]) for name, value in initial.items())
    assert native.optimizer.state  # Actual native Adam state after optimization.

    def original_save(path, infos):
        torch.save(dict(native.save(), infos=infos), path)
    context['original_save'] = original_save
    execute([function('save')], context)
    context['save'](str(tmp_path / 'model.pt'))
    checkpoint = torch.load(tmp_path / 'model.pt', weights_only=True)
    seal = json.loads((tmp_path / 'model.checkpoint.json').read_text())
    for key in ('integrated_lineage', 'integrated_runtime_clocks'):
        assert checkpoint['infos'][key] == seal[key] == progress[key] == context['report'][key]
    assert checkpoint['infos']['critic_warmup_remaining'] == 0
    assert checkpoint['infos']['integrated_lineage']['domain_complete'] is False

    # Execute the actual completed-save branch with a bounded cap fixture. The
    # native CPU optimizer committed all 10 critic + 1 actor updates above.
    receipt['allowance']['maximum_actor_updates'] = 1
    context['report'].update(status='completed', integrated_admission_sha256='CPU-fixture-only')
    context['runner'].save = context['save']
    main = function('main')
    outer = next(n for n in main.body if isinstance(n, ast.Try) and n.finalbody)
    final = next(n for n in outer.finalbody if isinstance(n, ast.If)
                 and 'runner is not None and report' in ast.unparse(n.test)
                 and "'completed', 'stopped'" in ast.unparse(n.test))
    execute(final.body[:3], context)
    checkpoint = torch.load(tmp_path / 'model_final.pt', weights_only=True)
    seal = json.loads((tmp_path / 'model_final.checkpoint.json').read_text())
    progress = json.loads((tmp_path / 'progress.json').read_text())
    completion_write = next(n for n in outer.finalbody if isinstance(n, ast.Expr)
                            and "'completion.json'" in ast.unparse(n))
    execute([completion_write], context)
    completion = json.loads((tmp_path / 'completion.json').read_text())
    for key in ('integrated_lineage', 'integrated_runtime_clocks'):
        assert checkpoint['infos'][key] == seal[key] == progress[key] == completion[key]
    assert completion['integrated_lineage']['domain_complete'] is True
    assert not torch.cuda.is_initialized()


def test_actor_clock_is_retained_when_a_post_optimizer_lr_invariant_fails(admitted, tmp_path):
    _, _, receipt = admitted
    context = update_context(receipt, tmp_path)
    context['report']['parent_updates'] = 10
    native = context['runner'].alg
    def committed_then_bad_lr():
        native.learning_rate = 2e-4
        return {}
    context['original_update'] = committed_then_bad_lr
    with pytest.raises(RuntimeError, match='learning rate'):
        committed_update(context)
    assert context['report']['successful_updates'] == 1
    assert context['env'].task_actor_update == 1
    assert context['report']['integrated_runtime_clocks']['component_actor_updates']['speed'] == 1


@pytest.mark.parametrize('endpoint,maximum,count,complete', [(1, 14990, 11, True), (15000, 1, 11, True),
    (15000, 14990, 11, False), (1, 14990, 10, False)])
def test_terminal_requires_completed_actual_work_and_endpoint_or_charged_cap(admitted, endpoint, maximum, count, complete):
    _, _, receipt = admitted
    # Helper boundary fixtures only. Admission/account provenance is tested separately.
    receipt = deepcopy(receipt)
    receipt['actor_endpoint'] = endpoint
    receipt['allowance']['maximum_actor_updates'] = maximum
    result = learning_metadata(receipt, 1, completed=True, successful_updates=count)
    assert result['integrated_lineage']['domain_complete'] is complete
    assert learning_metadata(receipt, 1, completed=False, successful_updates=count)['integrated_lineage']['domain_complete'] is False
    with pytest.raises(ValueError, match='actual admitted'):
        learning_metadata(receipt, 2)


def test_static_geometry_marker_never_constructs_a_native_environment(admitted):
    _, config, _ = admitted
    config = deepcopy(config)
    config['integrated_production'] = dict(schema=production.WORKER_SCHEMA)
    config['integrated_runtime_candidate']['static_recipe_preflight'] = True
    env_tree = ast.parse((ROOT / 'src/wheeled_tasks/chassis/env.py').read_text())
    constructor = next(n for n in ast.walk(env_tree) if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    with pytest.raises(ValueError, match='Static joint recipe'):
        execute([constructor.body[0]], dict(config=config))

# This fixture uses real named static recipes, a native 35/81 CPU model seal,
# durable account/state files and the actual admission helper, without a scene.
from test_integrated_production import static_bundle, frozen, admitted as production_admitted, arguments
from wheeled_tasks.chassis.preflight import design_preflight
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.integrated_preflight import validate_integrated_runtime_state


def production_design(f):
    receipt = production_admitted(f)
    args = arguments(f, receipt)
    args.contract = f.contracts['speed']
    args.transfer = None
    args.stage = f.worker['enabled_stages'][0]
    args.run_dir = f.run / 'external-probe'
    static = production.static_training_config(f.worker, f.source)
    # The production entry inventories transitive actuator evidence separately.
    evidence = json.loads((f.source / static['actuator_response_evidence']['file']).read_text())
    for name in evidence['files_sha256']:
        target = f.source / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    control = load_chassis_control(f.source / static['control_math_source'])
    baseline = design_preflight(static, control, f.source, stage=args.stage,
        role='train', seed=args.seed, num_envs=args.num_envs, block_index=0)
    effective = production.attach_runtime(f.worker, receipt, f.source)
    hook = make_design_hook(baseline, effective, receipt, args)
    actual_baseline = design_preflight(effective, control, f.source, stage=args.stage,
        role='train', seed=args.seed, num_envs=args.num_envs, block_index=0)
    return receipt, args, effective, hook, actual_baseline


def test_real_admission_builds_independent_scoped_design_before_native_construction(frozen):
    receipt, args, effective, hook, baseline = production_design(frozen)
    result = validate_design_hook(effective, frozen.source, baseline, hook,
        stage=args.stage, num_envs=4096, seed=args.seed, block_index=0)
    expected = result['runtime_expected']['integrated_curriculum']
    assert sum(expected['row_population'].values()) == 4096
    assert expected['clocks_at_source'] == receipt['component_clocks_at_source']
    assert expected['clocks_at_source']['component_actor_updates'] == dict(recovery=15000, speed=0)
    assert result['design_sha256'] == baseline['design_sha256']
    assert effective['integrated_runtime_candidate']['runtime_ready'] is False
    assert effective['integrated_runtime_candidate']['qualification'] is None
    assert 'static_recipe_preflight' not in effective['integrated_runtime_candidate']
    assert frozen.worker == json.loads(frozen.contracts['speed'].read_text())
    inventory = production.production_source_files(frozen.worker, receipt, frozen.source)
    for name in json.loads((frozen.source / production.SCUT_DIRECTORY / 'source_manifest.json').read_text())['files']:
        assert str(Path(production.SCUT_DIRECTORY) / name) in inventory

    # Execute the real entry's complete source inventory, then hash every file.
    # Native implementation files are copied from the current checkout; frozen
    # contracts/raw SCUT/evidence retain the fixture's independently pinned bytes.
    import train_chassis as entry
    for name in ('resume_physics_change', 'resume_response_change'):
        setattr(args, name, False)
    manifest = json.loads((ROOT / effective['asset_directory'] / 'manifest.json').read_text())
    nodes = function('main').body
    start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                 and ast.unparse(node.targets[0]) == 'source_files')
    end = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
               and ast.unparse(node.targets[0]) == 'source_hashes')
    context = dict(entry.__dict__, c=effective, contract_config=frozen.worker,
        manifest=manifest, args=args, ROOT=frozen.source, integrated_production=production,
        integrated_receipt=receipt)
    execute(nodes[start:end], context)
    files = context['source_files']
    assert 'src/wheeled_tasks/chassis/step_assist.py' in files
    assert 'src/wheeled_tasks/chassis/jump_assist.py' not in files
    for name in files:
        target = frozen.source / name
        assert not Path(name).is_absolute() and target.resolve().is_relative_to(frozen.source.resolve())
        if not target.exists():
            assert (ROOT / name).is_file(), name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
    execute([nodes[end]], context)
    assert len(context['source_hashes']) == len(files)
    assert all(context['source_hashes'][name] == digest(frozen.source / name) for name in files)
    assert len(context['source_hashes']) > 50


def test_hook_rejects_full_receipt_scope_and_population_mismatch(frozen):
    _, args, source_effective, source_hook, source_baseline = production_design(frozen)
    for bad in ('seed', 'count', 'baseline_population', 'receipt_clock', 'scope', 'effective_clock', 'selection_sha', 'fixed_eval'):
        effective, hook, baseline = map(deepcopy, (source_effective, source_hook, source_baseline))
        count, seed = 4096, args.seed
        if bad == 'seed':
            seed += 1
        elif bad == 'count':
            count -= 1
        elif bad == 'baseline_population':
            baseline['num_envs'] = 1
        elif bad == 'receipt_clock':
            hook['source_receipt']['component_clocks_at_source']['component_actor_updates']['recovery'] = 14999
        elif bad == 'scope':
            name = next(iter(hook['runtime_expected']['integrated_curriculum']['row_population']))
            hook['runtime_expected']['integrated_curriculum']['row_population'][name] += 1
        elif bad == 'effective_clock':
            effective['integrated_runtime_candidate']['clocks_at_source']['component_actor_updates']['speed'] = 1
        elif bad == 'selection_sha':
            hook['selection']['sha256'] = '0' * 64
        else:
            effective['evaluation_exact_cases'] = True
        with pytest.raises(ValueError):
            validate_design_hook(effective, frozen.source, baseline, hook,
                stage=args.stage, num_envs=count, seed=seed, block_index=0)


def test_external_scene_only_one_step_admission_preserves_charged_allowance(frozen):
    receipt = production_admitted(frozen)
    args = arguments(frozen, receipt)
    before = deepcopy(receipt)
    args.validate_scene, args.updates = 1, 1
    assert production.authenticate_invocation(frozen.worker, args, frozen.source) == receipt == before
    assert receipt['allowance']['requested_total_updates'] > 1
    assert json.loads(frozen.claim.read_text())['consumed_updates'] == 34000


def test_static_runout_geometry_has_no_clock_and_is_rejected_by_runtime(frozen):
    config = production.static_training_config(json.loads(frozen.contracts['terrain'].read_text()), frozen.source)
    audit = config['integrated_runtime_candidate']
    assert audit['static_recipe_preflight'] is True
    assert 'learning_source' not in audit and 'clocks_at_source' not in audit
    assert validate_runout(config) == dict(enabled=True, support_end_x_m=6.2, positive_boundary_x_m=5.6)
    with pytest.raises(ValueError, match='static recipe'):
        validate_integrated_runtime_state({}, {}, config=config, num_envs=4096, actor_update=0)
    config['integrated_production']['schema'] = 'wrong'
    with pytest.raises(ValueError, match='named recipe'):
        validate_runout(config)
