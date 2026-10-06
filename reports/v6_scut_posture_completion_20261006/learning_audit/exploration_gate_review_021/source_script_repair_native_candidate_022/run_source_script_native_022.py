#!/usr/bin/env python3
"""Source022/v20/REC3076 native observer. CPU default; no control hooks or PPO."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
EXPLORATION = HERE.parent
AUDIT = EXPLORATION.parent
PYTHON = '/home/yukikaze/isaacsim60-venv/bin/python'
BASE_COMMIT = 'd0dce6181df83b03fe166abcbadef3c81de2e57e'
PT_SHA = '286b9e26cdd7fe8726f9deb1ce5f24e005d5d918d3abc2b122fdb85b6b671c31'
RECOVERY_SHA = '04ff676ad98244b549e82506d760e0d7686163dcaf80bdb570f2a87717ac1ae6'
WORKER_SHA = 'a64eb6c03231e56d8efd8d83b87f70ee88c198de55fee0c2809aa766c7e19f62'
PARENT_CONTRACT_SHA = '7918e7850e16b96edfa54a8854d3f3571e8101dfa6527cad78a06b4e15baea95'
PAIR = EXPLORATION/'REC3076_pair_001'
TEACHING = AUDIT/'hold_cost_candidate_021/validate_native_teaching_021_002.py'
TEACHING_SHA = '4fed2e4f85f4d8e92c2af804aa98975dc13381c0a6bfcb02f7f49f92d35c84e9'
CAPTURE = EXPLORATION/'mechanical_capture_candidate_021/run_capture_021.py'
MECHANICAL = EXPLORATION/'mechanical_remaining_capture_candidate_021/mechanical_remaining_capture.py'
TARGETS = EXPLORATION/'pd_target_payment_candidate_021/target_payment_capture.py'


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_candidate(expected_sha):
    require(sha(HERE/'candidate_manifest.json') == expected_sha, 'Observer candidate manifest SHA differs')
    m = read(HERE/'candidate_manifest.json')
    require(m['schema'] == 'source022-native-five-lane-observer-candidate-v1', 'Observer schema differs')
    for name, digest in m['files_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and sha(HERE/path) == digest,
            'Observer file changed: '+name)
    for name, digest in m['external_sha256'].items():
        require(Path(name).is_absolute() and sha(name) == digest, 'Bound producer/evidence changed: '+name)
    root = Path(m['source_root']).resolve()
    inventory = read(HERE/'source_inventory.json')
    require(sha(HERE/'source_inventory.json') == m['source_inventory_sha256'], 'Source022 inventory differs')
    require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in inventory), 'Source path escape')
    require({n:sha(root/n) for n in inventory} == inventory, 'Actual candidate source bytes changed')
    require(sha(root/'src/wheeled_tasks/chassis/recovery.py') == RECOVERY_SHA
        and sha(root/'contracts/v6_task_campaign_recovery_worker_v20.json') == WORKER_SHA,
        'Actual source predicate or v20 worker differs')
    return m


def authenticate(args, admission, base, teaching):
    """New source identity, reused full-state/native protocol, actual v20 preflight."""
    import torch
    m = verify_candidate(args.observer_candidate_sha)
    root = Path(m['source_root']).resolve()
    inventory = read(HERE/'source_inventory.json')
    old = read(PAIR/'admission.json')
    for key in ('checkpoint_sha256', 'checkpoint_contract_sha256', 'checkpoint_seal_sha256', 'seed', 'block_index'):
        require(admission[key] == old[key], 'Historical sealed parent changed: '+key)
    require(admission['checkpoint_sha256'] == PT_SHA and admission['checkpoint_contract_sha256'] == PARENT_CONTRACT_SHA,
        'REC3076 source/v19 parent differs')
    pt = Path(admission['checkpoint']).resolve()
    for path, key in ((pt,'checkpoint_sha256'), (pt.with_suffix('.contract.json'),'checkpoint_contract_sha256'),
                      (pt.with_suffix('.checkpoint.json'),'checkpoint_seal_sha256')):
        require(sha(path) == admission[key], 'Immutable full parent bytes differ')
    seal = read(pt.with_suffix('.checkpoint.json'))
    require(seal['sealed'] is True and seal['checkpoint_sha256'] == PT_SHA
        and seal['task_actor_update'] == 3076 and seal['learning_updates'] == 30007, 'Parent seal clocks differ')
    require(not any(k == 'train_chassis' or k.startswith('wheeled_tasks.') for k in sys.modules),
        'Source candidate must be imported in a clean process')
    sys.path[:0] = [str(root/'src'), str(root/'scripts')]
    from train_chassis import preflight
    from wheeled_tasks.chassis import recovery_script_revision as recipe
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.task_campaign import validate_task_resume
    contract = root/'contracts/v6_task_campaign_recovery_worker_v20.json'
    config, manifest = preflight(contract)
    require(config == recipe.canonical_worker('recovery', root) and config['task_campaign']['revision_migration'] == recipe.revision_binding(),
        'Actual canonical v20 recovery recipe differs')
    predecessor = read(pt.with_suffix('.contract.json'))
    migration = validate_task_resume(predecessor, config)
    require(config['recovery_training']['side_speed_rad_s'] == 1.5 and not config.get('evaluation_exact_cases')
        and config['record_diagnostics'] is True and config.get('auto_reset', True) is True
        and config['enabled_stages'] == ['mixed'], 'Actual train/reset semantics changed')
    require(tuple(config[k] for k in ('physics_dt', 'policy_dt', 'actor_dim', 'critic_dim', 'action_dim')) == (.005,.02,35,81,6),
        'Native source ABI/timing differs')
    control = load_chassis_control(root/config['control_math_source'])
    receipt = design_preflight(config, control, root, stage='mixed', role='train', seed=admission['seed'], num_envs=4096, block_index=0)
    saved = torch.load(pt, map_location='cpu', weights_only=True)
    clocks = teaching.validate_state(saved, seal, predecessor, config, admission)
    require(clocks['critic_warmup_remaining'] == 0, 'Historical seal remaining differs')
    args.source_root, args.source_manifest, args.checkpoint = root, HERE/'source_inventory.json', pt
    args.formal_probe = teaching.load_bound(root/'scripts/probe_scut_curriculum.py', inventory['scripts/probe_scut_curriculum.py'], 'source022_native_formal_structure')
    base.SOURCE_COMMIT, base.CHECKPOINT_SHA, base.CONTRACT_SHA = m['source_commit'], PT_SHA, WORKER_SHA
    base.CLOCKS, base.SEED, base.BLOCK = clocks, admission['seed'], 0
    identity = dict(source_commit_label=m['source_commit'],
        source_identity_scope='actual_frozen022_archive_inventory; parent_seal_is_frozen021_v19',
        source_root=str(root), source_manifest=str(HERE/'source_inventory.json'),
        source_manifest_sha256=sha(HERE/'source_inventory.json'), archive_source_files=inventory,
        source_files_sha256=base.snapshot_sources(root, config, inventory), wrapper_sha256=teaching.BASE_SHA,
        contract_sha256=WORKER_SHA, checkpoint=str(pt), checkpoint_sha256=PT_SHA,
        checkpoint_contract_sha256=PARENT_CONTRACT_SHA, checkpoint_seal_sha256=admission['checkpoint_seal_sha256'],
        checkpoint_clocks=clocks, checkpoint_saved_iter_field=saved['iter'], observation_context='sealed_recovery_resume',
        observation_actor_clocks={k:clocks[k] for k in ('task_actor_update','stage_actor_update','global_actor_update')},
        forced_engineering_actor_update=None, forced_clock_is_not_resume_or_learning_progress=False,
        actual_parent_seal_infos_not_modified=True, formal_probe_provenance=None,
        observation_rng_protocol='Original55 saved CPU/environment/CUDA RNG; one native reset; private sampler unchanged.',
        actor_factory='original55 native saved Gaussian; no optimizer/PPO factory',
        saved_std=saved['actor_state_dict']['distribution.std_param'].tolist(),
        actor_tensor_sha256={k:base.tensor_sha(v) for k,v in saved['actor_state_dict'].items()},
        seed=admission['seed'], block_index=0, design_preflight=receipt,
        source022_runtime_expectation=recipe.runtime_expectation(config),
        source022_parent_migration_validation=migration,
        native_is_inference_not_critic_adaptation=True, qualification=None, PPO_updates=0)
    return config, manifest, control, saved, identity


def authenticate_callbacks(env, parent, source_root):
    root = Path(source_root)/'src/wheeled_tasks/chassis'
    for owner, modname, cls, file, methods in (
        (env.v5,'wheeled_tasks.chassis.v5_control','V5Control','v5_control.py',('decode','motor_efforts')),
        (env.signal_delay,'wheeled_tasks.chassis.signal_delay','PhysicalSignalDelay','signal_delay.py',('targets',)),
        (env.recovery.fsm,'wheeled_tasks.chassis.recovery','RecoveryFSM','recovery.py',())):
        mod = sys.modules.get(modname)
        require(mod is not None and type(owner).__module__ == modname and getattr(mod,cls,None) is type(owner)
            and Path(inspect.getsourcefile(type(owner))).resolve() == root/file, 'Constructed native class/source differs')
        for name in methods:
            fn = getattr(owner,name)
            require(fn.__self__ is owner and fn.__func__ is getattr(type(owner),name), 'Original native callback differs')
    fsm = env.recovery.fsm
    require(parent.original_update.__self__ is fsm and parent.original_update.__func__ is type(fsm).update
        and sha(root/'recovery.py') == RECOVERY_SHA, 'Native repaired source implementation differs')
    fn = fsm.update
    closure = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__)))
    require(fn.__code__.co_qualname == 'NativeObserver.__init__.<locals>.update'
        and Path(inspect.getsourcefile(fn)).resolve() == AUDIT/'probe_prefix_mean_stochastic_019.py'
        and closure.get('self') is parent and closure.get('r') is env.recovery, 'Original55 observer closure differs')
    reward = sys.modules.get('wheeled_tasks.chassis.rewards')
    require(reward is not None and Path(reward.__file__).resolve() == root/'rewards.py'
        and Path(inspect.getsourcefile(parent.cost_prior)).resolve() == root/'rewards.py', 'Original reward producer differs')
    recipe = sys.modules.get('wheeled_tasks.chassis.recovery_script_revision')
    require(recipe is not None and env.startup_report.get('recovery_script_revision')
        == recipe.native_runtime_binding(fsm,env.cfg,source_root),
        'Native source/runtime binding absent or different')
    return reward


def make_source_observer(parent, capture, helper, writer, source_root):
    entries = {}
    mechanical = helper.configure_mechanical(load(MECHANICAL,'source022_private_mechanical'), entries)
    inherited = capture.make_capture_observer(parent, None, mechanical, writer)
    class SourceObserver(inherited):
        def __init__(self, env, output):
            self.source_watch = None
            self.target_entries = []
            self.source_cleanup_done = False
            super().__init__(env, output)
            try:
                reward = authenticate_callbacks(env, self, source_root)
                self.source_watch = helper.SourceChangeWatch(env, mechanical.HookSet(), output, writer, mechanical.plain)
                self.source_watch.install()
                target_dir = output/'targets'; target_dir.mkdir()
                for lane, level, pid, name in zip(helper.LANES,helper.LEVELS,helper.PROFILE_IDS,helper.PROFILES):
                    directory = target_dir/f'lane_{lane}'; directory.mkdir()
                    frozen = load(TARGETS,f'source022_private_PD_{lane}')
                    entry = helper.lane_capture(frozen,lane,level,pid,name,env,self,reward,mechanical.HookSet(),directory,writer)
                    self.target_entries.append(entry); entries[lane] = entry
                    entry.install()
            except BaseException:
                self._before_close()
                raise

        def _source_cleanup(self):
            if self.source_cleanup_done:
                return
            errors = []
            for entry in reversed(self.target_entries):
                errors += entry.cleanup()
            if self.source_watch is not None:
                errors += self.source_watch.cleanup()
            self.source_cleanup_done = True
            require(not errors, 'Source read-only callback restoration rejected: '+str(errors))

        def _before_close(self):
            try:
                self._source_cleanup()
            finally:
                super()._before_close()

        def finish(self, output):
            try:
                snapshots = {}
                for entry in reversed(self.target_entries):
                    result = entry.finish(); snapshots[str(result['selected_lane'])] = result
                    writer(entry.output/'summary.json',result)
                change = self.source_watch.finish()
                self.source_cleanup_done = True
                original = super().finish(output)
                summary = dict(schema='source022-five-lane-source-observer-preclose-v1', complete=True,
                    policy_steps=600, physical_steps=2400, source_execution=True, control_hooks_injected=False,
                    selected_lanes=list(helper.LANES), targets=snapshots, global_source_change=change,
                    callbacks_restored_preclose=True, PPO_updates=0, formal_updates_consumed=0, qualification=None)
                writer(output/'source_script_observer_summary.json',summary)
            except BaseException:
                if not (output/'source_script_observer_failure.json').exists():
                    writer(output/'source_script_observer_failure.json',dict(error=traceback.format_exc(),
                        status='source_observer_failed_preclose',qualification=None,app_close_returned=None))
                self._before_close()
                raise
            original.update(source_script_observer_summary_sha256=sha(output/'source_script_observer_summary.json'),
                source_script_observer_complete=True, control_hooks_injected=False)
            return original
    return SourceObserver


def decoded(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream]


def frame_projection(row):
    """Original mechanical values only; added measurement metadata is explicit."""
    value = dict(row)
    value['state'] = {k:v for k,v in row['state'].items()
        if k != 'native_FSM_after_reference_before_current_physics'}
    if row['applied_control'] is not None:
        value['applied_control'] = {k:v for k,v in row['applied_control'].items()
            if k not in ('once_decoded_and_actual_delayed_PD','side_reference_audit')}
    value.pop('contacts',None)
    return value


def compare_trace(old, new, helper, lanes, cutoff=None):
    reference = {(r['state']['lane'],r['physical_step']):r for r in decoded(old)
        if r['state']['lane'] in lanes and (cutoff is None or r['physical_step'] <= cutoff)}
    actual = {(r['state']['lane'],r['physical_step']):r for r in new
        if r['state']['lane'] in lanes and (cutoff is None or r['physical_step'] <= cutoff)}
    require(len(reference) > 0, 'Actual baseline selected trace absent')
    if reference.keys() != actual.keys():
        return dict(complete_record_comparison_passed=False,first_mismatch='lane-physical-step-set',
            compared_rows=min(len(reference),len(actual)),cutoff=cutoff)
    for key in sorted(reference):
        a,b = reference[key],actual[key]
        if helper.bits(frame_projection(a)) != helper.bits(frame_projection(b)):
            return dict(complete_record_comparison_passed=False,first_mismatch=['native-state-control',*key],
                compared_rows=len(reference),cutoff=cutoff)
        mismatch = helper.compare_contacts(a['contacts'],b['contacts'])
        if mismatch is not None:
            return dict(complete_record_comparison_passed=False,first_mismatch=[mismatch,*key],
                compared_rows=len(reference),cutoff=cutoff)
    return dict(complete_record_comparison_passed=True,first_mismatch=None,
        compared_rows=len(reference),cutoff=cutoff,full_contact_record_multiset=True,
        numeric_tolerance_used=False,derived_contact_aggregation_order_is_not_contact_record_identity=True)


def review_entry(args, manifest):
    output = args.output.resolve()
    try:
        process,plan = read(args.process_receipt),read(output/'run_plan.json')
        require(process['process_exit_observed'] is True and process['process_exit_code'] == 0
            and process.get('process_present_after_wait') is False
            and process['candidate_manifest_sha256'] == args.candidate_sha
            and process['command'] == plan['native_command']
            and process['PPO_updates'] == process['formal_updates_consumed'] == 0, 'Actual completed process/argv required')
        native = output/'native'
        report,identity = read(native/'report.json'),read(native/'identity.json')
        teaching = load(TEACHING,'source022_terminal_readback')
        integrity = teaching.validate_terminal(report,read(native/'payment_summary.json'),read(native/'teaching_summary.json'))
        require(report['checkpoint_unchanged'] and report['source_files_unchanged']
            and report['native_effective_std_matches_saved'] and report['source_script_observer_complete'],
            'Native immutable source/state terminal integrity incomplete')
        require(identity['source_commit_label'] == manifest['source_commit']
            and identity['source_manifest_sha256'] == manifest['source_inventory_sha256']
            and identity['contract_sha256'] == WORKER_SHA and identity['checkpoint_sha256'] == PT_SHA,
            'Actual022/v20 vs historical021 parent identities conflated')
        require(report['native_startup']['recovery_script_revision'] == identity['source022_runtime_expectation'],
            'Source022 actual native binding differs from realv20 design')
        old = read(EXPLORATION/'mechanical_side_speed_candidate_021/native_SIDE_001/native/report.json')
        exact_keys = ('initial_physical_birth_tensor_sha256','initial_physical_birth_sha256','initial_env_rng_sha256',
            'native_group_population','sampling_rng_initial_sha256','sampling_rng_final_sha256',
            'first_native_sampling','final_native_sampling','native_effective_std')
        require(all(report[k] == old[k] for k in exact_keys), 'SameREC3076 initial birth/private sampling/std differs')
        summary = read(native/'source_script_observer_summary.json')
        require(report['source_script_observer_summary_sha256'] == sha(native/'source_script_observer_summary.json')
            and summary['complete'] and summary['policy_steps'] == 600 and summary['physical_steps'] == 2400
            and summary['callbacks_restored_preclose'] and not summary['control_hooks_injected'], 'Source observer preclose receipt incomplete')
        helper = load(HERE/'source_script_capture.py','source022_readback_helper')
        mechanical = read(native/'mechanical_summary.json')
        require(mechanical['complete'] and mechanical['physics_steps'] == 2400 and mechanical['policy_steps'] == 600
            and mechanical['hooks_restored'] and not mechanical['contact_count_at_capacity_indicators'], 'Mechanical clock/contact capacity integrity incomplete')
        traces = decoded(native/'mechanical_5ms.jsonl.gz')
        selected = {lane:[] for lane in helper.LANES}
        for row in traces:
            lane = row['state']['lane']
            require(lane in selected and row['state']['birth_id'] == 1 and len(row['state']['pin_gaps_m']) == 6,
                'First-birth lane/actual6pin trace differs')
            selected[lane].append(row['physical_step'])
        for lane in helper.LANES:
            count = mechanical['lane_row_counts'][str(lane)]
            require(selected[lane] == list(range(count)), 'Selected physical trace gaps/duplicates')
            entry = summary['targets'][str(lane)]
            require(entry['complete'] and entry['original_calls_attempted'] == entry['original_calls_completed']
                == dict(decode=600,delay=2400,pd=2400,fsm=2400,cost=600) and entry['hooks_restored'],
                'Single-call native target/payment chain incomplete')
            journal = decoded(native/f'targets/lane_{lane}/pd_target_payment_2216.jsonl.gz')
            require([r['policy_tick'] for r in journal] == list(range(1,len(journal)+1))
                and count == 4*len(journal)+1 and all(r['lane'] == lane and r['birth_id'] == 1 for r in journal),
                'Selected target/payment first birth/cadence differs')
        argmax = decoded(native/'mechanical_all4096_argmax_5ms.jsonl.gz')
        require([r['physical_step'] for r in argmax] == list(range(1,2401))
            and all(0 <= r['lane'] < 4096 and 0 <= r['pin'] < 6 for r in argmax), 'Full2400 all4096 native argmax coverage absent')
        require(not any(native.rglob('*failure.json')) and not (native/'side_override_applied.json').exists()
            and not (native/'capture_prepare_hook_applied.json').exists(), 'Failure/control-hook sidecar present')
        cutoff = summary['global_source_change']['strict_pre_intervention_last_physical_step']
        whole_hook = EXPLORATION/'capture_prepare_candidate_021/native_hook_candidate_002/native_CAPTURE_PREPARE_002/native/mechanical_5ms.jsonl.gz'
        original = EXPLORATION/'mechanical_capture_candidate_021/native_capture_003/native/mechanical_5ms.jsonl.gz'
        remaining = EXPLORATION/'mechanical_remaining_capture_candidate_021/native_remaining_001/native/mechanical_5ms.jsonl.gz'
        comparisons = dict(
            original_SOURCE021_global_preintervention=compare_trace(original,traces,helper,(1708,2216),cutoff),
            source021_SIDEonly_remaining_global_preintervention=compare_trace(remaining,traces,helper,(881,1877,2121),cutoff),
            equivalent_SIDEplusgyro_hook_selected_fullwindow=compare_trace(whole_hook,traces,helper,(1708,2216)))
        comparison_passed = all(x['complete_record_comparison_passed'] for x in comparisons.values())
        result = dict(schema='source022-native-five-lane-source-readback-v1',
            status='source_native_readback_complete_contact_multiset_compared_not_qualified',
            native_integrity=integrity,actual_source022_runtime=identity['source022_runtime_expectation'],
            actual_source_commit=manifest['source_commit'],actual_worker_sha256=WORKER_SHA,
            historical_parent_actor=3076,initial_birth_private_sampling_saved_std_exact=True,
            observer=summary,mechanical_capture=mechanical,
            complete_contact_multiset_mechanical_comparison_passed=comparison_passed,
            comparisons=comparisons,old_ordered_strict_rejection_preserved=True,
            source_global_preintervention_cutoff=cutoff,after_global_intervention_selected_differences_are_descriptive=True,
            all4096_argmax_rows=2400,PPO_updates=0,formal_updates_consumed=0,qualification=None,
            app_close_returned=None,
            inputs_sha256={str(p):sha(p) for p in [Path(args.process_receipt),output/'run_plan.json',HERE/'candidate_manifest.json',whole_hook,original,remaining,*native.rglob('*')] if p.is_file()},
            limitations=['Normal/friction multiset is exact within each body/filter/tick. It does not match invented normal/friction pairs.',
                'Original ordered-contact/aggregate-payment rejection remains historical and is not overwritten.',
                'Global source-sensitive reference algebra is same-input evidence, not an old physical counterfactual trajectory.',
                'L2 population/count changes, survival and12s right-censoring do not grant recovery function or mechanical safety qualification.'])
        write(output/'source_script_native_review.json',result)
    except BaseException:
        write(output/'source_script_native_rejection.json',dict(status='preserved_completed_native_readback_rejected',
            error=traceback.format_exc(),PPO_updates=0,qualification=None))
        return 2
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('cpu','native','review'), default='cpu')
    parser.add_argument('--candidate-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--process-receipt', type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(HERE in output.parents, 'Candidate-only output required')
    manifest = verify_candidate(args.candidate_sha)
    if args.mode == 'review':
        require(args.process_receipt is not None and output.is_dir(), 'Externally completed native output required')
        return review_entry(args,manifest)
    require(args.process_receipt is None and not output.exists(), 'New output required')
    capture = load(CAPTURE,'source022_frozen_capture_factory')
    teaching = load(TEACHING,'source022_private_teaching_driver')
    require(sha(TEACHING) == TEACHING_SHA, 'Frozen native driver changed')
    helper = load(HERE/'source_script_capture.py','source022_helper')
    output.mkdir(parents=True)
    old = read(PAIR/'admission.json')
    admission = old | dict(schema='source022-native-source-script-admission-v1',source_commit=manifest['source_commit'],
        source_root=manifest['source_root'],source_manifest=str(HERE/'source_inventory.json'),
        source_manifest_sha256=manifest['source_inventory_sha256'],
        contract_relative='contracts/v6_task_campaign_recovery_worker_v20.json',contract_sha256=WORKER_SHA)
    write(output/'admission.json',admission)
    resources = None
    if args.mode == 'native':
        pair = load(EXPLORATION/'run_sealed_REC3076_pair_021.py','source022_resource_check')
        resources = pair.local_GPU_empty()
    command = [PYTHON,'-B',str(Path(__file__).resolve()),'--mode','native','--candidate-sha',args.candidate_sha,'--output',str(output)]
    write(output/'run_plan.json',dict(schema='source022-native-source-script-plan-v1',mode=args.mode,
        candidate_manifest_sha256=args.candidate_sha,native_command=command,
        base_source_commit=BASE_COMMIT,source_commit=manifest['source_commit'],source_root=manifest['source_root'],
        source_inventory_sha256=manifest['source_inventory_sha256'],
        source_runtime_controls='Actual v20 recipe SIDE1.5 and actual native source CAPTURE gyro; no replacement/hook',
        checkpoint_sha256=PT_SHA,context='sealed_recovery_resume',resource_check=resources,
        lanes=4096,policy_steps=600,physical_steps=2400,selected_lanes=list(helper.LANES),
        PPO_updates=0,formal_updates_consumed=0,qualification=None))
    def admitted(path,digest):
        require(sha(path) == digest and read(path) == admission, 'Source022 parent admission changed')
        return admission
    teaching.read_admission = admitted
    def authenticate_source(inner,a,base):
        inner.observer_candidate_sha = args.candidate_sha
        return authenticate(inner,a,base,teaching)
    teaching.authenticate = authenticate_source
    original_factory = teaching.make_observer
    if args.mode == 'native':
        def factory(base,producer,composition,inner,identity,report,binding,writer):
            parent = original_factory(base,producer,composition,inner,identity,report,binding,writer)
            return make_source_observer(parent,capture,helper,writer,manifest['source_root'])
        teaching.make_observer = factory
    inherited = ['--admission',str(output/'admission.json'),'--admission-sha',sha(output/'admission.json'),
        '--arm','mean','--output',str(output/'native')]
    if args.mode == 'cpu':
        inherited += ['--preflight-only']
    code = teaching.main(inherited)
    verify_candidate(args.candidate_sha)
    if code:
        return code
    if args.mode == 'cpu':
        report = read(output/'native/report.json')
        require(report['status'] == 'CPU_preflight_passed_scene_not_run' and not report['native_scene_run'], 'Actual source022 App-free preflight failed')
        write(output/'CPU_admission.json',dict(status='source022_v20_parent3076_CPU_authenticated_not_native_tested',
            identity_sha256=sha(output/'native/identity.json'),candidate_manifest_sha256=args.candidate_sha,
            native_scene_run=False,control_hooks_injected=False,PPO_updates=0,qualification=None))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
