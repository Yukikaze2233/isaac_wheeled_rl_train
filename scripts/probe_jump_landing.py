#!/usr/bin/env python3
"""Exact train-role physical jump probe, no PPO or qualification credit."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from probe_recovery_takeover import digest, require, write_report, snapshot_sources, load_actor


def authenticate(args):
    from train_chassis import preflight
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    from wheeled_tasks.chassis.full_curriculum import checkpoint_contract_path, checkpoint_update_count
    config, manifest = preflight(args.contract)
    binding = config['design_preflight']
    design_path = ROOT / binding['file']
    require(digest(design_path) == binding['sha256'], 'Independent design SHA differs')
    row = json.loads(design_path.read_text())['stages'][binding['stage']]['roles']['train']['invocation']
    rule = row['seed']
    seed = rule if type(rule) is int else rule['root_seed'] if rule.get('rule') == 'root_plus_block_index' else None
    require(type(seed) is int, 'Probe requires exact train seed')
    control = load_chassis_control(ROOT / config['control_math_source'])
    stage = config['enabled_stages'][0]
    receipt = design_preflight(config, control, ROOT, stage=stage, role='train',
        num_envs=row['num_envs'], seed=seed, block_index=0)
    require((config['physics_dt'],config['policy_dt'],config['actor_dim'],config['critic_dim']) == (.005,.02,35,81), 'Control/observation ABI differs')
    require(config.get('jump_landing',{}).get('enabled') is True, 'Physical jump contact module absent')
    source_path = checkpoint_contract_path(args.checkpoint)
    source = json.loads(source_path.read_text())
    for key in ('asset_manifest_sha256','control_math_sha256','physics_dt','policy_dt','actor_dim','critic_dim',
                'action_dim','actor_observation_source','policy_action_order','policy_architecture','v5_control'):
        require(source.get(key) == config.get(key), f'Actor inference ABI differs: {key}')
    provenance = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        checkpoint_contract=str(source_path),checkpoint_contract_sha256=digest(source_path),
        checkpoint_learning_updates=checkpoint_update_count(args.checkpoint),actor_only_deterministic_inference=True)
    return config, manifest, control, stage, seed, row['num_envs'], receipt, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--actor-update',type=int,required=True)
    parser.add_argument('--seconds',type=float,default=12.)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    require(args.actor_update>=0 and math.isfinite(args.seconds) and 0<args.seconds<=30., 'Invalid engineering clock/duration')
    args.output.mkdir(parents=True,exist_ok=False)
    path=args.output/'report.json'
    report=dict(schema='jump-contact-native-probe-v1',status='starting',scope='engineering_only_no_PPO_no_qualification',
        formal_updates_consumed=0,optimizer_steps=0,qualification=None,started_at_utc=datetime.now(timezone.utc).isoformat(),
        target_contract=str(args.contract),target_contract_sha256=digest(args.contract),
        source_files_sha256=snapshot_sources(),forced_engineering_actor_update=args.actor_update,
        requested_physics_seconds=args.seconds,randomization_or_delay_overrides=[])
    write_report(path,report)
    launcher=env=None
    try:
        cfg,manifest,control,stage,seed,count,receipt,provenance=authenticate(args)
        report.update(design_preflight=receipt,checkpoint=provenance,num_envs=count,run_role='train',seed=seed)
        if args.preflight_only:
            report['status']='preflight_passed_scene_not_run'
            return 0
        os.environ.update(ENABLE_CAMERAS='0',LIVESTREAM='0')
        from isaaclab.app import AppLauncher
        launcher=AppLauncher(dict(headless=True,enable_cameras=False,device='cuda:0',kit_args='--/exts/omni.kit.telemetry/skipDeferredStartup=true'))
        import torch
        from wheeled_tasks.chassis.env import ChassisEnv
        torch.set_num_threads(4);torch.manual_seed(seed)
        env=ChassisEnv(cfg,manifest,control,ROOT,stage_name=stage,num_envs=count,device='cuda:0',level=1.,seed=seed,run_role='train')
        require(env.startup_report.get('runtime_preflight',{}).get('status')=='runtime_configuration_passed','Runtime design differs')
        actor,before=load_actor(env,args.checkpoint,cfg,provenance,seed)
        env.task_actor_update=env.stage_actor_update=env.global_actor_update=args.actor_update
        env.reset(torch.arange(count,device=env.device));observations=env.get_observations()
        counts=dict(verified_flight_frames=0,touchdown_events=0,paid_landing_frames=0,released_paid_frames=0,whole_task_success=0)
        ranges={};began=time.monotonic();steps=math.ceil(args.seconds/env.policy_dt)
        for tick in range(1,steps+1):
            with torch.inference_mode():
                action=actor(observations['policy'])
                observations,reward,done,extras=env.step(action)
            require(all(torch.isfinite(x).all() for x in (*observations.values(),reward)), 'Nonfinite native state/reward')
            require(tuple(observations['policy'].shape)==(count,35) and tuple(observations['critic'].shape)==(count,81),'Observation ABI changed')
            timing=env.control_audit.last
            require(timing['pd_updates']==timing['feedback_refreshes']==4 and timing['target_decodes']==1,'200 Hz fresh-feedback control differs')
            data=extras['diagnostics'];w=data['jump_landing_active']
            counts['verified_flight_frames']+=int(data['jump_landing_verified_flight'].sum())
            counts['touchdown_events']+=int(data['jump_landing_touchdown'].sum())
            counts['paid_landing_frames']+=int(w.sum())
            counts['released_paid_frames']+=int((w & ~data['jump_requested']).sum())
            counts['whole_task_success']+=int((done.bool() & data['success']).sum())
            for name in ('jump_landing_height','jump_recovery_height','jump_landing_vertical_velocity',
                         'jump_landing_velocity','jump_landing_attitude','jump_landing_position'):
                require(extras['native_reward_components'][name].eq(0).all(),f'Legacy landing payment survived: {name}')
            for name,value in extras['native_reward_components'].items():
                if name.startswith(('jump_landing_','jump_impact_','jump_precontact_','jump_down_speed_')):
                    require(torch.isfinite(value).all(),f'Nonfinite physical contact component {name}')
                    low,high=float(value.min()),float(value.max())
                    require(high<=0.,f'Positive contact plateau: {name}')
                    old=ranges.setdefault(name,dict(minimum=low,maximum=high,nonzero_frames=0))
                    old.update(minimum=min(old['minimum'],low),maximum=max(old['maximum'],high))
                    old['nonzero_frames']+=int((value!=0).sum())
            if tick==1 or tick%100==0:
                report.update(status='running',policy_steps=tick,counts=counts,cost_densities=ranges,
                    wall_seconds=time.monotonic()-began,startup=deepcopy(env.startup_report))
                write_report(path,report)
        require(all(torch.equal(before[n],v.detach().cpu()) for n,v in actor.state_dict().items()),'Engineering probe changed actor')
        require(counts['touchdown_events']>0 and counts['released_paid_frames']>0,'Physical touchdown or trigger-independent payment not observed')
        require(any(v['nonzero_frames'] for v in ranges.values()),'No physical contact costs observed')
        report.update(status='passed',policy_steps=steps,counts=counts,cost_densities=ranges,
            wall_seconds=time.monotonic()-began,actor_unchanged=True,
            cancellation_repeat_bounce_timeout='CPU lifecycle regression; unobserved native events receive no pass credit')
        return 0
    except Exception:
        report.update(status='failed',error=traceback.format_exc());traceback.print_exc();return 1
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat();write_report(path,report)
        if env is not None:env.close()
        if launcher is not None:launcher.app.close()

if __name__=='__main__':
    raise SystemExit(main())
