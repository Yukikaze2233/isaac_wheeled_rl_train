#!/usr/bin/env python3
"""Read-only CPU geometry supplement; no simulator, PPO or qualification."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import time

OUT=Path(__file__).resolve().parent
META=json.loads((OUT/'source_candidate_019.json').read_text())
PARSER=argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--source-root',type=Path,default=Path(META['clean_root']))
PARSER.add_argument('--output',type=Path,default=OUT/'full_ramp_geometry_019.json')
PARSER.add_argument('--allow-working-helper-amendment',action='store_true')
ARGS=PARSER.parse_args()
ROOT=ARGS.source_root.resolve()
sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import torch
from wheeled_tasks.chassis.ordinary_posture_curriculum import OrdinaryPostureCurriculum,ordinary_eligibility,rotate_vectors
from wheeled_tasks.chassis.nonstandard_start import NonstandardStart,nonstandard_start_settings,support_height_for_footprint
from wheeled_tasks.chassis.task import choose_scene_groups,terrain_surfaces,Surface
from wheeled_tasks.chassis.release_stop import release_stop_surfaces

COUNT=4096
DOMAINS=('recovery','speed','terrain','jump','stepjump')
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def summary(angles):
    if not len(angles):return dict(count=0,roll_min_rad=None,roll_max_rad=None,pitch_min_rad=None,pitch_max_rad=None,abs_max_rad=None)
    return dict(count=len(angles),roll_min_rad=float(angles[:,0].min()),roll_max_rad=float(angles[:,0].max()),
        pitch_min_rad=float(angles[:,1].min()),pitch_max_rad=float(angles[:,1].max()),abs_max_rad=float(angles.abs().max()))


def terrain_buffers(config,groups,kinds):
    rows=[];offsets=[]
    tiers=config.get('terrain_difficulty_tiers',[1.])
    for i,(group,kind) in enumerate(zip(groups,kinds)):
        spec=config['skill_specs'][group];limits=spec.get('terrain_limits',config['terrain_limits'])
        level=spec.get('terrain_level',tiers[(i//len(tiers))%len(tiers)])
        surfaces=terrain_surfaces(kind,limits,level,i)
        if config.get('release_stop',{}).get('enabled') and spec.get('kind')=='release_stop' and kind!='flat':
            surfaces=release_stop_surfaces(kind,limits,level,i,config['release_stop'])
        if config.get('evaluation_long_corridors') and kind in ('flat','jump'):
            half=config.get('flat_half_length_m',44.)
            surfaces=[Surface(-half,half)] if config.get('flat_triangle_mesh') else [Surface(x-4.,x+4.) for x in range(-40,41,8)]
        x=0. if config.get('centered_locomotion_resets') and kind in ('flat','jump') else -1.5
        if config.get('release_stop',{}).get('enabled') and spec.get('kind')=='release_stop':x=0.
        if group in ('stand','translate','rotate') and kind in ('slope','material','rough'):x=0.
        elif kind=='platform':x=1.
        elif group in ('step_up','step_down'):x=-.8
        if spec.get('body_climb') or spec.get('step_jump'):x=-.8
        rows.append([[s.x0,s.x1,s.z0,s.slope,s.cross_slope] for s in surfaces]);offsets.append(x)
    width=max(map(len,rows));surfaces=torch.zeros(COUNT,width,5);valid=torch.zeros(COUNT,width,dtype=torch.bool)
    for i,row in enumerate(rows):surfaces[i,:len(row)]=torch.tensor(row);valid[i,:len(row)]=True
    origins=torch.zeros(COUNT,3);root=torch.zeros(COUNT,7);root[:,0]=torch.tensor(offsets);root[:,6]=1.
    root[:,2]=support_height_for_footprint(surfaces,valid,root[:,:2],torch.zeros(COUNT,4))+.32
    return root,origins,surfaces,valid


def verify_support(lesson,root,profiles,checked,surfaces,valid):
    lane=checked.nonzero().flatten();q=root[lane,3:7];p=profiles[lane]+1
    wheels=rotate_vectors(q,lesson.wheel_points[p]);axes=rotate_vectors(q,lesson.wheel_axes[p])
    extents=lesson.wheel_radius[None,:,None]*(1.-axes.square()).clamp_min(0.).sqrt()
    extents+=lesson.wheel_halfwidth[None,:,None]*axes.abs()
    wheel_gaps=[]
    for side in (0,1):
        bounds=torch.stack((-extents[:,side,0],extents[:,side,0],-extents[:,side,1],extents[:,side,1]),-1)
        plane=support_height_for_footprint(surfaces[lane],valid[lane],root[lane,:2]+wheels[:,side,:2],bounds)
        wheel_gaps.append(root[lane,2]+wheels[:,side,2]-extents[:,side,2]-plane)
    gaps=torch.stack(wheel_gaps,-1)
    assert gaps.min()>=-1e-6
    settled=lesson.selected[lane];minimum=gaps.amin(-1)
    assert torch.allclose(minimum[settled],torch.zeros(int(settled.sum())),atol=1e-6,rtol=0.)
    assert minimum[~settled].numel()==0 or float(minimum[~settled].max())<=.001+2e-6
    points=rotate_vectors(q,lesson.collision_corners[p]);lo,hi=points.amin(-2),points.amax(-2)
    clearance=[]
    for shape in range(lo.shape[1]):
        bounds=torch.stack((lo[:,shape,0],hi[:,shape,0],lo[:,shape,1],hi[:,shape,1]),-1)
        plane=support_height_for_footprint(surfaces[lane],valid[lane],root[lane,:2],bounds)
        clearance.append(root[lane,2]+lo[:,shape,2]-plane)
    clearance=torch.stack(clearance,-1)
    assert clearance.min()>=.001-1e-6
    assert torch.allclose(q.norm(dim=-1),torch.ones(len(q)),atol=1e-6,rtol=0.)
    return dict(checked_lane_count=len(lane),both_wheels_minimum_support_gap_m=float(gaps.min()),
        one_wheel_support_max_abs_gap_m=float(gaps.amin(-1).abs().max()),
        rejected_fallback_reset_lift_above_wheel_support_max_m=float(minimum[~settled].max()) if (~settled).any() else 0.,
        nonwheel_AABB_minimum_terrain_clearance_m=float(clearance.min()),
        nonwheel_AABB_shape_count=clearance.shape[1],quaternion_unit_norm=True)


def run_case(config,manifest,names,asset,*,shared_bank_matrix=False):
    groups,kinds=zip(*choose_scene_groups(config['scene_groups'],COUNT));specs=config['skill_specs']
    spin=torch.tensor([specs[group].get('kind')=='spin_translate' for group in groups])
    recovery=torch.tensor([bool(specs[group].get('recovery_pose')) or specs[group].get('kind')=='recovery' for group in groups])
    eligible=ordinary_eligibility(config,groups,kinds,'cpu',recovery=recovery,spin=spin)
    settings=config['ordinary_posture_curriculum'];clock=settings['origin_actor_update']+3000
    if shared_bank_matrix:eligible=torch.ones(COUNT,dtype=torch.bool)
    lesson=OrdinaryPostureCurriculum(COUNT,'cpu',settings,eligible)
    bank_settings=config.get('nonstandard_start')
    if shared_bank_matrix:bank_settings=nonstandard_start_settings(settings['origin_actor_update'])
    bank=(NonstandardStart(COUNT,'cpu',.005,bank_settings,ROOT,names,asset_manifest_sha256=config['asset_manifest_sha256']) if bank_settings else None)
    lesson.bind_geometry(asset,manifest,names,bank)
    if shared_bank_matrix:
        root=torch.zeros(COUNT,7);root[:,2]=.32;root[:,6]=1.;origins=torch.zeros(COUNT,3)
        surfaces=torch.tensor([[[-4.,4.,0.,0.,0.]]]).repeat(COUNT,1,1);valid=torch.ones(COUNT,1,dtype=torch.bool)
    else:root,origins,surfaces,valid=terrain_buffers(config,groups,kinds)
    ids=torch.arange(COUNT);joints=torch.tensor([[manifest['nominal_joint_pos'][name] for name in names]]).repeat(COUNT,1)
    generator=torch.Generator().manual_seed(290927)
    lesson.reset(ids,clock,generator=generator)
    boundary_lanes=eligible.nonzero().flatten()[:4]
    lesson.selected[boundary_lanes]=True
    lesson.angles[boundary_lanes]=torch.tensor([[-.15,-.15],[-.15,.15],[.15,-.15],[.15,.15]])[:len(boundary_lanes)]
    if shared_bank_matrix:
        lesson.selected.fill_(True)
        lesson.angles=(torch.rand(COUNT,2,generator=generator)*2.-1.)*.15
        lesson.angles[:4]=torch.tensor([[-.15,-.15],[-.15,.15],[.15,-.15],[.15,.15]])
    before_angles=lesson.angles.clone();checked=lesson.selected.clone()
    lesson.initial_orientation(ids,root)
    profiles=torch.full((COUNT,),-1,dtype=torch.long)
    if bank is not None:
        bank_eligible=eligible&~torch.tensor([bool(specs[group].get('body_climb') or specs[group].get('step_jump')) for group in groups])
        bank.reset(ids,clock,eligible=bank_eligible,generator=generator)
        if shared_bank_matrix:
            bank.profile_ids.copy_((torch.arange(COUNT)%(len(bank.profile_names)+1))-1)
        support=support_height_for_footprint(surfaces,valid,root[:,:2],bank.collision_footprints(ids))
        bank.initial_state(ids,root,joints,torch.zeros(COUNT,6),support,preserve_orientation=True)
        profiles=bank.profile_ids.clone()
    before_joints=joints.clone();legacy_root=root.clone()
    try:
        lesson.align_support(ids,root,joints,origins,surfaces,valid,profile_ids=profiles if bank is not None else None)
    except ValueError as error:
        trace=error.__traceback__;local=None
        while trace:
            if trace.tb_frame.f_code.co_name=='align_support':local=trace.tb_frame.f_locals
            trace=trace.tb_next
        if local is None or 'body_z' not in local or 'wheel_z' not in local:raise
        invalid=(local['body_z']>local['wheel_z']+1e-6).nonzero().flatten()
        lanes=local['ids'][local['lane'][invalid]]
        details=[]
        for j,i in zip(invalid.tolist(),lanes.tolist()):
            profile=int(local['profiles'][j]);names=['nominal']+(bank.profile_names if bank is not None else [])
            details.append(dict(lane=i,scene_group=groups[i],terrain_kind=kinds[i],profile=names[profile],
                root_local_xy_m=local['local'][j].tolist(),legacy_reset_root_z_m=float(legacy_root[i,2]),
                legacy_reset_root_z_above_zero_body_required_m=float(legacy_root[i,2]-local['body_z'][j]),
                legacy_root_height_meets_zero_angle_1mm_AABB_margin=bool(legacy_root[i,2]>=local['body_z'][j]-1e-6),
                quaternion_after_rejection_xyzw=local['quat'][j].tolist(),
                sampled_roll_pitch_rad=before_angles[i].tolist(),wheel_support_required_root_z_m=float(local['wheel_z'][j]),
                body_AABB_required_root_z_with_1mm_margin_m=float(local['body_z'][j]),
                missing_1mm_support_margin_m=float(local['body_z'][j]-local['wheel_z'][j]),
                body_AABB_clearance_at_wheel_support_m=float(.001-local['body_z'][j]+local['wheel_z'][j])))
        return dict(status='failed',scope='CPU_helper_worker_mask_and_configured_bank_reset_geometry_only',
            num_envs=COUNT,actor_update=clock,relative_actor_update=3000,eligible_count=int(eligible.sum()),
            configured_bank=bool(config.get('nonstandard_start')),test_bank_instantiated=bank is not None,
            spin_excluded_count=int(spin.sum()),native_population_count_claim=False,formal_updates_consumed=0,qualification=None,
            source_alignment_completed=False,error=str(error),explicit_boundary_angles=[[-.15,-.15],[-.15,.15],[.15,-.15],[.15,.15]],
            proposed_angles=summary(before_angles[checked]),sampled_angle_cap_rad=lesson.amplitudes(clock)[1],
            rejected_candidate_count=int(local['rejected'].sum()),
            all_oriented_candidate_count=len(local['lane']),zero_angle_fallback_invalid_count=len(lanes),
            zero_angle_fallback_invalid_lanes=details,joint_configuration_exactly_preserved=bool(torch.equal(before_joints,joints)),
            terrain_families=sorted(set(kinds)))
    assert torch.equal(before_joints,joints)
    accepted=checked&lesson.selected;rejected=checked&~lesson.selected
    assert int(accepted.sum())==lesson.birth_selected_count and int(rejected.sum())==lesson.birth_rejected_count
    assert not lesson.angles[rejected].any()
    assert torch.equal(root[rejected,3:7],torch.tensor([0.,0.,0.,1.]).expand(int(rejected.sum()),4))
    proof=verify_support(lesson,root,profiles,checked,surfaces,valid)
    rejected_proof=verify_support(lesson,root,profiles,rejected,surfaces,valid) if rejected.any() else dict(checked_lane_count=0,scope='no sampled tilt rejection')
    labels=['nominal']+(bank.profile_names if bank is not None else [])
    per_profile={name:dict(population=int((profiles+1==i).sum()),proposed=int((checked&(profiles+1==i)).sum()),
        accepted=int((accepted&(profiles+1==i)).sum()),rejected=int((rejected&(profiles+1==i)).sum())) for i,name in enumerate(labels)}
    return dict(status='passed',scope='all_lane_shared_asset_nominal_plus_all_bank_rows_static_matrix' if shared_bank_matrix else 'CPU_helper_worker_mask_and_configured_bank_reset_geometry_only',
        num_envs=COUNT,actor_update=clock,relative_actor_update=3000,configured_bank=bool(config.get('nonstandard_start')),
        test_bank_instantiated=bank is not None,eligible_count=int(eligible.sum()),spin_excluded_count=int(spin.sum()) if not shared_bank_matrix else 0,
        native_population_count_claim=False,formal_updates_consumed=0,qualification=None,
        sampled_angle_cap_rad=lesson.amplitudes(clock)[1],sampled_push_cap_m_s=lesson.amplitudes(clock)[2],
        explicit_boundary_angles=[[-.15,-.15],[-.15,.15],[.15,-.15],[.15,.15]],
        proposed_angles=summary(before_angles[checked]),accepted_angles=summary(before_angles[accepted]),
        rejected_proposed_angles=summary(before_angles[rejected]),rejected_final_angles=summary(lesson.angles[rejected]),
        per_profile=per_profile,all_selected_geometry=proof,rejected_zero_angle_geometry=rejected_proof,
        joint_configuration_exactly_preserved=True,terrain_families=sorted(set(kinds)) if not shared_bank_matrix else ['flat'])


def main():
    started=time.monotonic();torch.set_num_threads(1)
    assert sha(META['archive_local'])==META['archive_sha256']
    configs={d:read(ROOT/f'contracts/v6_task_campaign_{d}_worker_v18.json') for d in DOMAINS}
    asset=ROOT/configs['speed']['asset_directory'];manifest=read(asset/'manifest.json');names=manifest['tree_joint_names']
    files={*(ROOT/'src/wheeled_tasks/chassis').glob('*.py'),*(ROOT/f'contracts/v6_task_campaign_{d}_worker_v18.json' for d in DOMAINS),
        ROOT/'contracts/v6_nonstandard_start_profiles_v1.json',asset/'manifest.json',*(asset/name for name in manifest['files_sha256'])}
    before={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)}
    with tarfile.open(META['archive_local'],'r:gz') as archive:
        assert archive.pax_headers.get('comment')==META['source_commit']
        matched={member.name:hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in archive if member.name in before and member.isfile()}
        amended={name:dict(archive_sha256=matched.get(name),candidate_sha256=value) for name,value in before.items() if matched.get(name)!=value}
        if ARGS.allow_working_helper_amendment:
            assert 'src/wheeled_tasks/chassis/ordinary_posture_curriculum.py' in amended
            assert set(amended)<={'src/wheeled_tasks/chassis/ordinary_posture_curriculum.py','src/wheeled_tasks/chassis/preflight.py'}
        else:assert matched==before
    domains={}
    for domain,config in configs.items():
        assert sha(ROOT/f'contracts/v6_task_campaign_{domain}_worker_v18.json')==META['worker_hashes'][domain]
        domains[domain]=run_case(config,manifest,names,asset)
    shared=run_case(configs['speed'],manifest,names,asset,shared_bank_matrix=True)
    spec=read(asset/'model_spec.json');loader=importlib.util.spec_from_file_location('full_ramp_verified_fk',asset/'tools/v6_mechanism.py')
    fk=importlib.util.module_from_spec(loader);loader.loader.exec_module(fk)
    bank=NonstandardStart(COUNT,'cpu',.005,configs['speed']['nonstandard_start'],ROOT,names)
    poses=[manifest['nominal_joint_pos']]+[dict(zip(names,row.tolist())) for row in bank.joint_positions]
    closure=[]
    for pose in poses:
        frames=fk.fk(spec,pose)
        error=np.asarray([fk.point(frames[c['body0']],c['local_pos0_m'])-fk.point(frames[c['body1']],c['local_pos1_m']) for c in spec['constraints']])
        closure.append(float(np.linalg.norm(error,axis=-1).max()))
    assert max(closure)<2e-7
    after={str(path.relative_to(ROOT)):sha(path) for path in sorted(files)};assert after==before
    result=dict(schema='v18-full-ramp-cpu-reset-geometry-supplement-v1',status='passed' if all(v['status']=='passed' for v in domains.values()) else 'failed_geometry_boundary_needs_review',verified_at_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=None if ARGS.allow_working_helper_amendment else META['source_commit'],parent_source_commit=META['source_commit'],
        candidate_source_unfrozen=ARGS.allow_working_helper_amendment,amended_files_from_parent_archive=amended,
        archive_sha256=META['archive_sha256'],clean_root=str(ROOT),
        formal_updates_consumed=0,optimizer_steps=0,qualification=None,native_or_GPU_validation=False,additional_deployment_gate=False,
        scope='real_sealed_FK_and_collision_AABB_cpu_helpers_at_relative_actor3000_with_sampled_and_exact_±.15_angles',
        domains=domains,shared_asset_nominal_plus_all_bank_matrix=shared,closed_chain_max_error_m=max(closure),closure_by_profile_m=closure,
        source_files_before_sha256=before,source_files_after_sha256=after,source_files_unchanged=True,
        all_selected_source_bytes_match_frozen_archive=not bool(amended),verification_runner_sha256=sha(__file__),duration_seconds=time.monotonic()-started,
        limitations=['CPU reset buffer and analytic support geometry only; no PhysX contacts, behavior quality, push actuation or qualification.',
            'Worker masks reconstructed from frozen scene and spin/recovery flags before simulation; native population receipts remain authoritative.',
            'Four boundary lanes deliberately selected at exactly ±.15; all-bank matrix deliberately tests every authenticated pose on a flat support plane.',
            'Recovery has no configured nonstandard bank; its worker sample uses nominal joints. Shared all-bank matrix is supplemental geometry coverage only.',
            'AABB rejection is conservative; rejection counts are retained rather than treated as failures when zero-angle fallback remains legal.'])
    write(ARGS.output,result)
    print(json.dumps(dict(status=result['status'],closure_max=result['closed_chain_max_error_m'],domains={d:dict(status=v['status'],eligible=v['eligible_count'],accepted=v.get('accepted_angles',{}).get('count'),rejected=v.get('rejected_proposed_angles',{}).get('count'),invalid_zero_fallback=v.get('zero_angle_fallback_invalid_count',0)) for d,v in domains.items()},all_bank=dict(accepted=shared['accepted_angles']['count'],rejected=shared['rejected_proposed_angles']['count']),duration_seconds=result['duration_seconds'])))

if __name__=='__main__':main()
