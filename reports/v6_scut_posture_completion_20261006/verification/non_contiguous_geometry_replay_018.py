#!/usr/bin/env python3
"""CPU replay of a captured native failure; supplemental, never admission."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile


OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=OUT.parent / 'deploy/source_candidate_018.json')
    parser.add_argument('--fixture', type=Path, default=OUT / 'native_failed_geometry_fixture_da8b11.json')
    parser.add_argument('--output', type=Path, default=OUT / 'non_contiguous_geometry_replay_018.json')
    args = parser.parse_args()
    meta, fixture = read(args.metadata), read(args.fixture)
    root = Path(meta['clean_root']).resolve(strict=True)
    archive_path = Path(meta['archive_local'])
    assert sha(archive_path) == meta['archive_sha256']
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    asset = root / 'model/纯底盘_v6_232mm_stop105/urdf'
    manifest, spec = read(asset / 'manifest.json'), read(asset / 'model_spec.json')
    selected_files = {str(path.relative_to(root))
        for folder in ('src', 'scripts', 'contracts')
        for path in (root / folder).rglob('*')
        if path.is_file() and path.suffix in ('.py', '.json')}
    selected_files |= {str((asset / name).relative_to(root)) for name in
        ('manifest.json', 'model_spec.json', 'tools/v6_mechanism.py')}
    selected_files |= {str((asset / shape['file']).relative_to(root))
        for body in spec['bodies'] for shape in body['collisions'] if shape['type'] == 'mesh'}
    before = {name: sha(root / name) for name in sorted(selected_files)}
    with tarfile.open(archive_path, 'r:gz') as archive:
        assert archive.pax_headers.get('comment') == meta['source_commit']
        matched = {name: hashlib.sha256(archive.extractfile(name).read()).hexdigest() for name in before}
    assert matched == before

    import torch
    from wheeled_tasks.chassis.ordinary_posture_curriculum import OrdinaryPostureCurriculum, posture_settings
    from wheeled_tasks.chassis.nonstandard_start import NonstandardStart, nonstandard_start_settings

    row = fixture['rows'][0]
    names = fixture['joint_names']
    lesson = OrdinaryPostureCurriculum(9, 'cpu', posture_settings(0), torch.ones(9, dtype=torch.bool))
    bank = NonstandardStart(9, 'cpu', .005, nonstandard_start_settings(0), root, names)
    lesson.bind_geometry(asset, manifest, names, bank)
    assert bank.profile_names[row['profile'] - 1] == 'gentle_05'
    assert torch.equal(bank.joint_positions[row['profile'] - 1], torch.tensor(row['nominal_joints']))
    # Remap original native lane 3870 to global lane 1 in a non-contiguous
    # reset batch. Its local row is 1; lanes 8 and 5 must remain unchanged.
    ids = torch.tensor([8, 1, 5])
    origins = torch.zeros(9, 3)
    origins[1, :2] = torch.tensor(row['root_pose'][:2])
    poses = torch.zeros(3, 7); poses[:, :3] = origins[ids]; poses[:, 2] = .32; poses[:, 6] = 1.
    joints = lesson.profile_positions[0].repeat(3, 1)
    surfaces = torch.zeros(9, len(row['surfaces']), 5)
    valid = torch.zeros(9, len(row['surfaces']), dtype=torch.bool)
    surfaces[:, 0] = torch.tensor([-4., 4., 0., 0., 0.]); valid[:, 0] = True
    surfaces[1] = torch.tensor(row['surfaces']); valid[1] = True
    lesson.selected[1] = True; lesson.angles[1] = .02
    lesson.initial_orientation(ids, poses)
    bank.profile_ids[1] = row['profile'] - 1; bank.level_ids[1] = 1
    bank.selected[1] = True; bank.remaining_steps[1] = bank.hold_steps
    bank.held_interval[1] = True; bank.learning_mask[1] = False
    bank.profile_reset_counts[row['profile'] - 1] = 7
    bank.level_reset_counts[:] = torch.tensor([14, 7, 3])
    joints[1] = bank.joint_positions[row['profile'] - 1]
    before_joints = joints.clone()
    before_profile_counts, before_level_counts = bank.profile_reset_counts.clone(), bank.level_reset_counts.clone()
    generator = torch.Generator().manual_seed(77)
    before_rng = generator.get_state().clone()
    cancelled = lesson.align_support(ids, poses, joints, origins, surfaces, valid, profile_ids=bank.profile_ids)
    bank.cancel_reset(cancelled)
    assert cancelled.tolist() == [1]
    assert torch.equal(joints[1], lesson.profile_positions[0])
    assert torch.equal(joints[[0, 2]], before_joints[[0, 2]])
    assert float(poses[1, 2]) == float(torch.tensor(.18994231522083282))
    assert torch.equal(poses[1, 3:7], torch.tensor([0., 0., 0., 1.]))
    assert not lesson.selected[1] and not lesson.angles[1].any()
    assert bank.profile_ids[1] == -1 and bank.level_ids[1] == 0
    assert not bank.selected[1] and not bank.holding[1] and not bank.capturing[1]
    assert bank.elapsed[1] == 0 and bank.episode_steps[1] == 0
    assert bank.held_interval[1] and not bank.learning_mask[1]
    assert torch.equal(bank.profile_reset_counts, before_profile_counts)
    assert torch.equal(bank.level_reset_counts, before_level_counts)
    assert torch.equal(generator.get_state(), before_rng)
    assert bool(bank.bootstrap_mask(torch.zeros(9, dtype=torch.bool))[1])
    bank.begin_policy()
    assert bank.learning_mask[1] and not bank.held_interval[1]
    assert torch.equal(bank.apply_torque(torch.ones(9, 6))[1], torch.ones(6))
    try:
        bank.cancel_reset(cancelled)
    except ValueError:
        assert bank.cancelled_reset_count == 1
    else:
        raise AssertionError('Duplicate cancellation must fail before changing counters')
    after = {name: sha(root / name) for name in sorted(selected_files)}
    assert before == after
    report = dict(schema='v18-non-contiguous-native-fixture-CPU-replay-v1', status='passed',
        scope='CPU_reset_helper_captured_native_fixture_only_not_native_admission',
        source_commit=meta['source_commit'], archive_sha256=meta['archive_sha256'],
        source_root=str(root), source_files_unchanged=True,
        source_files_sha256_before=before, source_files_sha256_after=after,
        matched_archive_file_count=len(before), fixture_source_commit=fixture['source_commit'],
        fixture_path=str(args.fixture.resolve()), fixture_sha256=sha(args.fixture),
        fixture_original_native_lane=row['lane'], native_captured_wheel_z_m=row['wheel_z'],
        native_captured_body_z_m=row['body_z'], native_captured_deficit_m=row['gap'],
        native_population_count_claim=False, formal_updates_consumed=0, optimizer_steps=0, qualification=None,
        reset_ids=ids.tolist(), cancelled_global_ids=cancelled.tolist(), cancelled_local_rows=[1],
        actual_native_joint_order=names, native_bank_coordinates_match=True, bank_name='gentle_05',
        nominal_joint_positions_exact=True, root_z_m=float(poses[1, 2]),
        quaternion_xyzw=poses[1, 3:7].tolist(), unchanged_local_rows=[0, 2],
        bank_profile_id=-1, bank_level_id=0, hold_steps=0, selected=False, capturing=False,
        sampling_attempt_counters_preserved=True, previous_interval_masks_preserved=True,
        next_policy_unmasked=True, returned_observation_can_bootstrap=True, rng_unchanged=True,
        duplicate_cancel_rejected_without_recount=True, cancelled_reset_count=bank.cancelled_reset_count,
        verified_at_utc=datetime.now(timezone.utc).isoformat())
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status=report['status'], source_commit=report['source_commit'],
        archive_sha256=report['archive_sha256'], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
