"""Unscored policy-rate geometry observations from native cached body poses."""
from __future__ import annotations

import math

import torch

LINK_PAIRS = tuple((left, right) for left, right in (
    ('L_link1', 'R_link1'), ('L_link2', 'R_link2'), ('L_link3', 'R_link3'),
    ('LLL_link1', 'RRR_link1'), ('LLL_link2', 'RRR_link2'),
    ('LL_link1', 'RR_link1'), ('LL_link2', 'RR_link2'),
    ('LL_link3', 'RR_link3'), ('LL_link4', 'RR_link4')))


def body_points(root_pose, world_points):
    """Reuse the native XYZW rotation for points relative to the root origin."""
    from wheeled_tasks.chassis.ordinary_posture_curriculum import rotate_vectors
    inverse = root_pose[:, 3:7].clone()
    inverse[:, :3].neg_()
    return rotate_vectors(inverse, world_points - root_pose[:, None, :3])


def paired_geometry(root_pose, wheel_centers, hip_origins, paired_link_origins):
    wheels = body_points(root_pose, wheel_centers)
    hips = body_points(root_pose, hip_origins)
    links = body_points(root_pose, paired_link_origins.flatten(1, 2)).reshape_as(paired_link_origins)
    spans = wheels - hips
    lengths = spans[..., (0, 2)].norm(dim=-1)
    angles = torch.atan2(spans[..., 0], -spans[..., 2])
    angle_delta = angles[:, 0] - angles[:, 1]
    angle_delta = torch.atan2(angle_delta.sin(), angle_delta.cos())
    relative_links = links - hips[:, None]
    right_mirror = relative_links[:, :, 1] * relative_links.new_tensor((1., -1., 1.))
    pair_delta = relative_links[:, :, 0] - right_mirror
    return dict(wheel_centers_body_m=wheels,
        hip_origins_body_m=hips, hip_to_wheel_body_m=spans,
        projected_leg_length_m=lengths, projected_leg_angle_rad=angles,
        projected_leg_length_delta_m=lengths[:, 0] - lengths[:, 1],
        projected_leg_angle_delta_rad=angle_delta,
        wheel_forward_delta_m=wheels[:, 0, 0] - wheels[:, 1, 0],
        wheel_vertical_delta_m=wheels[:, 0, 2] - wheels[:, 1, 2],
        paired_link_hip_relative_delta_m=pair_delta,
        paired_link_hip_relative_error_m=pair_delta.norm(dim=-1))


class SpinGeometryObserver:
    """First-episode spin frames only; no reset, command, reward or FSM writes."""
    def __init__(self, groups):
        self.groups = list(groups)
        self.rows = [dict(case=name, policy_frames=0, first_request_s=None,
            applied_target_frames=0, planar_speed_sum_m_s=0.,
            requested_yaw_error_sum_rad_s=0., applied_yaw_error_sum_rad_s=0.,
            leg_length_error_sum_m=0., leg_angle_error_sum_rad=0.,
            leg_length_error_max_m=0., leg_angle_error_max_rad=0.,
            root_displacement_max_m=0., wheel_midpoint_displacement_max_m=0.,
            both_wheel_force_above_2n_frames=0) for name in groups]
        self.origins = {}
        self.pair_ids = None
        self.frames = []

    def snapshot(self, env):
        if self.pair_ids is None:
            names = env.robot.body_names
            self.pair_ids = [[names.index(left), names.index(right)] for left, right in LINK_PAIRS]
        root = env.robot.data.root_link_pose_w.torch
        poses = env.robot.data.body_link_pose_w.torch
        pairs = poses[:, self.pair_ids, :3]
        wheels = env.wheel_centers()
        geometry = paired_geometry(root, wheels, pairs[:, 0], pairs)
        geometry['wheel_contact_force_n'] = env.contact_force[:, env.wheel_ids].norm(dim=-1)
        return {key: value.detach().cpu() for key, value in geometry.items()}

    def observe(self, time_s, data, active, policy_mask, requested, applied, geometry,
                root_before, wheel_midpoint_before, wheel_midpoint_after):
        for i, row in enumerate(self.rows):
            # Pure policy ownership and an actual yaw-only request are required.
            # The terminal frame is counted, all later held-terminal frames are excluded.
            if (not bool(active[i]) or not bool(policy_mask[i]) or row['case'] != 'stand_305mm'
                    or requested[i][0] != 0. or requested[i][1] == 0.):
                continue
            if i not in self.origins:
                self.origins[i] = (root_before[i].clone(), wheel_midpoint_before[i].clone())
                row['first_request_s'] = time_s
            root_origin, wheel_origin = self.origins[i]
            length_error = abs(float(geometry['projected_leg_length_delta_m'][i]))
            angle_error = abs(float(geometry['projected_leg_angle_delta_rad'][i]))
            row['policy_frames'] += 1
            row['applied_target_frames'] += int(abs(float(applied[i][1]) - requested[i][1]) <= .1)
            row['planar_speed_sum_m_s'] += float(data['planar_speed_world'][i])
            row['requested_yaw_error_sum_rad_s'] += abs(float(data['omega'][i, 2]) - requested[i][1])
            row['applied_yaw_error_sum_rad_s'] += abs(float(data['omega'][i, 2]) - float(applied[i][1]))
            row['leg_length_error_sum_m'] += length_error
            row['leg_angle_error_sum_rad'] += angle_error
            row['leg_length_error_max_m'] = max(row['leg_length_error_max_m'], length_error)
            row['leg_angle_error_max_rad'] = max(row['leg_angle_error_max_rad'], angle_error)
            row['root_displacement_max_m'] = max(row['root_displacement_max_m'],
                float((data['position'][i, :2] - root_origin).norm()))
            row['wheel_midpoint_displacement_max_m'] = max(row['wheel_midpoint_displacement_max_m'],
                float((wheel_midpoint_after[i] - wheel_origin).norm()))
            row['both_wheel_force_above_2n_frames'] += int(bool((geometry['wheel_contact_force_n'][i] > 2.).all()))
            self.frames.append(dict(replica_index=i, time_s=time_s,
                requested=list(requested[i]), applied=applied[i].tolist(),
                planar_speed_world_m_s=float(data['planar_speed_world'][i]),
                yaw_actual_rad_s=float(data['omega'][i, 2]),
                leg_length_delta_m=float(geometry['projected_leg_length_delta_m'][i]),
                leg_angle_delta_rad=float(geometry['projected_leg_angle_delta_rad'][i]),
                root_displacement_m=float((data['position'][i, :2] - root_origin).norm()),
                wheel_midpoint_displacement_m=float((wheel_midpoint_after[i] - wheel_origin).norm()),
                wheel_contact_force_n=geometry['wheel_contact_force_n'][i].tolist()))

    def report(self):
        rows = []
        for source in self.rows:
            row = dict(source)
            frames = row['policy_frames']
            for source_key, target_key in (
                ('planar_speed_sum_m_s', 'planar_speed_mean_m_s'),
                ('requested_yaw_error_sum_rad_s', 'requested_yaw_mae_rad_s'),
                ('applied_yaw_error_sum_rad_s', 'applied_yaw_mae_rad_s'),
                ('leg_length_error_sum_m', 'leg_length_mae_m'),
                ('leg_angle_error_sum_rad', 'leg_angle_mae_rad')):
                row[target_key] = row.pop(source_key) / frames if frames else None
            rows.append(row)
        return dict(schema='native-spin-geometry-v1', qualification=None, replicas=rows, policy_trace=self.frames,
            paired_link_names=LINK_PAIRS,
            frame_scope='Yaw-only requested frames with native pure-policy mask in the first episode; no warmup exclusion',
            displacement_anchor='Native root and wheel midpoint immediately before the first included command step',
            geometry_scope='Root-frame wheel centers, hip-to-wheel projected spans, and hip-relative link origins; not raw motor-angle equality',
            contact_scope='Native filtered wheel force magnitude >2 N; not a terrain contact classification',
            mirror_scope='Reflection of hip-relative link origins in the body XZ plane; CAD asymmetry is retained, no zero-error admission gate')


def validate_report(report, count):
    """Recompute every spin summary from actual policy-rate observations."""
    if (report.get('schema') != 'native-spin-geometry-v1' or report.get('qualification', 'missing') is not None
            or len(report.get('replicas', [])) != count
            or report.get('paired_link_names') != [list(pair) for pair in LINK_PAIRS]):
        raise ValueError('Incomplete native spin geometry report')
    previous = {}
    grouped = [[] for _ in range(count)]
    for frame in report.get('policy_trace', []):
        i, time_s = frame.get('replica_index'), frame.get('time_s')
        if (type(i) is not int or not 0 <= i < count or type(time_s) not in (int, float)
                or not math.isfinite(time_s) or time_s < 0 or time_s <= previous.get(i, -1.)):
            raise ValueError('Spin trace requires monotonic physical replica times')
        previous[i] = time_s
        if (len(frame.get('requested', [])) != 3 or len(frame.get('applied', [])) != 3
                or len(frame.get('wheel_contact_force_n', [])) != 2
                or frame['requested'][0] != 0. or frame['requested'][1] == 0.):
            raise ValueError('Spin trace requires a yaw-only command and two measured wheel forces')
        values = [*frame['requested'], *frame['applied'], *frame['wheel_contact_force_n'],
            *(frame.get(key) for key in ('planar_speed_world_m_s', 'yaw_actual_rad_s',
                'leg_length_delta_m', 'leg_angle_delta_rad', 'root_displacement_m', 'wheel_midpoint_displacement_m'))]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError('Spin trace contains missing or nonfinite physical values')
        grouped[i].append(frame)
    for row, frames in zip(report['replicas'], grouped):
        n = len(frames)
        if (type(row.get('policy_frames')) is not int or row['policy_frames'] != n
                or row['first_request_s'] != (frames[0]['time_s'] if n else None)
                or n and row['case'] != 'stand_305mm'):
            raise ValueError('Spin summary differs from actual policy frames')
        counters = dict(applied_target_frames=sum(abs(f['applied'][1] - f['requested'][1]) <= .1 for f in frames),
            both_wheel_force_above_2n_frames=sum(all(force > 2. for force in f['wheel_contact_force_n']) for f in frames))
        for key, expected in counters.items():
            if type(row.get(key)) is not int or row[key] != expected:
                raise ValueError('Spin summary counter differs from native observations')
        series = dict(planar_speed_mean_m_s=[f['planar_speed_world_m_s'] for f in frames],
            requested_yaw_mae_rad_s=[abs(f['yaw_actual_rad_s'] - f['requested'][1]) for f in frames],
            applied_yaw_mae_rad_s=[abs(f['yaw_actual_rad_s'] - f['applied'][1]) for f in frames],
            leg_length_mae_m=[abs(f['leg_length_delta_m']) for f in frames],
            leg_angle_mae_rad=[abs(f['leg_angle_delta_rad']) for f in frames])
        for key, values in series.items():
            expected = sum(values) / n if n else None
            if expected is None:
                passed = row.get(key, 'missing') is None
            else:
                passed = type(row.get(key)) in (int, float) and math.isclose(row[key], expected, abs_tol=1e-12, rel_tol=1e-12)
            if not passed:
                raise ValueError('Spin summary mean differs from native observations')
        for key, values in (
            ('leg_length_error_max_m', series['leg_length_mae_m']),
            ('leg_angle_error_max_rad', series['leg_angle_mae_rad']),
            ('root_displacement_max_m', [f['root_displacement_m'] for f in frames]),
            ('wheel_midpoint_displacement_max_m', [f['wheel_midpoint_displacement_m'] for f in frames])):
            if (type(row.get(key)) not in (int, float)
                    or not math.isclose(row[key], max(values, default=0.), abs_tol=1e-12, rel_tol=1e-12)):
                raise ValueError('Spin summary peak differs from native observations')
