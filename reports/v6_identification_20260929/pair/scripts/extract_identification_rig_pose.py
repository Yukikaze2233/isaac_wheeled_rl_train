#!/usr/bin/env python3
"""Inspect supported-chassis IMU pose directly from the MCAP's embedded schema.

Requires mcap-ros2-support, numpy and scipy; no installed ROS message overlay.
The mean gravity vector is a diagnostic until IMU-to-model alignment is checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mcap', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    quaternion, gyro, accel, ages, times, repetition = [], [], [], [], [], []
    total = running = 0
    schema_sha = None
    with args.mcap.open('rb') as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, _, message in reader.iter_decoded_messages(
                topics=['/wheel_leg/identification/sample']):
            total += 1
            if message.phase != 2:
                continue
            running += 1
            if running % 100:
                continue
            schema_sha = hashlib.sha256(schema.data).hexdigest()
            quaternion.append(message.imu_quaternion_xyzw)
            gyro.append(message.imu_gyro)
            accel.append(message.imu_acceleration_mps2)
            ages.append((message.control_steady_ns-message.imu_last_ns)/1e9)
            times.append(message.control_steady_ns)
            repetition.append(message.repetition_id)
    q = np.array(quaternion)
    valid = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > .9)
    if not valid.any():
        raise ValueError('No valid running IMU orientation; do not assume a level rig.')
    rotations = Rotation.from_quat(q[valid])
    gravity = rotations.inv().apply(np.tile([0., 0., -9.81], (valid.sum(), 1)))
    report = {
        'source_mcap': str(args.mcap), 'schema_sha256': schema_sha,
        'messages': total, 'running_messages': running,
        'running_repetition_ids': sorted(set(repetition)),
        'diagnostic_sample_stride': 100, 'valid_pose_fraction': float(valid.mean()),
        'gravity_body_mean_mps2': gravity.mean(axis=0).tolist(),
        'gravity_body_std_mps2': gravity.std(axis=0).tolist(),
        'imu_age_p99_s': float(np.quantile(ages, .99)),
        'gyro_rms_rad_s': np.sqrt(np.nanmean(np.array(gyro)**2, axis=0)).tolist(),
        'acceleration_mean_mps2': np.nanmean(accel, axis=0).tolist(),
        'rig_tilt_from_level_deg': float(np.degrees(np.arccos(np.clip(
            -gravity.mean(axis=0)[2]/np.linalg.norm(gravity.mean(axis=0)), -1., 1.)))),
        'model_frame_alignment_verified': False,
        'note': 'IMU body frame only; orientation is not evidence of CAD-to-encoder zero alignment.',
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'rig_pose.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.output/'rig_pose.npz', control_steady_ns=times,
                        quaternion_xyzw=q, gyro=gyro, acceleration=accel, valid=valid)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
