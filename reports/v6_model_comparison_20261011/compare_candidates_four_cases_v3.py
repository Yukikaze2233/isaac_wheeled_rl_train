"""Frozen actors on one existing native recovery/flat diagnostic platform."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ABI_KEYS = ("physics_dt", "policy_dt", "actor_dim", "critic_dim", "action_dim",
    "history_length", "asset_manifest_sha256", "control_math_sha256",
    "actor_observation_source", "manual_context35", "policy_action_order",
    "v5_control", "observation_scales", "command_reference", "command_slew", "motion_limits")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if revision != args.source_revision:
        raise ValueError('Unexpected frozen native evaluation source')
    if subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'scripts', 'contracts', 'assets'], cwd=root):
        raise ValueError('Native evaluation source has modifications')
    sys.path[:0] = [str(root / 'scripts'), str(root / 'src')]
    spec = spec_from_file_location('original_two_lane_comparison', args.driver)
    driver = module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.ROOT = root
    from train_chassis import preflight
    from play_chassis_suite import validated_session

    contracts = [Path(str(p) + '.contract.json') for p in args.onnx]
    sources = [preflight(p)[0] for p in contracts]
    common = sources[0]
    for candidate in sources:
        for key in ABI_KEYS:
            if candidate.get(key) != common.get(key):
                raise ValueError('Cross-model observation/control mismatch: ' + key)
    # REC41374/REC38000 use the same frozen recovery platform; other actors
    # retain their own contract identity and are assessed on this common platform.
    source, config, manifest, control, seed, receipt = driver.prepare(
        contracts[0], args.output, wide_floor=True)
    candidates = []
    for policy, contract, own_source in zip(args.onnx, contracts, sources, strict=True):
        session, metadata = validated_session(policy, contract, own_source)
        actions = session.run(None, {'obs': np.zeros((4, 35), dtype=np.float32)})[0]
        if actions.shape != (4, 6) or not np.isfinite(actions).all():
            raise ValueError('Native ONNX ABI probe failed')
        identity = dict(policy_file=str(policy.resolve()), policy_sha256=sha(policy),
            source_checkpoint_sha256=metadata.get('source_checkpoint_sha256'),
            source_contract_sha256=sha(contract),
            actor_task_update=metadata.get('checkpoint_infos', {}).get('task_actor_update'))
        candidates.append((session, metadata, identity))
    args.steps, args.device = 600, 'cpu'
    report = dict(schema='four-model-common-platform-native-comparison-v1',
        status='CPU_identity_design_ORT_preflight_passed_scene_not_run',
        source_root=str(root), source_revision=revision, driver=str(args.driver.resolve()),
        native_source_sha256={str(p.relative_to(root)):sha(p)
            for p in sorted((root/'src/wheeled_tasks/chassis').glob('*.py'))},
        script_sha256=sha(Path(__file__)), original_driver_sha256=sha(args.driver),
        started_at=datetime.now(timezone.utc).isoformat(),
        design_preflight=receipt, scene_run=False, steps_per_actor=args.steps,
        max_total_policy_ticks=args.steps * len(candidates), max_lane_transitions=args.steps * len(candidates) * 4, max_sim_seconds_per_actor=12.,
        selected_cases=[c['name'] for c in config['evaluation']['cases']],
        identities=[row[2] for row in candidates], candidates=[], qualification=None,
        formal_updates_consumed=0, optimizer_constructed=False, deterministic_mean=True,
        inference_backend='onnxruntime_CPU', identical_observation_control_keys=list(ABI_KEYS),
        common_platform_contract_sha256=sha(contracts[0]),
        scope='Single fixed birth per actor/case, four independent common-platform cases; not full skill acceptance',
        omitted_cases=['fallen_right','fallen_belly','fallen_back','reverse_translation','reverse_yaw',
                       'rough','stairs','jump'],
        horizon_rule='First terminal frame retained; surviving short-horizon episodes are right-censored',
        command_schedule={'flat': [[0,1,0,0],[1,3,2,0],[3,12,0,0]], 'yaw_2pi_stop': [[0,1,0,0],[1,5,0,2*np.pi],[5,12,0,0]], 'stand_305mm_10s': [[0,12,0,0]]},
        source_control=source['control_math_source'], command_slew=config['command_slew'],
        motion_limits=config.get('motion_limits'), diagnostic_runway_width_m=8.,
        diagnostic_boundary_x_m=config['flat_half_length_m']-4., diagnostic_boundary_y_m=3.7, playback_open_ground=False)
    driver.write_json(args.output / 'report.json', report)
    (args.output / 'source' / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':report['status'],'identities':report['identities']}), flush=True)
    if args.preflight_only:
        return 0
    report['scene_run'] = True
    return driver.run(args, config, manifest, control, seed, candidates, report)


if __name__ == '__main__':
    raise SystemExit(main())
