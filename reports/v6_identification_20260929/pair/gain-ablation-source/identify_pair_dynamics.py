"""Fixed-chassis, installed-leg MuJoCo replay of recorded 50 Hz references.

The normalized CAD-to-hardware mapping remains a conditional prior. Only the
selected leg is retained: a rigid support decouples the opposite leg dynamically.
No measured state is injected after each segment's initial condition. This is an
offline identification experiment, never a controller or training release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

TRAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN / 'scripts'))
from audit_v5_motor_coupling import MotorCouplingAudit

BUNDLE = TRAIN / 'model/纯底盘_v5_232mm/urdf'
ROOT = Path(__file__).resolve().parent


class Replay:
    def __init__(self, run, side='left', physics_dt=.0005, bundle=BUNDLE,
                 coordinate_sign=1., gravity=(0., 0., -9.81)):
        self.bundle = bundle
        if coordinate_sign not in (-1., 1.):
            raise ValueError('Coordinate sign must be +1 (old asset) or -1 (V6 asset).')
        raw = dict(np.load(run / 'response_1khz.npz'))
        self.a = {k: v[raw['phase'] == 2] for k, v in raw.items()}
        self.meta = json.loads((run / 'response_1khz.json').read_text())
        self.cfg = self.meta['controller']
        if (self.cfg['side'] != side or self.cfg['control_law'] != 'rl_pd'
                or self.cfg['control_frequency_hz'] != 1000
                or self.cfg['reference_frequency_hz'] != 50):
            raise ValueError('Expected selected-side PC PD at 1000 Hz with 50 Hz held targets.')
        if not np.isfinite(physics_dt) or physics_dt <= 0 or physics_dt > .001:
            raise ValueError('Physics timestep must be finite and in (0, 0.001].')
        if 'repetition_id' in self.a and len(np.unique(self.a['repetition_id'])) != 1:
            raise ValueError('Select exactly one running repetition before identification.')
        self.coordinate_sign = coordinate_sign
        self.real_q = coordinate_sign * (self.a['q_ref_model'] - self.a['position_error_model'])
        self.reference = coordinate_sign * self.a['q_ref_model']
        self.side = side
        self.letter = 'L' if side == 'left' else 'R'
        self.joints = [self.letter + '_joint1', self.letter * 2 + '_joint1']
        axes = [0, 1] if side == 'left' else [2, 3]
        self.kp = np.array(self.cfg['pd_kp'])[axes]
        self.kd = np.array(self.cfg['pd_kd'])[axes]
        self.cap = np.array(self.cfg['max_torque'])[axes]
        self.api_to_model = coordinate_sign * np.array(self.cfg['model_sign'])[axes]
        self.real_dq = self.api_to_model * self.a['dq_api']
        submitted = self.api_to_model * self.a['tau_frame_api']
        self.missing_frame_samples = (~np.isfinite(submitted)).sum(axis=0).tolist()
        # Periodic enable/system submissions have no new torque payload. The
        # input-replay hypothesis holds the last host-submitted effort; it is
        # not a claim that the board acknowledged CAN delivery.
        self.real_torque = np.empty_like(submitted)
        for axis in range(2):
            indices = np.maximum.accumulate(np.where(
                np.isfinite(submitted[:, axis]), np.arange(len(submitted)), -1))
            self.real_torque[:, axis] = np.where(
                indices >= 0, submitted[np.maximum(indices, 0), axis], 0.)
        self.audit = MotorCouplingAudit(bundle)
        self.spec = self.audit.spec
        root = ET.parse(bundle / 'robot.xml').getroot()
        world = root.find('worldbody')
        for geom in list(world.findall('geom')):
            world.remove(geom)
        base = world.find('body')
        base.remove(base.find('freejoint'))
        base.set('pos', '0 0 0')
        base.set('quat', '1 0 0 0')
        for body in list(base.findall('body')):
            if not body.get('name').startswith(self.letter):
                base.remove(body)
        # Visual/collision meshes are irrelevant to an airborne, fixed rig.
        # Original collision masks permit ground contact only, no self-contact.
        for body in root.iter('body'):
            for geom in list(body.findall('geom')):
                body.remove(geom)
        root.remove(root.find('asset'))
        root.remove(root.find('keyframe'))
        for tag in ('equality', 'actuator'):
            parent = root.find(tag)
            for element in list(parent):
                if not element.get('name').startswith(self.letter):
                    parent.remove(element)
        self.xml = ET.tostring(root, encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.model.opt.timestep = physics_dt
        self.model.opt.gravity[:] = gravity
        self.substeps = round(.001 / physics_dt)
        if self.substeps < 1 or abs(self.substeps * physics_dt - .001) > 1e-12:
            raise ValueError('Physics timestep must divide the 1 ms control period exactly.')
        self.data = mujoco.MjData(self.model)
        self.qi = np.array([self.model.joint(n).qposadr[0] for n in self.joints])
        self.vi = np.array([self.model.joint(n).dofadr[0] for n in self.joints])
        self.ui = np.array([self.model.actuator(n + '_equivalent_torque').id for n in self.joints])
        self.spring = self.letter + '_spring_slide'
        self.si = self.model.joint(self.spring).qposadr[0]
        self.su = self.model.actuator(self.spring + '_gas_force').id
        self.binding = self.spec['spring_binding'][self.spring]
        self.curve = json.loads((bundle / 'fit_10mpa.json').read_text())
        self.site_pairs = np.array([[self.model.site(e.get('site1')).id,
                                     self.model.site(e.get('site2')).id]
                                    for e in root.find('equality')])
        self.initial = {}
        self.time_audits = {}

    def state_at(self, sample):
        target = self.real_q[sample]
        nom = self.spec['nominal_joint_pos']
        active = {n: nom[n] for n in self.audit.manifest['control_joint_names']}
        q = dict(nom)
        # Continue the two input axes to preserve the CAD assembly branch.
        origin = np.array([nom[n] for n in self.joints])
        for r in np.linspace(0, 1, max(24, int(np.max(abs(target-origin))/.08)+1)):
            prescribed = {**active, **dict(zip(self.joints, origin+r*(target-origin)))}
            q, gap = self.audit.solve(prescribed, q)
        full = np.zeros(self.model.nq)
        for j in range(self.model.njnt):
            joint = self.model.joint(j)
            full[joint.qposadr[0]] = q[joint.name]
        # Initial velocity follows the local closure tangent; measurements are
        # used here only, not as a persistent constraint during integration.
        tangent = np.zeros((self.model.nv, 2))
        for j, name in enumerate(self.joints):
            prescribed = {**active, **dict(zip(self.joints, target))}
            prescribed[name] += 1e-5
            perturbed, _ = self.audit.solve(prescribed, q)
            for k in range(self.model.njnt):
                joint = self.model.joint(k)
                tangent[joint.dofadr[0], j] = (perturbed[joint.name]-q[joint.name])/1e-5
        return full, tangent @ self.real_dq[sample], gap, tangent

    def initial_state(self, segment):
        if segment in self.initial:
            return self.initial[segment]
        ix = np.flatnonzero(self.a['segment_id'] == segment)
        if len(ix) < 2 or np.any(np.diff(ix) != 1):
            raise ValueError(f'Segment {segment} requires at least two contiguous samples.')
        dt = np.diff(self.a['control_steady_ns'][ix].astype(np.int64)) / 1e9
        elapsed = (self.a['control_steady_ns'][ix].astype(np.int64)
                   - int(self.a['control_steady_ns'][ix[0]])) / 1e9
        grid_error = float(np.max(np.abs(elapsed-np.arange(len(ix))*.001)))
        if (np.any(np.diff(self.a['tick'][ix]) != 1) or np.any(dt <= 0)
                or np.max(dt, initial=0.) > .002 or grid_error > .001):
            raise ValueError(f'Segment {segment} has omitted ticks; nominal 1 ms replay rejected.')
        self.time_audits[int(segment)] = {'max_dt_s': float(np.max(dt)),
            'min_dt_s': float(np.min(dt)), 'max_nominal_grid_error_s': grid_error}
        full, vel, gap, _ = self.state_at(ix[0])
        self.initial[segment] = (ix, full, vel, gap)
        return self.initial[segment]

    def static_load_audit(self):
        """Project CAD gravity/spring force through closure tangents at dwells.

        This is a load diagnostic, not a forward-dynamics fit. A static residual
        cannot be explained by increasing armature or a motor response time.
        """
        rows = []
        for segment in np.unique(self.a['segment_id']):
            ix = np.flatnonzero(self.a['segment_id'] == segment)
            if len(ix) < 1000 or np.max(np.ptp(self.reference[ix], axis=0)) > 1e-8:
                continue
            window = ix[-500:]
            if np.max(np.sqrt(np.mean(self.real_dq[window]**2, axis=0))) > .08:
                continue
            sample = window[len(window)//2]
            full, _, gap, tangent = self.state_at(sample)
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = full
            compression = self.binding['compression_at_q_zero_m'] - full[self.si]
            force = np.polynomial.polynomial.polyval(
                np.clip(compression/self.curve['stroke_m'], 0., 1.),
                self.curve['monomial_coefficients_n'])
            self.data.ctrl[self.su] = force
            mujoco.mj_forward(self.model, self.data)
            gravity = tangent.T @ self.data.qfrc_bias
            gas = -(tangent.T @ self.data.qfrc_actuator)
            actual = self.real_torque[window].mean(axis=0)
            rows.append({'segment': int(segment), 'q_rad': full[self.qi].tolist(),
                         'gravity_hold_nm': gravity.tolist(), 'gas_hold_nm': gas.tolist(),
                         'predicted_hold_nm': (gravity+gas).tolist(),
                         'measured_command_nm': actual.tolist(),
                         'residual_nm': (actual-gravity-gas).tolist(),
                         'velocity_rms_rad_s': np.sqrt(np.mean(self.real_dq[window]**2, axis=0)).tolist(),
                         'compression_m': float(compression), 'closure_gap_m': gap,
                         'knee_inner_deg': self.audit.inner_angle_deg(self.letter)})
        return rows

    def replay(self, segment, params, max_samples=None, mode='closed_loop'):
        ix, q0, v0, gap0 = self.initial_state(segment)
        if max_samples:
            ix = ix[:max_samples]
        m, data = self.model, self.data
        # Passive engine damping is fixed except at the identified motor axes.
        m.dof_armature[self.vi] = params.get('armature', [.015795, .015795])
        m.dof_damping[self.vi] = params.get('damping', [0., 0.])
        m.dof_frictionloss[self.vi] = params.get('frictionloss', [0., 0.])
        mujoco.mj_resetData(m, data)
        data.qpos[:] = q0
        data.qvel[:] = v0
        mujoco.mj_forward(m, data)
        tau_s = np.broadcast_to(params.get('time_constant_s', 0.), (2,))
        alpha = np.exp(-m.opt.timestep / np.maximum(tau_s, 1e-12))
        gain = np.broadcast_to(params.get('gain', 1.), (2,))
        effort = self.api_to_model * self.a['tau_cmd_api'][ix[0]]
        prediction = np.empty((len(ix), 2))
        velocity = np.empty_like(prediction)
        torque = np.empty_like(prediction)
        compression = np.empty(len(ix))
        max_gap = gap0
        for k, sample in enumerate(ix):
            prediction[k] = data.qpos[self.qi]
            velocity[k] = data.qvel[self.vi]
            request = (np.clip(self.kp*(self.reference[sample]-prediction[k])
                               - self.kd*velocity[k], -self.cap, self.cap)
                       if mode == 'closed_loop' else self.real_torque[sample])
            torque[k] = request
            for _ in range(self.substeps):
                effort = alpha*effort + (1-alpha)*gain*request
                data.ctrl[self.ui] = effort
                s = self.binding['compression_at_q_zero_m'] - data.qpos[self.si]
                u = np.clip(s / self.curve['stroke_m'], 0., 1.)
                data.ctrl[self.su] = np.polynomial.polynomial.polyval(
                    u, self.curve['monomial_coefficients_n'])
                mujoco.mj_step(m, data)
            compression[k] = s
            if k % 10 == 0:
                gaps = data.site_xpos[self.site_pairs[:, 0]] - data.site_xpos[self.site_pairs[:, 1]]
                max_gap = max(max_gap, float(np.max(np.linalg.norm(gaps, axis=1))))
                if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel)) > 500:
                    raise RuntimeError(f'Divergent replay in segment {segment} at sample {k}')
        trim = slice(min(1000, len(ix)//4), None)
        err = prediction[trim] - self.real_q[ix][trim]
        score = {'segment': segment, 'samples': len(ix),
                 'axis_rmse_rad': np.sqrt(np.mean(err**2, axis=0)).tolist(),
                 'common_rmse_rad': float(np.sqrt(np.mean(err.mean(axis=1)**2))),
                 'difference_rmse_rad': float(np.sqrt(np.mean(np.diff(err, axis=1)**2))),
                 'max_closure_gap_m': max_gap,
                 'compression_range_m': [float(compression.min()), float(compression.max())],
                 'peak_request_nm': np.max(np.abs(torque), axis=0).tolist(),
                 'warnings': {str(i): int(w.number) for i,w in enumerate(data.warning) if w.number}}
        score['mode'] = mode
        score['velocity_rmse_rad_s'] = np.sqrt(np.mean(
            (velocity[trim] - self.real_dq[ix][trim])**2, axis=0)).tolist()
        score['mean_error_rad'] = err.mean(axis=0).tolist()
        return score, {'indices': ix, 'q_sim': prediction, 'dq_sim': velocity, 'tau_request': torque}


def fit_parameters(replay, segments, output, max_evaluations, writer, fit_torque_gain=False):
    """Fit only effective joint losses/inertia/lag; fixed gas, zeros, gain and PD.

    Equal residual counts per segment prevent long dwells from dominating. The
    complete segment is simulated before sampling residuals. All fits remain
    conditional on unverified physical zeros and rigid-rig frame alignment.
    """
    for segment in segments:
        ix, *_ = replay.initial_state(segment)
        if np.any(replay.a['segment_role'][ix] != 0):
            raise ValueError(f'Segment {segment} is marked validation; cannot fit on it.')

    def unpack(x):
        return {'armature': [float(x[0]*.02)]*2,
                'damping': (x[1:3]*.2).tolist(),
                'frictionloss': (x[3:5]*2.).tolist(),
                'time_constant_s': float(x[5]*.005),
                'gain': float(x[6]) if fit_torque_gain else 1.}

    counter = 0
    best = float('inf')

    def residual(x):
        nonlocal counter, best
        params = unpack(x)
        chunks = []
        for segment in segments:
            _, trace = replay.replay(segment, params)
            indices = trace['indices']
            select = np.linspace(100, len(indices)-1, 300).astype(int)
            dq = trace['dq_sim'][select]-replay.real_dq[indices[select]]
            q = trace['q_sim'][select]-replay.real_q[indices[select]]
            # 0.02 rad and 1 rad/s are declared objective scales, not empirical
            # sensor noise estimates or pass/fail tolerances.
            chunks.extend([(q/.02).ravel(), dq.ravel()])
        values = np.concatenate(chunks)
        score = float(np.mean(2*(np.sqrt(1+values*values)-1)))
        writer.add_scalar('identification/robust_objective', score, counter)
        if score < best:
            best = score
            (output/'candidate_parameters.json').write_text(json.dumps(params, indent=2)+'\n')
            print(json.dumps({'fit_evaluation': counter, 'best_objective': best,
                              'parameters': params}), flush=True)
        counter += 1
        return values

    initial = np.array([.015795/.02, .1/.2, .1/.2, 1./2, 1./2, .002/.005])
    lower = np.array([.05, 0., 0., 0., 0., 0.])
    upper = np.array([5., 10., 10., 6., 6., 4.])
    if fit_torque_gain:
        initial = np.append(initial, 1.)
        lower = np.append(lower, .6)
        upper = np.append(upper, 1.4)
    result = least_squares(residual, initial, bounds=(lower, upper), loss='soft_l1',
                           diff_step=.03, max_nfev=max_evaluations,
                           ftol=1e-4, xtol=1e-4, gtol=1e-4)
    params = unpack(result.x)
    report = {'parameters': params, 'fit_segments': segments,
              'optimizer_success': bool(result.success), 'message': result.message,
              'nfev': result.nfev, 'total_residual_evaluations': counter,
              'active_bounds': result.active_mask.tolist(),
              'jacobian_singular_values': np.linalg.svd(result.jac, compute_uv=False).tolist(),
              'parameter_bounds_scaled': {'lower': lower.tolist(), 'upper': upper.tolist()},
              'shared_parameters': ['motor-output armature', 'effective first-order response time'],
              'torque_gain_ablation': fit_torque_gain,
              'fixed_parameters': ([] if fit_torque_gain else ['torque gain = 1']) + ['gas curve', 'CAD inertials',
                                   'encoder zero/axis prior', 'recorded Kp/Kd'],
              'objective_scales': {'angle_rad': .02, 'velocity_rad_s': 1.},
              'training_release': False}
    (output/'fit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'parameters.json').write_text(json.dumps(params, indent=2, allow_nan=False)+'\n')
    return params


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--segments', type=int, nargs='+', default=[2, 4, 6, 8, 9, 11])
    p.add_argument('--params', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--physics-dt', type=float, default=.0005)
    p.add_argument('--bundle', type=Path, default=BUNDLE)
    p.add_argument('--side', choices=['left', 'right'], required=True)
    p.add_argument('--coordinate-sign', type=float, choices=[-1., 1.], default=1.)
    p.add_argument('--gravity', type=float, nargs=3, default=[0., 0., -9.81])
    p.add_argument('--static-only', action='store_true')
    p.add_argument('--max-samples', type=int)
    p.add_argument('--fit-segments', type=int, nargs='+')
    p.add_argument('--max-evaluations', type=int, default=25)
    p.add_argument('--mode', choices=['closed_loop', 'torque_input'], default='closed_loop')
    p.add_argument('--fit-torque-gain', action='store_true',
                   help='Conditional gain ablation; does not calibrate physical output torque.')
    args = p.parse_args()
    replay = Replay(args.run, side=args.side, physics_dt=args.physics_dt,
                    bundle=args.bundle, coordinate_sign=args.coordinate_sign,
                    gravity=args.gravity)
    params = json.loads(args.params.read_text()) if args.params else {}
    args.output.mkdir(parents=True, exist_ok=False)
    static_load = replay.static_load_audit()
    (args.output/'static_load.json').write_text(json.dumps(static_load, indent=2)+'\n')
    if args.static_only:
        print(json.dumps({'static_dwells': len(static_load), 'output': str(args.output)}))
        return
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(args.output/'events'))
    if args.fit_segments:
        if set(args.fit_segments) & set(args.segments):
            raise ValueError('Fitting and held-out segment IDs must be disjoint.')
        params = fit_parameters(replay, args.fit_segments, args.output, args.max_evaluations,
                                writer, args.fit_torque_gain)
    scores, traces = [], {}
    for segment in args.segments:
        start = time.monotonic()
        score, trace = replay.replay(segment, params, args.max_samples, args.mode)
        scores.append(score)
        traces.update({f'{segment}_{k}': v for k,v in trace.items()})
        ix = trace['indices']
        for k in range(0, len(ix), 10):
            for axis in range(2):
                prefix = f'replay/{args.side}/segment_{segment}/axis_{axis}'
                writer.add_scalar(prefix+'/q_real_rad', replay.real_q[ix[k], axis], k)
                writer.add_scalar(prefix+'/q_ref_rad', replay.reference[ix[k], axis], k)
                writer.add_scalar(prefix+'/q_sim_rad', trace['q_sim'][k, axis], k)
                writer.add_scalar(prefix+'/dq_real_rad_s', replay.real_dq[ix[k], axis], k)
                writer.add_scalar(prefix+'/dq_sim_rad_s', trace['dq_sim'][k, axis], k)
        for axis in range(2):
            writer.add_scalar(f'validation/axis_{axis}_q_rmse_rad', score['axis_rmse_rad'][axis], segment)
        print(json.dumps({**score, 'runtime_s': time.monotonic()-start}), flush=True)
    (args.output/'model.xml').write_text(replay.xml)
    report = {'source_run': replay.meta, 'mujoco_version': mujoco.__version__,
              'physics_dt_s': replay.model.opt.timestep, 'parameters': params, 'scores': scores,
              'manifest_sha256': hashlib.sha256((args.bundle/'manifest.json').read_bytes()).hexdigest(),
              'coordinate_sign': args.coordinate_sign, 'gravity': args.gravity,
              'mode': args.mode,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_npz_sha256': hashlib.sha256((args.run/'response_1khz.npz').read_bytes()).hexdigest(),
              'missing_torque_frame_samples_held': replay.missing_frame_samples,
              'nominal_grid_time_audit': replay.time_audits,
              'validation_scope': 'whole held-out segments in same existing recording; not an independent acquisition',
              'training_release': False,
              'fixed_priors': ['CAD mass/inertia/geometry', 'nominal gas curve',
                               f'rigid chassis; supplied gravity={args.gravity}', 'recorded PD and held targets'],
              'limitations': ['CAD-to-hardware zero/axis mapping is not independently verified.',
                              'No identified torque gain or physical output torque sensor.',
                              'Nominal 1 ms replay, no modeled encoder/velocity estimator.']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.output/'traces.npz', **traces)
    writer.close()


if __name__ == '__main__':
    main()
