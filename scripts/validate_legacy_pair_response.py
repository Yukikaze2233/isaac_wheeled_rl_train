"""Model-blind short-window validation of legacy host-command leg telemetry.

This adapter never claims submitted MIT torque or reuses an RL PD controller.
The fixed-chassis builder mirrors Replay's fixture, and the integration and
initial closure-tangent methods are inherited unchanged. No optimizer is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from identify_pair_dynamics import Replay, MotorCouplingAudit
from audit_pair_stop_coverage import nominal_free_mask


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LegacyReplay(Replay):
    def __init__(self, run, bundle, physics_dt=.001):
        if not np.isfinite(physics_dt) or physics_dt <= 0 or physics_dt > .001:
            raise ValueError('Physics timestep must be finite and in (0, 0.001].')
        self.meta = json.loads((run / 'response_1khz.json').read_text())
        if (self.meta.get('schema') != 'legacy-host-command-pair-telemetry'
                or self.meta.get('torque_source') != 'host_command'):
            raise ValueError('Explicit legacy host-command source required.')
        if sha256(run / 'response_1khz.npz') != self.meta['export_npz_sha256']:
            raise ValueError('Legacy telemetry hash mismatch.')
        self.cfg = self.meta['controller']
        if (self.cfg['side'] != 'left' or self.cfg['control_frequency_hz'] != 1000
                or self.cfg['model_sign'] != [1., 1., 1., 1.]
                or self.cfg['model_offset'] != [-1.6, -2.93, 1.6, 2.93]):
            raise ValueError('Reviewed legacy left-pair mapping required.')
        if self.cfg.get('control_law') == 'rl_pd':
            raise ValueError('This adapter is specifically for the legacy PI export.')
        raw = dict(np.load(run / 'response_1khz.npz'))
        self.source_rows = np.flatnonzero(raw['phase'] == 2)
        self.a = {k: v[self.source_rows] for k, v in raw.items()}
        if 'tau_frame_api' in self.a:
            raise ValueError('Host-command adapter must not masquerade as frame input.')
        self.coordinate_sign = -1.
        self.real_q = -(self.a['q_ref_model'] - self.a['position_error_model'])
        np.testing.assert_allclose(self.real_q, -(self.a['q_api'] + [-1.6,-2.93]), atol=1e-12)
        self.reference = -self.a['q_ref_model']
        side = 'left'
        self.side, self.letter = side, 'L'
        self.joints = ['L_joint1', 'LL_joint1']
        self.api_to_model = np.array([-1., -1.])
        self.real_dq = -self.a['dq_api']
        self.real_torque = -self.a['tau_cmd_api']
        np.testing.assert_array_equal(self.a['tau_cmd_api'], np.clip(self.a['tau_preclip_model'], -20., 20.))
        if not np.isfinite(self.real_torque).all():
            raise ValueError('Host command contains missing effort.')
        self.missing_frame_samples = None  # Unavailable is not zero missing frames.
        self.bundle = bundle
        gravity = (0., 0., -9.81)
        self.audit = MotorCouplingAudit(bundle)
        self.spec = self.audit.spec
        root = ET.parse(bundle / 'robot.xml').getroot()
        self.knee = 'L_joint2' if side == 'left' else 'R_jonit2'
        self.inner_limits_deg = self.audit.manifest['knee_inner_limits_deg']
        knee_joint = next(j for j in root.iter('joint') if j.get('name') == self.knee)
        self.original_knee_range = np.fromstring(knee_joint.get('range'), sep=' ')
        self.knee_upper_deg = self.inner_limits_deg[1]
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
        self.knee_id = self.model.joint(self.knee).id
        self.knee_qi = self.model.joint(self.knee).qposadr[0]
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

    def replay(self, segment, params, max_samples=None, mode='host_command'):
        if mode != 'host_command':
            raise ValueError('Legacy PI data cannot be replayed as RL PD or confirmed MIT input.')
        score, trace = super().replay(segment, params, max_samples, mode='torque_input')
        score['mode'] = 'host_command'
        score['input_delivery_confirmed'] = False
        return score, trace


def window_sum(values, count):
    total = np.r_[0., np.cumsum(np.asarray(values, dtype=float))]
    return total[count:] - total[:-count]


def choose_windows(arrays, beta, count=201, per_direction=4):
    """Freeze earliest nonoverlapping windows before examining any predictions.

    201 states span 200 ms. Each window remains within one angle bin, one source
    segment, fresh finite feedback and scope=1. At least 75% of relative velocity
    samples have the requested sign above 0.05 rad/s; mean signed speed >0.1.
    """
    t = arrays['control_steady_ns'].astype(np.int64)
    n = len(t)
    if len(beta) != n or count < 2 or count > n:
        raise ValueError('Coverage length/count mismatch.')
    vrel = -np.diff(arrays['dq_api'], axis=1)[:, 0]
    age = t[:, None] - arrays['feedback_steady_ns'].astype(np.int64)
    valid = ((arrays['actuation_scope'] == 1) & (beta < 104.)
             & np.all((age >= 0) & (age <= 5_000_000), axis=1)
             & np.isfinite(arrays['tau_cmd_api']).all(axis=1)
             & np.isfinite(arrays['dq_api']).all(axis=1))
    valid &= nominal_free_mask(beta, t * 1e-9)
    whole = window_sum(valid, count) == count
    transition_bad = np.r_[False,
        (np.diff(arrays['segment_id']) != 0) | (np.diff(arrays['tick'].astype(np.int64)) != 1)]
    whole &= (window_sum(transition_bad, count) - transition_bad[:n-count+1]) == 0
    gap_bad = np.r_[False, (np.diff(t) <= 0) | (np.diff(t) > 2_000_000)]
    # An invalid interval immediately BEFORE a window does not affect its horizon.
    whole &= (window_sum(gap_bad, count) - gap_bad[:n-count+1]) == 0
    selected, quota = [], []
    for low, high in ((76.,85.),(85.,95.),(95.,104.)):
        inbin = window_sum((beta >= low) & (beta < high), count) == count
        for sign in (1, -1):
            candidates = np.flatnonzero(whole & inbin
                & (window_sum(sign*vrel > .05, count) >= .75*count)
                & (window_sum(sign*vrel, count) / count > .1))
            kept = []
            for start in candidates:
                stop = int(start + count)
                if any(start < w['stop'] and stop > w['start'] for w in selected):
                    continue
                grid_error = np.max(abs((t[start:stop]-t[start])*1e-9 - np.arange(count)*.001))
                if grid_error > .001:
                    continue
                row = dict(start=int(start), stop=stop, bin_deg=[low,high], sign=sign,
                           segment=int(arrays['segment_id'][start]),
                           beta_range_deg=[float(beta[start:stop].min()),float(beta[start:stop].max())],
                           mean_signed_relative_speed_rad_s=float(np.mean(sign*vrel[start:stop])),
                           elapsed_s=float((t[stop-1]-t[start])*1e-9),
                           max_nominal_grid_error_s=float(grid_error))
                selected.append(row);kept.append(row)
                if len(kept) == per_direction:
                    break
            quota.append(dict(bin_deg=[low,high],sign=sign,requested=per_direction,selected=len(kept)))
    return selected, quota


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--params',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    replay=LegacyReplay(args.run,args.bundle)
    coverage=json.loads((args.run/'nominal_coverage.json').read_text())
    table=coverage['table'];delta=np.diff(replay.real_q,axis=1)[:,0]
    if delta.min()<table['delta_grid_rad'][0] or delta.max()>table['delta_grid_rad'][-1]:
        raise ValueError('Coverage table would extrapolate.')
    beta=np.interp(delta,table['delta_grid_rad'],table['knee_inner_deg'])
    windows,quota=choose_windows(replay.a,beta)
    for window in windows:
        window['source_rows']=[int(replay.source_rows[window['start']]),int(replay.source_rows[window['stop']-1]+1)]
    (args.output/'selection.json').write_text(json.dumps({'windows':windows,'quota':quota},indent=2)+'\n')
    params=json.loads(args.params.read_text())
    nominal=dict(armature=[0.,0.],damping=[0.,0.],frictionloss=[0.,0.],time_constant_s=0.,gain=1.)
    report={'scope':'legacy PI host-command 200 ms conditional prediction; no fit or release',
            'input_source':'host_command','delivery_confirmed':False,
            'source_hashes':replay.meta['source_hashes'],'adapter_npz_sha256':replay.meta['export_npz_sha256'],
            'script_sha256':sha256(Path(__file__)),'parent_replay_sha256':sha256(Path(__file__).with_name('identify_pair_dynamics.py')),
            'asset_manifest_sha256':sha256(args.bundle/'manifest.json'),
            'candidate_sha256':sha256(args.params),'parameters':params,'nominal':nominal,
            'physics_dt_s':.001,'gravity':[0.,0.,-9.81],
            'scoring':'all 201 measured states at 0..200 ms, no burn-in or within-window state injection',
            'quota':quota,'results':[]}
    for i,window in enumerate(windows):
        indices=np.arange(window['start'],window['stop'])
        pos,vel,gap,_=replay.state_at(indices[0]);replay.initial[i]=(indices,pos,vel,gap)
        for label,parameters in [('nominal',nominal),('candidate',params)]:
            score,trace=replay.replay(i,parameters)
            assert len(trace['q_sim'])==201
            e=trace['q_sim']-replay.real_q[indices];ve=trace['dq_sim']-replay.real_dq[indices]
            row={'window':i,'model':label,**window,
                 'axis_rmse_rad':np.sqrt(np.mean(e**2,axis=0)).tolist(),
                 'relative_rmse_rad':float(np.sqrt(np.mean(np.diff(e,axis=1)**2))),
                 'velocity_rmse_rad_s':np.sqrt(np.mean(ve**2,axis=0)).tolist(),
                 'end_error_rad':e[-1].tolist(),'knee_stop':score['knee_stop'],
                 'max_closure_gap_m':score['max_closure_gap_m'],'warnings':score['warnings']}
            report['results'].append(row)
            np.savez_compressed(args.output/f'{label}-window-{i}.npz',q_real=replay.real_q[indices],dq_real=replay.real_dq[indices],beta_real_nominal=beta[indices],**trace)
            print(json.dumps({'window':i,'model':label,'bin':window['bin_deg'],'sign':window['sign'],'q_rmse':row['axis_rmse_rad'],'relative_rmse':row['relative_rmse_rad'],'sim_stop_fraction':score['knee_stop']['limit_active_fraction']}),flush=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
