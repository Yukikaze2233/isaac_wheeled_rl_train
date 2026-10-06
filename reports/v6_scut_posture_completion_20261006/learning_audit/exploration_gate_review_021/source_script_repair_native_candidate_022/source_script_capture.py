"""Read-only five-lane source022 measurement; frozen producers remain single owners."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import traceback

import torch

LANES = (881, 1708, 1877, 2121, 2216)
LEVELS = (2, 2, 2, 1, 2)
PROFILE_IDS = (8, 16, 18, 19, 21)
PROFILES = ('front_down_90', 'step200_wheels_split', 'step200_lip_near_shell',
            'step200_upright_top', 'double_step_first_riser_split')
GROUPS = ('recovery_front_down_90_forward', 'recovery_step200_wheels_split_stand',
          'recovery_step200_lip_near_shell_stand',
          'recovery_step200_prone_belly_top_forward',
          'recovery_double_step_first_riser_split_forward')


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def bits(value):
    """Exact scalar identity, including signed zero; never numerical tolerance."""
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite contact value')
        return ('float64', struct.pack('!d', value).hex())
    if isinstance(value, list):
        return ('list', tuple(bits(x) for x in value))
    if isinstance(value, dict):
        return ('dict', tuple((k, bits(v)) for k, v in sorted(value.items())))
    return (type(value).__name__, value)


def contact_records(patch, kind):
    """All native normal/friction fields stay paired within each actual record."""
    if kind == 'normal':
        fields = ('force_scalar_n', 'point_physx_m', 'normal_physx', 'separation_m',
                  'impulse_scalar_ns_derived_force_times_dt')
    else:
        require(kind == 'friction', 'Unknown contact patch kind')
        fields = ('force_physx_n', 'point_physx_m', 'impulse_physx_ns_derived_force_times_dt')
    count = patch['count']
    require(type(count) is int and count >= 0
        and all(len(patch[k]) == count for k in fields), 'Incomplete full contact records')
    return Counter(bits({k: patch[k][i] for k in fields}) for i in range(count))


def compare_contacts(a, b):
    """Only patch order is ignored; body/filter/tick attribution is never relaxed."""
    if a is None or b is None:
        return None if a is b else 'initial-contact-absence'
    aa, bb = a['sensor_body_rows'], b['sensor_body_rows']
    if len(aa) != len(bb):
        return 'body-row-count'
    for x, y in zip(aa, bb):
        if x['sensor_index'] != y['sensor_index']:
            return 'sensor-index'
        for key in ('matrix_force_n', 'net_force_all_contacts_n'):
            if bits(x[key]) != bits(y[key]):
                return key
        for key, kind in (('normal_patches', 'normal'), ('friction_patches', 'friction')):
            if len(x[key]) != len(y[key]):
                return key+'-filter-count'
            for p, q in zip(x[key], y[key]):
                if p['surface_index'] != q['surface_index'] or p['count'] != q['count']:
                    return key+'-filter-or-multiplicity'
                if contact_records(p, kind) != contact_records(q, kind):
                    return key+'-full-record-multiset'
    return None


class SourceChangeWatch:
    """Observe all4096 once-called updates; no second FSM/physics/control call."""
    def __init__(self, env, hooks, output, writer, plain):
        self.env, self.hooks, self.output, self.writer, self.plain = env, hooks, output, writer, plain
        self.calls = 0
        self.first = None
        self.counts = Counter()
        self.complete = self.closed = False
        self.stream = gzip.open(output/'source_sensitive_events.jsonl.gz', 'xt')

    def install(self):
        fsm = self.env.recovery.fsm
        prior = fsm.update
        require(fsm.config['side_speed_rad_s'] == 1.5 and fsm.config['handover_gyro_max'] == .75,
            'Source/cfg repair was not natively constructed')
        def update(q, dq, gravity, gyro, height, support, clear):
            self.calls += 1
            oldphase = fsm.phase.clone()
            oldref = fsm.script_targets[:, fsm.LEG_IDS].clone()
            oldside = fsm.side_start.clone()
            olddirection = fsm.side_direction.clone()
            oldticks = fsm.phase_ticks.clone()
            valid = (torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
                & torch.isfinite(gravity).all(-1) & torch.isfinite(gyro).all(-1)
                & torch.isfinite(height) & (gravity.norm(dim=-1) > .5))
            result = prior(q, dq, gravity, gyro, height, support, clear)
            require(result is fsm.script_targets, 'Original FSM return identity differs')
            # This is an explicitly derived same-input old SIDE reference, not
            # an old physical trajectory or another controller invocation.
            elapsed = (oldticks+1)*fsm.dt
            duration = fsm.config['side_angle_rad']/6.
            stroke = torch.where(elapsed <= duration, elapsed*6.,
                fsm.config['side_angle_rad']-(elapsed-duration-.08).clamp_min(0.)*6.)
            stroke = stroke.clamp(0., fsm.config['side_angle_rad'])
            goal = oldside+olddirection*stroke[:, None]/fsm.axis_signs
            delta = goal-oldref
            fraction = (6.*fsm.dt/delta.abs().amax(-1).clamp_min(1e-6)).clamp_max(1.)
            previous_speed_ref = oldref+delta*fraction[:, None]
            side_change = (oldphase == 5) & fsm.enabled & valid & (previous_speed_ref != result[:, fsm.LEG_IDS]).any(-1)
            safe_gravity = torch.nan_to_num(gravity)
            safe_gravity = safe_gravity/safe_gravity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            tilt = torch.acos((-safe_gravity[:, 2]).clamp(-1., 1.))
            capture_change = ((oldphase == 6) & (fsm.phase == 6) & fsm.enabled & valid
                & (tilt < math.radians(20.)) & support & (height > .22)
                & (gyro.norm(dim=-1) >= fsm.config['handover_gyro_max']))
            for name, mask in (('source_SIDE_reference_differs_from_same_input_speed6', side_change),
                               ('source_CAPTURE_gyro_condition_blocks_old_PREPARE', capture_change)):
                ids = mask.nonzero().flatten()
                if not len(ids):
                    continue
                event = dict(reference=self.calls, input_physical_step=self.calls-1,
                    kind=name, affected_all_population_lanes=self.plain(ids),
                    native_phase_before=self.plain(oldphase[ids]), native_phase_after=self.plain(fsm.phase[ids]),
                    native_gyro_norm_rad_s=self.plain(gyro[ids].norm(dim=-1)),
                    scope='source-sensitive same-input change; subsequent shared reset/RNG may diverge')
                self.stream.write(json.dumps(event, allow_nan=False)+'\n')
                self.counts[name] += len(ids)
                if self.first is None:
                    self.first = event
            return result
        self.hooks.install(fsm, 'update', update)

    def cleanup(self):
        if self.closed:
            return []
        errors = self.hooks.restore()
        self.stream.close()
        self.closed = True
        return errors

    def finish(self):
        require(self.calls == 2400, 'Global source-change reference cadence incomplete')
        require(not self.cleanup(), 'Source-change read-only callback restoration failed')
        self.complete = True
        result = dict(complete=True, reference_calls=self.calls, first_global_source_sensitive_change=self.first,
            strict_pre_intervention_last_physical_step=(self.first['reference']-1 if self.first else 2400),
            all_population_event_counts=dict(self.counts), hooks_restored=True,
            no_second_FSM_or_controller_calls=True, no_control_override=True,
            counterfactual_reference_is_not_old_physical_trajectory=True)
        self.writer(self.output/'source_change_summary.json', result)
        return result


def configure_mechanical(module, pd_entries):
    """Private frozen remaining helper instance; selection/extra reads only."""
    module.LANES, module.LEVELS, module.PROFILES, module.GROUPS = LANES, LEVELS, PROFILES, GROUPS
    module.TRACE_STEPS = 2400
    original_pin = module.pin_geometry
    cached = {}
    def pin(env):
        world, gaps = original_pin(env)
        cached['gaps'] = gaps
        return world, gaps
    module.pin_geometry = pin
    parent = module.MechanicalCapture
    class FullSourceCapture(parent):
        def __init__(self, *args, **kwargs):
            self.argmax_stream = None
            super().__init__(*args, **kwargs)
            self.argmax_stream = gzip.open(self.output/'mechanical_all4096_argmax_5ms.jsonl.gz', 'xt')

        def state(self, lane, world, gaps):
            result = super().state(lane, world, gaps)
            fsm = self.env.recovery.fsm
            result['native_FSM_after_reference_before_current_physics'] = module.plain(dict(
                stable_masks={name:value[lane] for name,value in fsm.stable_pass_masks.items()},
                stable_ticks=fsm.stable_ticks[lane], stable_peak_ticks=fsm.stable_peak_ticks[lane],
                stable_break_count=fsm.stable_break_count[lane], ready=fsm.ready[lane],
                holding_rl=fsm.holding_rl[lane], motion_released=fsm.motion_released[lane],
                phase_ticks=fsm.phase_ticks[lane], side_start=fsm.side_start[lane],
                side_direction=fsm.side_direction[lane], axis_signs=fsm.axis_signs))
            return result

        def control(self, target):
            result = super().control(target)
            result['once_decoded_and_actual_delayed_PD'] = []
            for lane in LANES:
                entry = pd_entries.get(lane)
                active = entry.active if entry else None
                result['once_decoded_and_actual_delayed_PD'].append(
                    None if not active or not entry.selected() else
                    dict(decode=active['decode'], substep=entry.substep()))
            return result

        def post_physics(self):
            super().post_physics()
            gaps = cached['gaps']
            flat = int(gaps.argmax()); lane, pin = divmod(flat, gaps.shape[1])
            fsm = self.env.recovery.fsm
            row = dict(physical_step=self.physics_steps, policy_tick=self.policy_steps, substep=self.substep,
                lane=lane, pin=pin, pin_gap_m=float(gaps[lane, pin]), birth_id=self.birth_ids[lane],
                phase=int(fsm.phase[lane]), route=int(fsm.route[lane]),
                whole_actor_mask_so_far=bool(self.env.recovery.learning_mask[lane]))
            self.argmax_stream.write(json.dumps(row, allow_nan=False)+'\n')

        def cleanup(self):
            errors = super().cleanup()
            if self.argmax_stream is not None:
                self.argmax_stream.close()
            return errors
    module.MechanicalCapture = FullSourceCapture
    return module


def lane_capture(frozen, lane, level, profile_id, name, env, parent, reward, hooks, output, writer):
    """Generalize frozen PD observer's selection/init, retain its original hooks."""
    frozen.LANE = lane
    class SelectedCapture(frozen.TargetPaymentCapture):
        def __init__(self):
            self.env, self.parent, self.module = env, parent, reward
            self.hooks, self.output, self.writer = hooks, output, writer
            require(env.num_envs == 4096 and env.dt == .005 and env.policy_dt == .02
                and env.decimation == 4 and env.usb_transport is None
                and env.signal_delay.cfg['feedback_pd'] == 'fresh_physics_feedback', 'Native ABI/feedback changed')
            r = env.recovery
            require(bool(r.enabled[lane]) and int(r.pose_curriculum.level_ids[lane]) == level
                and int(r.pose_curriculum.profile_ids[lane]) == profile_id
                and r.pose_curriculum.profile_names[profile_id] == name, 'Selected first-birth identity differs')
            require(r.fsm.config['side_speed_rad_s'] == 1.5 and r.fsm.config['stable_seconds'] == 1.
                and r.fsm.config['handover_gyro_max'] == .75
                and env.cfg['closure_gap_termination_m'] == .003, 'Native source/cfg or safety thresholds changed')
            self.tick = self.rows = 0
            self.alive = True
            self.first_terminal = self.active = self.stream = None
            self.in_step = self.complete = self.closed = False
            self.error = None
            self.attempted = {k:0 for k in ('decode', 'delay', 'pd', 'fsm', 'cost')}
            self.completed = dict(self.attempted)

        def summary(self):
            result = super().summary()
            result.pop('source_code_modified', None)
            result.update(executed_source_repair_revision='022', observer_control_override=False,
                legacy_leaf_filename='pd_target_payment_2216.jsonl.gz',
                actual_lane_is_bound_by_directory_and_every_record=lane)
            return result
    return SelectedCapture()
