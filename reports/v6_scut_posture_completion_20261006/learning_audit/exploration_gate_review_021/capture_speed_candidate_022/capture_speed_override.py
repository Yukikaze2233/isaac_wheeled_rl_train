"""One constructed native config override; original source producers stay owners."""
from copy import deepcopy
import gzip
import math

import torch

KEY = 'capture_speed_rad_s'
BEFORE, AFTER = 4., 1.


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def same_input_references(fsm, q, dq, gravity, gyro, height, support):
    """Derived CAPTURE reference algebra only, never another FSM/controller call.

    Mirrors frozen source022's current-tick validity/reroute, aligned goal,
    hip-selected pair winding and one common fraction. Returns comparisons on
    precisely those lanes whose cloned native phase consumes CAPTURE speed.
    It is not an old physical trajectory or a second recovery update.
    """
    valid = (torch.isfinite(q).all(-1) & torch.isfinite(dq).all(-1)
        & torch.isfinite(gravity).all(-1) & torch.isfinite(gyro).all(-1)
        & torch.isfinite(height) & (gravity.norm(dim=-1) > .5))
    g = torch.nan_to_num(gravity)
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    tilt = torch.acos((-g[:, 2]).clamp(-1., 1.))
    pitch = torch.atan2(g[:, 0], -g[:, 2])
    unsupported = fsm.enabled & valid & (fsm.phase == 6) & ~support & (tilt > fsm.config['reroute_tilt_rad'])
    reroute_ticks = math.ceil(fsm.config['reroute_support_lost_seconds']/fsm.dt - 1e-9)
    reroute = unsupported & (fsm.unsupported_ticks + 1 >= reroute_ticks) & (fsm.reroute_count < fsm.config['max_reroutes'])
    selected = fsm.enabled & valid & (fsm.phase == 6) & ~reroute
    old = fsm.script_targets[:, fsm.LEG_IDS].clone()
    goal = fsm.nominal.expand(fsm.count, -1).clone()
    aligned = selected & support & (tilt < math.radians(85.))
    extension = ((tilt-math.radians(12.))/math.radians(43.)).clamp(0.,1.)
    aligned_goal = fsm.nominal + extension[:, None]*(fsm.support-fsm.nominal) - pitch[:, None]/fsm.axis_signs
    goal[aligned] = aligned_goal[aligned]
    difference = (goal-old).reshape(-1,2,2)
    turns = (difference[..., :1]/math.tau).round()
    delta = (difference-turns*math.tau).reshape_as(old)
    elapsed = (fsm.phase_ticks+1)*fsm.dt
    refs = []
    for speed in (AFTER, BEFORE):
        # Native uses tensor/tensor division, not Python-scalar/tensor reciprocal
        # multiplication; keep its operation order to avoid a false bitwise veto.
        speed_tensor = elapsed.new_full(elapsed.shape, speed)
        fraction = (speed_tensor*fsm.dt/delta.abs().amax(-1).clamp_min(1e-6)).clamp_max(1.)
        refs.append(old + delta*fraction[:,None])
    return selected, refs[0], refs[1]


class CaptureSpeedOverride:
    """Only config[KEY] is changed; step/update wrappers call originals once."""
    def __init__(self, env, hooks, output, writer, plain, authenticate):
        self.env, self.fsm, self.hooks = env, env.recovery.fsm, hooks
        self.output, self.writer, self.plain = output, writer, plain
        authenticate(self.fsm)
        require(type(self.fsm.config) is dict and self.fsm.config is not self.fsm.DEFAULTS,
            'FSM config must be instance-owned')
        self.config_object = self.fsm.config
        self.before = deepcopy(self.fsm.config)
        require(self.before[KEY] == BEFORE and self.before['orbit_speed_rad_s'] == 4.
            and self.before['side_speed_rad_s'] == 1.5 and self.before['handover_gyro_max'] == .75
            and self.before['dt'] == .005 and self.before['stable_seconds'] == 1.
            and self.before['max_script_seconds'] == 8., 'Baseline rate/safety/clocks differ')
        self.effective = self.before | {KEY: AFTER}
        self.policy_attempted = self.policy_completed = self.reference_attempted = self.reference_completed = 0
        self.sensitive_lane_ticks = 0
        self.first = None
        self.complete = self.applied = self.closed = False
        self.stream = None

    def assert_config(self):
        require(self.applied and not self.closed and self.fsm.config is self.config_object
            and self.fsm.config == self.effective, 'Foreign config mutation or CAPTURE override absent')

    def apply(self):
        require(not self.applied and not self.closed and self.fsm.config == self.before, 'Duplicate/foreign override')
        self.stream = gzip.open(self.output/'capture_speed_sensitive_events.jsonl.gz', 'xt')
        self.fsm.config[KEY] = AFTER
        self.applied = True
        self.assert_config()
        prior_update = self.fsm.update
        def update(q, dq, gravity, gyro, height, support, clear):
            self.assert_config()
            self.reference_attempted += 1
            selected, prediction1, prediction4 = same_input_references(self.fsm,q,dq,gravity,gyro,height,support)
            oldphase = self.fsm.phase.clone()
            result = prior_update(q,dq,gravity,gyro,height,support,clear)
            self.reference_completed += 1
            self.assert_config()
            require(result is self.fsm.script_targets, 'Original FSM return object differs')
            legs = result[:, self.fsm.LEG_IDS]
            require(torch.equal(prediction1[selected],legs[selected]), 'Derived one-rate CAPTURE algebra differs from native source')
            change = selected & (prediction4 != legs).any(-1)
            ids = change.nonzero().flatten()
            self.sensitive_lane_ticks += len(ids)
            if len(ids):
                event = dict(reference=self.reference_completed, input_physical_step=self.reference_completed-1,
                    all_population_lanes=self.plain(ids), native_phase_before=self.plain(oldphase[ids]),
                    native_phase_after=self.plain(self.fsm.phase[ids]),
                    native_reference_leg_rad=self.plain(legs[ids]),
                    same_input_reference4_leg_rad_derived=self.plain(prediction4[ids]),
                    meaning='same-input returned reference differs; not an old physical counterfactual or another FSM call')
                self.stream.write(__import__('json').dumps(event,separators=(',',':'),allow_nan=False)+'\n')
                if self.first is None:
                    self.first = event
            return result
        self.hooks.install(self.fsm,'update',update)
        prior_step = self.env.step
        def step(actions):
            self.assert_config(); self.policy_attempted += 1
            result = prior_step(actions)
            self.policy_completed += 1
            self.assert_config()
            return result
        self.hooks.install(self.env,'step',step)
        self.writer(self.output/'capture_speed_override_applied.json',self.receipt('applied_before_first_policy'))

    def receipt(self, status):
        return dict(schema='source022-CAPTURE-speed-one-config-override-v1',status=status,
            before=self.before,effective=self.effective,sole_changed_keys=[KEY],
            policy_calls_attempted=self.policy_attempted,policy_calls_completed=self.policy_completed,
            reference_calls_attempted=self.reference_attempted,reference_calls_completed=self.reference_completed,
            first_global_sensitive_reference=self.first,
            strict_preintervention_last_physical_step=None if self.first is None else self.first['input_physical_step'],
            sensitive_all_population_lane_ticks=self.sensitive_lane_ticks,
            complete=self.complete,config_override_applied=self.applied,
            config_restored=self.closed and self.fsm.config is self.config_object and self.fsm.config==self.before,
            hooks_restored=self.hooks.closed,first_global_cutoff_not_selected_lane_cutoff=True,
            no_extra_policy_decode_delay_FSM_reward_calls=True,PPO_updates=0,formal_updates_consumed=0,qualification=None)

    def finish(self):
        self.assert_config()
        require(self.policy_attempted == self.policy_completed == 600
            and self.reference_attempted == self.reference_completed == 2400, 'Original600/2400 calls incomplete')
        require(self.first is not None, 'No actual config-sensitive CAPTURE reference; no rate counterfactual observed')
        self.complete = True
        return self.cleanup()

    def cleanup(self):
        if self.closed:
            return []
        errors = self.hooks.restore()
        if self.stream is not None:
            self.stream.flush(); self.stream.close()
        if self.applied:
            if self.fsm.config is self.config_object and self.fsm.config == self.effective:
                self.fsm.config[KEY] = self.before[KEY]
            else:
                errors.append('Foreign config owner/value; do not overwrite')
        self.closed = True
        self.writer(self.output/'capture_speed_override_preclose.json',self.receipt(
            'complete_restored_preclose' if self.complete and not errors else 'incomplete_or_restore_rejected') | {'errors':errors})
        return errors
