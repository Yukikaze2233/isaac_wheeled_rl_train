"""One first-birth observer of original decode/delay/PD/FSM/payment values.

No policy evaluation, target write, predicate reconstruction or reward producer
is added. Each wrapped original is called once and its returned object is kept.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import traceback

import torch

LANE = 2216
GATES = ('valid', 'support', 'body_clear', 'tilt', 'gyro', 'height', 'wheel_speed', 'leg_speed')
TERMS = ('leg_velocity', 'wheel_velocity', 'horizontal_angular_velocity')


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def pick(value):
    """Immutable selected-lane copy before another original call can overwrite it."""
    require(isinstance(value, torch.Tensor) and value.shape[0] == 4096,
            'Selected native tensor population differs')
    return value[LANE].detach().clone().cpu().tolist()


class TargetPaymentCapture:
    def __init__(self, env, parent, reward_module, hooks, output, writer):
        self.env, self.parent, self.module = env, parent, reward_module
        self.hooks, self.output, self.writer = hooks, output, writer
        require(env.num_envs == 4096 and env.dt == .005 and env.policy_dt == .02
                and env.decimation == 4, 'Original4096/600/2400 clock differs')
        require(env.usb_transport is None and env.signal_delay is not None
                and env.signal_delay.cfg['feedback_pd'] == 'fresh_physics_feedback',
                'Admitted fresh PD feedback/signal-delay context differs')
        r = env.recovery
        require(bool(r.enabled[LANE]) and int(r.pose_curriculum.level_ids[LANE]) == 2
                and int(r.pose_curriculum.profile_ids[LANE]) == 21
                and r.pose_curriculum.profile_names[21] == 'double_step_first_riser_split',
                'Actual2216 first-birth level/profile differs')
        require(r.fsm.config['side_speed_rad_s'] == 1.5
                and r.fsm.config['stable_seconds'] == 1.
                and r.fsm.config['handover_gyro_max'] == .75
                and env.cfg['closure_gap_termination_m'] == .003,
                'SIDE1.5/original stability/3mm context differs')
        self.tick = self.rows = 0
        self.alive = True
        self.first_terminal = None
        self.active = None
        self.in_step = self.complete = self.closed = False
        self.error = None
        self.attempted = {k: 0 for k in ('decode', 'delay', 'pd', 'fsm', 'cost')}
        self.completed = dict(self.attempted)
        self.stream = None

    def enter_call(self, name):
        require(self.in_step and self.active is not None and not self.closed,
                'Original '+name+' outside admitted policy interval')
        expected = 1 if name in ('decode', 'cost') else 4
        self.active['counts'][name] += 1
        require(self.active['counts'][name] <= expected, 'Duplicate original '+name+' call')
        self.attempted[name] += 1

    def end_call(self, name):
        self.completed[name] += 1

    def selected(self):
        return self.active is not None and self.active['first_birth_live']

    def substep(self):
        k = self.active['counts']['pd']
        require(1 <= k <= 4, 'Native PD substep index differs')
        return self.active['substeps'][k-1]

    def install(self):
        require(self.stream is None and not self.closed, 'Observer duplicate/closed')
        self.stream = gzip.open(self.output/'pd_target_payment_2216.jsonl.gz', 'xt', encoding='utf-8')
        e, r = self.env, self.env.recovery
        prior_step, prior_decode = e.step, e.v5.decode
        prior_delay, prior_pd, prior_fsm = e.signal_delay.targets, e.v5.motor_efforts, r.fsm.update
        prior_cost = self.module.apply_recovery_hold_costs

        def decode(actions, q):
            self.enter_call('decode')
            require(self.active['counts']['delay'] == 0, 'Decode occurred after original delay')
            payment = self.parent.payment.active
            require(payment is not None and payment['tick'] == self.tick,
                    'Original begin-policy payment snapshot missing')
            if self.selected():
                self.active['old_episode'] = dict(enabled=pick(payment['enabled']),
                    lesson_id=pick(payment['lesson']), profile_id=pick(payment['profile']),
                    hold_at_policy_start=pick(payment['hold']), first_birth_live=pick(payment['live']))
                inputs = dict(applied_action_CONTROL_order=pick(actions),
                              continuous_encoder_input_rad=pick(q))
            result = prior_decode(actions, q)
            self.end_call('decode')
            require(isinstance(result, tuple) and len(result) == 3, 'Original decode ABI differs')
            if self.selected():
                self.active['decode'] = inputs | dict(leg_targets_rad=pick(result[0]),
                    wheel_targets_rad_s=pick(result[1]), clipped_action_CONTROL_order=pick(result[2]),
                    policy_dt_s=.02, input_physical_step=4*(self.tick-1),
                    target_scope='original once-decoded returns; distinct from delayed PD targets')
            return result

        def delay(legs, wheels):
            self.enter_call('delay')
            k = self.active['counts']['delay']
            require(self.active['counts']['decode'] == 1 and self.active['counts']['pd'] == k-1,
                    'Original delay/PD call order differs')
            if self.selected():
                inputs = dict(decoded_legs_rad=pick(legs), decoded_wheels_rad_s=pick(wheels),
                              target_lags_physics_steps=pick(e.signal_delay.lags[:, 4:]))
            result = prior_delay(legs, wheels)
            self.end_call('delay')
            require(isinstance(result, tuple) and len(result) == 2, 'Original delay ABI differs')
            if self.selected():
                self.active['substeps'][k-1]['delay'] = inputs | dict(
                    delayed_legs_rad=pick(result[0]), delayed_wheels_rad_s=pick(result[1]))
            return result

        def pd(q, dq, legs, wheels):
            self.enter_call('pd')
            k = self.active['counts']['pd']
            require(self.active['counts']['delay'] == k and self.active['counts']['fsm'] == k-1,
                    'Original PD/reference call order differs')
            if self.selected():
                inputs = dict(continuous_control_q_rad=pick(q), fresh_motor_dq_rad_s=pick(dq),
                    actual_PD_leg_targets_rad=pick(legs), actual_PD_wheel_targets_rad_s=pick(wheels))
            result = prior_pd(q, dq, legs, wheels)
            self.end_call('pd')
            if self.selected():
                self.substep()['pd'] = inputs | dict(
                    original_RL_PD_preclip_request_nm=pick(e.v5.requested_motor_effort),
                    original_RL_PD_clipped_torque_nm=pick(result),
                    native_motor_bounds_nm=pick(e.v5.current_motor_bounds),
                    scope='original V5 motor_efforts return before recovery mixes/overwrites request; not final shaft effort')
            return result

        def fsm(q, dq, gravity, gyro, height, support, body_clear):
            self.enter_call('fsm')
            k = self.active['counts']['fsm']
            require(self.active['counts']['pd'] == k, 'Reference did not follow original PD')
            if self.selected():
                inputs = dict(continuous_q_rad=pick(q), motor_dq_rad_s=pick(dq),
                    gravity=pick(gravity), gyro_rad_s=pick(gyro),
                    conditional_height_m=pick(height), support_proxy=pick(support),
                    body_clear_proxy=pick(body_clear), phase_before=pick(r.fsm.phase))
            result = prior_fsm(q, dq, gravity, gyro, height, support, body_clear)
            self.end_call('fsm')
            if self.selected():
                self.substep()['fsm'] = inputs | dict(native_gate_masks={name: pick(r.fsm.stable_pass_masks[name]) for name in GATES},
                    native_stable_pass=pick(r.fsm.stable_pass), native_holding_rl=pick(r.fsm.holding_rl),
                    native_ready=pick(r.fsm.ready), stable_ticks=pick(r.fsm.stable_ticks),
                    stable_peak_ticks=pick(r.fsm.stable_peak_ticks), stable_break_count=pick(r.fsm.stable_break_count),
                    motion_released=pick(r.fsm.motion_released), phase_after=pick(r.fsm.phase),
                    release_finished=pick(r.release_finished),
                    source='existing native fields after original update chain once; no gate reconstruction')
            return result

        def cost(components, dq, omega, enabled, owned, held, settings, *, diagnostics=None):
            self.enter_call('cost')
            require(all(self.active['counts'][name] == 4 for name in ('delay', 'pd', 'fsm')),
                    'Hold payment producer occurred before physical interval end')
            require(settings is e.recovery_hold_cost_settings
                    and diagnostics is e.recovery_hold_cost_diagnostics,
                    'Original hold settings/diagnostics identity differs')
            if self.selected():
                inputs = dict(dq_end_rad_s=pick(dq), gyro_end_rad_s=pick(omega),
                    recovery_enabled=pick(enabled), whole_interval_actor_owned=pick(owned),
                    hold_at_policy_start=pick(held))
            result = prior_cost(components, dq, omega, enabled, owned, held, settings, diagnostics=diagnostics)
            self.end_call('cost')
            require(result is components, 'Original hold dictionary identity differs')
            if self.selected():
                self.active['hold_producer'] = inputs | dict(native_hold_cost_mask=pick(diagnostics['mask']),
                    native_diagnostics={name: pick(value) for name, value in diagnostics.items()},
                    actual_component_densities={name: pick(result[name]) for name in TERMS},
                    physical_end_step=4*self.tick, last_gate_input_physical_step=4*self.tick-1,
                    density_dt_s=.02, scope='native end-state producer outputs; not a 5ms payment')
            return result

        def step(actions):
            require(not self.in_step and self.active is None and not self.closed, 'Nested/unclosed native policy interval')
            self.tick += 1
            require(self.tick <= 600 and bool(self.parent.live[LANE]) == self.alive,
                    'Selected first birth was omitted/reopened or policy budget differs')
            self.in_step = True
            self.active = dict(policy_tick=self.tick, lane=LANE, birth_id=1,
                first_birth_live=self.alive, counts={name: 0 for name in self.completed},
                substeps=[dict(substep=k, reference_and_PD_before_physical_step=4*(self.tick-1)+k,
                    input_physical_step=4*(self.tick-1)+k-1, postphysics_step=4*(self.tick-1)+k,
                    reference_dt_s=.005) for k in range(1, 5)])
            try:
                result = prior_step(actions)
                require(isinstance(result, tuple) and len(result) == 4, 'Original step ABI differs')
                require(self.active['counts'] == dict(decode=1, delay=4, pd=4, fsm=4, cost=1),
                        'Native original-call cadence differs')
                if self.selected():
                    extras, d = result[3], result[3]['diagnostics']
                    require(torch.equal(result[1], d['reward']) and torch.equal(result[2], d['done']),
                            'Original returns differ from pre-reset diagnostics')
                    producer = self.active['hold_producer']
                    actual = {name: pick(extras['native_reward_components'][name]) for name in TERMS}
                    require(actual == producer['actual_component_densities'],
                            'Hold component changed after original producer')
                    require(pick(extras['policy_learning_mask']) == producer['whole_interval_actor_owned'],
                            'Terminal whole-interval payment ownership differs')
                    self.active['policy_end'] = dict(physical_end_step=4*self.tick,
                        whole_interval_actor_valid=pick(extras['policy_learning_mask']),
                        returned_reward=pick(result[1]), done=pick(result[2]),
                        terminated=pick(d['terminated']), timeout=pick(extras['time_outs']),
                        success=pick(d['success']), commands=pick(d['commands']),
                        native_motion_released=pick(d['recovery_motion_released']),
                        actual_component_densities=actual,
                        actual_component_payments={name: value*.02 for name, value in actual.items()},
                        density_dt_s=.02,
                        old_episode_attribution=self.active['old_episode'],
                        source='original returned pre-reset diagnostics/components; full reward/event reconciliation is inherited')
                    self.stream.write(json.dumps(self.active, allow_nan=False)+'\n')
                    self.rows += 1
                    if self.active['policy_end']['done']:
                        self.first_terminal = self.active['policy_end'] | {'policy_tick': self.tick}
                        self.alive = False
                self.active = None
                return result
            except BaseException:
                self.error = traceback.format_exc()
                raise
            finally:
                self.in_step = False

        try:
            for owner, name, callback in ((e.v5, 'decode', decode), (e.signal_delay, 'targets', delay),
                    (e.v5, 'motor_efforts', pd), (r.fsm, 'update', fsm),
                    (self.module, 'apply_recovery_hold_costs', cost), (e, 'step', step)):
                self.hooks.install(owner, name, callback)
        except BaseException:
            self.cleanup()
            raise

    def summary(self):
        trace_path = self.output/'pd_target_payment_2216.jsonl.gz'
        trace_sha = None
        if self.closed and trace_path.exists():
            with trace_path.open('rb') as stream:
                trace_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(schema='source021-native2216-target-payment-summary-v1', complete=self.complete,
            policy_steps=self.tick, expected_physical_steps=4*self.tick,
            original_calls_attempted=self.attempted, original_calls_completed=self.completed,
            first_birth_policy_rows=self.rows, first_terminal=self.first_terminal,
            right_censored_unfinished=self.alive, hooks_restored=self.closed and self.hooks.closed,
            unfinished_interval=self.active, error=self.error,
            selected_lane=LANE, selected_birth_id=1, source_code_modified=False,
            trace_artifact_sha256=trace_sha,
            extra_policy_decode_delay_FSM_reward_calls=0, PPO_updates=0, formal_updates_consumed=0,
            qualification=None)

    def finish(self):
        require(not self.in_step and self.active is None and self.tick == 600,
                'Original600 policy interval observation incomplete')
        expected = dict(decode=600, delay=2400, pd=2400, fsm=2400, cost=600)
        require(self.attempted == self.completed == expected, 'Original600/2400 producer cadence incomplete')
        require(self.rows == (self.first_terminal['policy_tick'] if self.first_terminal else 600),
                'Selected first-cohort policy rows incomplete')
        self.complete = True
        require(not self.cleanup(), 'Target/payment callbacks could not restore')
        return self.summary()

    def cleanup(self):
        if self.closed:
            return []
        errors = self.hooks.restore()
        self.closed = True
        if self.stream is not None:
            self.stream.close()
        if not self.complete or errors:
            self.writer(self.output/'pd_target_payment_failure.json',
                dict(status='incomplete_or_restore_rejected_preclose', errors=errors,
                     observation=self.summary(), app_close_returned=None, qualification=None))
        return errors


def make_observer(parent, reward_module, hooks_class, writer, sha):
    class WithTargetPayments(parent):
        def __init__(self, env, output):
            self.targets_capture = None
            super().__init__(env, output)
            try:
                module = reward_module(env, self) if callable(reward_module) else reward_module
                self.targets_capture = TargetPaymentCapture(env, self, module, hooks_class(), output, writer)
                self.targets_capture.install()
            except BaseException:
                self._before_close()
                raise

        def _before_close(self):
            try:
                if self.targets_capture is not None:
                    require(not self.targets_capture.cleanup(), 'Target/payment preclose restore rejected')
            finally:
                super()._before_close()

        def finish(self, output):
            try:
                summary = self.targets_capture.finish()
                writer(output/'pd_target_payment_summary.json', summary)
                result = super().finish(output)
            except BaseException:
                writer(output/'pd_target_payment_failure.json', dict(status='target_payment_or_parent_finish_failed_preclose',
                    error=traceback.format_exc(), app_close_returned=None, qualification=None, PPO_updates=0))
                self._before_close()
                raise
            result.update(pd_target_payment_capture_complete=True,
                pd_target_payment_summary_sha256=sha(output/'pd_target_payment_summary.json'))
            return result
    return WithTargetPayments
