"""Isolated SIDE-rate override and read-only reference/baseline checks.

No controller, physical step, birth, RNG, terminal or learning operation lives
here. The original native FSM and 003 mechanical capture retain those owners.
"""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import math
import struct

KEY = 'side_speed_rad_s'
BEFORE, AFTER = 6., 1.5
LANES = (1708, 2216)
SIDE = 5
TRACE_STEPS = 2400


def require(value, message):
    if not bool(value): raise ValueError(message)


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class SideOverride:
    """Only a constructed FSM's owned config dictionary is changed."""
    def __init__(self, env, hooks, writer, output, authenticate):
        self.env, self.hooks, self.writer, self.output = env, hooks, writer, output
        self.fsm = env.recovery.fsm
        authenticate(self.fsm)
        require(type(self.fsm.config) is dict and self.fsm.config is not self.fsm.DEFAULTS,
            'Native FSM requires an instance-owned config')
        self.before = deepcopy(self.fsm.config)
        require(type(self.before.get(KEY)) in (int, float) and self.before[KEY] == BEFORE,
            'SIDE baseline value is not 6 rad/s')
        require(self.before['dt'] == .005 and self.before['side_angle_rad'] == math.pi/2
            and self.before['max_script_seconds'] == 8., 'Native SIDE clocks/angle/global limit differ')
        self.expected = self.before | {KEY: AFTER}
        self.calls = 0
        self.complete = self.closed = False
        self.applied = False

    def apply(self):
        require(not self.applied and not self.closed, 'SIDE override duplicate/closed')
        require(self.fsm.config == self.before, 'FSM config changed before override')
        self.fsm.config[KEY] = AFTER
        self.applied = True
        self.assert_config()
        prior = self.env.step
        def step(actions):
            self.assert_config()
            result = prior(actions)
            self.calls += 1
            self.assert_config()
            return result
        self.hooks.install(self.env, 'step', step)
        self.writer(self.output/'side_override_applied.json', self.receipt('applied_before_first_policy'))

    def assert_config(self):
        require(self.applied and not self.closed and self.fsm.config == self.expected,
            'Only admitted SIDE 6->1.5 config is permitted throughout observation')

    def receipt(self, status):
        return dict(schema='source021-isolated-SIDE-override-v1', status=status,
            before=self.before, effective=self.expected, sole_changed_keys=[KEY],
            before_config_sha256=digest_value(self.before), effective_config_sha256=digest_value(self.expected),
            side_outward_duration_s=self.before['side_angle_rad']/AFTER,
            side_native_timeout_s=2*self.before['side_angle_rad']/AFTER+.4,
            turnaround_hold_s=.08, global_script_limit_s=self.before['max_script_seconds'],
            native_formula_source_unchanged=True, policy_intervals_completed=self.calls,
            complete=self.complete, runtime_override_applied=self.applied,
            config_restored=self.closed and self.fsm.config == self.before,
            override_hooks_restored=self.hooks.closed, PPO_updates=0, formal_updates_consumed=0, qualification=None)

    def finish(self):
        self.assert_config()
        require(self.calls == 600, 'SIDE override did not observe the original600 intervals')
        self.complete = True
        return self.cleanup()

    def cleanup(self):
        if self.closed: return []
        errors = self.hooks.restore()
        if self.applied:
            if self.fsm.config == self.expected:
                self.fsm.config[KEY] = self.before[KEY]
            else:
                errors.append('Foreign FSM config mutation; no restoration over unknown owner')
        self.closed = True
        self.writer(self.output/'side_override_preclose.json', self.receipt(
            'complete_restored_preclose' if self.complete and not errors else 'incomplete_or_restore_rejected') | {'errors': errors})
        return errors


def expand_capture_module(helper):
    """Change only one private observer module instance's measurement horizon."""
    require(helper.TRACE_STEPS == 260 and helper.GUARD_M == .003 and tuple(helper.LANES) == LANES,
        'Frozen003 measurement/guard identity differs')
    helper.TRACE_STEPS = TRACE_STEPS
    parent = helper.MechanicalCapture
    class SideMechanicalCapture(parent):
        def control(self, target):
            result = super().control(target)
            fsm = self.env.recovery.fsm
            result['side_reference_audit'] = [helper.plain(dict(
                phase=int(fsm.phase[lane]), phase_ticks=int(fsm.phase_ticks[lane]),
                side_start_rad=fsm.side_start[lane], side_direction=fsm.side_direction[lane],
                axis_signs=fsm.axis_signs, native_config_side_speed_rad_s=fsm.config[KEY],
                source='native FSM state at the original effort setter; no update/reconstruction'))
                for lane in LANES]
            return result
    helper.MechanicalCapture = SideMechanicalCapture
    return dict(baseline_trace_cutoff_physics_steps=260, candidate_trace_cutoff_physics_steps=TRACE_STEPS,
        measurement_only=True, selected_first_birth_only=True, physical_parameters_changed=[KEY])


def rows(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream: yield json.loads(line)


def compare_bits(a, b, path=''):
    """Decoded native numeric values compared without tolerance, including -0."""
    if type(a) is not type(b): return path+': type differs'
    if isinstance(a, float):
        if not (math.isfinite(a) and math.isfinite(b)): return path+': nonfinite'
        return None if struct.pack('!d', a) == struct.pack('!d', b) else path+': float bits differ'
    if isinstance(a, dict):
        if a.keys() != b.keys(): return path+': keys differ'
        for key in a:
            mismatch = compare_bits(a[key], b[key], path+'/'+key)
            if mismatch: return mismatch
        return None
    if isinstance(a, list):
        if len(a) != len(b): return path+': length differs'
        for index, (x, y) in enumerate(zip(a, b)):
            mismatch = compare_bits(x, y, path+'/'+str(index))
            if mismatch: return mismatch
        return None
    return None if a == b else path+': value differs'


def comparable(row):
    value = deepcopy(row)
    if value.get('applied_control') is not None:
        value['applied_control'].pop('side_reference_audit', None)
    if value.get('contacts') is not None:
        # Counts sum over ALL lanes in a contact-view group; other SIDE lanes
        # may have diverged already. Selected contact patches/forces stay checked.
        value['contacts']['capacity_indicator'].pop('sum_reported_normal_patch_counts', None)
        value['contacts']['capacity_indicator'].pop('observed_count_at_capacity', None)
    return value


def pre_side_review(baseline_path, candidate_path):
    baseline = {lane: [] for lane in LANES}
    for row in rows(baseline_path): baseline[row['state']['lane']].append(row)
    candidate = {lane: {} for lane in LANES}
    for row in rows(candidate_path):
        lane = row['state']['lane']; step = row['physical_step']
        require(lane in candidate and step not in candidate[lane], 'Duplicate/unknown selected row')
        candidate[lane][step] = row
    verdicts = []
    for lane in LANES:
        prior = baseline[lane]
        entry = next((row['physical_step'] for row in prior if row['state']['phase'] == SIDE), None)
        require(entry is not None, 'Actual003 selected baseline never enters SIDE')
        selected = [row for row in prior if row['physical_step'] <= entry]
        require([row['physical_step'] for row in selected] == list(range(entry+1)), 'Actual003 preSIDE cadence incomplete')
        first = None
        for row in selected:
            step = row['physical_step']; actual = candidate[lane].get(step)
            mismatch = 'missing row' if actual is None else compare_bits(comparable(row), comparable(actual))
            if mismatch and first is None: first = dict(physical_step=step, reason=mismatch)
        verdicts.append(dict(lane=lane, baseline_first_SIDE_entry_physical_step=entry,
            compared_physics_steps_including_initial_and_entry=len(selected), bitwise_identical=first is None,
            first_mismatch=first))
    return dict(status='preSIDE_equal' if all(v['bitwise_identical'] for v in verdicts) else 'preSIDE_rejected',
        lanes=verdicts, exact_comparison_scope='all selected native state/control/contact fields through first SIDE entry, decoded float bits',
        declared_exclusions=['new side_reference_audit telemetry', 'contact capacity count/at-capacity indicator over all view lanes'],
        tolerance_used=False)


def coupled_reference_review(path):
    counts = {lane: 0 for lane in LANES}; violations = []; peak = common_peak = 0.
    for row in rows(path):
        control = row.get('applied_control')
        if control is None: continue
        lane = row['state']['lane']; audit = control['side_reference_audit']
        require(audit['native_config_side_speed_rad_s'] == AFTER, 'Recorded SIDE speed differs')
        if audit['phase'] != SIDE: continue
        counts[lane] += 1
        q = [control['script_targets_control6'][i] for i in (0, 1, 3, 4)]
        start, direction, signs = (audit[k] for k in ('side_start_rad', 'side_direction', 'axis_signs'))
        require(len(start) == len(direction) == len(signs) == 4
            and all(math.isfinite(x) for x in q+start+direction+signs)
            and all(abs(x) == 1 for x in signs) and signs[0] == signs[1] and signs[2] == signs[3]
            and all(abs(x) == 1 for x in direction)
            and direction[0] == direction[1] and direction[2] == direction[3]
            and direction[0] == -direction[2], 'Native paired signs/directions invalid')
        tolerance = max(1e-6, 8*2**-23*max(1., *(abs(x) for x in q+start)))
        errors = [abs(math.remainder((q[j+1]-q[j])-(start[j+1]-start[j]), math.tau)) for j in (0, 2)]
        stroke = [(q[j]-start[j])*signs[j]/direction[j] for j in range(4)]
        common_error = max(stroke)-min(stroke)
        peak = max(peak, *errors)
        common_peak = max(common_peak, common_error)
        if (max(errors) > tolerance or common_error > tolerance
                or min(stroke) < -tolerance or max(stroke) > math.pi/2+tolerance):
            violations.append(dict(lane=lane, physical_step=row['physical_step'], errors_rad=errors,
                common_stroke_rad=stroke, common_stroke_spread_rad=common_error, tolerance_rad=tolerance))
    return dict(status='paired_reference_preserved' if not violations and all(counts.values()) else 'paired_reference_rejected',
        SIDE_frames_by_lane=counts, violations=violations, maximum_periodic_pair_relative_error_rad=peak,
        maximum_signed_common_stroke_spread_rad=common_peak,
        tolerance='max(1e-6,8*float32-epsilon*max(1,abs(native ref/start))); float32 addition roundoff only',
        scope='actual script reference while setter phase is SIDE; converted exit interval is not inferred')
