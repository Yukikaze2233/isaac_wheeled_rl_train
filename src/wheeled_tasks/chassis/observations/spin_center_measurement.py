"""Per-turn kinematic decomposition of a spinning root-link reference.

This is a measurement model, not a reward, a chassis COM estimate, or an
acceptance threshold. It fits each complete monotonic yaw turn independently:
    xy = center_start + center_delta * normalized_time
         + cosine_offset * cos(yaw - start_yaw)
         + sine_offset * sin(yaw - start_yaw).
The trend describes a fitted rotation center; the periodic component describes
reference-point eccentricity. Other motion appears in the residual and must
not be silently called slip. Endpoints use linear interpolation between policy
samples; this cannot recover motion above the sampling frequency.
"""
from copy import deepcopy
import math
import sys


VERSION = "spin-center-linear-harmonic-per-turn-v1"
TAU = 2 * math.pi


def _sample(position, yaw, seconds):
    if (not isinstance(position, (list, tuple)) or len(position) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   for x in (*position, yaw, seconds))):
        raise ValueError("Spin samples require finite root-link xyz, unwrapped yaw and time")
    return (float(seconds), float(yaw), float(position[0]), float(position[1]))


def _fit_turn(points, segment_id, direction):
    start, end = points[0], points[-1]
    duration = end[0] - start[0]
    result = dict(segment_id=segment_id, direction=direction, sample_count=len(points),
        start_seconds=start[0], end_seconds=end[0], duration_seconds=duration,
        start_unwrapped_yaw_rad=start[1], end_unwrapped_yaw_rad=end[1])
    if len(points) < 5 or duration <= 0:
        return dict(result, status="insufficient_samples")
    rows = [(1., (p[0] - start[0]) / duration,
             math.cos(p[1] - start[1]), math.sin(p[1] - start[1])) for p in points]
    # Reorthogonalized QR avoids squaring the condition number with normal
    # equations. The tolerance is a float64 rank check, not a capability limit.
    columns = [[row[j] for row in rows] for j in range(4)]
    q, r = [], [[0.] * 4 for _ in range(4)]
    norm = lambda x: math.sqrt(math.fsum(v * v for v in x))
    tolerance = 128 * sys.float_info.epsilon * max(norm(c) for c in columns)
    for j, column in enumerate(columns):
        v = list(column)
        for _ in range(2):
            for i, qi in enumerate(q):
                projection = math.fsum(x * y for x, y in zip(qi, v))
                r[i][j] += projection
                v = [x - projection * y for x, y in zip(v, qi)]
        r[j][j] = norm(v)
        if r[j][j] <= tolerance:
            return dict(result, status="rank_deficient", qr_rank_tolerance=tolerance)
        q.append([x / r[j][j] for x in v])
    coefficients = []
    for axis in (2, 3):
        target = [p[axis] for p in points]
        # Subtract the first sample before the solve to retain small-motion
        # precision when a stage uses large world-coordinate offsets.
        origin = target[0]
        rhs = [math.fsum(qi[k] * (target[k] - origin) for k in range(len(points))) for qi in q]
        answer = [0.] * 4
        for j in range(3, -1, -1):
            answer[j] = (rhs[j] - math.fsum(r[j][k] * answer[k] for k in range(j + 1, 4))) / r[j][j]
        answer[0] += origin
        coefficients.append(answer)
    cx, cy = coefficients
    residuals = [math.hypot(p[2] - math.fsum(a * b for a, b in zip(row, cx)),
                           p[3] - math.fsum(a * b for a, b in zip(row, cy)))
                 for p, row in zip(points, rows)]
    aa = cx[2] ** 2 + cx[3] ** 2
    bb = cy[2] ** 2 + cy[3] ** 2
    ab = cx[2] * cy[2] + cx[3] * cy[3]
    discriminant = math.hypot(aa - bb, 2 * ab)
    return dict(result, status="fitted", center_start_world_xy_m=[cx[0], cy[0]],
        center_end_world_xy_m=[cx[0] + cx[1], cy[0] + cy[1]],
        center_midpoint_world_xy_m=[cx[0] + .5 * cx[1], cy[0] + .5 * cy[1]],
        center_delta_world_xy_m=[cx[1], cy[1]],
        center_velocity_world_xy_m_s=[cx[1] / duration, cy[1] / duration],
        center_displacement_m=math.hypot(cx[1], cy[1]),
        periodic_cosine_world_xy_m=[cx[2], cy[2]], periodic_sine_world_xy_m=[cx[3], cy[3]],
        periodic_offset_rms_m=math.sqrt(.5 * (aa + bb)),
        periodic_offset_major_m=math.sqrt(max(0., .5 * (aa + bb + discriminant))),
        periodic_offset_minor_m=math.sqrt(max(0., .5 * (aa + bb - discriminant))),
        residual_planar_rms_m=math.sqrt(math.fsum(x * x for x in residuals) / len(residuals)),
        residual_planar_max_m=max(residuals),
        qr_diagonal_ratio=max(r[i][i] for i in range(4)) / min(r[i][i] for i in range(4)))


class SpinCenterMeasurements:
    """Complete observed turns only; no stitching over resets or invalid data."""

    def __init__(self):
        self.turns = []
        self.partial_segments = []
        self.segment_id = 0
        self._points = None
        self._direction = 0
        self._segment_turns = 0

    def begin_segment(self, position, unwrapped_yaw, seconds):
        point = _sample(position, unwrapped_yaw, seconds)
        if self._points is not None:
            raise ValueError("End the previous spin segment before beginning another")
        self.segment_id += 1
        self._points, self._direction, self._segment_turns = [point], 0, 0

    def observe(self, position, unwrapped_yaw, seconds):
        point = _sample(position, unwrapped_yaw, seconds)
        if self._points is None:
            raise ValueError("Spin sample requires an explicitly opened segment")
        previous = self._points[-1]
        if point[0] <= previous[0]:
            raise ValueError("Spin sample time must increase strictly")
        delta = point[1] - previous[1]
        direction = 1 if delta > 0 else -1 if delta < 0 else 0
        if direction and self._direction and direction != self._direction:
            self.end_segment("observed_yaw_reversal")
            self.begin_segment([previous[2], previous[3], 0.], previous[1], previous[0])
        if direction:
            self._direction = direction
        # Several revolutions in one policy interval are unresolvable. Do not
        # fabricate endpoint-only turns from an aliased or corrupted input.
        if abs(delta) >= math.pi:
            self.end_segment("yaw_interval_unresolved")
            self.begin_segment(position, unwrapped_yaw, seconds)
            return
        start = self._points[0]
        target = start[1] + self._direction * TAU
        if self._direction and self._direction * (point[1] - target) >= 0:
            alpha = (target - previous[1]) / delta
            endpoint = (previous[0] + alpha * (point[0] - previous[0]), target,
                        previous[2] + alpha * (point[2] - previous[2]),
                        previous[3] + alpha * (point[3] - previous[3]))
            if endpoint[0] > previous[0]:
                self._points.append(endpoint)
            self.turns.append(_fit_turn(self._points, self.segment_id, self._direction))
            self._segment_turns += 1
            self._points = [endpoint]
            if point[0] > endpoint[0]:
                self._points.append(point)
        else:
            self._points.append(point)

    def end_segment(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError("Spin segment end requires a named reason")
        if self._points is not None:
            start, end = self._points[0], self._points[-1]
            self.partial_segments.append(dict(segment_id=self.segment_id, end_reason=reason,
                complete_turns=self._segment_turns, remaining_sample_count=len(self._points),
                remaining_observed_seconds=end[0] - start[0],
                remaining_yaw_change_rad=end[1] - start[1]))
        self._points, self._direction, self._segment_turns = None, 0, 0

    def report(self):
        fitted = [turn for turn in self.turns if turn["status"] == "fitted"]
        # Do not measure center steps between different command/validity
        # segments, even if world coordinates happen to be close.
        steps = [math.hypot(*(b - a for a, b in zip(left["center_midpoint_world_xy_m"],
                                                  right["center_midpoint_world_xy_m"])))
                 for left, right in zip(fitted, fitted[1:]) if left["segment_id"] == right["segment_id"]]
        return dict(version=VERSION, assessment="measurement_only", center_reference="root_link_origin_world",
            model="linear_center_trend_plus_yaw_first_harmonic_per_complete_monotonic_turn",
            physical_COM_center_identified=False, capability_thresholds=None,
            endpoint_method="linear_interpolation_between_policy_samples",
            fit_residual_interpretation="unmodelled_motion_and_sampling_error_not_automatically_slip",
            completed_observed_turns=len(self.turns), fitted_turns=len(fitted),
            adjacent_fitted_center_midpoint_steps_m=steps,
            turns=deepcopy(self.turns), partial_segments=deepcopy(self.partial_segments))
