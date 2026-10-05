"""Named, bounded command protocols for the native inference observer only.

No environment, reset, motor, reward, random stream or PPO is constructed here.
The existing native trajectory supplies ordinary acceleration/release timing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

DEFAULT_SUITE = "low_speed_305mm"
HIGH_SPEED_SUITE = "high_speed_braking_v1"
SUITE_NAMES = (DEFAULT_SUITE, HIGH_SPEED_SUITE)
POLICY_DT, PHYSICS_DT = .02, .005


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    vx_m_s: float = 0.
    yaw_rad_s: float = 0.
    height_m: float = .305
    kind: str = "ordinary"


@dataclass(frozen=True)
class DiagnosticSuite:
    name: str
    horizon_s: float
    cases: tuple[DiagnosticCase, ...]
    replicas: int = 4

    @property
    def policy_steps(self):
        return round(self.horizon_s / POLICY_DT)

    @property
    def physics_steps(self):
        return self.policy_steps * 4

    @property
    def num_envs(self):
        return len(self.cases) * self.replicas

    @property
    def case_names(self):
        return tuple(case.name for case in self.cases)

    @property
    def profiles(self):
        # Compatibility with the original scalar-speed observer metadata.
        # Actual requests, including yaw and height, come from suite_schedule.
        return {case.name: case.vx_m_s for case in self.cases}

    @property
    def primary_windows(self):
        return {case.name: (("stand",) if case.vx_m_s == case.yaw_rad_s == 0.
                            else ("steady", "stop")) for case in self.cases}

    def case(self, name):
        for case in self.cases:
            if case.name == name:
                return case
        raise ValueError(f"Case {name!r} is not in frozen diagnostic suite {self.name}")

    def as_dict(self):
        return {"schema": "native-locomotion-diagnostic-suite-v1", "name": self.name,
                "horizon_s": self.horizon_s, "policy_steps": self.policy_steps,
                "physics_steps": self.physics_steps, "policy_dt_s": POLICY_DT,
                "physics_dt_s": PHYSICS_DT, "num_envs": self.num_envs, "replicas": self.replicas,
                "cases": [asdict(case) for case in self.cases],
                "primary_windows": {key: list(value) for key, value in self.primary_windows.items()},
                "birth": "native_nominal_305mm; no readiness waits or pose writes",
                "command_slew": {"vx_m_s2": 1.5, "yaw_rad_s2": 4.},
                "high_height_settle_s": 2., "boundary_semantics": "native unchanged; partial windows null",
                "scope": "flat native pipeline inference only; not full DR or qualification"}


_LOW = DiagnosticSuite(DEFAULT_SUITE, 20., (
    DiagnosticCase("stand_305mm"), DiagnosticCase("forward_0p5", .5),
    DiagnosticCase("backward_0p5", -.5)))
_HIGH = DiagnosticSuite(HIGH_SPEED_SUITE, 32., (
    DiagnosticCase("stand_305mm"), DiagnosticCase("stand_430mm", height_m=.43),
    *(DiagnosticCase(f"{direction}_{speed:g}_{height}mm", sign * speed, height_m=height / 1000)
      for height in (305, 430) for speed in (3., 5.)
      for direction, sign in (("forward", 1.), ("backward", -1.))),
    *(DiagnosticCase(f"yaw_{direction}_{rev}rps_{height}mm", yaw_rad_s=sign * rev * 2 * math.pi,
                     height_m=height / 1000)
      for height, rev in ((305, 1), (430, 3))
      for direction, sign in (("positive", 1.), ("negative", -1.))),
    DiagnosticCase("cancel_forward_5_305mm", 5., kind="cancel_and_rerequest"),
    DiagnosticCase("cancel_backward_5_305mm", -5., kind="cancel_and_rerequest")))


def get_suite(name=DEFAULT_SUITE):
    """Reject ad-hoc timing/population changes: only these two frozen catalogs."""
    if name == DEFAULT_SUITE:
        return _LOW
    if name == HIGH_SPEED_SUITE:
        return _HIGH
    raise ValueError(f"Unknown diagnostic suite {name!r}")


def suite_schedule(case_name, suite_name=DEFAULT_SUITE):
    """Return (phase, requested [vx, yaw, height]) for every policy tick.

    Zero is requested at the first braking tick. The caller always applies the
    native command slew; no generated intermediate target bypasses that helper.
    High posture is reached through actions after a fixed two-second transition.
    All phase names are contiguous so censored exposure cannot be hidden.
    """
    from probe_locomotion_quality import trajectory

    suite = get_suite(suite_name)
    case = suite.case(case_name)
    scalar = case.yaw_rad_s if case.yaw_rad_s else case.vx_m_s
    rate = 4. if case.yaw_rad_s else 1.5
    phases = trajectory(scalar, POLICY_DT, rate)
    plan = [("settle", (0., 0., .305))] * phases[0][1]
    if case.height_m != .305:
        plan.extend([("height_settle", (0., 0., case.height_m))] * 100)
    if case.kind == "cancel_and_rerequest":
        ramp = math.ceil(abs(case.vx_m_s) / (1.5 * POLICY_DT) - 1e-10)
        phases = [("cancelled_acceleration", 50, scalar), ("cancelled_braking", 50, 0.),
                  ("cancelled_wait", 100, 0.), ("acceleration", ramp, scalar),
                  ("steady", 50, scalar), ("interrupted_braking", 25, 0.),
                  ("renewed_drive", 25, scalar), ("braking", ramp, 0.),
                  ("stop_warmup", 100, 0.), ("stop", 500, 0.)]
    else:
        phases = phases[1:]
    for phase, count, target in phases:
        command = ((0., target, case.height_m) if case.yaw_rad_s
                   else (target, 0., case.height_m))
        plan.extend([(phase, command)] * count)
    if len(plan) > suite.policy_steps:
        raise ValueError("Frozen diagnostic trajectory exceeds its bounded horizon")
    plan.extend([("hold", (0., 0., case.height_m))] * (suite.policy_steps - len(plan)))
    return plan
