"""Explicit chassis motor priors, distinct from the legacy V4 policy ABI."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .preflight import control_decimation


SCHEMA = "chassis-explicit-motor-control-v1"
ACTIVE_ORDER = ["L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"]
COARSE_TIMING = {"physics_dt": .005, "policy_dt": .02, "pc_control_dt": .005, "decimation": 4}
COARSE_SCOPE = {"model": "200hz_feedback_pd_with_1ms_transport_impulse",
                "hardware_pd_dt": .001, "raw_bag_dt": .001,
                "fresh_feedback_per_policy_step": 4, "hardware_equivalence_verified": False}


def load_chassis_control(path):
    def reject(value):
        raise ValueError(f"Nonfinite motor-control value: {value}")
    control = json.loads(Path(path).read_text(), parse_constant=reject)
    if control.get("schema") != SCHEMA:
        from wheeled_tasks.v40.contract import load_contract
        return load_contract(path)
    return validate_chassis_control(control)


def validate_chassis_control(control):
    """Validate explicit output-side priors without claiming hardware identification."""
    capacity = control.get("engineering_probe_only") is True and not control.get("research_training", False)
    research = (control.get("engineering_probe_only") is False and control.get("research_training") is True
                and control.get("hardware_deployment_ready") is False)
    if control.get("schema") != SCHEMA or not (capacity or research) or control.get("training_release") is not False:
        raise ValueError("Explicit motor prior must declare capacity or simulation-research scope, without hardware release")
    if control.get("control_frame") != "Xforward_Yleft_Zup" or control.get("active_order") != ACTIVE_ORDER:
        raise ValueError("Explicit motor prior requires canonical six active-output axes")
    timing = control["timing"]
    coarse = control.get("simulation_timing_approximation") == COARSE_SCOPE
    expected = COARSE_TIMING if coarse else {"physics_dt": .001, "policy_dt": .02, "pc_control_dt": .001, "decimation": 20}
    if (control_decimation(timing["physics_dt"], timing["policy_dt"]) != expected["decimation"]
            or timing != expected or (coarse and not research)
            or ("simulation_timing_approximation" in control and not coarse)):
        raise ValueError("Motor prior requires authenticated 1 kHz or explicit 200 Hz research PD with 50 Hz targets")
    leg, wheel = control["actuators"]["leg"], control["actuators"]["wheel"]
    for key, expected in (("kp", 160.), ("kd", 2.5), ("effort_limit", 40.)):
        if leg.get(key) != expected:
            raise ValueError(f"Engineering leg prior {key} differs from selected PD")
    required = {"model", "gear_ratio", "gearbox_efficiency", "kd", "armature", "effort_limit",
                "curve_side", "curve_kind", "torque_reference", "scope", "source_note"}
    if set(wheel) != required:
        raise ValueError("Explicit wheel-output cap must not include an ambiguous motor-side curve or conversion")
    for key, expected in (("model", "M3508"), ("gear_ratio", 15.8), ("gearbox_efficiency", 1.),
                          ("kd", .6), ("effort_limit", 4.5), ("curve_side", "wheel_output"),
                          ("torque_reference", "wheel_output"),
                          ("curve_kind", "software_cap_only_not_identified"),
                          ("scope", "engineering_capacity_only" if capacity else "simulation_research_prior")):
        if wheel[key] != expected:
            raise ValueError(f"Engineering wheel prior {key} differs from the explicit output-side cap")
    for module in (leg, wheel):
        value = module["armature"]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Armature must be finite and nonnegative")
    return control
