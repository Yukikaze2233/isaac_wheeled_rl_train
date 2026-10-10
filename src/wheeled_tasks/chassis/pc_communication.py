"""Explicit authoring and validation of the no-loss PC communication prior.

The feedback bound covers sensor-to-PC delivery.  The command bound covers
USB arrival plus the existing periodic CAN cache.  Neither is a measured RTT,
and the held 20 ms policy target is not another communication delay.  These
helpers never patch a running configuration or authenticate a deployment.
"""
from copy import deepcopy
import json
import math
import re

from .feedback_transport_config import validate_feedback_transport


FEEDBACK_CONSUMERS = "policy_pc_pd_recovery"
FEEDBACK_VERSION = "complete_one_way_feedback_v1"
COMMAND_VERSION = "complete_one_way_command_v1"
RETIREMENT_SCHEMA = "pc-communication-case-retirement-v1"
DIAGNOSTIC_TRANSPORT_MODE = "training_domain_prior"
_BASE_CASES = ("stand_305mm", "forward_05", "backward_05", "rotate_1")
_FAULT_PROFILES = {
    "link_loss_005": dict(delay_ms=.08, can_phase_ms=.5, drop_probability=.005),
    "link_burst_3": dict(delay_ms=.08, can_phase_ms=.5,
                         burst_every_packets=2000, burst_packets=3),
    "link_combined": dict(delay_ms=.08, can_phase_ms=.08, jitter_ms=.02,
                          jitter_correlation=.9, drop_probability=.005,
                          burst_every_packets=5000, burst_packets=3),
}
LEGACY_FAULT_CASE_NAMES = tuple(f"{base}__{suffix}"
    for suffix in _FAULT_PROFILES for base in _BASE_CASES)
_FAULT_NAMES = frozenset(LEGACY_FAULT_CASE_NAMES)
_PROFILE_KEYS = frozenset(("delay_ms", "can_phase_ms", "jitter_ms",
    "jitter_correlation", "drop_probability", "burst_every_packets", "burst_packets"))


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} requires an explicit boolean")
    return value


def _finite(value, name):
    if type(value) not in (int, float):
        raise ValueError(f"{name} requires a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} requires a finite number")
    return value


def _finite_tree(value, name):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif type(value) in (int, float):
        _finite(value, name)


def _same_json(left, right):
    """Keep numeric types and complete surviving case definitions unchanged."""
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def feedback_transport_settings(enabled=True):
    """Canonical complete one-way return path, in seconds."""
    return dict(version=FEEDBACK_VERSION, enabled=_boolean(enabled, "feedback_transport.enabled"),
        units="seconds", scope="engineering_prior_complete_sensor_to_host_path_not_measured_rtt",
        ordered_delivery="monotonic_sensor_timestamp", reconstruction="linear_physics_history",
        delay_range_s=[.00004, .015], enabled_fraction=.75,
        jitter_s=.00002, jitter_correlation=.9)


def command_transport_settings(enabled=True):
    """Canonical USB-arrival prior plus CAN phase: at most 14 + 1 ms."""
    return dict(version=COMMAND_VERSION, enabled=_boolean(enabled, "command_transport.enabled"),
        delay_ms=[.04, 14.], can_period_ms=1., enabled_fraction=.75,
        drop_probability_max=0., burst_probability=0., burst_packets=[2, 3],
        scope="engineering_prior_complete_host_to_drive_path_not_measured_rtt;"
              "USB_arrival_plus_CAN_phase;substep_torque_impulse_approximation",
        delivery_model="phase_aware_hold_v1", can_phase_ms=[0., 1.],
        jitter_ms=.02, jitter_correlation=.9)


def _canonical(settings, factory, name):
    if not isinstance(settings, dict):
        raise ValueError(f"{name} requires complete canonical settings")
    expected = factory(_boolean(settings.get("enabled"), f"{name}.enabled"))
    if set(settings) != set(expected):
        raise ValueError(f"{name} contains missing or unsupported settings")
    _finite_tree(settings, name)
    for key, value in expected.items():
        actual = settings[key]
        if type(value) in (int, float):
            _finite(actual, f"{name}.{key}")
        elif isinstance(value, list) and all(type(item) in (int, float) for item in value):
            if not isinstance(actual, list) or len(actual) != len(value):
                raise ValueError(f"{name}.{key} differs from the canonical prior")
            for item, expected_item in zip(actual, value):
                _finite(item, f"{name}.{key}")
                if type(expected_item) is int and type(item) is not int:
                    raise ValueError(f"{name}.{key} requires integer packet counts")
        if actual != value:
            raise ValueError(f"{name}.{key} differs from the canonical prior")
    return deepcopy(settings)


def _profile(profile, name):
    """Fixed cases cannot override the no-loss contract with old fault stress."""
    if profile is None:
        return
    if not isinstance(profile, dict) or set(profile) - _PROFILE_KEYS or "delay_ms" not in profile:
        raise ValueError(f"{name} requires a supported fixed communication profile")
    for key, value in profile.items():
        _finite(value, f"{name}.{key}")
    for key in ("drop_probability", "burst_every_packets", "burst_packets"):
        if profile.get(key, 0.) != 0:
            raise ValueError(f"{name}.{key}: fixed profile packet loss and bursts are excluded")
    delay = profile["delay_ms"]
    phase = profile.get("can_phase_ms", .5)
    jitter = profile.get("jitter_ms", 0.)
    correlation = profile.get("jitter_correlation", .9)
    if (not .04 <= delay <= 14. or not 0. <= phase <= 1.
            or jitter < 0. or not 0. <= correlation <= 1.
            or delay + phase > 15.):
        raise ValueError(f"{name} exceeds the complete one-way command prior")


def _validate_cases(document, path="config"):
    if isinstance(document, dict):
        if "communication_profile" in document:
            _profile(document["communication_profile"], f"{path}.communication_profile")
        if isinstance(document.get("name"), str) and document["name"] in _FAULT_NAMES:
            raise ValueError(f"{path}: retired legacy fault case cannot re-enter the new roles")
        for key, value in document.items():
            if key.endswith("case_names") and isinstance(value, list):
                if any(name in _FAULT_NAMES for name in value if isinstance(name, str)):
                    raise ValueError(f"{path}.{key}: retired legacy fault case cannot re-enter the new roles")
            _validate_cases(value, f"{path}.{key}")
    elif isinstance(document, list):
        for index, value in enumerate(document):
            _validate_cases(value, f"{path}[{index}]")


def _case_catalogue(config):
    evaluation = config.get("evaluation")
    cases = evaluation.get("cases") if isinstance(evaluation, dict) else None
    if (not isinstance(cases, list) or not cases or any(not isinstance(case, dict)
            or not isinstance(case.get("name"), str) or not case["name"] for case in cases)
            or len({case["name"] for case in cases}) != len(cases)):
        raise ValueError("Transport profiles require unique explicit evaluation cases")
    return {case["name"]: case for case in cases}


def _diagnostic_transport_mode(config):
    """An authored unscored probe may deliberately sample the training prior."""
    if "evaluation_transport_mode" not in config:
        return False
    if config["evaluation_transport_mode"] != DIAGNOSTIC_TRANSPORT_MODE:
        raise ValueError("Unsupported explicit evaluation_transport_mode")
    if (config.get("run_scope") != "interactive_unscored"
            or config.get("evaluation_exact_cases") is not True):
        raise ValueError("Diagnostic transport mode requires an exact interactive_unscored probe")
    if "feedback_transport" not in config:
        raise ValueError("Diagnostic transport mode requires explicit PC feedback settings")
    for name in ("qualification", "domain_qualified", "qualified"):
        if config.get(name) is not None:
            raise ValueError("Diagnostic transport mode cannot grant qualification or acceptance credit")
    for name in ("role", "run_role"):
        if name in config and config[name] != "playback":
            raise ValueError("Diagnostic transport mode cannot be used for training or scored evaluation roles")
    cases = _case_catalogue(config)
    if any(case.get("communication_profile") is not None for case in cases.values()):
        raise ValueError("Diagnostic transport mode conflicts with a fixed communication_profile")
    if any("transport_enabled" in case for case in cases.values()):
        raise ValueError("Diagnostic transport mode forbids explicit transport_enabled overrides")
    return True


def _group_cases(config, scene_groups):
    cases = _case_catalogue(config)
    if (not isinstance(scene_groups, (list, tuple))
            or any(not isinstance(name, str) or name not in cases for name in scene_groups)):
        raise ValueError("Transport profile selection requires declared scene-group case names")
    return [cases[name] for name in scene_groups]


def _validate_case_activation(config, feedback):
    exact = _boolean(config.get("evaluation_exact_cases", False), "evaluation_exact_cases")
    evaluation = config.get("evaluation", {})
    if exact:
        cases = list(_case_catalogue(config).values())
    else:
        cases = evaluation.get("cases", [])
        if not isinstance(cases, list) or any(not isinstance(case, dict) for case in cases):
            raise ValueError("Transport activation requires explicit evaluation case mappings")
    for case in cases:
        if "transport_enabled" not in case:
            continue
        enabled = _boolean(case["transport_enabled"], "case.transport_enabled")
        expected = feedback["enabled"] and case.get("communication_profile") is not None
        if exact and enabled is not expected:
            raise ValueError("Fixed case transport_enabled differs from paired native profile activation")


def command_profiles(config, scene_groups):
    """Select native downlink profiles without changing fixed-case defaults.

    Legacy exact cases use their declared per-case profile or ``None`` for a
    clean lane. Only the explicit unscored diagnostic mode returns ``None``
    for the whole suite, allowing the native command class to sample its
    canonical prior. This never edits a case or materializes runtime state.
    """
    if not isinstance(config, dict):
        raise ValueError("Transport profile selection requires a configuration mapping")
    diagnostic = "evaluation_transport_mode" in config
    if diagnostic:
        validate_pc_communication(config)
    if not config.get("evaluation_exact_cases"):
        return None
    cases = _group_cases(config, scene_groups)
    if diagnostic:
        return None
    return [deepcopy(case.get("communication_profile")) for case in cases]


def feedback_profile_enablement(config, scene_groups):
    """Pair returned-feedback activation with the existing exact-case downlink.

    Absent feedback keeps legacy behavior. Ordinary new fixed roles have
    clean feedback where the command profile is absent and enabled feedback
    where it is present. The explicit unscored diagnostic instead keeps the
    shared domain draw; globally disabled exact roles return all false.
    """
    if not isinstance(config, dict):
        raise ValueError("Feedback activation requires a configuration mapping")
    if "feedback_transport" not in config:
        return None
    validate_pc_communication(config)
    if not config.get("evaluation_exact_cases"):
        return None
    cases = _group_cases(config, scene_groups)
    if not config["feedback_transport"]["enabled"]:
        return [False] * len(cases)
    if "evaluation_transport_mode" in config:
        return None
    return [case.get("communication_profile") is not None for case in cases]


def validate_pc_communication(config):
    """Validate only opted-in new configs; legacy absent-field configs are no-ops.

    All enabled/disabled settings are explicit.  Validation never changes the
    supplied object, samples randomness or grants source-migration admission.
    """
    if not isinstance(config, dict):
        raise ValueError("PC communication requires a configuration mapping")
    _diagnostic_transport_mode(config)
    if "feedback_transport" not in config:
        return None
    for name, expected in (("physics_dt", .005), ("policy_dt", .02)):
        if _finite(config.get(name), name) != expected:
            raise ValueError("PC communication retains 200 Hz physics and the 50 Hz policy clock")
    feedback = _canonical(config["feedback_transport"], feedback_transport_settings, "feedback_transport")
    validate_feedback_transport(feedback, config["physics_dt"])
    command = _canonical(config.get("command_transport"), command_transport_settings, "command_transport")
    if command["enabled"] is not feedback["enabled"]:
        raise ValueError("PC communication requires paired feedback and command transport enablement")
    for name in ("signal_delay", "usb_transport"):
        settings = config.get(name)
        if not isinstance(settings, dict) or settings.get("enabled") is not False:
            raise ValueError(f"{name} must be explicitly disabled; communication delay cannot stack")
    if config["signal_delay"].get("version") != "physical_signal_delay_v1":
        raise ValueError("Retain the explicit disabled physical_signal_delay_v1 object")
    perturbations = config.get("signal_perturbations")
    if (not isinstance(perturbations, dict) or type(perturbations.get("max_delay_steps")) is not int
            or perturbations["max_delay_steps"] != 0):
        raise ValueError("signal_perturbations.max_delay_steps must be zero; no policy-clock delay stack")
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("PC communication requires an evaluation mapping")
    frozen = evaluation.get("frozen_module_enablement")
    if frozen is not None:
        if not isinstance(frozen, dict):
            raise ValueError("Evaluation module enablement must be explicit")
        for name, expected in (("feedback_transport", feedback["enabled"]),
                ("command_transport", command["enabled"]), ("signal_delay", False),
                ("usb_transport", False)):
            if name in frozen and (type(frozen[name]) is not bool or frozen[name] is not expected):
                raise ValueError(f"Evaluation override would change {name} or stack a legacy delay")
    _validate_case_activation(config, feedback)
    _validate_cases(config)
    return dict(feedback_transport=feedback, command_transport=command,
        feedback_consumers=FEEDBACK_CONSUMERS,
        complete_one_way_feedback_range_s=[.00004, .015],
        complete_one_way_command_range_s=[.00004, .015],
        synthetic_rtt_range_s=[.00008, .030],
        rtt_provenance="sum_of_two_engineering_priors_not_measured_rtt")


def runtime_expectation(config):
    """Declare complete independent runtime values before instance creation."""
    if validate_pc_communication(config) is None:
        return {}
    settings = deepcopy(config["feedback_transport"])
    return dict(feedback_transport=settings,
        feedback_transport_model=settings["version"] if settings["enabled"] else None,
        feedback_consumers=FEEDBACK_CONSUMERS)


def apply_pc_communication(config, *, enabled=True):
    """Author a new copy explicitly; the old contract and runtime stay intact.

    Retire legacy fault cases separately with their predecessor provenance
    before calling this helper.  A clean counterfactual disables both paths
    while retaining the exact versioned prior definitions.
    """
    enabled = _boolean(enabled, "PC communication enabled")
    result = deepcopy(config)
    if not isinstance(result, dict):
        raise ValueError("PC communication authoring requires a configuration mapping")
    for name in ("signal_delay", "usb_transport", "signal_perturbations"):
        if not isinstance(result.get(name), dict):
            raise ValueError(f"PC communication authoring requires the explicit legacy {name} object")
    result["feedback_transport"] = feedback_transport_settings(enabled)
    result["command_transport"] = command_transport_settings(enabled)
    result["signal_delay"]["enabled"] = False
    result["usb_transport"]["enabled"] = False
    result["signal_perturbations"]["max_delay_steps"] = 0
    evaluation = result.get("evaluation", {})
    if not isinstance(evaluation, dict):
        raise ValueError("PC communication authoring requires an evaluation mapping")
    frozen = evaluation.get("frozen_module_enablement")
    if frozen is not None:
        if not isinstance(frozen, dict):
            raise ValueError("Evaluation module enablement must be explicit")
        frozen.update(feedback_transport=enabled, command_transport=enabled, signal_delay=False)
        if "usb_transport" in frozen:
            frozen["usb_transport"] = False
    validate_pc_communication(result)
    return result


def retire_legacy_fault_cases(document, predecessor_file, predecessor_sha256, *, require_full_catalogue=True):
    """Retire the twelve named legacy cases and nested case-name references.

    The caller authenticates the predecessor bytes.  This pure function retains
    its file/SHA pin in a separate provenance receipt.  Role or plan documents
    may explicitly set ``require_full_catalogue=False`` after their canonical
    322-case source has been authenticated and retired.
    """
    if (not isinstance(document, dict) or not isinstance(predecessor_file, str) or not predecessor_file
            or not isinstance(predecessor_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", predecessor_sha256) is None
            or type(require_full_catalogue) is not bool):
        raise ValueError("Case retirement requires an explicit predecessor file and full lowercase SHA-256")
    locations = {}
    full_count = 0

    def project(value, path):
        nonlocal full_count
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                child = f"{path}.{key}" if path else key
                if key == "cases":
                    if (not isinstance(item, list) or any(not isinstance(row, dict)
                            or not isinstance(row.get("name"), str) for row in item)
                            or len({row["name"] for row in item}) != len(item)):
                        raise ValueError(f"{child} requires unique explicit case definitions")
                    names = {row["name"] for row in item}
                    if len(item) == 322:
                        if not _FAULT_NAMES <= names:
                            raise ValueError("The 322-case predecessor must contain exactly the twelve named legacy faults")
                        full_count += 1
                    for row in item:
                        if row["name"] in _FAULT_NAMES:
                            suffix = row["name"].rsplit("__", 1)[1]
                            if not _same_json(row.get("communication_profile"), _FAULT_PROFILES[suffix]):
                                raise ValueError("Named legacy fault case differs from its predecessor profile")
                    retained = [deepcopy(row) for row in item if row["name"] not in _FAULT_NAMES]
                    locations[child] = dict(predecessor_count=len(item), successor_count=len(retained),
                        retired_names=[row["name"] for row in item if row["name"] in _FAULT_NAMES])
                    result[key] = retained
                elif key.endswith("case_names"):
                    if not isinstance(item, list) or any(not isinstance(name, str) for name in item):
                        raise ValueError(f"{child} requires an explicit case-name list")
                    result[key] = [name for name in item if name not in _FAULT_NAMES]
                else:
                    result[key] = project(item, child)
            return result
        if isinstance(value, list):
            return [project(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return deepcopy(value)

    result = project(document, "")
    if require_full_catalogue and not full_count:
        raise ValueError("Retirement requires the authenticated full 322-case predecessor catalogue")
    _validate_cases(result)
    provenance = dict(schema=RETIREMENT_SCHEMA,
        predecessor=dict(file=predecessor_file, sha256=predecessor_sha256),
        retired_names=list(LEGACY_FAULT_CASE_NAMES),
        predecessor_full_case_count=322, successor_full_case_count=310,
        surviving_case_definitions="unchanged", locations=locations)
    return result, provenance


def validate_no_loss_projection(config, predecessor_evaluation=None):
    """Reject fault re-entry and optionally authenticate an exact role projection.

    Pass the matching predecessor role's evaluation, rather than a different
    subset: its surviving cases must retain their order and complete values.
    """
    if validate_pc_communication(config) is None:
        raise ValueError("No-loss projection requires explicit PC communication settings")
    if config.get("evaluation_exact_cases") is not True:
        raise ValueError("No-loss projection requires an explicit fixed evaluation role")
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("cases"), list):
        raise ValueError("No-loss projection requires explicit fixed cases")
    if predecessor_evaluation is not None:
        if not isinstance(predecessor_evaluation, dict) or not isinstance(predecessor_evaluation.get("cases"), list):
            raise ValueError("No-loss projection requires the matching predecessor evaluation")
        expected = [case for case in predecessor_evaluation["cases"] if case.get("name") not in _FAULT_NAMES]
        if not _same_json(evaluation["cases"], expected):
            raise ValueError("No-loss projection changed surviving predecessor case definitions or coverage")
    return dict(status="no_loss_projection_passed", qualification=None,
        surviving_case_count=len(evaluation["cases"]), retired_case_names=list(LEGACY_FAULT_CASE_NAMES))
