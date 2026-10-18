"""Opt-in startup expectations for the local component-routed candidate.

The independent design retains the native physics/module baseline and adds one
explicit scoped declaration. No expectation is inferred from observed instances.
This startup-only check neither qualifies a policy nor runs a functional probe.
"""
from collections import Counter
from copy import deepcopy
import math

from . import integrated_runtime
from .nonstandard_start import runtime_expectation as birth_expectation
from .ordinary_posture_curriculum import ordinary_eligibility, runtime_expectation as posture_expectation
from .preflight import _compare, validate_runtime_state as validate_native_runtime_state
from .task import choose_scene_groups

SCHEMA = "v6-integrated-preflight-candidate-v1"
FIELD = "integrated_curriculum"
OBSERVED = "integrated_curriculum_observed"


def _active(config):
    return bool(config.get("integrated_runtime_candidate")) and not config.get("evaluation_exact_cases")


def scoped_expectation(config, *, num_envs):
    """Build a named design declaration from config/source clocks, before startup."""
    if not _active(config):
        raise ValueError("Scoped preflight requires an explicit training candidate")
    audit = config["integrated_runtime_candidate"]
    if (audit.get("schema") != integrated_runtime.SCHEMA or audit.get("role") != "train"
            or audit.get("domain") != config.get("task_campaign", {}).get("domain")
            or audit.get("runtime_ready") is not False or audit.get("qualification") is not None):
        raise ValueError("Scoped preflight requires the matching unqualified runtime candidate")
    clocks = integrated_runtime._validate_clock_document(audit["clocks_at_source"], audit["learning_source"])
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("Scoped preflight requires a positive native population")
    ownership, components = audit["row_ownership"], audit["component_runtime"]
    if set(ownership) != set(config["skill_specs"]):
        raise ValueError("Scoped preflight row ownership must cover the complete skill table")
    rows = choose_scene_groups(config["scene_groups"], num_envs)
    population = Counter(name for name, _ in rows)
    fields = {}
    for field, key, factory in (
            ("ordinary_posture_curriculum", "posture_component", posture_expectation),
            ("nonstandard_start", "nonstandard_component", birth_expectation)):
        if field not in config:
            continue
        settings = {}
        for group in population:
            owner = ownership[group][key]
            if owner is not None:
                if owner not in clocks["component_actor_updates"] or owner not in components:
                    raise ValueError("Scoped preflight has an unknown component owner")
                value = components[owner].get(field)
                if value is not None:
                    # Validate every complete native schema, including physical priors.
                    factory(value, config["policy_dt"]) if field == "nonstandard_start" else factory(value)
                    settings[owner] = deepcopy(value)
        base = (factory(config[field], config["policy_dt"]) if field == "nonstandard_start"
                else factory(config[field]))
        fields[field] = dict(base, component_settings=settings)
    if "locomotion_difficulty" in config:
        fields["locomotion_difficulty"] = dict(enabled=True,
            settings=deepcopy(config["locomotion_difficulty"]),
            row_components={name: owner["difficulty_component"] for name, owner in ownership.items()})
    populations = {}
    for field, key in (("ordinary_posture_curriculum", "posture_component"),
                       ("nonstandard_start", "nonstandard_component")):
        if field not in fields:
            continue
        settings = fields[field]["component_settings"]
        masks = Counter(ownership[name][key] for name, _ in rows if ownership[name][key] in settings)
        value = dict(component_mask_counts={name: masks[name] for name in settings})
        if field == "ordinary_posture_curriculum":
            import torch
            names, kinds = map(list, zip(*rows))
            # Native spin activations are false at the initial reset (stable=false).
            spin = torch.tensor([config["skill_specs"][name].get("kind") == "spin_translate"
                and not config.get("special_mode_activation", {}).get("spin_translate") for name in names])
            recovery = torch.tensor([bool(config["skill_specs"][name].get("recovery_pose")) for name in names])
            eligible = ordinary_eligibility(config, names, kinds, "cpu", recovery=recovery, spin=spin)
            counts = Counter(ownership[name][key] for (name, _), active in zip(rows, eligible.tolist(), strict=True)
                if active and ownership[name][key] in settings and settings[ownership[name][key]]["enabled"])
            value.update(eligible_count=sum(counts.values()),
                component_eligible_counts={name: counts[name] for name in settings})
        populations[field] = value
    return dict(schema=SCHEMA, domain=audit["domain"], num_envs=num_envs,
        clocks_at_source=clocks, row_ownership=deepcopy(ownership),
        row_population=dict(sorted(population.items())), runtime_fields=fields, module_populations=populations,
        runtime_ready=False, qualification=None)


def with_scoped_expectation(native_expected, config, *, num_envs):
    """Add the declaration to a future frozen design; never consume actual state."""
    result = deepcopy(native_expected)
    result[FIELD] = scoped_expectation(config, num_envs=num_envs)
    return result


def observe_component_clocks(env):
    """Read installed native clock providers and their actual module ownership."""
    from .training_schedule import stage_actor_updates
    components = env.cfg["integrated_runtime_candidate"]["clocks_at_source"]["component_actor_updates"]
    result = dict(schema=SCHEMA, task_actor_updates=stage_actor_updates(env),
        component_actor_updates={name: integrated_runtime.component_actor_updates(env, name) for name in components})
    for field, attribute in (("ordinary_posture_curriculum", "ordinary_posture"),
                             ("nonstandard_start", "nonstandard_start")):
        module = getattr(env, attribute, None)
        if module is not None:
            result[field] = env._curriculum_clock(field)
            value = dict(component_mask_counts={name: int(mask.sum()) for name, mask in module.component_masks.items()})
            if field == "ordinary_posture_curriculum":
                value.update(eligible_count=int(module.eligible.sum()),
                    component_eligible_counts={name: int((mask & module.eligible).sum())
                        for name, mask in module.component_masks.items()})
            result.setdefault("module_populations", {})[field] = value
    return result


def _clocks(scope, actor_update):
    """Independent arithmetic from the sealed design, not the actual clock helper."""
    source = scope["clocks_at_source"]
    if type(actor_update) is not int or actor_update < source["domain_actor_updates"]:
        raise ValueError("Scoped preflight rejected a regressed task actor clock")
    delta = actor_update - source["domain_actor_updates"]
    joint = source["joint_actor_updates"] + delta
    origin = source.get("origin_recovery_actor_update", 15000)
    if type(origin) is not int or origin < 0:
        raise ValueError("Scoped preflight rejected an invalid sealed REC origin")
    return {name: origin + joint if name == "recovery" else joint - start
            for name, start in {"recovery": None, **source["component_start_joint_actor_updates"]}.items()}


def _validate_posture(expected, actual, clocks, population, errors):
    observed = actual.pop("runtime_observed", None)
    _compare(expected, actual, "runtime.ordinary_posture_curriculum", errors)
    if not expected["enabled"]:
        if observed is not None:
            errors.append("runtime.ordinary_posture_curriculum: disabled candidate has no population observation")
        return
    if not isinstance(observed, dict):
        errors.append("runtime.ordinary_posture_curriculum: actual scoped population observation required")
        return
    integers = ("eligible_count", "birth_eligible_population", "birth_selected_count", "birth_rejected_count", "push_events")
    floats = ("birth_abs_angle_max_rad", "push_clock_min_s", "push_clock_max_s", "push_clock_std_s", "push_velocity_delta_max_m_s")
    keys = set(integers + floats) | {"geometry_bound"}
    for component, settings in expected["component_settings"].items():
        prefix = f"component/{component}/"
        relative = max(0, clocks[component] - settings["origin_actor_update"])
        ramp = settings["ramp"]
        alpha = min(1., max(0., (relative - ramp["start_actor_update"])
            / (ramp["end_actor_update"] - ramp["start_actor_update"])))
        caps = {"relative_actor_updates": relative,
            "birth_angle_cap_rad": settings["birth"]["start_roll_pitch_rad"] + alpha * (
                settings["birth"]["end_roll_pitch_rad"] - settings["birth"]["start_roll_pitch_rad"]),
            "push_speed_cap_m_s": settings["push"]["start_speed_m_s"] + alpha * (
                settings["push"]["end_speed_m_s"] - settings["push"]["start_speed_m_s"])}
        keys.update(prefix + key for key in caps)
        for key, value in caps.items():
            got = observed.get(prefix + key)
            valid = (type(got) is int and got == value if key == "relative_actor_updates" else
                     type(got) in (int, float) and math.isfinite(got)
                     and math.isclose(got, value, rel_tol=1e-6, abs_tol=1e-8))
            if not valid:
                errors.append(f"runtime.ordinary_posture_curriculum.{prefix}{key}: actual component-clock cap differs")
    if set(observed) != keys:
        errors.append("runtime.ordinary_posture_curriculum: complete component observations required; no global cap substitution")
    for key in integers:
        if type(observed.get(key)) is not int or observed[key] < 0:
            errors.append(f"runtime.ordinary_posture_curriculum.{key}: nonnegative integer required")
    for key in ("eligible_count", "birth_eligible_population"):
        _compare(population["eligible_count"], observed.get(key),
                 f"runtime.ordinary_posture_curriculum.{key}", errors)
    for key in floats:
        value = observed.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            errors.append(f"runtime.ordinary_posture_curriculum.{key}: finite nonnegative value required")
    for key in ("eligible_count", "birth_eligible_population", "birth_selected_count", "birth_abs_angle_max_rad"):
        if type(observed.get(key)) in (int, float) and observed[key] <= 0:
            errors.append(f"runtime.ordinary_posture_curriculum.{key}: enabled lesson has zero population")
    if observed.get("geometry_bound") is not True:
        errors.append("runtime.ordinary_posture_curriculum.geometry_bound: authenticated geometry required")
    lo, hi = expected["settings"]["push"]["interval_seconds"]
    timer_values = [observed.get(key) for key in ("push_clock_min_s", "push_clock_max_s", "push_clock_std_s")]
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in timer_values)
            or not lo <= timer_values[0] <= timer_values[1] <= hi or timer_values[2] <= 0):
        errors.append("runtime.ordinary_posture_curriculum: independent native push timers differ")


def validate_integrated_runtime_state(expected, actual, *, config, num_envs, actor_update):
    """Validate scoped clocks while delegating all other physical/module checks."""
    if not _active(config):
        return validate_native_runtime_state(expected, actual)
    if config["integrated_runtime_candidate"].get("static_recipe_preflight"):
        raise ValueError("Scoped preflight rejected: static recipe has no actual learning clocks")
    scope = expected.get(FIELD)
    if not isinstance(scope, dict):
        raise ValueError("Scoped preflight rejected: explicit independent integrated_curriculum expectation required")
    independent = scoped_expectation(config, num_envs=num_envs)
    errors = []
    _compare(independent, scope, "design.integrated_curriculum", errors)
    clocks = _clocks(independent, actor_update)
    clock_expected = dict(schema=SCHEMA, task_actor_updates=actor_update, component_actor_updates=clocks,
        module_populations=deepcopy(independent["module_populations"]))
    for field, value in independent["runtime_fields"].items():
        if field in ("ordinary_posture_curriculum", "nonstandard_start"):
            clock_expected[field] = {name: clocks[name] for name in value["component_settings"]}
    actual_clocks = actual.get(OBSERVED)
    _compare(clock_expected, actual_clocks, "runtime.integrated_curriculum_observed", errors)
    if isinstance(actual_clocks, dict):
        for field in ("task_actor_updates", "component_actor_updates", "ordinary_posture_curriculum", "nonstandard_start"):
            values = actual_clocks.get(field)
            if isinstance(values, dict):
                values = values.values()
            else:
                values = (values,) if field in clock_expected else ()
            if any(type(value) is not int or value < 0 for value in values):
                errors.append(f"runtime.integrated_curriculum_observed.{field}: integer component clocks required")
    before, observed = deepcopy(expected), deepcopy(actual)
    before.pop(FIELD)
    observed.pop(OBSERVED, None)
    # Overrides come only from the independently declared design after its check.
    before.update(deepcopy(independent["runtime_fields"]))
    posture = before.pop("ordinary_posture_curriculum", None)
    if posture is not None:
        actual_posture = observed.pop("ordinary_posture_curriculum", None)
        if not isinstance(actual_posture, dict):
            errors.append("runtime.ordinary_posture_curriculum: actual scoped mapping required")
        else:
            _validate_posture(posture, actual_posture, clocks,
                independent["module_populations"]["ordinary_posture_curriculum"], errors)
        if type(before.get("modules", {}).get("ordinary_posture_curriculum")) is not bool:
            errors.append("runtime.modules.ordinary_posture_curriculum: explicit boolean expectation required")
        elif before["modules"]["ordinary_posture_curriculum"] != posture["enabled"]:
            errors.append("runtime.modules.ordinary_posture_curriculum: scoped enabled setting differs")
    try:
        validate_native_runtime_state(before, observed)
    except ValueError as error:
        errors.append(str(error))
    if errors:
        raise ValueError("Scoped runtime preflight rejected:\n" + "\n".join(errors))
    return dict(status="runtime_configuration_passed", observed=actual,
        functional_transport_precision_randomization_probes="not_run",
        scoped_validation=SCHEMA, runtime_ready=False, qualification=None)
