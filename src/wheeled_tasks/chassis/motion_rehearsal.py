"""Opt-in authoring of pure basic motion atoms in the integrated speed recipe.

The native sampler, feasibility projection, row quotas and frontier remain in
place. The native heading boundary selects ``configure_heading`` only for an
explicit motion-rehearsal candidate. This module does not enable a recipe,
qualify a policy or change a learning clock.
"""
from copy import deepcopy

from .heading_command import configure_heading as _inherited_heading


SCHEMA = "v6-motion-rehearsal-v1"
BASE_COMMANDS = {
    "forward_05": (.5, 0., .305),
    "backward_05": (-.5, 0., .305),
    "forward_1": (1., 0., .305),
    "backward_1": (-1., 0., .305),
    "rotate_1": (0., 1., .305),
    "rotate_1__reverse_train": (0., -1., .305),
}
PURE_NAMES = tuple(name for base in BASE_COMMANDS
                   for name in (base, base + "__height_train"))
CONTINUOUS_BASES = ("curve_low", "curve_low__reverse_train",
                    "forward_continuous_low", "backward_continuous_low",
                    "yaw_continuous_low", "yaw_continuous_low__reverse_train")
CONTINUOUS_NAMES = tuple(name for base in CONTINUOUS_BASES
                         for name in (base, base + "__height_train"))


def pure_height_settings():
    """Return the native full-height distribution and transition semantics."""
    return dict(range_m=[.23, .43], nominal_fraction=0., endpoint_fraction=.2,
                transition_seconds=4., start_at_target_fraction=.5,
                boundary_fraction_of_range=.1, boundary_weight=2.)


def motion_rehearsal_settings():
    """Fresh, explicit settings suitable for a frozen versioned candidate."""
    return dict(schema=SCHEMA, enabled=True,
                pure_rows={name: list(command) for base, command in BASE_COMMANDS.items()
                           for name in (base, base + "__height_train")},
                nominal_height_m=.305, height_sampling=pure_height_settings(),
                heading_owner_names=list(CONTINUOUS_NAMES),
                pure_difficulty_component=None,
                preserve="scene rows, fractions, coupled feasibility and frontier")


def _structure(config):
    runtime = config.get("integrated_runtime_candidate", {})
    specs = config.get("skill_specs", {})
    rows = config.get("scene_groups", [])
    names = [row["name"] for row in rows]
    owners = runtime.get("row_ownership", {})
    mapped = config.get("scut_reuse_candidate", {}).get("mapped_groups", {})
    if (runtime.get("domain") != "speed" or config.get("task_campaign", {}).get("domain") != "speed"
            or len(names) != 189 or len(set(names)) != 189
            or set(names) != set(specs) or set(owners) != set(specs)
            or not set(PURE_NAMES + CONTINUOUS_NAMES) <= set(mapped)):
        raise ValueError("Motion rehearsal requires the effective 189-row integrated speed recipe")
    return runtime, specs, mapped


def _expected_kind(base):
    return "rotate" if base.startswith("rotate_") else (
        "backward" if base.startswith("backward_") else "forward")


def validate(config):
    """Check actual command producers and ownership, not legacy row labels."""
    runtime, specs, mapped = _structure(config)
    if config.get("motion_rehearsal") != motion_rehearsal_settings():
        raise ValueError("Motion rehearsal settings differ from the explicit pure contract")
    for base, command in BASE_COMMANDS.items():
        for name in (base, base + "__height_train"):
            spec = specs[name]
            pool = "basic_yaw" if base.startswith("rotate_") else "basic_translation"
            if (spec.get("kind") != _expected_kind(base) or spec.get("command") != list(command)
                    or config["behavior_pool_membership"].get(name) != pool
                    or spec.get("planar_command_sampling") is not None
                    or spec.get("sample_amplitude") is not False
                    or spec.get("sample_yaw_sign") is not False
                    or spec.get("standing_command_probability", 0.) != 0.
                    or spec.get("command_curriculum") or spec.get("height_motion")
                    or spec.get("command_sequence") or spec.get("request_events")
                    or runtime["row_ownership"][name].get("difficulty_component") is not None
                    or mapped[name].get("kind") != "protected_motion_rehearsal"):
                raise ValueError("Pure motion has a competing command producer or owner: " + name)
            if name.endswith("__height_train"):
                if (spec.get("height_sampling") != pure_height_settings()
                        or spec.get("height_transition_seconds") != 4.):
                    raise ValueError("Pure motion height coverage differs: " + name)
            elif spec.get("height_sampling") or spec.get("height_transition_seconds"):
                raise ValueError("Nominal pure motion must keep its fixed height: " + name)
    normal = {name for name, row in mapped.items() if row.get("kind") == "normal"}
    if (normal != set(CONTINUOUS_NAMES)
            or config.get("scut_heading_command", {}).get("ordinary_names") != list(CONTINUOUS_NAMES)):
        raise ValueError("Heading ownership overlaps pure motion or omits an extended row")
    return deepcopy(config["motion_rehearsal"])


def apply(config, settings=None):
    """Deep-copy the effective speed runtime; retain every row and allocation.

    This is an authoring operation. The native heading boundary selects the
    explicit helper below; the inherited heading configurator rejects this
    changed ownership instead of silently applying heading to pure yaw rows.
    """
    if settings is None:
        settings = motion_rehearsal_settings()
    if settings != motion_rehearsal_settings():
        raise ValueError("Motion rehearsal requires its explicit versioned settings")
    candidate = deepcopy(config)
    runtime, specs, mapped = _structure(candidate)
    normal = {name for name, row in mapped.items() if row.get("kind") == "normal"}
    if normal != set(PURE_NAMES + CONTINUOUS_NAMES):
        raise ValueError("Motion rehearsal needs the inherited 24 normal motion rows")
    for base, command in BASE_COMMANDS.items():
        for name in (base, base + "__height_train"):
            spec = specs[name]
            if spec.get("command") != list(command) or spec.get("kind") != _expected_kind(base):
                raise ValueError("Inherited named motion command differs: " + name)
            spec.pop("planar_command_sampling", None)
            spec.update(sample_amplitude=False, sample_yaw_sign=False,
                        standing_command_probability=0.)
            if name.endswith("__height_train"):
                spec["height_sampling"] = pure_height_settings()
                spec["height_transition_seconds"] = 4.
            else:
                spec.pop("height_sampling", None)
                spec.pop("height_transition_seconds", None)
            runtime["row_ownership"][name]["difficulty_component"] = None
            mapped[name]["kind"] = "protected_motion_rehearsal"
    candidate["scut_heading_command"]["ordinary_names"] = list(CONTINUOUS_NAMES)
    candidate["scut_reuse_candidate"]["group_statistics"] = (
        "Continuous joint commands belong to the extended rows; named basic rows retain pure commands")
    candidate["motion_rehearsal"] = deepcopy(settings)
    validate(candidate)
    return candidate


def configure_heading(config, root=None):
    """Authenticate inherited heading settings and bind only extended rows.

    Reconstruct only the prior sampler claims for source authentication. The
    returned ownership comes from the explicit pure-motion contract; no sampled
    commands or inherited source pins are rewritten by this read-only helper.
    """
    if config.get("evaluation_exact_cases") or not config.get("scut_heading_command", {}).get("enabled"):
        return None
    validate(config)
    inherited = deepcopy(config)
    prototype = inherited["skill_specs"][CONTINUOUS_NAMES[0]]
    for name in PURE_NAMES:
        spec = inherited["skill_specs"][name]
        spec["planar_command_sampling"] = deepcopy(prototype["planar_command_sampling"])
        spec["standing_command_probability"] = prototype["standing_command_probability"]
        inherited["scut_reuse_candidate"]["mapped_groups"][name]["kind"] = "normal"
    configured = _inherited_heading(inherited, root)
    configured["ordinary_names"] = tuple(CONTINUOUS_NAMES)
    return configured
