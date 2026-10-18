"""Explicit V7 measurement wiring; no experiment or promotion authorization.

The historical default course is intentionally unavailable through this entry.
Physical thresholds require measurements and a later reviewed scoring schema.
Native admission and the shared BudgetOwner remain independent prerequisites.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .autonomous_curriculum import validate_definition, measurement_only


MARKER = "v7_curriculum_runtime"
SCHEMA = "v7-source-flat-measurement-selection-v1"
SOURCE_FILES = [
    "src/wheeled_tasks/chassis/curriculum_selection.py",
    "src/wheeled_tasks/chassis/curriculum_sampling.py",
    "src/wheeled_tasks/chassis/curriculum_runtime.py",
    "src/wheeled_tasks/chassis/autonomous_curriculum.py",
    "src/wheeled_tasks/chassis/autonomous_exposure.py",
    "src/wheeled_tasks/chassis/training_budget.py",
    "src/wheeled_tasks/chassis/flat_episode_runtime.py",
    "src/wheeled_tasks/chassis/flat_command_sampling.py",
    "src/wheeled_algo/recovery_rollout.py",
]


def validate_curriculum_selection(config):
    if MARKER not in config:
        return None
    value = config[MARKER]
    keys = {"schema", "definition_file", "definition_sha256", "capabilities"}
    if not isinstance(value, dict) or set(value) != keys or value["schema"] != SCHEMA:
        raise ValueError("Unknown/incomplete V7 curriculum measurement selection")
    path, sha = value["definition_file"], value["definition_sha256"]
    if (not isinstance(path, str) or not path or Path(path).is_absolute()
            or ".." in Path(path).parts or Path(path).suffix != ".json"):
        raise ValueError("Course definition must be a local workspace JSON path")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Course definition needs an exact byte SHA256")
    if not isinstance(value["capabilities"], dict):
        raise ValueError("Source and chassis capabilities must be explicitly bound")
    from .training_profile import uses_profile
    if not uses_profile(config) or "flat_episode_runtime" not in config:
        raise ValueError("V7 course measurement requires the named78 source-flat owner")
    if any(config.get(key) for key in ("performance_curriculum", "task_campaign", "integrated_production",
                                    "evaluation_exact_cases", "flat_evaluation_setup")):
        raise ValueError("V7 training exposure cannot mix another course or fixed evaluator owner")
    return deepcopy(value)


def load_curriculum_definition(config, root):
    settings = validate_curriculum_selection(config)
    if settings is None:
        return None
    root = Path(root).resolve()
    path = root / settings["definition_file"]
    if not path.is_file() or path.resolve() != path or not path.resolve().is_relative_to(root):
        raise ValueError("Course definition is missing, escaped or symlinked")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != settings["definition_sha256"]:
        raise ValueError("Course definition bytes changed")
    definition = json.loads(content)
    if not measurement_only(definition):
        raise ValueError("Old fixtures and unreviewed physical scoring cannot select V7 runtime")
    validate_definition(definition)
    if definition["clock"]["reference_transitions_per_update"] != 4096 * 24:
        raise ValueError("V7 source-flat definition must bind the actual4096x24 rollout")
    for phase in definition["phases"]:
        for node in phase["nodes"]:
            if (node["recovery_lesson_fractions"] or
                    set(node["minimum_exposure"]["key_phase_counts"]) != {"source_horizon_complete"}):
                raise ValueError("Source-flat observer provides only complete source-horizon occurrences")
            for case_id in node["required_cases"]:
                case = definition["cases"][case_id]
                if case["kind"] not in ("source_flat_ordinary", "stand", "pure_translation", "pure_spin"):
                    raise ValueError("Unsupported lesson cannot be credited by source-flat commands")
                if case["minimum_observed_duration_s"] != 999 * .02:
                    raise ValueError("Source flat horizon is 999 policy ticks, not a fabricated full20 seconds")
    return definition


def curriculum_source_files(config):
    settings = validate_curriculum_selection(config)
    return [] if settings is None else [*SOURCE_FILES, settings["definition_file"]]


def require_curriculum_launch(config, *, resume=None, transfer=None, num_envs=None, rollout_steps=None):
    settings = validate_curriculum_selection(config)
    if settings is None:
        return None
    if resume is not None or transfer is not None:
        raise ValueError("Curriculum native restart awaits coordinated physical/environment restoration")
    if num_envs != 4096 or rollout_steps != 24:
        raise ValueError("V7 measurement launch must retain the reviewed4096x24 rollout")
    return settings
