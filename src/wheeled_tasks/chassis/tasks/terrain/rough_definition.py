"""Bound terrain definitions and column allocation without simulator imports."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

REFERENCE = Path(__file__).resolve().parents[5] / "references/scut_curriculum_b8ff79f3"


ENV_FILE = "pretrained/26_infantry/rough_rotation_stair/2026-07-30_09-16-27/params/env.yaml"


ENV_SHA = "a6117e55b3ebb5a03870981bd2d65d889c38850063008a9341d8346b997bb1ee"


GRID_FILE = "source/agent_world/agent_world/terrains/height_field.py"


GRID_SHA = "5f444236205f6ab4a56e0c76387b388d5d2d5a7a83ab63386f5cf15ae5b68d0e"


SOURCE_COMMIT = "b8ff79f3df855faf9dc92f4a282bd80c42649466"


def _read_bound(relative, expected, root=REFERENCE):
    raw = (Path(root) / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("reference rough source SHA differs: " + relative)
    return raw


def source_parameters(root=REFERENCE):
    """Read the saved effective terrain block; Python YAML tags remain data."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    for tag in ("python/tuple", "python/object/apply:builtins.slice"):
        Loader.add_constructor("tag:yaml.org,2002:" + tag,
                               lambda loader, node: loader.construct_sequence(node))
    env = yaml.load(_read_bound(ENV_FILE, ENV_SHA, root), Loader=Loader)
    return deepcopy(env["terrain"])


def column_names(parameters):
    generator = parameters["terrain_generator"]
    entries = generator["sub_terrains"]
    weights = np.asarray([entry["proportion"] for entry in entries.values()], dtype=float)
    cumulative = np.cumsum(weights / weights.sum())
    names = list(entries)
    return [names[int(np.flatnonzero(i / generator["num_cols"] + .001 < cumulative)[0])]
            for i in range(generator["num_cols"])]
