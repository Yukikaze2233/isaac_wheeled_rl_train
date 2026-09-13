"""Reject asset substitution even when the CLI preflight is bypassed."""
import ast
import __future__
from pathlib import Path

import pytest


@pytest.mark.parametrize("seed", ["replacement.usd", ""])
def test_direct_asset_factory_rejects_unbound_seed_before_sdk_calls(seed):
    source = Path(__file__).resolve().parents[2] / "src/wheeled_world/assets/v40.py"
    tree = ast.parse(source.read_text())
    factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "make_v40_articulation")
    namespace = {}
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(source), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    # No SDK constructors are supplied: rejection must precede asset creation.
    with pytest.raises(ValueError, match="not bound to the audited asset manifest"):
        namespace["make_v40_articulation"](
            urdf_path="canonical.urdf", joint_names=[], nominal_positions=[],
            effort_limits=[], armatures=[], nominal_base_height=.32,
            asset_manifest_sha256="0" * 64, usd_seed=seed,
        )
