"""CPU geometry and native outcome checks for the training-only route runout."""
import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.full_tasks import FullTaskSemantics
from wheeled_tasks.chassis.evaluation import fixed_suite_contract
from wheeled_tasks.chassis.task import Phase, Surface, episode_outcome, terrain_surfaces
from wheeled_tasks.chassis.terrain_runout import (
    ROUTE_TERRAIN_KINDS, append_route_runout, route_runout_selection, validate_runout,
)


WORKER_FILE = ROOT / "contracts/v6_task_campaign_terrain_worker_v18.json"
# Same immutable contract bytes as frozen source revision 023, not a test fixture
# reconstructed from the helper's selection logic.
WORKER_SHA = "d3c0658b7b5b51aa89fc08d1503e91ae57a4682173bcb076bdbd4d83ede3f611"
WORKER = json.loads(WORKER_FILE.read_text())
MECHANICAL_INPUT_SHA = {
    "model/纯底盘_v6_232mm_stop105/urdf/model_spec.json":
        "7429f6e59369e590e0fd96cba8b45fd60c362a0231c4bcc2e80a4c4688eadd88",
    "model/纯底盘_v6_232mm_stop105/urdf/robot.urdf":
        "12488abd510c1b6e395b88a786e9f5b542f929d10d4a3f1ce1f7b68699e35635",
    "tools/v5_mechanism.py":
        "c73c5add74f08a71c51bdf42ec629a2dee372664938ead7c180ccafd607249b8",
    "tools/build_v5_closedchain.py":
        "1e010538a9113a007d330f5e869fe6832d2a5b2c87c6739226ab46887d6b5d25",
}
ROUTE_ROWS = [(index, row) for index, row in enumerate(WORKER["scene_groups"])
              if WORKER["skill_specs"][row["name"]].get("mode") in (2, 3)]


def enabled_config():
    config = deepcopy(WORKER)
    config["terrain_route_runout"] = dict(enabled=True, support_end_x_m=6.2,
                                         positive_boundary_x_m=5.6)
    return config


def height_at(surfaces, x, y=0.):
    return max((surface.height(x, y) for surface in surfaces
                if surface.x0 <= x <= surface.x1), default=0.)


def rotate(point, quaternion):
    w, x, y, z = quaternion
    cross = (y * point[2] - z * point[1], z * point[0] - x * point[2],
             x * point[1] - y * point[0])
    cross2 = (y * cross[2] - z * cross[1], z * cross[0] - x * cross[2],
              x * cross[1] - y * cross[0])
    return tuple(point[i] + 2 * (w * cross[i] + cross2[i]) for i in range(3))


def assert_box_top_matches(surface, *, solid_bottom=None):
    size, position, quaternion = surface.box(width=4., solid_bottom=solid_bottom)
    for fx in (-.5, -.2, .2, .5):
        for fy in (-.5, 0., .5):
            point = rotate((fx * size[0], fy * size[1], size[2] / 2), quaternion)
            world = tuple(point[i] + position[i] for i in range(3))
            assert surface.x0 - 1e-10 <= world[0] <= surface.x1 + 1e-10
            assert -2. - 1e-10 <= world[1] <= 2. + 1e-10
            assert world[2] == pytest.approx(surface.height(*world[:2]), abs=1e-12)


def test_real_frozen_v18_selects_54_of_129_groups_without_touching_contract():
    assert hashlib.sha256(WORKER_FILE.read_bytes()).hexdigest() == WORKER_SHA
    config = enabled_config()
    before = deepcopy(config)
    selected = [row for row in config["scene_groups"] if
                route_runout_selection(config, row["name"], row["terrain"][0])]
    assert len(config["scene_groups"]) == 129
    assert len(selected) == 54
    grades = Counter(config["skill_specs"][row["name"]]["terrain_training_row"]["grade"]
                     for row in selected)
    assert grades == {"easy": 18, "medium": 18, "full": 18}
    assert {row["terrain"][0] for row in selected} == ROUTE_TERRAIN_KINDS
    for row in config["scene_groups"]:
        if "body_climb" in config["skill_specs"][row["name"]]:
            assert not route_runout_selection(config, row["name"], row["terrain"][0])
    assert config == before


@pytest.mark.parametrize("index,row", ROUTE_ROWS, ids=[row["name"] for _, row in ROUTE_ROWS])
def test_every_native_row_preserves_old_support_boxes_and_matches_runout_seam(index, row):
    config = enabled_config()
    settings = validate_runout(config)
    spec = config["skill_specs"][row["name"]]
    kind = row["terrain"][0]
    old = terrain_surfaces(kind, spec["terrain_limits"], spec["terrain_level"], index)
    bottom = (min(surface.z0 for surface in old) - .1
              if kind in ("step_up", "step_down", "stairs", "stairs_down") else None)
    boxes = [surface.box(width=4., solid_bottom=bottom) for surface in old]
    new = append_route_runout(old, settings)
    assert all(first is second for first, second in zip(new[:-1], old))
    assert new[:-1] == old
    assert [surface.box(width=4., solid_bottom=bottom) for surface in new[:-1]] == boxes
    assert len(new) == len(old) + 1
    last, extension = old[-1], new[-1]
    assert (extension.x0, extension.x1) == (4., 6.2)
    assert extension.slope == last.slope
    assert extension.cross_slope == last.cross_slope
    assert extension.friction == last.friction
    for y in (-1.7, -.2, 0., .2, 1.7):
        assert extension.height(4., y) == pytest.approx(last.height(4., y), abs=1e-12)
        for x in (-4., -1., -.5, 0., .6, 1.2, 1.8, 3.99, 4.):
            assert height_at(new, x, y) == pytest.approx(height_at(old, x, y), abs=1e-12)
        for x in (4., 4.3, 5.6, 6.2):
            assert height_at(new, x, y) == pytest.approx(last.height(x, y), abs=1e-12)
    assert_box_top_matches(extension, solid_bottom=bottom)
    # Appending never alters platform_delta's native samples at x=2 and -1.5.
    assert new[-1].height(2.) - new[0].height(-1.5) == pytest.approx(
        old[-1].height(2.) - old[0].height(-1.5), abs=1e-12)


@pytest.mark.parametrize("surface", [Surface(1., 4., .2, slope=.2, friction=.4),
    Surface(1., 4., -.3, slope=-.3, friction=.7),
    Surface(-4., 4., .1, cross_slope=.267949, friction=.6),
    Surface(-4., 4., -.2, cross_slope=-.267949, friction=.3)])
def test_terminal_plane_continuation_preserves_nonzero_slope_and_box_top(surface):
    new = append_route_runout([surface], validate_runout(enabled_config()))
    for y in (-1.7, 0., 1.7):
        assert new[-1].height(4., y) == pytest.approx(surface.height(4., y))
        assert new[-1].height(6.2, y) == pytest.approx(surface.height(6.2, y))
    assert_box_top_matches(surface)
    assert_box_top_matches(new[-1])


@pytest.mark.parametrize("change", ["absent", "disabled", "exact", "other_domain"])
def test_disabled_exact_evaluation_and_other_domains_are_identity_noops(change):
    config = enabled_config()
    if change == "absent":
        config.pop("terrain_route_runout")
    elif change == "disabled":
        config["terrain_route_runout"] = dict(enabled=False, support_end_x_m=float("nan"))
    elif change == "exact":
        config["evaluation_exact_cases"] = True
        config["evaluation_long_corridors"] = False
        config["terrain_route_runout"] = dict(enabled=True, support_end_x_m=float("nan"))
    else:
        config["task_campaign"]["domain"] = "jump"
    assert validate_runout(config) is None
    assert not route_runout_selection(config, "slope_up", "slope_up")
    old = terrain_surfaces("slope_up", WORKER["terrain_limits"], 1.)
    assert append_route_runout(old, validate_runout(config)) is old


def test_actual_fixed_suite_projection_preserves_all_322_cases_and_old_geometry():
    baseline = fixed_suite_contract(deepcopy(WORKER))
    candidate = fixed_suite_contract(enabled_config())
    settings = candidate.pop("terrain_route_runout")
    assert candidate == baseline
    assert len(candidate["evaluation"]["cases"]) == 322
    candidate["terrain_route_runout"] = settings
    assert validate_runout(candidate) is None
    for row in candidate["scene_groups"]:
        assert not route_runout_selection(candidate, row["name"], row["terrain"][0])
    for index, row in ROUTE_ROWS:
        spec = WORKER["skill_specs"][row["name"]]
        old = terrain_surfaces(row["terrain"][0], spec["terrain_limits"], spec["terrain_level"], index)
        assert append_route_runout(old, validate_runout(candidate)) is old


def test_explicit_enabled_setting_has_reviewed_geometric_defaults():
    config = enabled_config()
    config["terrain_route_runout"] = dict(enabled=True)
    assert validate_runout(config) == dict(enabled=True, support_end_x_m=6.2,
                                          positive_boundary_x_m=5.6)


@pytest.mark.parametrize("fault", ["missing_marker", "empty_marker", "mode_zero",
    "mode_one", "mode_four", "float_mode", "body_climb", "unknown_group", "unknown_kind"])
def test_only_native_marked_mode_two_or_three_rows_can_be_selected(fault):
    config = enabled_config()
    group, kind = "slope_up", "slope_up"
    spec = config["skill_specs"][group]
    if fault == "missing_marker":
        spec.pop("terrain_training_row")
    elif fault == "empty_marker":
        spec["terrain_training_row"] = {}
    elif fault == "body_climb":
        spec["kind"] = "body_climb"
    elif fault == "unknown_group":
        group = "missing"
    elif fault == "unknown_kind":
        kind = "platform"
    else:
        spec["mode"] = {"mode_zero": 0, "mode_one": 1, "mode_four": 4, "float_mode": 2.}[fault]
    assert not route_runout_selection(config, group, kind)


@pytest.mark.parametrize("key,value", [("enabled", 1), ("enabled", "true"),
    ("support_end_x_m", float("nan")), ("support_end_x_m", float("inf")),
    ("support_end_x_m", True), ("support_end_x_m", 4.),
    ("positive_boundary_x_m", float("nan")), ("positive_boundary_x_m", float("inf")),
    ("positive_boundary_x_m", True), ("positive_boundary_x_m", 3.6),
    ("positive_boundary_x_m", 5.61), ("positive_boundary_x_m", 6.1),
    ("extra", 1)])
def test_invalid_enabled_settings_are_rejected(key, value):
    config = enabled_config()
    config["terrain_route_runout"][key] = value
    with pytest.raises(ValueError):
        validate_runout(config)


@pytest.mark.parametrize("key,value", [("evaluation_long_corridors", False),
    ("evaluation_long_corridors", 1), ("flat_half_length_m", 0.),
    ("flat_half_length_m", float("nan")), ("flat_half_length_m", True),
    ("flat_floor_width_m", 3.99), ("flat_floor_width_m", 8.01),
    ("flat_floor_width_m", float("inf"))])
def test_unvalidated_layout_is_rejected(key, value):
    config = enabled_config()
    config[key] = value
    with pytest.raises(ValueError):
        validate_runout(config)


def test_spacing_checks_real_mixed_corridor_extents_and_support_padding():
    config = enabled_config()
    settings = validate_runout(config)
    assert settings["support_end_x_m"] - settings["positive_boundary_x_m"] == pytest.approx(.6)
    spacing = 2 * config["flat_half_length_m"] + 12.
    assert spacing == 332.
    assert config["flat_floor_width_m"] + 1. == 9.
    assert spacing - 2 * max(config["flat_half_length_m"], 20., 6.2) == 12.
    config["flat_half_length_m"] = 1.
    with pytest.raises(ValueError, match="overlaps"):
        validate_runout(config)  # Existing release-stop terrain still spans +-20m.
    config["release_stop"]["enabled"] = False
    assert validate_runout(config) == settings
    config["terrain_route_runout"]["support_end_x_m"] = 7.
    with pytest.raises(ValueError, match="overlaps"):
        validate_runout(config)


@pytest.mark.parametrize("old", [[], [Surface(0., 3.)], [Surface(0., 4., float("nan"))],
    [Surface(4., 4.)], [Surface(0., 4., slope=.1, cross_slope=.1)], [object()]])
def test_missing_or_unrepresentable_terminal_support_is_rejected(old):
    with pytest.raises(ValueError):
        append_route_runout(old, validate_runout(enabled_config()))


def native_assignment(prefix):
    path = ROOT / "src/wheeled_tasks/chassis/env.py"
    source = path.read_text()
    matches = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
               and ast.get_source_segment(source, node).startswith(prefix)]
    assert len(matches) == 1, prefix
    return compile(ast.Expression(matches[0].value), str(path), "eval")


@pytest.mark.parametrize("mode", [2, 3])
@pytest.mark.parametrize("speed", [1.2, 1.5])
@pytest.mark.parametrize("wheel_min_offset", [0., -.510])
def test_runout_allows_native_continuous_two_second_completion_before_collection_cut(mode, speed, wheel_min_offset):
    config = enabled_config()
    dt = config["policy_dt"]
    settings = validate_runout(config)
    hold_expression = native_assignment("self.success_hold = torch.where(success_now,")
    success_expression = native_assignment("success = self.success_hold >= self.cfg.get(")
    full = FullTaskSemantics(1, "cpu", dt, config["task_semantics"])
    contact = torch.ones(1, 2, dtype=torch.bool)
    stable = torch.ones(1, dtype=torch.bool)
    phase = torch.tensor([Phase.GROUND])
    route_goal = torch.tensor([1.8])
    mode_tensor = torch.tensor([mode])
    command_height = torch.tensor([.305])

    def replay(positive_boundary):
        state = SimpleNamespace(cfg=config, policy_dt=dt, success_hold=torch.zeros(1))
        first_gate = None
        for tick in range(1, 1000):
            # These are ideal CPU trajectories, including the conservative
            # rear-wheel/root offset. No simulated contact qualification follows.
            x = -1.5 + speed * tick * dt
            local = torch.tensor([[x, 0., .305]])
            wheels = torch.tensor([[x + wheel_min_offset, x + wheel_min_offset]])
            success_now = full.completion(mode_tensor, local, wheels, route_goal,
                contact, stable, phase, command_height, torch.tensor([speed]))
            if bool(success_now.item()) and first_gate is None:
                first_gate = tick
            namespace = dict(self=state, success_now=success_now, torch=torch)
            state.success_hold = eval(hold_expression, namespace)
            success = eval(success_expression, namespace)
            boundary = torch.tensor([x > positive_boundary])
            terminated, timeout, success = episode_outcome(
                dict(boundary=boundary, fall=torch.zeros(1, dtype=torch.bool)),
                success, torch.zeros(1, dtype=torch.bool))
            if bool((terminated | timeout | success).item()):
                return dict(success=success.item(), timeout=timeout.item(),
                    x=x, hold=state.success_hold.item(), ticks=tick-first_gate+1)
        raise AssertionError("No native episode outcome")

    before = replay(3.6)
    after = replay(settings["positive_boundary_x_m"])
    assert before["timeout"] and not before["success"] and before["hold"] < 2.
    assert after["success"] and not after["timeout"]
    assert after["ticks"] == 101  # Native float32 needs the 101st 20ms increment.
    assert after["x"] < settings["positive_boundary_x_m"]
    assert after["x"] < settings["support_end_x_m"]
    assert config["task_semantics"] == WORKER["task_semantics"]


def mechanical_spec():
    for relative, expected in MECHANICAL_INPUT_SHA.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    return json.loads((ROOT / WORKER["asset_directory"] / "model_spec.json").read_text())


@pytest.mark.parametrize("side,names,expected", [
    ("L", ("L_joint1", "L_joint2", "L_joint3"), .5095449309016841),
    ("R", ("R_joint1", "R_jonit2", "R_joint3"), .5091155938588481),
])
def test_current_asset_rigid_chain_and_whole_wheel_fit_the_six_tenths_padding(side, names, expected):
    """Bound applies to these SHA-bound rigid chains, not dynamic qualification.

    All downstream rotations preserve the local z axis. Root-to-hip translation
    in the hip zero frame and the knee, wheel, collision offsets therefore split
    into orthogonal planar and axial components. The planar triangle inequality
    gives center norm <= hypot(planar_sum, axial_sum). Any root rotation preserves
    that norm and hence bounds the world x projection.

    Left center bound .509544931m + cylinder circumradius .061288254m
    < .572m < the 6.2-5.6=.6m support padding. Cylinder half-thickness participates
    in its circumradius; checking radius alone would understate arbitrary tilt.
    """
    spec = mechanical_spec()
    joints = {joint["name"]: joint for joint in spec["joints"]}
    bodies = {body["name"]: body for body in spec["bodies"]}
    hip, knee, wheel = [joints[name] for name in names]
    assert (hip["parent"], hip["child"], knee["parent"], knee["child"],
            wheel["parent"], wheel["child"]) == (
        "base_link", side + "_link1", side + "_link1", side + "_link2",
        side + "_link2", side + "_link3")
    H, K, W = [np.asarray(joint["origin"]) for joint in (hip, knee, wheel)]
    collision = bodies[side + "_link3"]["collisions"][0]
    C = np.asarray(collision["origin"])
    for joint in (hip, knee, wheel):
        axis = np.asarray(joint["axis"])
        assert np.linalg.norm(axis[:2]) < 1e-15
        assert abs(abs(axis[2]) - 1.) < 1e-15
    for matrix in (K, W, C):
        np.testing.assert_allclose(matrix[:2, 2], 0., atol=1e-15)
        np.testing.assert_allclose(matrix[2, :2], 0., atol=1e-15)
        assert abs(matrix[2, 2] - 1.) < 1e-15
    for matrix in (H, K, W, C):
        np.testing.assert_allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-15)
    translation = H[:3, :3].T @ H[:3, 3]
    planar = (np.linalg.norm(translation[:2]) + np.linalg.norm(K[:2, 3])
              + np.linalg.norm(W[:2, 3]) + np.linalg.norm(C[:2, 3]))
    axial = translation[2] + K[2, 3] + W[2, 3] + C[2, 3]
    center_bound = math.hypot(planar, axial)
    assert center_bound == pytest.approx(expected, abs=1e-12)
    assert center_bound < .510
    urdf = ET.parse(ROOT / WORKER["asset_directory"] / "robot.urdf").getroot()
    cylinder = urdf.find(f"./link[@name='{side}_link3']/collision/geometry/cylinder")
    assert cylinder is not None
    assert float(cylinder.get("radius")) == collision["radius"]
    assert float(cylinder.get("length")) == collision["length"]
    collision_bound = math.hypot(collision["radius"], collision["length"] / 2.)
    assert collision_bound == pytest.approx(.06128825339864575, abs=1e-12)
    padding = validate_runout(enabled_config())["support_end_x_m"] - 5.6
    assert center_bound + collision_bound < .572 < padding


def test_legal_closed_chain_pose_is_a_counterexample_to_arbitrary_four_tenths_offset():
    """A valid pose can exceed .4m after rotating the root; no stable/contact claim.

    Execute the existing source functions through AST to avoid importing mesh
    generation dependencies. This preserves their actual FK and closure solve.
    """
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    spec = mechanical_spec()
    namespace = dict(np=np, Rotation=Rotation, least_squares=least_squares)
    for relative, names in (("tools/v5_mechanism.py", ("point", "fk")),
        ("tools/build_v5_closedchain.py", ("closure_error", "solve_pose"))):
        path = ROOT / relative
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    joints = {joint["name"]: joint for joint in spec["joints"]}
    knee = float(joints["L_joint2"]["limit"]["upper"])
    names = ("L_joint1", "L_joint2", "L_joint3", "R_joint1", "R_jonit2", "R_joint3")
    q = namespace["solve_pose"](spec,
        dict(zip(names, (.42, knee, 0., -.42, -knee, 0.))), spec["nominal_joint_pos"])
    frames = namespace["fk"](spec, q)
    assert np.abs(namespace["closure_error"](spec, q)).max() < 1e-12
    for joint in spec["joints"]:
        if joint["type"] in ("revolute", "prismatic"):
            assert float(joint["limit"]["lower"]) <= q[joint["name"]] <= float(joint["limit"]["upper"])
    bodies = {body["name"]: body for body in spec["bodies"]}
    offset = np.asarray(bodies["L_link3"]["collisions"][0]["origin"])[:3, 3]
    center = namespace["point"](frames["L_link3"], offset)
    distance = np.linalg.norm(center)
    assert distance == pytest.approx(.4263813927626719, abs=1e-9)
    rotation, _ = Rotation.align_vectors([[1., 0., 0.]], [center / distance])
    root = np.eye(4)
    root[:3, :3] = rotation.as_matrix()
    rotated = namespace["fk"](spec, q, root=root)
    rotated_center = namespace["point"](rotated["L_link3"], offset)
    assert rotated_center[0] == pytest.approx(distance, abs=1e-12)
    assert .4 < rotated_center[0] < .510
