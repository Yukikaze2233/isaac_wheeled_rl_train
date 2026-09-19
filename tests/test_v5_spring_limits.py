"""Independent compiled mount-frame check of the knee/spring usable range."""
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/纯底盘_v5/urdf"


def test_reachability_matches_mujoco_mount_frames():
    module_spec = importlib.util.spec_from_file_location("limits_audit", ROOT / "tools/analyze_v5_spring_limits.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = json.loads((BUNDLE / "model_spec.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(BUNDLE / "robot.xml"))
    data = mujoco.MjData(model)
    for side, knee in (("L", "L_joint2"), ("R", "R_jonit2")):
        values, result = module.side_geometry(spec, side)
        assert result["mechanical_minimum_knee_deg"] == pytest.approx(35.668681, abs=1e-5)
        assert result["recommended_minimum_knee_deg"] == pytest.approx(48.671033, abs=1e-5)
        for angle in (35., 45., 60., 80.):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[model.joint(knee).qposadr[0]] = values(angle)["knee_raw_rad"]
            # Downstream passive states are intentionally not re-solved: both
            # mount centers depend only on the thigh and shank joint frames.
            mujoco.mj_forward(model, data)
            upper = data.site(side + "_spring_upper_mount_0").xpos
            lower = data.body(side * 3 + "_link2").xpos
            assert np.linalg.norm(upper - lower) == pytest.approx(values(angle)["pin_distance_m"], abs=1e-10)
        assert values(35.)["compression_m"] > result["stroke_m"]
        assert values(50.)["compression_m"] < .9 * result["stroke_m"]


def test_standing_height_matches_compiled_wheel_contact_geometry():
    module_spec = importlib.util.spec_from_file_location("standing_limits_audit", ROOT / "tools/analyze_v5_spring_limits.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = json.loads((BUNDLE / "model_spec.json").read_text())
    _, limits = module.side_geometry(spec, "L")
    model = mujoco.MjModel.from_xml_path(str(BUNDLE / "robot.xml"))
    data = mujoco.MjData(model)
    heights = []
    for angle in (limits["mechanical_minimum_knee_deg"], limits["recommended_minimum_knee_deg"], 80.):
        pose = module.balanced_standing_pose(spec, angle)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        data.qpos[:3] = [0., 0., pose["base_frame_height_m"]]
        roll = pose["base_roll_rad"]
        data.qpos[3:7] = [np.cos(roll / 2), np.sin(roll / 2), 0., 0.]
        for name, value in pose["joint_positions"].items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        wheel_x = []
        for side in ("L", "R"):
            name = side + "_link3_collision_000"
            geometry = data.geom(name)
            size = model.geom(name).size
            axial_z = geometry.xmat.reshape(3, 3)[2, 2]
            extent = size[0] * np.sqrt(1 - axial_z ** 2) + size[1] * abs(axial_z)
            assert geometry.xpos[2] - extent == pytest.approx(0., abs=1e-9)
            wheel_x.append(geometry.xpos[0])
        com_x = sum(b["mass"] * data.body(b["name"]).xipos[0] for b in spec["bodies"]) / sum(b["mass"] for b in spec["bodies"])
        assert com_x == pytest.approx(np.mean(wheel_x), abs=1e-9)
        heights.append(pose["base_frame_height_m"])
    assert heights[0] < heights[1] < heights[2]
