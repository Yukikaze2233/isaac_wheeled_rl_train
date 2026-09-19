"""Check the bench feedforward against finite work in the compiled mechanism."""
import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "model/纯底盘_v5/urdf"


def test_load_feedforward_matches_compiled_virtual_work():
    module_spec = importlib.util.spec_from_file_location("load_bench", ROOT / "scripts/compare_v5_spring_load.py")
    bench = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(bench)
    spec = json.loads((BUNDLE / "model_spec.json").read_text())
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    fit = json.loads((BUNDLE / "fit_10mpa.json").read_text())
    trial = bench.prepare_trial(spec, manifest, fit, 67.85497)
    assert abs(trial["wheel_height_difference_m"]) < 1e-10
    model = mujoco.MjModel.from_xml_path(str(BUNDLE / "robot.xml"))
    data = mujoco.MjData(model)
    names = manifest["control_joint_names"]
    step = 1e-5

    def work(q, gas):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        roll = trial["root_roll_rad"]
        data.qpos[3:7] = [np.cos(roll / 2), np.sin(roll / 2), 0., 0.]
        for joint, value in q.items():
            data.qpos[model.joint(joint).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        gravity = sum(model.body(b["name"]).mass[0] * 9.81 * data.body(b["name"]).xipos[2]
                      for b in spec["bodies"] if gas or b["name"] not in bench.SPRING_BODIES)
        ground = sum(force * data.geom(name + "_collision_000").xpos[2]
                     for force, name in zip(trial["expected_wheel_loads_n"][gas], ("L_link3", "R_link3")))
        spring = sum(force * q[name] for force, name in zip(trial["spring_force_n"], manifest["spring_joint_names"]))
        return gravity - ground - gas * spring

    for index in (0, 1, 3, 4):
        solved = []
        for sign in (-1, 1):
            prescribed = {n: trial["joint_positions"][n] for n in names}
            prescribed[names[index]] += sign * step
            solved.append(bench.solve_pose(spec, prescribed, trial["joint_positions"]))
        for gas in (0, 1):
            finite_work = (work(solved[1], gas) - work(solved[0], gas)) / (2 * step)
            assert trial["feedforward_nm"][gas][index] == pytest.approx(finite_work, abs=2e-4)
