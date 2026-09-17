#!/usr/bin/env python3
"""Read actual SCUT USD spring/loop connections without changing its assets."""
import argparse
import hashlib
import json
from pathlib import Path

from pxr import Usd, UsdPhysics


def inspect(path):
    stage = Usd.Stage.Open(str(path))
    rows = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        bodies = [[str(p) for p in rel.GetTargets()] for rel in (joint.GetBody0Rel(), joint.GetBody1Rel())]
        if "spring" not in str(prim.GetPath()).lower() and not any("spring" in p.lower() for pair in bodies for p in pair):
            continue
        row = {"path": str(prim.GetPath()), "type": prim.GetTypeName(), "bodies": bodies,
               "excluded_from_articulation": joint.GetExcludeFromArticulationAttr().Get(),
               "local_pos0": list(joint.GetLocalPos0Attr().Get()),
               "local_pos1": list(joint.GetLocalPos1Attr().Get())}
        if prim.IsA(UsdPhysics.PrismaticJoint):
            prism = UsdPhysics.PrismaticJoint(prim)
            row.update(axis=prism.GetAxisAttr().Get(), lower_m=prism.GetLowerLimitAttr().Get(), upper_m=prism.GetUpperLimitAttr().Get())
        rows.append(row)
    return {"asset": str(path), "root_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "spring_joints": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    path = root / "scut-wheeled-legged-rl/source/agent_world/agent_world/assets/usd_files/wheelbipeV14_2_1/wheelbipeV14_2.usd"
    report = inspect(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
