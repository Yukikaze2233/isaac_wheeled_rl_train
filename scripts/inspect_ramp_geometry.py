#!/usr/bin/env python3
"""Author and read back local ramp collision meshes without a PPO worker."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from wheeled_tasks.chassis.ramp_flight import RampFlightGeometry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(output):
    output.mkdir(parents=True, exist_ok=False)
    usd_file = output / "ramp_collision_candidate.usda"
    stage = Usd.Stage.CreateNew(str(usd_file))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomData({"scope": "geometry_authoring_only", "qualification": "unqualified"})
    cases = []
    for index, (direction, gap) in enumerate(((1, .15), (-1, .15), (1, .650123), (-1, .650123))):
        geometry = RampFlightGeometry(direction=direction, gap_m=gap)
        name = f"route_{index}"
        path = f"/World/{name}"
        UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(0., 3. * index, 0.))
        for slot, (points, faces) in enumerate(geometry.meshes()):
            mesh = UsdGeom.Mesh.Define(stage, path + f"/Terrain/surface_{slot}/geometry/mesh")
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
            mesh.CreateFaceVertexCountsAttr([4] * 6)
            mesh.CreateFaceVertexIndicesAttr(faces)
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateDoubleSidedAttr(False)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        cases.append({"name": name, "geometry": asdict(geometry)})
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(usd_file))
    collision_count, max_error = 0, 0.
    for case in cases:
        geometry = RampFlightGeometry(**case["geometry"])
        for slot, (points, faces) in enumerate(geometry.meshes()):
            path = f"/World/{case['name']}/Terrain/surface_{slot}/geometry/mesh"
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
            actual = np.asarray(mesh.GetPointsAttr().Get())
            error = float(np.max(np.abs(actual - np.asarray(points))))
            max_error = max(max_error, error)
            if error > 3e-7:
                raise RuntimeError(f"Ramp vertex readback mismatch at {path}: {error}")
            if list(mesh.GetFaceVertexCountsAttr().Get()) != [4] * 6:
                raise RuntimeError("Ramp face count readback mismatch")
            if tuple(mesh.GetFaceVertexIndicesAttr().Get()) != faces:
                raise RuntimeError("Ramp face winding readback mismatch")
            prim = mesh.GetPrim()
            if (not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none"):
                raise RuntimeError("Ramp must retain native static mesh collision")
            collision_count += 1
    root = Path(__file__).resolve().parents[1]
    report = {"status": "native_usd_geometry_readback_passed", "qualification": None,
              "at_utc": datetime.now(timezone.utc).isoformat(), "cases": cases,
              "collision_meshes": collision_count, "vertex_max_abs_error_m": max_error,
              "usd_file": usd_file.name, "usd_sha256": sha(usd_file),
              "source_files_sha256": {str(path.relative_to(root)): sha(path) for path in (
                  Path(__file__).resolve(), root / "src/wheeled_tasks/chassis/ramp_flight.py")},
              "physics_steps": 0, "formal_updates_consumed": 0,
              "limitations": ["No robot or PhysX motion was run", "Not a deployed curriculum",
                              "Does not establish flight or landing capability"]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "status", "collision_meshes", "vertex_max_abs_error_m", "usd_file", "formal_updates_consumed")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    inspect(parser.parse_args().output)
