"""Independent CPU-only quaternion-boundary reproduction; no Kit or PhysX.

Run with Python -B. This script prints JSON and never writes source or results.
Small cases use the installed Isaac Lab standardize_xform_ops function and the
frozen runtime's USD readback. The 4096-lane enumeration is a CPU reconstruction
of terrain descriptions and transformed corners, not a native scene readback.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


FROZEN = Path('/tmp/terrain-runout-frozen-v22-1j8x0cux')
REPO = Path('/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule')
INSTALLED = Path('/home/yukikaze/Documents/workspace/robot_rl/.deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab/isaaclab')
NATIVE_REPORT = Path('/tmp/terrain-runout-native-v22-001/report.json')
WORKER = 'contracts/v6_task_campaign_terrain_worker_v22.json'
TOLERANCE_M = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def orientation_expression(path):
    tree = ast.parse(path.read_text())
    expressions = [keyword.value for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'func' and isinstance(node.func.value, ast.Name)
        and node.func.value.id == 'cfg' for keyword in node.keywords
        if keyword.arg == 'orientation']
    assert len(expressions) == 1
    expression = expressions[0]
    code = compile(ast.fix_missing_locations(ast.Expression(expression)), str(path), 'eval')
    return ast.unparse(expression), lambda quat: eval(code, {'__builtins__': {}}, {'quat': quat})


def errors(surface, points, width=4.):
    bounds = [min(p[0] for p in points), max(p[0] for p in points),
              min(p[1] for p in points), max(p[1] for p in points)]
    targets = [surface.x0, surface.x1, -width / 2, width / 2]
    return dict(bounds_m=bounds,
        support_error_m=max(abs(a - b) for a, b in zip(bounds, targets)),
        top_plane_error_m=max(abs(z - surface.height(x, y)) for x, y, z in points))


def main():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    sys.path.insert(0, str(FROZEN / 'src'))
    from wheeled_tasks.chassis import task, terrain_runout, terrain_runout_runtime
    assert Path(task.__file__).resolve() == FROZEN / 'src/wheeled_tasks/chassis/task.py'
    Surface = task.Surface
    transforms = INSTALLED / 'sim/utils/transforms.py'
    spec = importlib.util.spec_from_file_location('independent_installed_transforms', transforms)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    installed_paths = [INSTALLED / 'sim/spawners/shapes/shapes.py',
                       INSTALLED / 'sim/utils/prims.py', transforms]
    frozen_paths = [FROZEN / name for name in (WORKER,
        'src/wheeled_tasks/chassis/task.py', 'src/wheeled_tasks/chassis/env.py',
        'src/wheeled_tasks/chassis/contact_domain.py',
        'src/wheeled_tasks/chassis/terrain_runout.py',
        'src/wheeled_tasks/chassis/terrain_runout_runtime.py')]
    review_paths = [REPO / name for name in ('src/wheeled_tasks/chassis/env.py',
        'src/wheeled_tasks/chassis/task.py', 'tests/test_terrain_runout_runtime.py')]
    paths = installed_paths + frozen_paths + review_paths + [NATIVE_REPORT, Path(__file__).resolve()]
    before = {str(path): sha(path) for path in paths}
    native = json.loads(NATIVE_REPORT.read_text())
    for path in frozen_paths:
        relative = str(path.relative_to(FROZEN))
        expected = native.get('source_files_sha256', {}).get(relative)
        if expected is not None:
            assert sha(path) == expected, relative
    assert sha(FROZEN / WORKER) == native['contract_sha256']
    assert native['source_directory'] == str(FROZEN)
    old_text, old = orientation_expression(FROZEN / 'src/wheeled_tasks/chassis/env.py')
    new_text, new = orientation_expression(REPO / 'src/wheeled_tasks/chassis/env.py')
    assert old((1., 0., 0., 0.)) == (1., 0., 0., 0.)
    assert new((1., 0., 0., 0.)) == (0., 0., 0., 1.)

    def usd_case(surface, convert, bottom=None):
        stage = Usd.Stage.CreateInMemory()
        env = '/World/envs/env_0'
        UsdGeom.Xform.Define(stage, env)
        size, position, quaternion = surface.box(solid_bottom=bottom)
        parent = UsdGeom.Xform.Define(stage, env + '/Terrain/surface_0')
        module.standardize_xform_ops(parent.GetPrim(), translation=position,
                                     orientation=convert(quaternion))
        cube = UsdGeom.Cube.Define(stage, env + '/Terrain/surface_0/geometry/mesh')
        side = min(size)
        cube.CreateSizeAttr(side)
        module.standardize_xform_ops(cube.GetPrim(), scale=tuple(v / side for v in size))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        material = UsdShade.Material.Define(stage, env + '/Terrain/Material')
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(.5)
        api.CreateDynamicFrictionAttr(.5)
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
        points, _, _ = terrain_runout_runtime._mesh_top_vertices(stage, str(cube.GetPath()), env)
        return errors(surface, points)

    tangent = math.tan(math.radians(15))
    cases = [('flat', Surface(-4., 4.), None),
             ('solid_step', Surface(0., 4., z0=.25), -.1),
             ('slope_up', Surface(-1., 1., slope=tangent), None),
             ('slope_down', Surface(-1., 1., slope=-tangent), None),
             ('cross_plus', Surface(-4., 4., cross_slope=tangent), None),
             ('cross_minus', Surface(-4., 4., cross_slope=-tangent), None),
             ('cross_extension', Surface(4., 6.2, cross_slope=tangent), None)]
    samples = []
    for label, surface, bottom in cases:
        previous, corrected = usd_case(surface, old, bottom), usd_case(surface, new, bottom)
        assert corrected['support_error_m'] <= TOLERANCE_M
        assert corrected['top_plane_error_m'] <= TOLERANCE_M
        if label.startswith('cross'):
            assert previous['support_error_m'] > TOLERANCE_M
        samples.append(dict(case=label, old=previous, new=corrected))

    config = json.loads((FROZEN / WORKER).read_text())
    assert not config.get('evaluation_exact_cases')
    assert not config.get('release_stop', {}).get('enabled')
    assert config.get('flat_triangle_mesh') and config.get('evaluation_long_corridors')
    assert not config.get('playback_open_ground')
    rows = task.choose_scene_groups(config['scene_groups'], 4096)
    tiers = config.get('terrain_difficulty_tiers', [1.])
    selected = count = skipped_triangle_lanes = 0
    failures = []
    corrected_support_max = corrected_plane_max = 0.
    for index, (group, kind) in enumerate(rows):
        skill = config['skill_specs'].get(group, {})
        limits = skill.get('terrain_limits', config['terrain_limits'])
        level = skill.get('terrain_level', tiers[(index // len(tiers)) % len(tiers)])
        surfaces = task.terrain_surfaces(kind, limits, level, index)
        if terrain_runout.route_runout_selection(config, group, kind):
            surfaces = terrain_runout.append_route_runout(surfaces, config['terrain_route_runout'])
            selected += 1
        if kind in ('flat', 'jump'):
            skipped_triangle_lanes += 1
            continue
        for surface_index, surface in enumerate(surfaces):
            bottom = (min(s.z0 for s in surfaces) - .1 if config.get('solid_step_platforms')
                and kind in ('low_step', 'step_up', 'step_down', 'platform', 'stairs', 'stairs_down') else None)
            size, position, quaternion = surface.box(width=4., solid_bottom=bottom)
            vertices = [(x, y, size[2] / 2) for x in (-size[0] / 2, size[0] / 2)
                        for y in (-size[1] / 2, size[1] / 2)]
            count += 1
            for mode, convert in [('old', old), ('new', new)]:
                orientation = convert(quaternion)
                # Exact scalar-last conversion found in installed transforms.py.
                rotation = Gf.Matrix3d(Gf.Quatd(orientation[3], *orientation[:3]))
                points = [tuple(Gf.Vec3d(*point) * rotation + Gf.Vec3d(*position))
                          for point in vertices]
                measured = errors(surface, points)
                if mode == 'new':
                    corrected_support_max = max(corrected_support_max, measured['support_error_m'])
                    corrected_plane_max = max(corrected_plane_max, measured['top_plane_error_m'])
                elif measured['support_error_m'] > TOLERANCE_M:
                    failures.append(dict(clone_index=index, group=group, terrain_kind=kind,
                        surface_index=surface_index, slope=surface.slope,
                        cross_slope=surface.cross_slope, **measured))
    assert selected == 1724
    assert failures and {row['terrain_kind'] for row in failures} == {'cross_slope'}
    assert corrected_support_max <= TOLERANCE_M
    assert corrected_plane_max <= TOLERANCE_M
    after = {str(path): sha(path) for path in paths}
    assert before == after
    return dict(status='passed', scope='CPU_USD_API_and_corner_reconstruction_only',
        Kit_started=False, PhysX_started=False, GPU_work_started=False,
        PPO_updates=0, optimizer_steps=0, qualification=None,
        executable=sys.executable, python=sys.version, usd_version=Usd.GetVersion(),
        frozen_source_directory=str(FROZEN),
        frozen_source_commit_from_native_report=native['source_commit'],
        native_failure_report=str(NATIVE_REPORT),
        installed_api_call_chain='spawn_cuboid -> _spawn_geom_from_prim_type -> create_prim -> standardize_xform_ops',
        installed_api_executed='standardize_xform_ops imported directly from installed source',
        api_scope='Small USD cases reproduce Cuboid size=min(size) and scale=size/min(size); no full Isaac spawner or simulator initialization',
        old_boundary_expression=old_text, new_boundary_expression=new_text,
        source_sha256=before, source_files_unchanged_during_execution=True,
        tolerance_m=TOLERANCE_M, usd_cases=samples,
        reconstructed_population=dict(lanes=4096, selected_runout_lanes=selected,
            cuboid_count=count, skipped_flat_or_jump_triangle_lanes=skipped_triangle_lanes,
            scope='Pure CPU reconstruction from frozen config and terrain builders; no USD population or PhysX clone order was read',
            contact_domain_note='ContactDomain.surfaces changes friction only; material validation is outside this geometry enumeration',
            raw_support_failure_count=len(failures),
            raw_support_failed_kinds=sorted({row['terrain_kind'] for row in failures}),
            first_in_construction_order=failures[0],
            first_if_clone_order_is_lexicographic=min(failures, key=lambda row: f"/World/envs/env_{row['clone_index']}"),
            first_lane_evidence='CPU prediction only; original native failure report does not record the failing prim',
            corrected_max_support_error_m=corrected_support_max,
            corrected_max_top_plane_error_m=corrected_plane_max))


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, allow_nan=False))
