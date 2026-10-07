"""CPU-only REC v21 occupied-geometry comparison, with immutable source binding."""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO = Path('/home/yukikaze/Documents/workspace/robot_rl/isaac_wheeled_rl_schedule')
FROZEN = Path('/tmp/scut-entropy-frozen-023-av6rrek4')
INSTALLED = Path('/home/yukikaze/Documents/workspace/robot_rl/.deployment_sources/IsaacLab-3.0.0-beta2/source/isaaclab/isaaclab')
OLD = '6f07ec298e01d2686216787cfba34b394151f1aa'
NEW = 'b0d2800'
WORKER = 'contracts/v6_task_campaign_recovery_worker_v21.json'
RECEIPT = REPO / 'reports/v6_scut_posture_completion_20261006/source_repair_023/first_PPO_verification_023.json'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def git_source(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=REPO)

def orientation(source):
    tree = ast.parse(source)
    nodes = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'func'
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'cfg'
        for kw in node.keywords if kw.arg == 'orientation']
    assert len(nodes) == 1
    code = compile(ast.fix_missing_locations(ast.Expression(nodes[0])), 'source_boundary', 'eval')
    return ast.unparse(nodes[0]), lambda q: eval(code, {'__builtins__': {}}, {'quat': q})

def function(source, name):
    return next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)

def main():
    import torch
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
    sys.path.insert(0, str(FROZEN / 'src'))
    from wheeled_tasks.chassis.task import Surface, choose_scene_groups, corridor_mesh, terrain_surfaces
    from wheeled_tasks.chassis.contact_domain import ContactDomain
    files = [WORKER, 'contracts/v6_task_campaign_v21.json',
             'src/wheeled_tasks/chassis/env.py', 'src/wheeled_tasks/chassis/task.py',
             'src/wheeled_tasks/chassis/contact_domain.py']
    frozen_bytes = {name: (FROZEN / name).read_bytes() for name in files}
    assert all(data == git_source(OLD, name) for name, data in frozen_bytes.items())
    correction = git_source(NEW, 'src/wheeled_tasks/chassis/env.py')
    corrected_task = git_source(NEW, 'src/wheeled_tasks/chassis/task.py')
    for name in ('terrain_surfaces', 'corridor_mesh', 'box'):
        previous = function(frozen_bytes['src/wheeled_tasks/chassis/task.py'], name)
        corrected = function(corrected_task, name)
        for node in (previous, corrected):
            if (isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:]
        assert ast.dump(previous, include_attributes=False) == ast.dump(corrected, include_attributes=False)
    def flat_mesh_branch(source):
        nodes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.If)
                 and any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                         and n.func.attr == 'CreateDoubleSidedAttr' for statement in node.body
                         for n in ast.walk(statement))]
        # Select the narrowest enclosing branch rather than the constructor's
        # optional outer branches, if any.
        return min(nodes, key=lambda node: len(ast.dump(node)))
    assert ast.dump(flat_mesh_branch(frozen_bytes['src/wheeled_tasks/chassis/env.py']),
                    include_attributes=False) == ast.dump(flat_mesh_branch(correction), include_attributes=False)
    old_text, old = orientation(frozen_bytes['src/wheeled_tasks/chassis/env.py'])
    new_text, new = orientation(correction)
    installed_paths = [INSTALLED / name for name in ('sim/utils/transforms.py',
        'sim/utils/prims.py', 'sim/spawners/shapes/shapes.py')]
    paths = [FROZEN / name for name in files] + installed_paths + [RECEIPT, Path(__file__).resolve()]
    before = {str(path): sha(path.read_bytes()) for path in paths}
    module_spec = importlib.util.spec_from_file_location('installed_xforms', installed_paths[0])
    transforms = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(transforms)
    config = json.loads(frozen_bytes[WORKER])
    count, seed = 4096, 290927
    actual_receipt = json.loads(RECEIPT.read_text())
    def commands(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'command' and isinstance(item, list) and any(
                        isinstance(s, str) and s.endswith('/scripts/train_chassis.py') for s in item):
                    yield item
                yield from commands(item)
        elif isinstance(value, list):
            for item in value:
                yield from commands(item)
    worker_commands = list(commands(actual_receipt))
    assert worker_commands and all(cmd[cmd.index('--seed') + 1] == str(seed)
        and cmd[cmd.index('--num-envs') + 1] == str(count) for cmd in worker_commands)
    assert config['flat_triangle_mesh'] and config['evaluation_long_corridors']
    assert config['solid_step_platforms'] and config['contact_domain']['enabled']
    assert not config.get('release_stop') and not config.get('terrain_route_runout')
    assert not config.get('evaluation_exact_cases') and not config.get('playback_open_ground')
    rows = choose_scene_groups(config['scene_groups'], count)
    domains = [ContactDomain(count, config['contact_domain'], seed) for _ in range(2)]
    assert torch.equal(domains[0].mu, domains[1].mu)
    assert torch.equal(domains[0].domain_draw, domains[1].domain_draw)
    conventions = {'old': old, 'corrected': new}
    signs = list(itertools.product((-1, 1), repeat=3))
    faces = [[i for i, xyz in enumerate(signs) if xyz[axis] == sign]
             for axis in range(3) for sign in (-1, 1)]

    @lru_cache(None)
    def collider_pair(size, position, quaternion, mu):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, '/World')
        result = {}
        for label, convert in conventions.items():
            root = '/World/' + label
            prim = UsdGeom.Xform.Define(stage, root)
            transforms.standardize_xform_ops(prim.GetPrim(), translation=position,
                                             orientation=convert(quaternion))
            cube = UsdGeom.Cube.Define(stage, root + '/geometry/mesh')
            side = min(size)
            cube.CreateSizeAttr(side)
            transforms.standardize_xform_ops(cube.GetPrim(), scale=tuple(d / side for d in size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            material = UsdShade.Material.Define(stage, root + '/geometry/material')
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr(mu)
            api.CreateDynamicFrictionAttr(mu)
            api.CreateRestitutionAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, materialPurpose='physics')
            resolved, _ = UsdShade.MaterialBindingAPI(cube.GetPrim()).ComputeBoundMaterial('physics')
            resolved_api = UsdPhysics.MaterialAPI(resolved.GetPrim())
            matrix, reset = UsdGeom.XformCache().ComputeRelativeTransform(cube.GetPrim(), stage.GetPrimAtPath('/World'))
            assert not reset
            points = [tuple(matrix.Transform(Gf.Vec3d(*(s * side / 2 for s in xyz)))) for xyz in signs]
            result[label] = dict(vertices=sorted(points),
                ordered_vertices=points,
                faces=sorted(tuple(sorted(points[i] for i in ids)) for ids in faces),
                local_positive_z_vertices=sorted(points[i] for i, s in enumerate(signs) if s[2] == 1),
                material=[resolved_api.GetStaticFrictionAttr().Get(),
                          resolved_api.GetDynamicFrictionAttr().Get(), resolved_api.GetRestitutionAttr().Get()],
                collision_enabled=UsdPhysics.CollisionAPI(cube.GetPrim()).GetCollisionEnabledAttr().Get(),
                material_relative_path=str(resolved.GetPath())[len(root):])
        # For the actual REC horizontal boxes, this is an exact sign permutation
        # from the stored quaternion, independent of Gf's sin(pi) matrix rounding.
        assert quaternion == (1., 0., 0., 0.)
        assert old(quaternion) == (1., 0., 0., 0.)
        assert new(quaternion) == (0., 0., 0., 1.)
        algebraic_old = [tuple(position[a] + xyz[a] * size[a] / 2 * (1 if a == 0 else -1)
                              for a in range(3)) for xyz in signs]
        algebraic_new = [tuple(position[a] + xyz[a] * size[a] / 2 for a in range(3)) for xyz in signs]
        assert sorted(algebraic_old) == sorted(algebraic_new)
        assert sorted(tuple(sorted(algebraic_old[i] for i in ids)) for ids in faces) == sorted(
            tuple(sorted(algebraic_new[i] for i in ids)) for ids in faces)
        mapping = [signs.index((xyz[0], -xyz[1], -xyz[2])) for xyz in signs]
        result['usd_corresponding_vertex_max_error_m'] = max(abs(a - b)
            for i, j in enumerate(mapping) for a, b in zip(result['old']['ordered_vertices'][i],
                                                          result['corrected']['ordered_vertices'][j]))
        result['algebraic_complete_vertex_and_face_sets_exact'] = True
        return result

    summary = defaultdict(lambda: dict(lanes=0, cuboids=0, triangle_meshes=0,
        vertices_exact=True, faces_exact=True, material_exact=True, filters_exact=True))
    all_surfaces = []
    cuboids = local_top_differences = 0
    exact_vertex_mismatches = exact_face_mismatches = 0
    usd_corresponding_vertex_max_error = 0.
    filters_old, filters_new = [], []
    nominal_to_actual_top_error = 0.
    local_top_label_max_error = 0.
    group_materials = defaultdict(list)
    for lane, (group, kind) in enumerate(rows):
        record = summary[group]
        record['terrain_kind'] = kind
        record['lanes'] += 1
        spec = config['skill_specs'].get(group, {})
        limits = spec.get('terrain_limits', config['terrain_limits'])
        tiers = config.get('terrain_difficulty_tiers', [1.])
        level = spec.get('terrain_level', tiers[(lane // len(tiers)) % len(tiers)])
        original = terrain_surfaces(kind, limits, level, lane)
        if kind in ('flat', 'jump'):
            original = [Surface(-config['flat_half_length_m'], config['flat_half_length_m'])]
        old_surfaces, new_surfaces = [domain.surfaces(original, lane) for domain in domains]
        assert old_surfaces == new_surfaces
        assert all(s.slope == s.cross_slope == 0. for s in original)
        all_surfaces.append(old_surfaces)
        record['surfaces'] = [dict(x0=s.x0, x1=s.x1, z0=s.z0, slope=s.slope,
                                   cross_slope=s.cross_slope) for s in original]
        group_materials[group].extend(s.friction for s in old_surfaces)
        lane_filters = [f'/World/envs/env_{lane}/Terrain/surface_{j}/geometry/mesh'
                        for j in range(len(old_surfaces))]
        filters_old.extend(lane_filters)
        filters_new.extend([f'/World/envs/env_{lane}/Terrain/surface_{j}/geometry/mesh'
                            for j in range(len(new_surfaces))])
        if kind in ('flat', 'jump'):
            points, indices = corridor_mesh(config['flat_half_length_m'], config['flat_floor_width_m'])
            assert all(point[2] == 0. for point in points) and indices
            record['triangle_meshes'] += 1
            continue
        bottom = min(s.z0 for s in original) - .1
        for surface in old_surfaces:
            size, position, quaternion = surface.box(width=4., solid_bottom=bottom)
            pair = collider_pair(size, position, quaternion, surface.friction)
            a, b = pair['old'], pair['corrected']
            record['vertices_exact'] &= a['vertices'] == b['vertices']
            record['faces_exact'] &= a['faces'] == b['faces']
            record['material_exact'] &= a['material'] == b['material']
            exact_vertex_mismatches += a['vertices'] != b['vertices']
            exact_face_mismatches += a['faces'] != b['faces']
            usd_corresponding_vertex_max_error = max(usd_corresponding_vertex_max_error,
                pair['usd_corresponding_vertex_max_error_m'])
            assert pair['algebraic_complete_vertex_and_face_sets_exact']
            assert a['material'] == b['material'] and a['collision_enabled'] == b['collision_enabled'] == True
            assert a['material_relative_path'] == b['material_relative_path']
            nominal_to_actual_top_error = max(nominal_to_actual_top_error,
                abs(max(p[2] for p in a['vertices']) - surface.z0))
            local_top_label_max_error = max(local_top_label_max_error,
                max(abs(p[2] - surface.z0) for p in a['local_positive_z_vertices']))
            local_top_differences += a['local_positive_z_vertices'] != b['local_positive_z_vertices']
            cuboids += 1
            record['cuboids'] += 1
    assert filters_old == filters_new
    for group, item in summary.items():
        item['friction_min'] = min(group_materials[group])
        item['friction_max'] = max(group_materials[group])
    query_fn = function(frozen_bytes['src/wheeled_tasks/chassis/env.py'], 'ground_height')
    assert ast.dump(query_fn, include_attributes=False) == ast.dump(function(correction, 'ground_height'), include_attributes=False)
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[query_fn], type_ignores=[])), 'frozen_ground_height', 'exec'), namespace)
    maximum = max(map(len, all_surfaces))
    data = torch.zeros(count, maximum, 5)
    valid = torch.zeros(count, maximum, dtype=torch.bool)
    for i, surfaces in enumerate(all_surfaces):
        data[i, :len(surfaces)] = torch.tensor([[s.x0, s.x1, s.z0, s.slope, s.cross_slope] for s in surfaces])
        valid[i, :len(surfaces)] = True
    old_env = SimpleNamespace(surface_data=data, surface_valid=valid)
    new_env = SimpleNamespace(surface_data=data.clone(), surface_valid=valid.clone())
    queries = 0
    for x in (-4., -.001, 0., .001, .799, .8, .801, 1.399, 1.4, 1.401, 3.999, 4.):
        for y in (-1.7, 0., 1.7):
            xy = data.new_tensor([x, y]).expand(count, 2)
            assert torch.equal(namespace['ground_height'](old_env, xy), namespace['ground_height'](new_env, xy))
            queries += count
    assert before == {str(path): sha(path.read_bytes()) for path in paths}
    return dict(status='passed', scope='CPU source-bound USD occupied-geometry and analytic query comparison',
        qualification=None, GPU_work_started=False, Kit_started=False, PhysX_started=False,
        PPO_updates=0, optimizer_steps=0, seed=seed, source_sha256=before,
        frozen_files_match_commit=OLD, corrected_commit=subprocess.check_output(
            ['git', 'rev-parse', NEW], cwd=REPO, text=True).strip(),
        corrected_env_sha256=sha(correction), old_orientation=old_text, corrected_orientation=new_text,
        corrected_task_sha256=sha(corrected_task),
        terrain_builders_and_box_AST_unchanged_except_docstrings=True,
        flat_triangle_mesh_spawn_branch_AST_unchanged=True,
        installed_api_executed='standardize_xform_ops; Cuboid sizing replicated from installed spawn_cuboid',
        group_count=len(summary), group_kind_counts=dict(Counter(r['terrain_kind'] for r in summary.values())),
        lane_count=count, lane_kind_counts=dict(Counter(k for _, k in rows)),
        cuboids_compared=cuboids, complete_vertices_per_cuboid=8, complete_faces_per_cuboid=6,
        cached_USD_pairs=collider_pair.cache_info().currsize,
        exact_USD_vertex_set_mismatches=exact_vertex_mismatches,
        exact_USD_face_set_mismatches=exact_face_mismatches, exact_material_mismatches=0,
        USD_corresponding_vertex_max_error_m=usd_corresponding_vertex_max_error,
        algebraic_complete_vertex_and_face_sets_exact=True,
        numerical_note='USD XformCache computes a sin(pi) residual for the old 180-degree X quaternion; actual floating-point vertex/face arrays are not bitwise equal. No geometric tolerance was loosened or used to hide this negative result.',
        unchanged_triangle_mesh_lanes=sum(s['triangle_meshes'] for s in summary.values()),
        local_positive_z_face_labels_changed=local_top_differences,
        old_local_positive_z_label_max_error_m=local_top_label_max_error,
        actual_occupied_top_vs_nominal_max_error_m=nominal_to_actual_top_error,
        analytic_query_samples=queries, analytic_queries_bitwise_equal=True,
        geometry_filter_paths_equal=True, unique_filter_path_count=len(filters_old),
        geometry_filter_paths_sha256=sha(json.dumps(filters_old).encode()),
        contact_scope='Algebraically same occupied convex boxes and complete face sets; reported USD rounding differences retained; same whole-prim isotropic friction/restitution, material-relative bindings and filter paths; no actual PhysX force or contact-slot comparison',
        production_material_settings=dict(friction_combine_mode='multiply', restitution=0.,
                                          static_equals_dynamic=True, per_face_material=False),
        scope_limit='Local CPU reconstruction bound to verified source; not a current remote USD/contact readback and not a guarantee of identical future PhysX floating-point trajectories',
        conclusion='This quaternion bug alone does not change occupied terrain geometry/material/query semantics for current REC v21. It supplies no geometry-based reason for an early REC migration.',
        groups=dict(sorted(summary.items())))

if __name__ == '__main__':
    print(json.dumps(main(), indent=2, allow_nan=False))
