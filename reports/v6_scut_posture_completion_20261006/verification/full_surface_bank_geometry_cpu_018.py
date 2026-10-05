#!/usr/bin/env python3
"""Deterministic CPU ordinary-surface × sealed-bank × angle reset geometry."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import time

OUT = Path(__file__).resolve().parent
DOMAINS = ("recovery", "speed", "terrain", "jump", "stepjump")
ANGLES = [(0., 0.), *[(r, p) for cap in (.03, .09, .15) for r in (-cap, cap) for p in (-cap, cap)],
          (-.15, 0.), (.15, 0.), (0., -.15), (0., .15)]

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

def catalog(config):
    scenes = choose_scene_groups(config["scene_groups"], 4096)
    groups, kinds = zip(*scenes)
    specs, tiers = config["skill_specs"], config.get("terrain_difficulty_tiers", [1.])
    excluded_recovery = torch.tensor([bool(specs[g].get("recovery_pose")) or specs[g].get("kind") == "recovery" for g in groups])
    excluded_spin = torch.tensor([specs[g].get("kind") == "spin_translate" for g in groups])
    eligible = ordinary_eligibility(config, groups, kinds, "cpu", recovery=excluded_recovery, spin=excluded_spin)
    width = config.get("flat_floor_width_m", 4.)
    spacing = 2 * config.get("flat_half_length_m", 44.) + 12.
    columns = max(1, math.ceil(math.sqrt(4096 * (width + 1.) / spacing)))
    rows = math.ceil(4096 / columns)
    unique, families = {}, Counter()
    for i, (group, kind) in enumerate(scenes):
        if not bool(eligible[i]): continue
        spec = specs[group]; limits = spec.get("terrain_limits", config["terrain_limits"])
        level = spec.get("terrain_level", tiers[(i // len(tiers)) % len(tiers)])
        surfaces = terrain_surfaces(kind, limits, level, i)
        if config.get("release_stop", {}).get("enabled") and spec.get("kind") == "release_stop" and kind != "flat":
            surfaces = release_stop_surfaces(kind, limits, level, i, config["release_stop"])
        if config.get("evaluation_long_corridors") and kind in ("flat", "jump"):
            half = config.get("flat_half_length_m", 44.)
            surfaces = [Surface(-half, half)] if config.get("flat_triangle_mesh") else [Surface(x - 4., x + 4.) for x in range(-40, 41, 8)]
        data = torch.tensor([[s.x0, s.x1, s.z0, s.slope, s.cross_slope] for s in surfaces], dtype=torch.float32)
        x = 0. if config.get("centered_locomotion_resets") and kind in ("flat", "jump") else -1.5
        if config.get("release_stop", {}).get("enabled") and spec.get("kind") == "release_stop": x = 0.
        if group in ("stand", "translate", "rotate") and kind in ("slope", "material", "rough"): x = 0.
        elif kind == "platform": x = 1.
        elif group in ("step_up", "step_down"): x = -.8
        if spec.get("body_climb") or spec.get("step_jump"): x = -.8
        if config.get("command_reference") and spec.get("kind") in ("step_up", "stairs", "step_down", "stairs_down"):
            lo, hi = config["command_reference"].get("step_start_x_range_m", [-1.5, -1.5])
            xs = torch.linspace(lo, hi, 9).tolist()
        else: xs = [x]
        origin = (((i % columns) - (columns - 1) / 2) * spacing,
                  ((i // columns) - (rows - 1) / 2) * (width + 1.), 0.) if config.get("evaluation_long_corridors") else ((i % 64) * 10., (i // 64) * 10., 0.)
        origin = torch.tensor(origin, dtype=torch.float32)
        for x in xs:
            world = origin.clone(); world[0] += x
            local = world[:2] - origin[:2]
            key = (tuple(data.flatten().tolist()), tuple(local.tolist()))
            if key not in unique:
                unique[key] = dict(lane=i, group=group, kind=kind, surface=data, origin=origin, world=world)
            families[kind] += 1
    return list(unique.values()), dict(ordinary_native_lanes=int(eligible.sum()),
        ordinary_lane_position_combinations=sum(families.values()), terrain_family_counts=dict(families),
        unique_surface_position_combinations=len(unique), deduplication="identical float32 terrain strips and realized local XY only")

def geometry_requirements(lesson, root, origins, profiles, surfaces, valid):
    q, local = root[:, 3:7], root[:, :2] - origins[:, :2]
    wheels = rotate_vectors(q, lesson.wheel_points[profiles]); axes = rotate_vectors(q, lesson.wheel_axes[profiles])
    extent = lesson.wheel_radius[None, :, None] * (1. - axes.square()).clamp_min(0.).sqrt()
    extent += lesson.wheel_halfwidth[None, :, None] * axes.abs()
    required_wheels = []
    for side in (0, 1):
        bounds = torch.stack((-extent[:, side, 0], extent[:, side, 0], -extent[:, side, 1], extent[:, side, 1]), -1)
        plane = support_height_for_footprint(surfaces, valid, local + wheels[:, side, :2], bounds)
        required_wheels.append(plane - wheels[:, side, 2] + extent[:, side, 2])
    points = rotate_vectors(q, lesson.collision_corners[profiles]); lo, hi = points.amin(-2), points.amax(-2)
    required_bodies = []
    for shape in range(lo.shape[1]):
        bounds = torch.stack((lo[:, shape, 0], hi[:, shape, 0], lo[:, shape, 1], hi[:, shape, 1]), -1)
        plane = support_height_for_footprint(surfaces, valid, local, bounds)
        required_bodies.append(plane - lo[:, shape, 2])
    return torch.stack(required_wheels, -1), torch.stack(required_bodies, -1)

def verify(lesson, root, joints, origins, profiles, surfaces, valid):
    assert torch.equal(joints, lesson.profile_positions[profiles])
    wheel, body = geometry_requirements(lesson, root, origins, profiles, surfaces, valid)
    z = root[:, 2] - origins[:, 2]; wheel_gap = z[:, None] - wheel
    body_clearance = z[:, None] - body
    assert float(wheel_gap.min()) >= -2e-6
    assert float(body_clearance.min()) >= .001 - 2e-6
    assert float(wheel_gap.amin(-1).max()) <= .001 + 2e-6
    assert torch.allclose(root[:, 3:7].norm(dim=-1), torch.ones(len(root)), atol=1e-6, rtol=0.)
    return float(wheel_gap.min()), float(body_clearance.min()), float(wheel_gap.amin(-1).max())

def exact_fixture(config, template, source, fixture):
    row = fixture["rows"][0]; names = fixture["joint_names"]
    bank = NonstandardStart(1, "cpu", .005, config["nonstandard_start"], source, names)
    lesson = OrdinaryPostureCurriculum(1, "cpu", config["ordinary_posture_curriculum"], torch.ones(1, dtype=torch.bool))
    for key in CACHE: setattr(lesson, key, getattr(template, key))
    joints = torch.tensor([row["nominal_joints"]]); index = int((lesson.profile_positions - joints).abs().amax(-1).argmin())
    assert float((lesson.profile_positions[index] - joints[0]).abs().max()) < 2e-6
    bank.profile_ids[0] = index - 1; bank.level_ids[0] = 1; bank.selected[0] = True
    bank.remaining_steps[0] = 9; bank.elapsed[0] = .03; bank.episode_steps[0] = 8
    bank.held_interval[0] = True; bank.learning_mask[0] = False
    root = torch.tensor([row["root_pose"]]); origins = torch.tensor([[root[0, 0] - row["local_xy"][0], root[0, 1] - row["local_xy"][1], 0.]])
    surfaces = torch.tensor([row["surfaces"]]); valid = torch.ones(1, len(row["surfaces"]), dtype=torch.bool)
    lesson.selected[0] = True
    cancelled = lesson.align_support(torch.tensor([0]), root, joints, origins, surfaces, valid, profile_ids=bank.profile_ids)
    assert cancelled.tolist() == [0]; bank.cancel_reset(cancelled)
    assert bank.profile_ids.item() == -1 and not bank.holding.any() and not bank.capturing.any()
    assert bank.elapsed.item() == 0. and bank.episode_steps.item() == 0
    assert bank.held_interval.item() and not bank.learning_mask.item()
    proof = verify(lesson, root, joints, origins, bank.profile_ids + 1, surfaces, valid)
    bank.begin_policy(); assert bank.learning_mask.item() and not bank.held_interval.item()
    lesson.selected[0] = True
    bad = torch.cat((surfaces, torch.tensor([[[.18, .20, .3, 0., 0.]]])), 1)
    try: lesson.align_support(torch.tensor([0]), root.clone(), joints.clone(), origins, bad, torch.ones(1, bad.shape[1], dtype=torch.bool), profile_ids=bank.profile_ids)
    except ValueError as error: assert "no collision-free wheel support" in str(error)
    else: raise AssertionError("Genuine nominal collision was accepted")
    return dict(status="passed", captured_lane=row["lane"], original_bank_profile=bank.profile_names[index - 1],
        captured_invalid_gap_m=row["gap"], fallback_full_nominal_q=True, final_profile_id=-1,
        final_root_pose=root.tolist(), bank_hold_capture_and_nonzero_timers_cleared=True,
        previous_interval_ownership_preserved=True, next_policy_interval_learning_enabled=True,
        true_nominal_collision_rejected=True, geometry_proof=proof)

CACHE = ("profile_positions", "collision_corners", "wheel_points", "wheel_axes", "wheel_radius", "wheel_halfwidth", "geometry_bound")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=OUT.parent / "deploy/source_candidate_018.json")
    parser.add_argument("--output", type=Path, default=OUT / "full_surface_bank_geometry_018.json")
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args(); meta = read(args.metadata); source = Path(meta["clean_root"])
    for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "" if name == "CUDA_VISIBLE_DEVICES" else "1"
    sys.dont_write_bytecode = True; sys.path.insert(0, str(source / "src"))
    global torch, OrdinaryPostureCurriculum, ordinary_eligibility, rotate_vectors, NonstandardStart, support_height_for_footprint, choose_scene_groups, terrain_surfaces, Surface, release_stop_surfaces
    import torch
    from wheeled_tasks.chassis.ordinary_posture_curriculum import OrdinaryPostureCurriculum, ordinary_eligibility, rotate_vectors
    from wheeled_tasks.chassis.nonstandard_start import NonstandardStart, support_height_for_footprint
    from wheeled_tasks.chassis.task import choose_scene_groups, terrain_surfaces, Surface
    from wheeled_tasks.chassis.release_stop import release_stop_surfaces
    torch.set_num_threads(1); started = time.monotonic()
    configs = {d: read(source / f"contracts/v6_task_campaign_{d}_worker_v18.json") for d in DOMAINS}
    asset = source / configs["terrain"]["asset_directory"]; manifest = read(asset / "manifest.json"); names = manifest["tree_joint_names"]
    files = sorted({*(source / "src/wheeled_tasks/chassis").glob("*.py"), *(source / f"contracts/v6_task_campaign_{d}_worker_v18.json" for d in DOMAINS), source / configs["terrain"]["nonstandard_start"]["profiles_file"], asset / "manifest.json", *(asset / name for name in manifest["files_sha256"])})
    before = {str(p.relative_to(source)): sha(p) for p in files}
    assert sha(meta["archive_local"]) == meta["archive_sha256"]
    with tarfile.open(meta["archive_local"], "r:gz") as archive:
        assert archive.pax_headers.get("comment") == meta["source_commit"]
        matched = {m.name: hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive if m.name in before and m.isfile()}
    assert matched == before
    template_bank = NonstandardStart(1, "cpu", .005, configs["terrain"]["nonstandard_start"], source, names)
    template = OrdinaryPostureCurriculum(1, "cpu", configs["terrain"]["ordinary_posture_curriculum"], torch.ones(1, dtype=torch.bool))
    template.bind_geometry(asset, manifest, names, template_bank)
    fixture_path = OUT / "native_failed_geometry_fixture_da8b11.json"
    exact = exact_fixture(configs["terrain"], template, source, read(fixture_path))
    domains = {}; failures = []; totals = Counter()
    for domain, config in configs.items():
        cases, coverage = catalog(config); combinations = itertools.product(range(len(cases)), range(len(template.profile_positions)), range(len(ANGLES)))
        counts = Counter(); minima = [math.inf, math.inf]; max_gap = 0.
        while batch := list(itertools.islice(combinations, args.chunk_size)):
            n = len(batch); indices = torch.tensor([b[1] for b in batch]); bank = NonstandardStart(n, "cpu", .005, configs["terrain"]["nonstandard_start"], source, names)
            bank.profile_ids.copy_(indices - 1); bank.selected.copy_(indices > 0); bank.remaining_steps[indices > 0] = bank.hold_steps
            lesson = OrdinaryPostureCurriculum(n, "cpu", config["ordinary_posture_curriculum"], torch.ones(n, dtype=torch.bool))
            for key in CACHE: setattr(lesson, key, getattr(template, key))
            origins = torch.stack([cases[b[0]]["origin"] for b in batch]); root = torch.zeros(n, 7); root[:, :3] = torch.stack([cases[b[0]]["world"] for b in batch]); root[:, 2] += .32; root[:, 6] = 1.
            width = max(len(cases[b[0]]["surface"]) for b in batch); surfaces = torch.zeros(n, width, 5); valid = torch.zeros(n, width, dtype=torch.bool)
            for i, b in enumerate(batch):
                data = cases[b[0]]["surface"]; surfaces[i, :len(data)] = data; valid[i, :len(data)] = True
            lesson.selected.fill_(True); lesson.angles[:] = torch.tensor([ANGLES[b[2]] for b in batch]); ids = torch.arange(n); lesson.initial_orientation(ids, root)
            joints = template.profile_positions[indices].clone()
            try:
                cancelled = lesson.align_support(ids, root, joints, origins, surfaces, valid, profile_ids=bank.profile_ids); bank.cancel_reset(cancelled)
                final_indices = bank.profile_ids + 1
                assert torch.equal(joints, template.profile_positions[final_indices])
                assert torch.equal(indices[~torch.isin(ids, cancelled)], final_indices[~torch.isin(ids, cancelled)])
                proof = verify(lesson, root, joints, origins, final_indices, surfaces, valid)
            except (ValueError, AssertionError) as error:
                failures.append(dict(domain=domain, error=str(error), first_case=cases[batch[0][0]]["lane"], attempted_combinations=n))
                counts["failed_batches"] += 1; continue
            counts["combinations"] += n; counts["cancelled_bank_to_nominal"] += len(cancelled); counts["cancelled_tilts"] += lesson.birth_rejected_count
            minima = [min(minima[0], proof[0]), min(minima[1], proof[1])]; max_gap = max(max_gap, proof[2])
        domains[domain] = dict(coverage=coverage, counts=dict(counts), min_wheel_gap_m=minima[0] if math.isfinite(minima[0]) else None, min_body_clearance_m=minima[1] if math.isfinite(minima[1]) else None, max_fallback_lift_m=max_gap, configured_bank=bool(config.get("nonstandard_start")))
        totals.update(counts); print(json.dumps(dict(domain=domain, counts=dict(counts), unique_surfaces=coverage["unique_surface_position_combinations"])), flush=True)
    after = {str(p.relative_to(source)): sha(p) for p in files}; assert before == after
    report = dict(status="passed" if not failures else "failed", source_commit=meta["source_commit"], archive_sha256=meta["archive_sha256"], exact_native_failure_fixture=exact, fixture_sha256=sha(fixture_path), domains=domains, total_counts=dict(totals), failures=failures, angle_pairs_rad=ANGLES, profile_names=["nominal", *template_bank.profile_names], all_declared_ordinary_surface_instances_enumerated=True, all_stored_bank_profiles_for_every_unique_surface_position=True, manual_step_start_range_points=9, source_files_sha256_before=before, source_files_sha256_after=after, source_files_unchanged=True, all_selected_source_bytes_match_archive=True, runner_sha256=sha(__file__), duration_seconds=time.monotonic() - started, formal_updates_consumed=0, native_or_GPU_validation=False, qualification=None, limitations=["Deterministic CPU reset-buffer geometry only; no PhysX contacts or skill qualification.", "Every declared 4096-lane ordinary terrain instance is enumerated; identical terrain strips and realized local XY are deduplicated.", "All sealed bank poses are forced geometrically even for worker rows that do not sample a bank; these are supplemental combinations, not native population counts.", "Seventeen exact angle pairs and nine manual-start positions are finite boundary/interior coverage, not proof over every continuous angle or start coordinate."])
    write(args.output, report); print(json.dumps(dict(status=report["status"], total_counts=dict(totals), duration_seconds=report["duration_seconds"])))
    return 0 if not failures else 1

if __name__ == "__main__": raise SystemExit(main())
