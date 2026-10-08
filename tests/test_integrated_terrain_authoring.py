"""Static terrain derivation; no checkpoint, plan migration or native startup."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import pytest

from wheeled_tasks.chassis import integrated_production as production
from wheeled_tasks.chassis import integrated_runtime as runtime
from wheeled_tasks.chassis import integrated_sampling as sampling
from wheeled_tasks.chassis.heading_command import configure_heading
from wheeled_tasks.chassis.ordinary_large_stair import ROWS, validate_row


ROOT = Path(__file__).resolve().parents[1]
NAMES = {row[0] for row in ROWS}


@pytest.mark.parametrize("version", [24, 25, 26])
def test_old_recipe_bytes_remain_exact(version):
    recipe = production.revised_recipe(ROOT, version=version)
    assert production._bytes(recipe) == (ROOT / production.artifact_files(version)["recipe"]).read_bytes()
    for domain, _ in production.lineage.CHAIN:
        assert production._bytes(production.revised_design(domain, ROOT, version=version)) == (
            ROOT / production.design_file(domain, version=version)).read_bytes()
        assert production._bytes(production.canonical_worker(domain, ROOT, version=version)) == (
            ROOT / production.worker_file(domain, version=version)).read_bytes()


@pytest.fixture(scope="module")
def candidate():
    return production.revised_recipe(ROOT, version=27)


def test_new_recipe_retains_speed_and_declares_static_clock_boundary(candidate):
    before = json.loads((ROOT / production.artifact_files(26)["recipe"]).read_text())
    assert candidate["domains"]["speed"] == before["domains"]["speed"]
    assert candidate["predecessor_recipe"] == dict(file=production.artifact_files(26)["recipe"],
        sha256=production.lineage.digest(ROOT / production.artifact_files(26)["recipe"]))
    assert "actual terrain introduction clock and remaining actor allowance" in candidate["pending"]
    assert candidate["freeze_ready"] is False and candidate["qualification"] is None


@pytest.mark.parametrize("domain,groups", [("terrain", 241), ("jump", 256), ("stepjump", 270)])
def test_derived_rows_allocator_immutable_coverage_and_exclusive_commands(candidate, domain, groups):
    patch = candidate["domains"][domain]["patch"]
    native = sampling.load_sources(ROOT)["terrain"]
    audit = patch["integrated_sampling_candidate"]
    transform = audit["source_transforms"]["terrain"]
    assert len(patch["scene_groups"]) == len(patch["skill_specs"]) == groups
    assert sum(audit["allocation"]["counts"].values()) == 4096
    assert audit["allocation"]["minimum_count"] >= 1
    assert math.fsum(row["fraction"] for row in patch["scene_groups"]) == pytest.approx(1.)
    assert set(audit["source_coverage"]["terrain"]) == set(native["skill_specs"])
    assert not NAMES & set(audit["source_coverage"]["terrain"])
    assert transform["input_source"] == dict(file=sampling.SOURCE_BINDINGS["terrain"][0],
        sha256=sampling.SOURCE_BINDINGS["terrain"][1], groups=101)
    assert transform["derived_groups"] == 109
    assert transform["sha256"] == production.lineage.digest(ROOT / transform["file"])
    assert transform["static_parameters"] == dict(origin_actor_update=0, actor_allowance=15000)
    assert transform["clock_status"] == "schema_only_actual_admission_parameters_required"
    owners = candidate["domains"][domain]["runtime"]["row_ownership"]
    mapped = patch["scut_reuse_candidate"]["mapped_groups"]
    for name in NAMES:
        spec, provenance = patch["skill_specs"][name], audit["row_provenance"][name]
        assert validate_row(spec) is not None
        assert owners[name]["component"] == provenance["component"] == "terrain"
        assert provenance["source_kind"] == "derived_row" and provenance["source_name"] is None
        assert provenance["derived_name"] == name and provenance["immutable_source_fraction"] is None
        assert "native_spec_sha256" not in provenance and "derived_spec_sha256" in provenance
        assert name not in mapped and name in patch["scut_reuse_candidate"]["preserved_groups"]
    for name in runtime.SMALL_STAIR_NAMES:
        assert mapped[name]["kind"] == "explicit_terrain"
    assert set(patch["integrated_sampling_candidate"]["source_bindings"]) == set(sampling.SOURCE_BINDINGS)
    # Composition never writes or extends the immutable native input contract.
    assert production.lineage.digest(ROOT / sampling.SOURCE_BINDINGS["terrain"][0]) == sampling.SOURCE_BINDINGS["terrain"][1]


def test_sampling_opt_in_is_train_only_and_cannot_modify_speed():
    sources = sampling.load_sources(ROOT)
    before = deepcopy(sources["terrain"])
    assert sampling.apply(before, ROOT / "missing", role="fixed_eval", ordinary_stairs=True) == before
    with pytest.raises(ValueError, match="completed speed"):
        sampling.apply(sources["speed"], ROOT, ordinary_stairs=True)
    with pytest.raises(ValueError, match="boolean"):
        sampling.apply(before, ROOT, ordinary_stairs=1)
    assert sources["terrain"] == before


def test_temporary_recipe_and_worker_designs_are_source_derived(tmp_path, candidate):
    """Validate real canonical workers without generating a v27 campaign/manifest."""
    shutil.copytree(ROOT / "contracts", tmp_path / "contracts")
    shutil.copytree(ROOT / production.SCUT_DIRECTORY, tmp_path / production.SCUT_DIRECTORY)
    for file in [*runtime._references(ROOT, sampling.load_sources(ROOT)), runtime.AUDIT[0],
                 candidate["ordinary_terrain_transform"]["file"]]:
        target = tmp_path / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, target)
    files = production.artifact_files(27)
    production._put(tmp_path, files["recipe"], production.revised_recipe(tmp_path, version=27))
    for domain, _ in production.lineage.CHAIN:
        design = production.revised_design(domain, tmp_path, version=27)
        production._put(tmp_path, production.design_file(domain, version=27), design)
        worker = production.canonical_worker(domain, tmp_path, version=27)
        production._put(tmp_path, production.worker_file(domain, version=27), worker)
        assert production.validate_static_worker(worker, tmp_path) == domain
        effective = production.static_training_config(worker, tmp_path)
        heading = configure_heading(effective, tmp_path)
        assert len(heading["ordinary_names"]) == 24 and not NAMES & set(heading["ordinary_names"])
        assert effective["evaluation"] == sampling.load_sources(tmp_path)[domain]["evaluation"]
        inventory = production.production_source_files(worker, root=tmp_path)
        assert candidate["ordinary_terrain_transform"]["file"] in inventory
        assert production.artifact_files(26)["recipe"] in inventory
        if domain != "speed":
            transform = effective["integrated_sampling_candidate"]["source_transforms"]["terrain"]
            assert design["revision_migration"]["training_source_transforms"] == dict(terrain=transform)
    assert not (tmp_path / files["plan"]).exists() and not (tmp_path / files["lineage"]).exists()
    assert not (tmp_path / files["manifest"]).exists()
    # A cached recipe cannot hide a changed transformation source.
    module = tmp_path / candidate["ordinary_terrain_transform"]["file"]
    module.write_bytes(module.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="exact native/SCUT source composition"):
        production.validate_static_worker(worker, tmp_path)
