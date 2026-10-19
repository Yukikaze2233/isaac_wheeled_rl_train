"""Versioned wheel material domains and CPU native-shaped write/readback."""

import hashlib

import pytest
import torch

from support_privileged_runtime import fixture, warp_runtime
from wheeled_tasks.chassis.privileged_runtime import (
    ContactMaterialPolicy, WheelMaterialCache, apply_startup_wheel_material_buckets,
)


V2 = "v7_low_grip_wheel_buckets_v2"
V3 = "wheel_low_grip_01_v3"


def make_cache(count=3):
    robot, sim, _ = fixture()
    view = robot.root_view
    if count != view.count:
        view.count = count
        view.link_paths = [[f"/World/env_{i}/Robot/{name}" for name in robot.body_names]
                           for i in range(count)]
        view.materials = view.materials[:1].repeat(count, 1, 1)
    return robot, WheelMaterialCache(robot, sim, "cpu")


def apply(cache, version=V2):
    return apply_startup_wheel_material_buckets(cache, seed=991,
        legacy_ground_randomization_enabled=False, version=version)


def test_legacy_default_preserves_frozen_buckets_and_assignment(warp_runtime):
    _, cache = make_cache()
    receipt = apply_startup_wheel_material_buckets(cache, seed=991,
        legacy_ground_randomization_enabled=False)
    assert receipt["version"] == "wheel_shape_buckets_v1"
    assert hashlib.sha256(receipt["buckets"].numpy().tobytes()).hexdigest() == (
        "fdc330a85e7c84f57b4e382e644ad4dff3d44bc5faeb21bd5842adedcfef47c5")
    assert receipt["bucket_ids"].tolist() == [[3, 63], [3, 39], [60, 4]]
    assert receipt["low_friction_wheel_count"] == receipt["lower_endpoint_wheel_count"] == 0


@pytest.mark.parametrize("count", [3, 4096])
@pytest.mark.parametrize("version,lower", [(V2, .2), (V3, .1)])
def test_versioned_wheels_write_and_report_actual_coverage(count, version, lower, warp_runtime):
    robot, cache = make_cache(count)
    before = robot.root_view.materials.clone()
    receipt = apply(cache, version)
    buckets = receipt["buckets"]
    assert receipt["version"] == version
    assert buckets.shape == (64, 3)
    assert torch.equal(buckets[0, :2], torch.full((2,), lower))
    assert bool(((buckets >= torch.tensor([lower, lower, .02])) &
                 (buckets <= torch.tensor([1.2, 1., .2]))).all())
    assert bool((buckets[:, 1] <= buckets[:, 0]).all())
    assert torch.equal(cache.values, buckets[receipt["bucket_ids"]])
    assert torch.equal(robot.root_view.materials[:, [0, 3]], before[:, [0, 3]])
    assert robot.root_view.writes == 1
    actual = cache.values.reshape(-1, 3)
    assert receipt["sampled_static_friction_range"] == [float(actual[:, 0].min()), float(actual[:, 0].max())]
    assert receipt["sampled_dynamic_friction_range"] == [float(actual[:, 1].min()), float(actual[:, 1].max())]
    assert receipt["low_friction_wheel_count"] == int(((actual[:, 0] < .5) | (actual[:, 1] < .4)).sum())
    assert receipt["lower_endpoint_wheel_count"] == int((receipt["bucket_ids"] == 0).sum())
    if count == 3:
        assert receipt["lower_endpoint_wheel_count"] == 0
        assert receipt["bucket_ids"].tolist() == [[3, 63], [3, 39], [60, 4]]
    else:
        assert receipt["lower_endpoint_wheel_count"] > 0
        assert receipt["low_friction_wheel_count"] > receipt["lower_endpoint_wheel_count"]


def test_v2_keeps_random_assignment_and_restitution_stream(warp_runtime):
    _, old = make_cache()
    _, new = make_cache()
    old_receipt, new_receipt = apply(old, "wheel_shape_buckets_v1"), apply(new)
    assert torch.equal(old_receipt["bucket_ids"], new_receipt["bucket_ids"])
    assert torch.equal(old_receipt["buckets"][:, 2], new_receipt["buckets"][:, 2])


def test_v3_keeps_v2_assignment_restitution_and_high_grip_coverage(warp_runtime):
    _, old = make_cache(4096)
    _, new = make_cache(4096)
    before, after = apply(old, V2), apply(new, V3)
    assert torch.equal(before["bucket_ids"], after["bucket_ids"])
    assert torch.equal(before["buckets"][:, 2], after["buckets"][:, 2])
    assert after["sampled_static_friction_range"][0] == pytest.approx(.1)
    assert after["sampled_dynamic_friction_range"][0] == pytest.approx(.1)
    assert after["sampled_static_friction_range"][1] > 1.
    assert after["sampled_dynamic_friction_range"][1] > .9


def test_v2_preserves_native_readback_and_double_application_guards(warp_runtime):
    robot, cache = make_cache()
    setter = robot.root_view.set_material_properties
    robot.root_view.set_material_properties = lambda *args: None
    with pytest.raises(RuntimeError, match="readback differs"):
        apply(cache)
    assert not getattr(cache, "_startup_buckets_applied", False)
    robot.root_view.set_material_properties = setter
    apply(cache)
    with pytest.raises(RuntimeError, match="already applied"):
        apply(cache)


def test_unknown_version_is_rejected_before_native_write():
    robot, cache = make_cache()
    with pytest.raises(ValueError, match="version"):
        apply(cache, "unreviewed_materials")
    assert robot.root_view.writes == 0
    config = {"wheel_material_priors": {"enabled": True, "version": "unreviewed_materials"},
              "friction_combine_mode": "multiply"}
    with pytest.raises(ValueError, match="version"):
        ContactMaterialPolicy.from_config(config, legacy_contact_domain_active=False)


@pytest.mark.parametrize("version", [V2, V3])
def test_low_grip_policy_requires_unit_ground_and_no_duplicate_ground_dr(version):
    config = {"wheel_material_priors": {"enabled": True, "version": version},
              "friction_combine_mode": "multiply"}
    policy = ContactMaterialPolicy.from_config(config, legacy_contact_domain_active=False)
    assert policy.ground_friction(1.) == 1.
    with pytest.raises(ValueError, match="unit ground"):
        policy.ground_friction(.2)
    with pytest.raises(ValueError, match="cannot stack"):
        ContactMaterialPolicy.from_config(config, legacy_contact_domain_active=True)
