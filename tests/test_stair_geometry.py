"""Explicit stair tops and landing widths must become physical geometry."""
import pytest

from wheeled_tasks.chassis.task import terrain_surfaces


def test_double_step_has_800mm_first_tread_and_cumulative_450mm_top():
    surfaces = terrain_surfaces("stairs", {
        "stair_heights_m": [.20, .45], "stair_tread_m": [.8, .6],
    }, 1.)
    assert [(s.x0, s.x1, s.z0) for s in surfaces] == [
        (-4., 0., 0.), (0., .8, .2), (.8, 1.4, .45), (1.4, 4., .45),
    ]
    # The last tread is a landing region; its unmeasured physical end must not
    # invent a 45 cm drop immediately after the requested two upward steps.
    assert surfaces[-1].z0 == surfaces[-2].z0
    for surface in surfaces:
        size, center, _ = surface.box(solid_bottom=-.2)
        assert center[2] + size[2] / 2 == pytest.approx(surface.z0)


def test_scalar_tread_and_curriculum_scale():
    surfaces = terrain_surfaces("stairs", {
        "stair_heights_m": [.20, .45], "stair_tread_m": .8,
    }, .5)
    assert [s.z0 for s in surfaces] == pytest.approx([0., .1, .225, .225])
    assert [s.x1 - s.x0 for s in surfaces[1:3]] == pytest.approx([.8, .8])


def test_descending_explicit_tops_and_treads_are_travel_order():
    surfaces = terrain_surfaces("stairs_down", {
        "stair_heights_m": [.20, .45], "stair_tread_m": [.8, .6],
    }, 1.)
    assert [s.z0 for s in surfaces] == pytest.approx([.45, .20, 0., 0.])
    assert [(s.x0, s.x1) for s in surfaces[1:3]] == [(0., .8), (.8, 1.4)]


def test_legacy_three_4cm_stairs_unchanged():
    limits = {"stair_rise_m": .04}
    up = terrain_surfaces("stairs", limits, 1.)
    down = terrain_surfaces("stairs_down", limits, 1.)
    assert [(s.x0, s.x1) for s in up] == [(-4., 0.), (0., .6), (.6, 1.2), (1.2, 4.)]
    assert [s.z0 for s in up] == pytest.approx([0., .04, .08, .12])
    assert [s.z0 for s in down] == pytest.approx([.12, .08, .04, 0.])


@pytest.mark.parametrize("values", [
    {"stair_heights_m": []}, {"stair_heights_m": [.2, .1]},
    {"stair_heights_m": [.2, .2]}, {"stair_heights_m": [.2, float("nan")]},
    {"stair_heights_m": [0.]}, {"stair_heights_m": [True]},
    {"stair_heights_m": .2}, {"stair_heights_m": [.2, .45], "stair_tread_m": [.8]},
    {"stair_heights_m": [.2], "stair_tread_m": [0.]},
    {"stair_heights_m": [.2], "stair_tread_m": float("inf")},
    {"stair_heights_m": [.2], "stair_tread_m": True},
    {"stair_heights_m": None, "stair_rise_m": .04},
    {"stair_tread_m": .8},
    {"stair_heights_m": [.2, .45], "stair_tread_m": [1e308, 1e308]},
])
def test_invalid_explicit_geometry_is_rejected(values):
    with pytest.raises(ValueError):
        terrain_surfaces("stairs", values, 1.)
