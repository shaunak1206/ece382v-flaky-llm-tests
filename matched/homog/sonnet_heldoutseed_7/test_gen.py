import numpy as np
import pytest

from homog.homog import (
    hray,
    hrot,
    is_valid_rays,
    rand_ray,
    line_line_distance,
    line_line_closest_points,
    hdot,
    hcross,
    hnorm,
)


def _ray(origin, direction):
    """Batch of one ray, shape (1, 4, 2)."""
    return hray(origin, direction)[None]


# ---------------------------------------------------------------- hray

def test_hray_shape_and_layout_3d_inputs():
    r = hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1, 2, 3, 1])
    assert np.allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = hray([0, 0, 0], [0, 0, 5])
    assert np.allclose(r[:, 1], [0, 0, 1, 0])
    r = hray([0, 0, 0], [3, 4, 0])
    assert np.allclose(r[:, 1], [0.6, 0.8, 0, 0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_accepts_homogeneous_inputs():
    r = hray([1, 2, 3, 1], [2, 0, 0, 0])
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1, 2, 3, 1])
    assert np.allclose(r[:, 1], [1, 0, 0, 0])


def test_hray_origin_w_forced_to_one():
    r = hray([1, 2, 3, 1], [0, 1, 0, 0])
    assert r[3, 0] == 1
    assert r[3, 1] == 0


def test_hray_batched_origins_single_direction():
    origins = np.random.RandomState(0).randn(5, 3)
    r = hray(origins, [0, 1, 0])
    assert r.shape == (5, 4, 2)
    assert np.allclose(r[:, :3, 0], origins)
    assert np.allclose(r[:, 3, 0], 1)
    assert np.allclose(r[:, :, 1], [0, 1, 0, 0])


def test_hray_batched_directions():
    rs = np.random.RandomState(1)
    origins = rs.randn(6, 3)
    dirs = rs.randn(6, 3)
    r = hray(origins, dirs)
    assert r.shape == (6, 4, 2)
    assert np.allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1.0)
    expected = dirs / np.linalg.norm(dirs, axis=-1)[:, None]
    assert np.allclose(r[:, :3, 1], expected)
    assert np.allclose(r[:, 3, 1], 0)


def test_hray_result_is_valid_ray():
    r = hray([0.3, -1.2, 5.0], [1, 2, 3])
    assert is_valid_rays(r)


# ---------------------------------------------------- line_line_distance

def test_distance_skew_perpendicular_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 3], [0, 1, 0])
    d = line_line_distance(r1, r2)
    assert d.shape == (1,)
    assert np.isclose(d[0], 3.0)


def test_distance_skew_oblique_lines():
    r1 = _ray([0, 0, 0], [1, 1, 0])
    r2 = _ray([0, 0, 7], [1, -1, 0])
    assert np.isclose(line_line_distance(r1, r2)[0], 7.0)


def test_distance_intersecting_lines_is_zero():
    r1 = _ray([1, 2, 3], [1, 0, 0])
    r2 = _ray([4, 2, 3], [0, 0, 1])
    assert np.isclose(line_line_distance(r1, r2)[0], 0.0, atol=1e-9)


def test_distance_parallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([10, 3, 4], [1, 0, 0])
    assert np.isclose(line_line_distance(r1, r2)[0], 5.0)


def test_distance_antiparallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([-2, 3, 4], [-1, 0, 0])
    assert np.isclose(line_line_distance(r1, r2)[0], 5.0)


def test_distance_symmetric_and_nonnegative():
    np.random.seed(3)
    r1 = rand_ray(15)
    r2 = rand_ray(15)
    d12 = line_line_distance(r1, r2)
    d21 = line_line_distance(r2, r1)
    assert d12.shape == (15,)
    assert np.all(d12 >= 0)
    assert np.allclose(d12, d21)


def test_distance_invariant_to_sliding_origin_along_line():
    np.random.seed(4)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    d = line_line_distance(r1, r2)
    r1b = r1.copy()
    r1b[:, :3, 0] += 2.5 * r1[:, :3, 1]
    r2b = r2.copy()
    r2b[:, :3, 0] -= 1.7 * r2[:, :3, 1]
    assert np.allclose(d, line_line_distance(r1b, r2b))


def test_distance_invariant_to_rigid_transform():
    np.random.seed(5)
    r1 = rand_ray(8)
    r2 = rand_ray(8)
    x = hrot([1, 2, 3], 0.7, center=[0.5, -1.0, 2.0], degrees=False)
    r1x = x @ r1
    r2x = x @ r2
    assert np.allclose(line_line_distance(r1, r2),
                       line_line_distance(r1x, r2x))


# ----------------------------------------------- line_line_closest_points

def test_closest_points_perpendicular_axes():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1[0], [0, 0, 0, 1])
    assert np.allclose(q2[0], [0, 0, 3, 1])


def test_closest_points_offset_origins():
    r1 = _ray([2, 0, 0], [1, 0, 0])
    r2 = _ray([0, 5, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1[0], [0, 0, 0, 1])
    assert np.allclose(q2[0], [0, 0, 3, 1])


def test_closest_points_unbatched_ray():
    r1 = hray([2, 0, 0], [1, 0, 0])
    r2 = hray([0, 5, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (4,)
    assert np.allclose(q1, [0, 0, 0, 1])
    assert np.allclose(q2, [0, 0, 3, 1])


def test_closest_points_geometric_properties_random():
    np.random.seed(7)
    r1 = rand_ray(20)
    r2 = rand_ray(20)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (20, 4)
    assert q2.shape == (20, 4)
    ax1, ax2 = r1[:, :, 1], r2[:, :, 1]
    pt1, pt2 = r1[:, :, 0], r2[:, :, 0]
    # points are actually on their lines
    assert np.allclose(hcross(q1 - pt1, ax1), 0, atol=1e-8)
    assert np.allclose(hcross(q2 - pt2, ax2), 0, atol=1e-8)
    # connecting segment perpendicular to both lines
    seg = q2 - q1
    assert np.allclose(hdot(seg, ax1), 0, atol=1e-8)
    assert np.allclose(hdot(seg, ax2), 0, atol=1e-8)
    # homogeneous w coordinate preserved
    assert np.allclose(q1[:, 3], 1)
    assert np.allclose(q2[:, 3], 1)


def test_closest_points_separation_matches_line_line_distance():
    np.random.seed(8)
    r1 = rand_ray(20)
    r2 = rand_ray(20)
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(hnorm(q2 - q1), line_line_distance(r1, r2))


def test_closest_points_unchanged_by_sliding_origin_along_line():
    np.random.seed(9)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    q1, q2 = line_line_closest_points(r1, r2)
    r1b = r1.copy()
    r1b[:, :3, 0] += 3.0 * r1[:, :3, 1]
    r2b = r2.copy()
    r2b[:, :3, 0] -= 2.0 * r2[:, :3, 1]
    q1b, q2b = line_line_closest_points(r1b, r2b)
    assert np.allclose(q1, q1b)
    assert np.allclose(q2, q2b)


def test_closest_points_parallel_lines_not_finite():
    # documented limitation: errors (non-finite output) when ax1 == ax2
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 1, 0], [1, 0, 0])
    with np.errstate(all='ignore'):
        q1, q2 = line_line_closest_points(r1, r2)
    assert not (np.all(np.isfinite(q1)) and np.all(np.isfinite(q2)))
