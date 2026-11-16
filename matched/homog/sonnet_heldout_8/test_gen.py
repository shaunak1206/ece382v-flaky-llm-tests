import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    is_valid_rays,
    rand_ray,
    hdot,
    hnorm,
)


def _stack(*rays):
    """Stack single (4, 2) rays into a batch of shape (N, 4, 2)."""
    return np.stack(rays, axis=0)


# --------------------------------------------------------------------------
# hray
# --------------------------------------------------------------------------

def test_hray_basic_shape_and_contents():
    r = hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = hray([0, 0, 0], [0, 0, 5])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])
    r = hray([0, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(r[:, 1], [0.6, 0.8, 0, 0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_accepts_homogeneous_origin_and_direction():
    r = hray([1, 2, 3, 1], [0, 2, 0, 0])
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 1, 0, 0])


def test_hray_origin_w_is_one_and_direction_w_is_zero():
    np.random.seed(0)
    r = hray(np.random.randn(6, 3), np.random.randn(6, 3))
    assert r.shape == (6, 4, 2)
    np.testing.assert_array_equal(r[:, 3, 0], np.ones(6))
    np.testing.assert_array_equal(r[:, 3, 1], np.zeros(6))


def test_hray_batch_directions_are_unit():
    np.random.seed(1)
    r = hray(np.random.randn(10, 3), np.random.randn(10, 3))
    norms = np.linalg.norm(r[:, :3, 1], axis=-1)
    np.testing.assert_allclose(norms, np.ones(10))


def test_hray_broadcasts_single_direction_over_many_origins():
    origins = np.arange(15, dtype=float).reshape(5, 3)
    r = hray(origins, [1, 0, 0])
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins)
    for i in range(5):
        np.testing.assert_allclose(r[i, :, 1], [1, 0, 0, 0])


def test_hray_result_is_valid_ray():
    np.random.seed(2)
    r = hray(np.random.randn(4, 3), np.random.randn(4, 3))
    assert is_valid_rays(r)


# --------------------------------------------------------------------------
# line_line_distance
# --------------------------------------------------------------------------

def test_line_line_distance_skew_axes():
    r1 = hray([0, 0, 0], [1, 0, 0])
    r2 = hray([0, 0, 3], [0, 1, 0])
    d = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(d, [3.0])


def test_line_line_distance_intersecting_lines_is_zero():
    r1 = hray([0, 0, 0], [1, 0, 0])
    r2 = hray([2, 5, 0], [0, 1, 0])
    d = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(d, [0.0], atol=1e-9)


def test_line_line_distance_parallel_lines():
    r1 = hray([0, 0, 0], [1, 0, 0])
    r2 = hray([5, 2, 0], [1, 0, 0])
    d = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(d, [2.0])


def test_line_line_distance_antiparallel_lines():
    r1 = hray([0, 0, 0], [0, 0, 1])
    r2 = hray([3, 4, 7], [0, 0, -1])
    d = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(d, [5.0])


def test_line_line_distance_is_symmetric():
    np.random.seed(3)
    r1 = rand_ray(8)
    r2 = rand_ray(8)
    np.testing.assert_allclose(line_line_distance(r1, r2),
                               line_line_distance(r2, r1))


def test_line_line_distance_matches_triple_product_formula():
    np.random.seed(4)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    p1, a1 = r1[:, :3, 0], r1[:, :3, 1]
    p2, a2 = r2[:, :3, 0], r2[:, :3, 1]
    cr = np.cross(a1, a2)
    expected = np.abs(np.sum((p2 - p1) * cr, axis=-1)) / np.linalg.norm(cr, axis=-1)
    np.testing.assert_allclose(line_line_distance(r1, r2), expected)


def test_line_line_distance_invariant_to_sliding_point_along_line():
    np.random.seed(5)
    r1 = rand_ray(6)
    r2 = rand_ray(6)
    shifted = r1.copy()
    shifted[:, :3, 0] += 2.5 * r1[:, :3, 1]
    np.testing.assert_allclose(line_line_distance(r1, r2),
                               line_line_distance(shifted, r2))


# --------------------------------------------------------------------------
# line_line_closest_points
# --------------------------------------------------------------------------

def test_closest_points_simple_skew_axes():
    r1 = hray([0, 0, 0], [1, 0, 0])
    r2 = hray([0, 0, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    np.testing.assert_allclose(q1, [[0, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[0, 0, 3, 1]], atol=1e-9)


def test_closest_points_offset_example():
    r1 = hray([0, 0, 0], [1, 0, 0])
    r2 = hray([2, 5, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    np.testing.assert_allclose(q1, [[2, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[2, 0, 3, 1]], atol=1e-9)


def test_closest_points_random_geometric_properties():
    np.random.seed(6)
    n = 12
    r1 = rand_ray(n)
    r2 = rand_ray(n)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (n, 4)
    assert q2.shape == (n, 4)
    np.testing.assert_allclose(q1[:, 3], np.ones(n))
    np.testing.assert_allclose(q2[:, 3], np.ones(n))

    pt1, ax1 = r1[:, :, 0], r1[:, :, 1]
    pt2, ax2 = r2[:, :, 0], r2[:, :, 1]
    # Q1 lies on line 1 and Q2 lies on line 2
    off1 = np.cross((q1 - pt1)[:, :3], ax1[:, :3])
    off2 = np.cross((q2 - pt2)[:, :3], ax2[:, :3])
    np.testing.assert_allclose(off1, np.zeros((n, 3)), atol=1e-8)
    np.testing.assert_allclose(off2, np.zeros((n, 3)), atol=1e-8)
    # connecting segment is perpendicular to both lines
    seg = q2 - q1
    np.testing.assert_allclose(hdot(seg, ax1), np.zeros(n), atol=1e-8)
    np.testing.assert_allclose(hdot(seg, ax2), np.zeros(n), atol=1e-8)


def test_closest_points_separation_equals_line_line_distance():
    np.random.seed(7)
    r1 = rand_ray(9)
    r2 = rand_ray(9)
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(hnorm(q2 - q1), line_line_distance(r1, r2),
                               atol=1e-8)


def test_closest_points_swapping_rays_swaps_results():
    np.random.seed(8)
    r1 = rand_ray(5)
    r2 = rand_ray(5)
    q1, q2 = line_line_closest_points(r1, r2)
    p1, p2 = line_line_closest_points(r2, r1)
    np.testing.assert_allclose(q1, p2, atol=1e-8)
    np.testing.assert_allclose(q2, p1, atol=1e-8)


def test_closest_points_intersecting_lines_coincide():
    target = np.array([1.0, -2.0, 0.5])
    r1 = hray(target - 2 * np.array([1.0, 0, 0]), [1, 0, 0])
    r2 = hray(target + 3 * np.array([0, 1.0, 1.0]), [0, 1, 1])
    q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    np.testing.assert_allclose(q1[0, :3], target, atol=1e-9)
    np.testing.assert_allclose(q2[0, :3], target, atol=1e-9)


def test_closest_points_parallel_lines_are_not_finite():
    r1 = hray([0, 0, 0], [1, 0, 0])
    r2 = hray([0, 1, 0], [1, 0, 0])
    with np.errstate(all='ignore'):
        q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    assert not np.all(np.isfinite(q1))
