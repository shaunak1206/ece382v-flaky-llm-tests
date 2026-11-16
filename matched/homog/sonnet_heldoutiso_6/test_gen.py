import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    is_valid_rays,
    hdot,
    hnorm,
    hcross,
    rand_ray,
)


def _ray(origin, direction):
    """Build a batch of one ray, shape (1, 4, 2)."""
    return hray(np.array([origin], dtype='f8'), np.array([direction], dtype='f8'))


# ---------------------------------------------------------------- hray


def test_hray_shape_and_values_single():
    r = hray([1, 2, 3], [0, 0, 2])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = hray([0, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(r[:, 1], [0.6, 0.8, 0, 0])
    np.testing.assert_allclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_accepts_homogeneous_inputs():
    r = hray([1, 2, 3, 1], [0, 3, 4, 0])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 0.6, 0.8, 0])


def test_hray_batch_shape():
    np.random.seed(0)
    origin = np.random.randn(5, 3)
    direction = np.random.randn(5, 3)
    r = hray(origin, direction)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origin)
    np.testing.assert_allclose(r[:, 3, 0], 1)
    np.testing.assert_allclose(r[:, 3, 1], 0)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1.0)


def test_hray_direction_parallel_to_input():
    np.random.seed(1)
    direction = np.random.randn(6, 3)
    r = hray(np.zeros((6, 3)), direction)
    expected = direction / np.linalg.norm(direction, axis=-1)[:, None]
    np.testing.assert_allclose(r[:, :3, 1], expected)


def test_hray_broadcasts_single_origin_over_directions():
    np.random.seed(2)
    direction = np.random.randn(5, 3)
    r = hray([1, 2, 3], direction)
    assert r.shape == (5, 4, 2)
    for i in range(5):
        np.testing.assert_allclose(r[i, :, 0], [1, 2, 3, 1])


def test_hray_result_is_valid_ray():
    np.random.seed(3)
    r = hray(np.random.randn(7, 3), np.random.randn(7, 3))
    assert is_valid_rays(r)


# ------------------------------------------------- line_line_distance


def test_distance_skew_axes():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 5], [0, 1, 0])
    d = line_line_distance(r1, r2)
    np.testing.assert_allclose(d, [5.0])


def test_distance_intersecting_lines_is_zero():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([3, -2, 0], [0, 1, 0])
    d = line_line_distance(r1, r2)
    np.testing.assert_allclose(d, [0.0], atol=1e-9)


def test_distance_parallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([7, 3, 4], [1, 0, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0])


def test_distance_antiparallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([-2, 3, 4], [-1, 0, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0])


def test_distance_is_symmetric():
    np.random.seed(4)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    np.testing.assert_allclose(
        line_line_distance(r1, r2), line_line_distance(r2, r1))


def test_distance_unchanged_by_sliding_origin_along_line():
    np.random.seed(5)
    r1 = rand_ray(8)
    r2 = rand_ray(8)
    d0 = line_line_distance(r1, r2)
    r1b = r1.copy()
    r1b[:, :3, 0] += 3.7 * r1[:, :3, 1]
    r2b = r2.copy()
    r2b[:, :3, 0] -= 1.3 * r2[:, :3, 1]
    np.testing.assert_allclose(line_line_distance(r1b, r2b), d0, atol=1e-8)


def test_distance_matches_reference_formula_random():
    np.random.seed(6)
    r1 = rand_ray(20)
    r2 = rand_ray(20)
    d = line_line_distance(r1, r2)
    n = np.cross(r1[:, :3, 1], r2[:, :3, 1])
    n /= np.linalg.norm(n, axis=-1)[:, None]
    expected = np.abs(np.sum((r2[:, :3, 0] - r1[:, :3, 0]) * n, axis=-1))
    assert d.shape == (20,)
    np.testing.assert_allclose(d, expected, atol=1e-8)


# ------------------------------------------- line_line_closest_points


def test_closest_points_simple_skew_axes():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, 3, 5], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[2, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[2, 0, 5, 1]], atol=1e-9)


def test_closest_points_output_shape_and_homogeneous_coord():
    np.random.seed(7)
    r1 = rand_ray(9)
    r2 = rand_ray(9)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (9, 4)
    assert q2.shape == (9, 4)
    np.testing.assert_allclose(q1[:, 3], 1)
    np.testing.assert_allclose(q2[:, 3], 1)


def test_closest_points_lie_on_their_lines():
    np.random.seed(8)
    r1 = rand_ray(15)
    r2 = rand_ray(15)
    q1, q2 = line_line_closest_points(r1, r2)
    off1 = hcross(q1 - r1[:, :, 0], r1[:, :, 1])
    off2 = hcross(q2 - r2[:, :, 0], r2[:, :, 1])
    np.testing.assert_allclose(off1, 0, atol=1e-8)
    np.testing.assert_allclose(off2, 0, atol=1e-8)


def test_closest_points_segment_perpendicular_to_both_lines():
    np.random.seed(9)
    r1 = rand_ray(15)
    r2 = rand_ray(15)
    q1, q2 = line_line_closest_points(r1, r2)
    seg = q2 - q1
    np.testing.assert_allclose(hdot(seg, r1[:, :, 1]), 0, atol=1e-8)
    np.testing.assert_allclose(hdot(seg, r2[:, :, 1]), 0, atol=1e-8)


def test_closest_points_separation_equals_line_line_distance():
    np.random.seed(10)
    r1 = rand_ray(15)
    r2 = rand_ray(15)
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(
        hnorm(q2 - q1), line_line_distance(r1, r2), atol=1e-8)


def test_closest_points_swapping_rays_swaps_points():
    np.random.seed(11)
    r1 = rand_ray(6)
    r2 = rand_ray(6)
    a1, a2 = line_line_closest_points(r1, r2)
    b2, b1 = line_line_closest_points(r2, r1)
    np.testing.assert_allclose(a1, b1, atol=1e-8)
    np.testing.assert_allclose(a2, b2, atol=1e-8)


def test_closest_points_intersecting_lines_coincide():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([3, -2, 0], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[3, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[3, 0, 0, 1]], atol=1e-9)


def test_closest_points_parallel_lines_not_finite():
    # documented limitation: fails (non-finite result) when axes are parallel
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 1, 0], [1, 0, 0])
    with np.errstate(all='ignore'):
        q1, q2 = line_line_closest_points(r1, r2)
    assert not (np.all(np.isfinite(q1)) and np.all(np.isfinite(q2)))
