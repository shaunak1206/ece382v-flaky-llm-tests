import numpy as np
import pytest

from homog.homog import (
    hray,
    is_valid_rays,
    line_line_closest_points,
    line_line_distance,
    rand_ray,
)


def _rays(origins, directions):
    """Build a batch of rays (N, 4, 2) from (N, 3) origins and directions."""
    return hray(np.array(origins, dtype=float), np.array(directions, dtype=float))


# ---------------------------------------------------------------- hray


def test_hray_single_shape_and_columns():
    np.random.seed(0)
    r = hray([1.0, 2.0, 3.0], [0.0, 3.0, 4.0])
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1, 2, 3, 1])
    assert np.allclose(r[:, 1], [0, 0.6, 0.8, 0])


def test_hray_direction_is_normalized():
    np.random.seed(0)
    r = hray([0, 0, 0], [10.0, 0.0, 0.0])
    assert np.allclose(r[:, 1], [1, 0, 0, 0])
    r = hray([0, 0, 0], [1.0, 1.0, 1.0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)
    assert np.allclose(r[:3, 1], np.ones(3) / np.sqrt(3))


def test_hray_accepts_length4_origin_and_direction():
    np.random.seed(0)
    r = hray([1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 2.0, 0.0])
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1, 2, 3, 1])
    assert np.allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_origin_w_forced_to_one():
    np.random.seed(0)
    r = hray([1.0, 2.0, 3.0, 5.0], [1.0, 0.0, 0.0])
    assert r[3, 0] == 1
    assert r[3, 1] == 0


def test_hray_batch_shape_and_validity():
    np.random.seed(0)
    origins = np.arange(12, dtype=float).reshape(4, 3)
    dirs = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1]], dtype=float)
    r = hray(origins, dirs)
    assert r.shape == (4, 4, 2)
    assert np.allclose(r[:, :3, 0], origins)
    assert np.allclose(r[:, 3, 0], 1)
    assert np.allclose(r[:, 3, 1], 0)
    assert np.allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1.0)
    assert is_valid_rays(r)


def test_hray_broadcasts_single_direction_over_origins():
    np.random.seed(0)
    origins = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 4]], dtype=float)
    r = hray(origins, [0.0, 0.0, 5.0])
    assert r.shape == (3, 4, 2)
    assert np.allclose(r[:, :3, 0], origins)
    for i in range(3):
        assert np.allclose(r[i, :, 1], [0, 0, 1, 0])


def test_hray_does_not_modify_inputs():
    np.random.seed(0)
    o = np.array([1.0, 2.0, 3.0])
    d = np.array([0.0, 0.0, 4.0])
    hray(o, d)
    assert np.array_equal(o, [1.0, 2.0, 3.0])
    assert np.array_equal(d, [0.0, 0.0, 4.0])


# ------------------------------------------------- line_line_distance


def test_distance_skew_axis_aligned_lines():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[0, 0, 3]], [[0, 1, 0]])
    d = line_line_distance(r1, r2)
    assert d.shape == (1,)
    assert np.allclose(d, 3.0)


def test_distance_intersecting_lines_is_zero():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[2, 5, 0]], [[0, 1, 0]])
    assert np.allclose(line_line_distance(r1, r2), 0.0, atol=1e-9)


def test_distance_parallel_and_antiparallel_lines():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0]])
    r2 = _rays([[5, 3, 4], [-2, 3, 4]], [[1, 0, 0], [-1, 0, 0]])
    d = line_line_distance(r1, r2)
    assert np.allclose(d, [5.0, 5.0])


def test_distance_is_symmetric():
    np.random.seed(0)
    r1 = _rays([[1, 2, 3]], [[1, 1, 0]])
    r2 = _rays([[-2, 0, 1]], [[0, 1, 1]])
    assert np.allclose(line_line_distance(r1, r2), line_line_distance(r2, r1))


def test_distance_matches_triple_product_formula():
    np.random.seed(0)
    o1 = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])
    d1 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    o2 = np.array([[-2.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    d2 = np.array([[0.0, 1.0, 1.0], [1.0, 2.0, 0.0]])
    r1, r2 = _rays(o1, d1), _rays(o2, d2)
    n1 = d1 / np.linalg.norm(d1, axis=-1)[:, None]
    n2 = d2 / np.linalg.norm(d2, axis=-1)[:, None]
    cr = np.cross(n1, n2)
    expected = np.abs(np.sum((o2 - o1) * cr, axis=-1)) / np.linalg.norm(cr, axis=-1)
    assert np.allclose(line_line_distance(r1, r2), expected)


def test_distance_invariant_to_sliding_origin_along_line():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2a = _rays([[0, 0, 3]], [[0, 1, 0]])
    r2b = _rays([[0, 7, 3]], [[0, 1, 0]])
    assert np.allclose(line_line_distance(r1, r2a), line_line_distance(r1, r2b))


def test_distance_broadcast_single_ray_against_batch():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[0, 0, 1], [0, 0, 2], [0, 5, -4]], [[0, 1, 0]] * 3)
    d = line_line_distance(r1, r2)
    assert d.shape == (3,)
    assert np.allclose(d, [1.0, 2.0, 4.0])


# ------------------------------------------- line_line_closest_points


def test_closest_points_axis_aligned_example():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[1, 2, 3]], [[0, 1, 0]])
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1, [[1, 0, 0, 1]])
    assert np.allclose(q2, [[1, 0, 3, 1]])


def test_closest_points_intersecting_lines_give_intersection():
    np.random.seed(0)
    p = np.array([1.0, -2.0, 0.5])
    d1 = np.array([1.0, 0.0, 0.0])
    d2 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    r1 = _rays([p + 2 * d1], [d1])
    r2 = _rays([p - 3 * d2], [d2])
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1[0, :3], p, atol=1e-8)
    assert np.allclose(q2[0, :3], p, atol=1e-8)


def test_closest_points_are_points_with_w_one():
    np.random.seed(0)
    r1 = _rays([[1, 2, 3]], [[1, 1, 0]])
    r2 = _rays([[-2, 0, 1]], [[0, 1, 1]])
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (1, 4)
    assert q2.shape == (1, 4)
    assert np.allclose(q1[:, 3], 1)
    assert np.allclose(q2[:, 3], 1)


def test_closest_points_lie_on_their_lines():
    np.random.seed(0)
    o1 = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0]])
    d1 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    o2 = np.array([[-2.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    d2 = np.array([[0.0, 1.0, 1.0], [1.0, 2.0, 0.0]])
    r1, r2 = _rays(o1, d1), _rays(o2, d2)
    q1, q2 = line_line_closest_points(r1, r2)
    c1 = np.cross(q1[:, :3] - o1, r1[:, :3, 1])
    c2 = np.cross(q2[:, :3] - o2, r2[:, :3, 1])
    assert np.allclose(c1, 0, atol=1e-8)
    assert np.allclose(c2, 0, atol=1e-8)


def test_closest_points_segment_perpendicular_to_both_lines_seeded_random():
    np.random.seed(1234)
    r1 = rand_ray(6)
    r2 = rand_ray(6)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (6, 4)
    diff = (q1 - q2)[:, :3]
    assert np.allclose(np.sum(diff * r1[:, :3, 1], axis=-1), 0, atol=1e-6)
    assert np.allclose(np.sum(diff * r2[:, :3, 1], axis=-1), 0, atol=1e-6)


def test_closest_points_separation_equals_line_distance():
    np.random.seed(0)
    o1 = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 2.0], [0.0, 0.0, 0.0]])
    d1 = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    o2 = np.array([[-2.0, 0.0, 1.0], [4.0, 1.0, 0.0], [1.0, 2.0, 3.0]])
    d2 = np.array([[0.0, 1.0, 1.0], [1.0, 2.0, 0.0], [0.0, 1.0, 0.0]])
    r1, r2 = _rays(o1, d1), _rays(o2, d2)
    q1, q2 = line_line_closest_points(r1, r2)
    sep = np.linalg.norm((q1 - q2)[:, :3], axis=-1)
    assert np.allclose(sep, line_line_distance(r1, r2), atol=1e-8)


def test_closest_points_swapping_rays_swaps_results():
    np.random.seed(0)
    r1 = _rays([[1, 2, 3]], [[1, 1, 0]])
    r2 = _rays([[-2, 0, 1]], [[0, 1, 1]])
    q1, q2 = line_line_closest_points(r1, r2)
    p2, p1 = line_line_closest_points(r2, r1)
    assert np.allclose(q1, p1)
    assert np.allclose(q2, p2)


def test_closest_points_batch_shape():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]] * 3, [[1, 0, 0]] * 3)
    r2 = _rays([[1, 2, 3], [2, 2, 4], [-1, 5, 6]], [[0, 1, 0]] * 3)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (3, 4)
    assert q2.shape == (3, 4)
    assert np.allclose(q1[:, 0], [1, 2, -1])
    assert np.allclose(q2[:, 2], [3, 4, 6])
