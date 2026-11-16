import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    is_valid_rays,
    hdot,
    hcross,
    hnorm,
    rand_ray,
)


def _ray(origin, direction):
    """Build a batched ray of shape (1, 4, 2)."""
    return hray(np.array([origin], dtype='f8'), np.array([direction], dtype='f8'))


# ---------------------------------------------------------------- hray


def test_hray_shape_and_origin_3vec():
    r = hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1, 2, 3, 1])


def test_hray_direction_is_normalized_with_zero_w():
    r = hray([0, 0, 0], [0, 0, 5])
    assert np.allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_direction_general_normalization():
    r = hray([0, 0, 0], [3, 0, 4])
    assert np.allclose(r[:, 1], [0.6, 0, 0.8, 0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_accepts_4vec_inputs():
    r = hray([1, 2, 3, 1], [0, 2, 0, 0])
    assert np.allclose(r[:, 0], [1, 2, 3, 1])
    assert np.allclose(r[:, 1], [0, 1, 0, 0])


def test_hray_origin_w_forced_to_one():
    r = hray([1, 2, 3, 7], [1, 0, 0, 0])
    assert r[3, 0] == 1


def test_hray_3vec_and_4vec_direction_equivalent():
    r3 = hray([1, 2, 3], [1, 2, 3])
    r4 = hray([1, 2, 3], [1, 2, 3, 0])
    assert np.allclose(r3, r4)


def test_hray_batched():
    np.random.seed(0)
    o = np.random.randn(5, 3)
    d = np.random.randn(5, 3)
    r = hray(o, d)
    assert r.shape == (5, 4, 2)
    assert np.allclose(r[:, :3, 0], o)
    assert np.allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1.0)
    assert np.all(r[:, 3, 0] == 1)
    assert np.all(r[:, 3, 1] == 0)
    assert is_valid_rays(r)


def test_hray_broadcasts_single_origin_over_directions():
    np.random.seed(1)
    d = np.random.randn(5, 3)
    r = hray([1, 2, 3], d)
    assert r.shape == (5, 4, 2)
    assert np.allclose(r[:, :3, 0], [1, 2, 3])


def test_hray_is_valid_ray():
    assert is_valid_rays(hray([0.5, -1, 2], [1, 1, 1]))


def test_hray_invalid_origin_length_raises():
    with pytest.raises(ValueError):
        hray([1, 2], [0, 0, 1])


# ---------------------------------------------------------------- line_line_distance


def test_distance_skew_orthogonal_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 3], [0, 1, 0])
    assert np.allclose(line_line_distance(r1, r2), [3.0])


def test_distance_skew_oblique_lines():
    r1 = _ray([0, 0, 0], [1, 1, 0])
    r2 = _ray([0, 0, 5], [1, -1, 0])
    assert np.allclose(line_line_distance(r1, r2), [5.0])


def test_distance_intersecting_lines_is_zero():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, -4, 0], [0, 1, 0])
    assert np.allclose(line_line_distance(r1, r2), [0.0], atol=1e-8)


def test_distance_parallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([7, 3, 4], [1, 0, 0])
    assert np.allclose(line_line_distance(r1, r2), [5.0])


def test_distance_antiparallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([-2, 3, 4], [-1, 0, 0])
    assert np.allclose(line_line_distance(r1, r2), [5.0])


def test_distance_symmetric_and_invariant_to_origin_shift():
    np.random.seed(2)
    r1 = rand_ray(8)
    r2 = rand_ray(8)
    d12 = line_line_distance(r1, r2)
    d21 = line_line_distance(r2, r1)
    assert d12.shape == (8,)
    assert np.allclose(d12, d21)
    # slide origin of ray2 along its own direction
    r2s = r2.copy()
    r2s[:, :3, 0] += 2.5 * r2[:, :3, 1]
    assert np.allclose(line_line_distance(r1, r2s), d12)


# ---------------------------------------------------------------- line_line_closest_points


def test_closest_points_simple_skew():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, 0, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (1, 4)
    assert q2.shape == (1, 4)
    assert np.allclose(q1, [[2, 0, 0, 1]])
    assert np.allclose(q2, [[2, 0, 3, 1]])


def test_closest_points_intersecting_lines_coincide():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, -4, 0], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1, [[2, 0, 0, 1]])
    assert np.allclose(q2, [[2, 0, 0, 1]])


def test_closest_points_random_properties():
    np.random.seed(3)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (10, 4)
    assert q2.shape == (10, 4)
    p1, a1 = r1[:, :, 0], r1[:, :, 1]
    p2, a2 = r2[:, :, 0], r2[:, :, 1]
    # points lie on their respective lines
    assert np.allclose(hnorm(hcross(q1 - p1, a1)), 0, atol=1e-8)
    assert np.allclose(hnorm(hcross(q2 - p2, a2)), 0, atol=1e-8)
    # connecting segment is perpendicular to both lines
    diff = q2 - q1
    assert np.allclose(hdot(diff, a1), 0, atol=1e-8)
    assert np.allclose(hdot(diff, a2), 0, atol=1e-8)
    # homogeneous w component is 1
    assert np.allclose(q1[:, 3], 1)
    assert np.allclose(q2[:, 3], 1)
    # segment length equals line-line distance
    assert np.allclose(hnorm(diff), line_line_distance(r1, r2), atol=1e-8)


def test_closest_points_swapping_rays_swaps_points():
    np.random.seed(4)
    r1 = rand_ray(6)
    r2 = rand_ray(6)
    q1, q2 = line_line_closest_points(r1, r2)
    s1, s2 = line_line_closest_points(r2, r1)
    assert np.allclose(q1, s2)
    assert np.allclose(q2, s1)
