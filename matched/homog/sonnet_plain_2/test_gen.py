import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    is_valid_rays,
    hcross,
    hdot,
    hnorm,
)


def _batch_ray(origin, direction):
    """Build a batch of one ray, shape (1, 4, 2)."""
    return hray(np.array([origin], dtype=float),
                np.array([direction], dtype=float))


def _rand_rays(seed, n):
    rs = np.random.RandomState(seed)
    return hray(rs.randn(n, 3), rs.randn(n, 3)), rs


# ---------------------------------------------------------------- hray


def test_hray_basic_values():
    r = hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1, 2, 3, 1])
    assert np.allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = hray([0, 0, 0], [0, 3, 4])
    assert np.allclose(r[:, 1], [0, 0.6, 0.8, 0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_accepts_four_vectors():
    r = hray([1, 2, 3, 1], [0, 2, 0, 0])
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1, 2, 3, 1])
    assert np.allclose(r[:, 1], [0, 1, 0, 0])


def test_hray_origin_w_is_forced_to_one():
    r = hray([1, 2, 3, 5], [1, 0, 0])
    assert r[3, 0] == 1
    assert r[3, 1] == 0


def test_hray_is_valid_ray():
    r = hray([0.5, -1.5, 2.0], [1, 2, 3])
    assert is_valid_rays(r)


def test_hray_batch_shape_and_rows():
    rs = np.random.RandomState(0)
    origins = rs.randn(5, 3)
    dirs = rs.randn(5, 3)
    r = hray(origins, dirs)
    assert r.shape == (5, 4, 2)
    assert np.allclose(r[:, :3, 0], origins)
    assert np.allclose(r[:, 3, 0], 1)
    assert np.allclose(r[:, 3, 1], 0)
    assert np.allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1.0)
    assert is_valid_rays(r)


def test_hray_broadcasts_single_origin_over_directions():
    dirs = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=float)
    r = hray([1, 1, 1], dirs)
    assert r.shape == (3, 4, 2)
    for i in range(3):
        assert np.allclose(r[i, :, 0], [1, 1, 1, 1])
    assert np.allclose(r[:, :3, 1], np.eye(3))


def test_hray_bad_origin_length_raises():
    with pytest.raises(ValueError):
        hray([1, 2], [1, 0, 0])


# ------------------------------------------------- line_line_distance


def test_distance_axis_aligned_skew_lines():
    r1 = _batch_ray([0, 0, 0], [1, 0, 0])
    r2 = _batch_ray([0, 0, 3], [0, 1, 0])
    d = line_line_distance(r1, r2)
    assert d.shape == (1,)
    assert np.allclose(d, 3.0)


def test_distance_general_skew_lines():
    r1 = _batch_ray([1, 2, 3], [1, 1, 0])
    r2 = _batch_ray([0, 0, 0], [0, 1, 1])
    d = line_line_distance(r1, r2)
    assert np.allclose(d, 2.0 / np.sqrt(3.0))


def test_distance_parallel_lines():
    r1 = _batch_ray([0, 0, 0], [1, 0, 0])
    r2 = _batch_ray([5, 2, 0], [1, 0, 0])
    assert np.allclose(line_line_distance(r1, r2), 2.0)


def test_distance_antiparallel_lines():
    r1 = _batch_ray([0, 0, 0], [1, 0, 0])
    r2 = _batch_ray([5, 0, -4], [-1, 0, 0])
    assert np.allclose(line_line_distance(r1, r2), 4.0)


def test_distance_intersecting_lines_is_zero():
    r1 = _batch_ray([0, 0, 0], [1, 0, 0])
    r2 = _batch_ray([2, -1, 0], [0, 1, 0])
    assert np.allclose(line_line_distance(r1, r2), 0.0)


def test_distance_batch_of_mixed_pairs():
    r1 = hray([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
              [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    r2 = hray([[0, 0, 3], [0, 2, 0], [2, -1, 0]],
              [[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    d = line_line_distance(r1, r2)
    assert d.shape == (3,)
    assert np.allclose(d, [3.0, 2.0, 0.0])


def test_distance_is_symmetric():
    r1, _ = _rand_rays(1234, 6)
    r2, _ = _rand_rays(4321, 6)
    assert np.allclose(line_line_distance(r1, r2), line_line_distance(r2, r1))


def test_distance_unchanged_by_sliding_origin_along_axis():
    o = np.array([[1.0, -2.0, 0.5]])
    d = np.array([[1.0, 2.0, -1.0]])
    r1 = hray(o, d)
    r1_slid = hray(o + 4.5 * d / np.linalg.norm(d), d)
    r2 = hray([[0.0, 0.0, 3.0]], [[0.0, 1.0, 1.0]])
    assert np.allclose(line_line_distance(r1, r2),
                       line_line_distance(r1_slid, r2))


# ------------------------------------------- line_line_closest_points


def test_closest_points_axis_aligned_skew():
    r1 = _batch_ray([0, 0, 0], [1, 0, 0])
    r2 = _batch_ray([2, -5, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (1, 4)
    assert q2.shape == (1, 4)
    assert np.allclose(q1, [[2, 0, 0, 1]])
    assert np.allclose(q2, [[2, 0, 3, 1]])


def test_closest_points_intersecting_lines_coincide():
    r1 = _batch_ray([0, 0, 0], [1, 0, 0])
    r2 = _batch_ray([2, -1, 0], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1, [[2, 0, 0, 1]])
    assert np.allclose(q2, [[2, 0, 0, 1]])


def test_closest_points_lie_on_lines_and_segment_is_perpendicular():
    r1, _ = _rand_rays(7, 8)
    r2, _ = _rand_rays(8, 8)
    pt1, ax1 = r1[:, :, 0], r1[:, :, 1]
    pt2, ax2 = r2[:, :, 0], r2[:, :, 1]
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (8, 4)
    assert q2.shape == (8, 4)
    # q1 lies on line 1 and q2 lies on line 2
    assert np.allclose(hnorm(hcross(q1 - pt1, ax1)), 0, atol=1e-9)
    assert np.allclose(hnorm(hcross(q2 - pt2, ax2)), 0, atol=1e-9)
    # the connecting segment is perpendicular to both lines
    seg = q2 - q1
    assert np.allclose(hdot(seg, ax1), 0, atol=1e-9)
    assert np.allclose(hdot(seg, ax2), 0, atol=1e-9)
    # points keep w == 1
    assert np.allclose(q1[:, 3], 1)
    assert np.allclose(q2[:, 3], 1)


def test_closest_points_segment_length_matches_line_line_distance():
    r1, _ = _rand_rays(21, 6)
    r2, _ = _rand_rays(22, 6)
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(hnorm(q2 - q1), line_line_distance(r1, r2))


def test_closest_points_swapping_rays_swaps_points():
    r1, _ = _rand_rays(31, 5)
    r2, _ = _rand_rays(32, 5)
    q1, q2 = line_line_closest_points(r1, r2)
    p2, p1 = line_line_closest_points(r2, r1)
    assert np.allclose(q1, p1)
    assert np.allclose(q2, p2)


def test_closest_points_translate_with_rays():
    shift = np.array([3.0, -2.0, 7.0])
    rs = np.random.RandomState(5)
    o1, d1 = rs.randn(4, 3), rs.randn(4, 3)
    o2, d2 = rs.randn(4, 3), rs.randn(4, 3)
    q1, q2 = line_line_closest_points(hray(o1, d1), hray(o2, d2))
    s1, s2 = line_line_closest_points(hray(o1 + shift, d1),
                                      hray(o2 + shift, d2))
    assert np.allclose(s1[:, :3], q1[:, :3] + shift)
    assert np.allclose(s2[:, :3], q2[:, :3] + shift)
    assert np.allclose(s1[:, 3], 1)


def test_closest_points_parallel_lines_are_not_finite():
    r1 = _batch_ray([0, 0, 0], [1, 0, 0])
    r2 = _batch_ray([0, 2, 0], [1, 0, 0])
    with np.errstate(all='ignore'):
        q1, q2 = line_line_closest_points(r1, r2)
    assert np.isnan(q1).any()
    assert np.isnan(q2).any()
