import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    is_valid_rays,
    hdot,
    hcross,
)


def _rays(origins, dirs):
    return hray(np.array(origins, dtype=float), np.array(dirs, dtype=float))


# A fixed set of non-parallel line pairs (origin1, dir1, origin2, dir2).
_PAIRS = [
    ((0, 0, 0), (1, 0, 0), (2, 5, 3), (0, 1, 0)),
    ((1, -2, 3), (1, 1, 0), (0, 4, -1), (0, 1, 1)),
    ((-3, 2, 1), (2, -1, 0.5), (4, 4, 4), (-1, 2, 3)),
    ((0.5, 0.5, 0.5), (0, 0, 1), (3, -2, 7), (1, 1, 1)),
]


def _pair_rays():
    o1 = [p[0] for p in _PAIRS]
    d1 = [p[1] for p in _PAIRS]
    o2 = [p[2] for p in _PAIRS]
    d2 = [p[3] for p in _PAIRS]
    return _rays(o1, d1), _rays(o2, d2)


# ---------------------------- hray ----------------------------

def test_hray_single_3vectors_shape_and_values():
    np.random.seed(0)
    r = hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1], atol=1e-12)
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0], atol=1e-12)


def test_hray_normalizes_direction():
    np.random.seed(0)
    r = hray([0, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(r[:, 1], [0.6, 0.8, 0, 0], atol=1e-12)
    assert abs(np.linalg.norm(r[:3, 1]) - 1) < 1e-12


def test_hray_accepts_4vectors():
    np.random.seed(0)
    r = hray([1, 2, 3, 1], [0, 0, 2, 0])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1], atol=1e-12)
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0], atol=1e-12)


def test_hray_origin_has_w_one_and_direction_w_zero():
    np.random.seed(0)
    r = hray([5, -6, 7], [1, 2, 3])
    assert r[3, 0] == 1
    assert r[3, 1] == 0


def test_hray_batch_shape_and_values():
    np.random.seed(0)
    origins = np.arange(15, dtype=float).reshape(5, 3)
    dirs = np.array([[1, 0, 0], [0, 2, 0], [0, 0, -3], [1, 1, 0], [1, 2, 2]],
                    dtype=float)
    r = hray(origins, dirs)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins, atol=1e-12)
    np.testing.assert_allclose(r[:, 3, 0], 1, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1,
                               atol=1e-12)
    np.testing.assert_allclose(r[2, :, 1], [0, 0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(r[4, :, 1], [1 / 3, 2 / 3, 2 / 3, 0],
                               atol=1e-12)


def test_hray_broadcast_single_origin_many_directions():
    np.random.seed(0)
    dirs = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    r = hray([1, 2, 3], dirs)
    assert r.shape == (2, 4, 2)
    for i in range(2):
        np.testing.assert_allclose(r[i, :, 0], [1, 2, 3, 1], atol=1e-12)
    np.testing.assert_allclose(r[0, :, 1], [1, 0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(r[1, :, 1], [0, 1, 0, 0], atol=1e-12)


def test_hray_broadcast_many_origins_single_direction():
    np.random.seed(0)
    origins = np.array([[0, 0, 0], [1, 1, 1], [2, 3, 4]], dtype=float)
    r = hray(origins, [0, 3, 0])
    assert r.shape == (3, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins, atol=1e-12)
    for i in range(3):
        np.testing.assert_allclose(r[i, :, 1], [0, 1, 0, 0], atol=1e-12)


def test_hray_result_is_valid_ray():
    np.random.seed(0)
    r = hray([[1, 2, 3], [-1, 0, 4]], [[1, 1, 1], [0, -2, 5]])
    assert is_valid_rays(r)


# ----------------------- line_line_distance -----------------------

def test_distance_perpendicular_skew_lines():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[0, 0, 3]], [[0, 1, 0]])
    d = line_line_distance(r1, r2)
    assert d.shape == (1,)
    np.testing.assert_allclose(d, [3.0], atol=1e-9)


def test_distance_skew_with_offset_origins():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[2, 5, 3]], [[0, 1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [3.0], atol=1e-9)


def test_distance_non_orthogonal_skew_lines():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[1, 2, 3]], [[1, 1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [3.0], atol=1e-9)


def test_distance_intersecting_lines_is_zero():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[1, -2, 0]], [[0, 1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [0.0], atol=1e-9)


def test_distance_parallel_lines():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[7, 2, 0]], [[1, 0, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [2.0], atol=1e-9)


def test_distance_antiparallel_lines():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[0, 0, 1]])
    r2 = _rays([[3, 4, 10]], [[0, 0, -1]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0], atol=1e-9)


def test_distance_symmetric_and_batched():
    np.random.seed(0)
    r1, r2 = _pair_rays()
    d12 = line_line_distance(r1, r2)
    d21 = line_line_distance(r2, r1)
    assert d12.shape == (len(_PAIRS),)
    np.testing.assert_allclose(d12, d21, atol=1e-9)
    assert np.all(d12 >= 0)


def test_distance_invariant_to_sliding_origin_along_axis():
    np.random.seed(0)
    r1, r2 = _pair_rays()
    base = line_line_distance(r1, r2)
    r1s = r1.copy()
    r2s = r2.copy()
    r1s[:, :3, 0] += 2.5 * r1[:, :3, 1]
    r2s[:, :3, 0] -= 4.0 * r2[:, :3, 1]
    np.testing.assert_allclose(line_line_distance(r1s, r2s), base, atol=1e-8)


# --------------------- line_line_closest_points ---------------------

def test_closest_points_perpendicular_skew():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[2, 5, 3]], [[0, 1, 0]])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[2, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[2, 0, 3, 1]], atol=1e-9)


def test_closest_points_non_orthogonal_skew():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[1, 2, 3]], [[1, 1, 0]])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[-1, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[-1, 0, 3, 1]], atol=1e-9)


def test_closest_points_intersecting_lines_coincide():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[1, -2, 0]], [[0, 1, 0]])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[1, 0, 0, 1]], atol=1e-9)
    np.testing.assert_allclose(q2, [[1, 0, 0, 1]], atol=1e-9)


def test_closest_points_geometric_properties_batched():
    np.random.seed(0)
    r1, r2 = _pair_rays()
    q1, q2 = line_line_closest_points(r1, r2)
    n = len(_PAIRS)
    assert q1.shape == (n, 4)
    assert q2.shape == (n, 4)
    np.testing.assert_allclose(q1[:, 3], 1, atol=1e-12)
    np.testing.assert_allclose(q2[:, 3], 1, atol=1e-12)
    ax1, ax2 = r1[:, :, 1], r2[:, :, 1]
    # Q1 lies on line 1, Q2 lies on line 2
    off1 = hcross(q1 - r1[:, :, 0], ax1)
    off2 = hcross(q2 - r2[:, :, 0], ax2)
    np.testing.assert_allclose(off1[:, :3], 0, atol=1e-8)
    np.testing.assert_allclose(off2[:, :3], 0, atol=1e-8)
    # connecting segment is perpendicular to both lines
    seg = q2 - q1
    np.testing.assert_allclose(hdot(seg, ax1), 0, atol=1e-8)
    np.testing.assert_allclose(hdot(seg, ax2), 0, atol=1e-8)


def test_closest_points_separation_matches_line_line_distance():
    np.random.seed(0)
    r1, r2 = _pair_rays()
    q1, q2 = line_line_closest_points(r1, r2)
    sep = np.linalg.norm((q2 - q1)[:, :3], axis=-1)
    np.testing.assert_allclose(sep, line_line_distance(r1, r2), atol=1e-8)


def test_closest_points_swapping_rays_swaps_results():
    np.random.seed(0)
    r1, r2 = _pair_rays()
    q1, q2 = line_line_closest_points(r1, r2)
    p1, p2 = line_line_closest_points(r2, r1)
    np.testing.assert_allclose(p1, q2, atol=1e-8)
    np.testing.assert_allclose(p2, q1, atol=1e-8)


def test_closest_points_parallel_lines_not_finite():
    np.random.seed(0)
    r1 = _rays([[0, 0, 0]], [[1, 0, 0]])
    r2 = _rays([[0, 1, 0]], [[1, 0, 0]])
    with np.errstate(all='ignore'):
        q1, q2 = line_line_closest_points(r1, r2)
    assert not np.all(np.isfinite(q1))
