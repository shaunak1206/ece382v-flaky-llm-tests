import numpy as np
import pytest

from homog import homog as hg


def _ray(origin, direction):
    return hg.hray(np.asarray(origin, dtype=float),
                   np.asarray(direction, dtype=float))


# ---------------------------------------------------------------------------
# hray
# ---------------------------------------------------------------------------

def test_hray_single_ray_shape():
    r = hg.hray([1.0, 2.0, 3.0], [0.0, 0.0, 5.0])
    assert r.shape == (4, 2)


def test_hray_origin_is_homogeneous_point_in_column0():
    r = hg.hray([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(r[:, 0], [1.0, 2.0, 3.0, 1.0])


def test_hray_direction_is_normalized_in_column1():
    r = hg.hray([0.0, 0.0, 0.0], [3.0, 0.0, 4.0])
    np.testing.assert_allclose(r[:3, 1], [0.6, 0.0, 0.8])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_direction_w_component_is_zero():
    r = hg.hray([1.0, 2.0, 3.0], [0.0, 1.0, 0.0])
    assert r[3, 1] == 0


def test_hray_batch_shape_and_matches_per_item():
    rng = np.random.RandomState(1)
    o = rng.randn(4, 3)
    d = rng.randn(4, 3)
    r = hg.hray(o, d)
    assert r.shape == (4, 4, 2)
    for i in range(4):
        np.testing.assert_allclose(r[i], hg.hray(o[i], d[i]))


def test_hray_broadcasts_single_origin_over_batch_of_directions():
    rng = np.random.RandomState(2)
    d = rng.randn(6, 3)
    r = hg.hray([1.0, 2.0, 3.0], d)
    assert r.shape == (6, 4, 2)
    for i in range(6):
        np.testing.assert_allclose(r[i, :, 0], [1.0, 2.0, 3.0, 1.0])
        np.testing.assert_allclose(r[i, :3, 1], d[i] / np.linalg.norm(d[i]))


def test_hray_result_is_valid_ray():
    rng = np.random.RandomState(3)
    r = hg.hray(rng.randn(5, 3), rng.randn(5, 3))
    assert hg.is_valid_rays(r)


def test_hray_returns_float64_for_integer_input():
    r = hg.hray([1, 2, 3], [0, 0, 1])
    assert r.dtype == np.float64
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])


# ---------------------------------------------------------------------------
# line_line_distance
# ---------------------------------------------------------------------------

def test_distance_skew_lines_known_value():
    # x-axis through origin vs. y-direction line through (0, 0, 1)
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 1], [0, 1, 0])
    assert np.isclose(hg.line_line_distance(r1, r2), 1.0)


def test_distance_intersecting_lines_is_zero():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 0], [0, 1, 0])
    assert np.isclose(hg.line_line_distance(r1, r2), 0.0)


def test_distance_parallel_lines_ignores_offset_along_axis():
    # Offset (5, 3, 4) along x-axis: perpendicular part is (0, 3, 4), norm 5
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([5, 3, 4], [1, 0, 0])
    assert np.isclose(hg.line_line_distance(r1, r2), 5.0)


def test_distance_parallel_lines_simple_offset():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 2, 0], [1, 0, 0])
    assert np.isclose(hg.line_line_distance(r1, r2), 2.0)


def test_distance_antiparallel_lines():
    # Direction sign is irrelevant for the distance between lines
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([7, 0, 2], [-1, 0, 0])
    assert np.isclose(hg.line_line_distance(r1, r2), 2.0)


def test_distance_invariant_to_moving_origins_along_their_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 1], [0, 1, 0])
    r1_moved = _ray([3, 0, 0], [1, 0, 0])
    r2_moved = _ray([0, 5, 1], [0, 1, 0])
    assert np.isclose(hg.line_line_distance(r1_moved, r2_moved), 1.0)
    assert np.isclose(hg.line_line_distance(r1, r2),
                      hg.line_line_distance(r1_moved, r2_moved))


def test_distance_is_symmetric():
    rng = np.random.RandomState(4)
    r1 = hg.hray(rng.randn(3), rng.randn(3))
    r2 = hg.hray(rng.randn(3), rng.randn(3))
    assert np.isclose(hg.line_line_distance(r1, r2),
                      hg.line_line_distance(r2, r1))


def test_distance_batch_matches_per_item():
    rng = np.random.RandomState(5)
    r1 = hg.hray(rng.randn(3, 3), rng.randn(3, 3))
    r2 = hg.hray(rng.randn(3, 3), rng.randn(3, 3))
    d = hg.line_line_distance(r1, r2)
    assert d.shape == (3,)
    for i in range(3):
        assert np.isclose(d[i], hg.line_line_distance(r1[i], r2[i]))


def test_distance_broadcast_single_ray_against_batch():
    # x-axis through origin vs. y-direction lines at heights z = 1..5
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = hg.hray(np.array([[0, 0, z] for z in range(1, 6)], dtype=float),
                 np.array([[0, 1, 0]] * 5, dtype=float))
    d = hg.line_line_distance(r1, r2)
    assert d.shape == (5,)
    np.testing.assert_allclose(d, [1, 2, 3, 4, 5])


# ---------------------------------------------------------------------------
# line_line_closest_points
# ---------------------------------------------------------------------------

def test_closest_points_known_configuration():
    # Line 1: x-axis through origin. Line 2: through (1, 0, 1) along y.
    # Closest points: Q1 = (1, 0, 0) on line 1, Q2 = (1, 0, 1) on line 2.
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([1, 0, 1], [0, 1, 0])
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    np.testing.assert_allclose(Q1, [1, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(Q2, [1, 0, 1, 1], atol=1e-12)


def test_closest_points_lie_on_lines_and_connector_is_perpendicular():
    rng = np.random.RandomState(6)
    for _ in range(20):
        r1 = hg.hray(rng.randn(3), rng.randn(3))
        r2 = hg.hray(rng.randn(3), rng.randn(3))
        pt1, ax1 = r1[:3, 0], r1[:3, 1]
        pt2, ax2 = r2[:3, 0], r2[:3, 1]
        Q1, Q2 = hg.line_line_closest_points(r1, r2)
        Q1, Q2 = Q1[:3], Q2[:3]
        # Q1 lies on line 1, Q2 lies on line 2
        np.testing.assert_allclose(np.cross(Q1 - pt1, ax1), 0, atol=1e-9)
        np.testing.assert_allclose(np.cross(Q2 - pt2, ax2), 0, atol=1e-9)
        # Connector is perpendicular to both lines
        assert abs(np.dot(Q1 - Q2, ax1)) < 1e-9
        assert abs(np.dot(Q1 - Q2, ax2)) < 1e-9


def test_closest_points_distance_matches_line_line_distance():
    rng = np.random.RandomState(7)
    for _ in range(20):
        r1 = hg.hray(rng.randn(3), rng.randn(3))
        r2 = hg.hray(rng.randn(3), rng.randn(3))
        Q1, Q2 = hg.line_line_closest_points(r1, r2)
        gap = np.linalg.norm(Q1[:3] - Q2[:3])
        assert np.isclose(gap, hg.line_line_distance(r1, r2), atol=1e-9)


def test_closest_points_intersecting_lines_coincide():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 0], [0, 1, 0])
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    np.testing.assert_allclose(Q1[:3], Q2[:3], atol=1e-12)
    np.testing.assert_allclose(Q1[:3], [0, 0, 0], atol=1e-12)


def test_closest_points_batch_shape_and_values():
    rng = np.random.RandomState(8)
    r1 = hg.hray(rng.randn(3, 3), rng.randn(3, 3))
    r2 = hg.hray(rng.randn(3, 3), rng.randn(3, 3))
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    assert Q1.shape == (3, 4)
    assert Q2.shape == (3, 4)
    for i in range(3):
        q1, q2 = hg.line_line_closest_points(r1[i], r2[i])
        np.testing.assert_allclose(Q1[i], q1, atol=1e-12)
        np.testing.assert_allclose(Q2[i], q2, atol=1e-12)


def test_closest_points_verbose_prints_intermediates(capsys):
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([1, 0, 1], [0, 1, 0])
    hg.line_line_closest_points(r1, r2, verbose=1)
    out = capsys.readouterr().out
    for label in ('C21', 'M', 'm2', 'R', 't1', 't2', 'Q1', 'Q2'):
        assert label in out


def test_closest_points_quiet_by_default(capsys):
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([1, 0, 1], [0, 1, 0])
    hg.line_line_closest_points(r1, r2)
    assert capsys.readouterr().out == ""


def test_closest_points_parallel_lines_currently_non_finite():
    # Documented limitation: the implementation errors for parallel axes
    # (division by zero in the cross-product normalization).
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 2, 0], [1, 0, 0])
    with np.errstate(all='ignore'):
        Q1, Q2 = hg.line_line_closest_points(r1, r2)
    assert not np.all(np.isfinite(Q1))
