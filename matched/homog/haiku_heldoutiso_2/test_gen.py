import numpy as np
import pytest

import homog.homog as hg


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ray(origin, direction):
    """Single ray, shape (4, 2)."""
    return hg.hray(np.asarray(origin, dtype=float),
                   np.asarray(direction, dtype=float))


def _batch(origins, directions):
    """Batch of rays from (N, 3) origins and (N, 3) directions, shape (N, 4, 2)."""
    return hg.hray(np.asarray(origins, dtype=float),
                   np.asarray(directions, dtype=float))


# Canonical configurations
X_AXIS_RAY = _ray((0, 0, 0), (1, 0, 0))               # x-axis through origin
Y_AXIS_RAY = _ray((0, 0, 0), (0, 1, 0))               # y-axis through origin
SKEW_RAY = _ray((0, 0, 5), (0, 1, 0))                 # distance 5 from X_AXIS_RAY
PARALLEL_RAY = _ray((5, 3, 4), (1, 0, 0))             # parallel to X, offset (0,3,4)
ANTIPARALLEL_RAY = _ray((-2, 3, 4), (-1, 0, 0))       # antiparallel, offset (0,3,4)


def _stack(*rays):
    return np.stack(rays, axis=0)


# ---------------------------------------------------------------------------
# hray tests
# ---------------------------------------------------------------------------

def test_hray_single_output_shape():
    r = _ray((1, 2, 3), (0, 0, 2))
    assert r.shape == (4, 2)


def test_hray_origin_stored_in_column_zero_with_w_one():
    r = _ray((1.5, -2.0, 3.0), (0, 1, 0))
    np.testing.assert_allclose(r[:3, 0], [1.5, -2.0, 3.0])
    assert r[3, 0] == 1


def test_hray_direction_normalized_and_w_zero():
    r = _ray((0, 0, 0), (0, 0, 2))
    np.testing.assert_allclose(r[:3, 1], [0, 0, 1], atol=1e-12)
    assert r[3, 1] == 0
    assert np.linalg.norm(r[:3, 1]) == pytest.approx(1.0)


def test_hray_scaled_direction_gives_same_ray_as_unit_direction():
    a = _ray((1, 1, 1), (3, 4, 0))
    b = _ray((1, 1, 1), (0.6, 0.8, 0))
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_hray_accepts_homogeneous_inputs():
    from_3 = _ray((1, 2, 3), (0, 0, 2))
    r = hg.hray(np.array([1.0, 2.0, 3.0, 1.0]),
                np.array([0.0, 0.0, 2.0, 0.0]))
    np.testing.assert_allclose(r, from_3, atol=1e-12)


def test_hray_result_is_valid_ray():
    r = _ray((0.3, -0.7, 2.0), (1, 2, -2))
    assert hg.is_valid_rays(r)


def test_hray_batch_shape_and_contents():
    rng = np.random.RandomState(0)
    origins = rng.randn(5, 3)
    dirs = rng.randn(5, 3)
    r = _batch(origins, dirs)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins)
    np.testing.assert_allclose(r[:, 3, 0], 1.0)
    np.testing.assert_allclose(r[:, 3, 1], 0.0)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1.0)
    assert hg.is_valid_rays(r)


def test_hray_single_origin_broadcasts_over_many_directions():
    origin = np.array([1.0, 2.0, 3.0])
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    r = hg.hray(origin, dirs)
    assert r.shape == (4, 4, 2)
    for i in range(4):
        np.testing.assert_allclose(r[i, :3, 0], origin)
        np.testing.assert_allclose(r[i, 3, 0], 1.0)


def test_hray_does_not_modify_inputs():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([0.0, 0.0, 5.0, 0.0])
    origin_copy = origin.copy()
    direction_copy = direction.copy()
    hg.hray(origin, direction)
    np.testing.assert_array_equal(origin, origin_copy)
    np.testing.assert_array_equal(direction, direction_copy)


def test_hray_zero_direction_gives_nan_direction():
    with np.errstate(all='ignore'):
        r = _ray((0, 0, 0), (0, 0, 0))
    assert np.all(np.isnan(r[:3, 1]))


def test_hray_origin_wrong_length_raises():
    with pytest.raises(ValueError):
        hg.hray(np.array([1.0, 2.0]), np.array([0.0, 0.0, 1.0]))


# ---------------------------------------------------------------------------
# line_line_distance tests
# ---------------------------------------------------------------------------

def test_distance_skew_lines_known_value():
    d = hg.line_line_distance(_stack(X_AXIS_RAY), _stack(SKEW_RAY))
    np.testing.assert_allclose(d, [5.0], atol=1e-12)


def test_distance_intersecting_lines_is_zero():
    d = hg.line_line_distance(_stack(X_AXIS_RAY), _stack(Y_AXIS_RAY))
    np.testing.assert_allclose(d, [0.0], atol=1e-12)


def test_distance_parallel_lines_known_value():
    d = hg.line_line_distance(_stack(X_AXIS_RAY), _stack(PARALLEL_RAY))
    np.testing.assert_allclose(d, [5.0], atol=1e-12)


def test_distance_antiparallel_lines_known_value():
    d = hg.line_line_distance(_stack(X_AXIS_RAY), _stack(ANTIPARALLEL_RAY))
    np.testing.assert_allclose(d, [5.0], atol=1e-12)


def test_distance_is_symmetric():
    rng = np.random.RandomState(7)
    a = _batch(rng.randn(10, 3), rng.randn(10, 3))
    b = _batch(rng.randn(10, 3), rng.randn(10, 3))
    np.testing.assert_allclose(hg.line_line_distance(a, b),
                               hg.line_line_distance(b, a), atol=1e-10)


def test_distance_invariant_to_sliding_point_along_line():
    # Moving the base point of ray2 along its own direction must not
    # change the distance between the two lines.
    slid = _ray((0, 7, 5), (0, 1, 0))
    d = hg.line_line_distance(_stack(X_AXIS_RAY), _stack(slid))
    np.testing.assert_allclose(d, [5.0], atol=1e-12)


def test_distance_batch_mixed_cases():
    ray1 = _stack(X_AXIS_RAY, X_AXIS_RAY, X_AXIS_RAY)
    ray2 = _stack(SKEW_RAY, Y_AXIS_RAY, PARALLEL_RAY)
    d = hg.line_line_distance(ray1, ray2)
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [5.0, 0.0, 5.0], atol=1e-12)


# ---------------------------------------------------------------------------
# line_line_closest_points tests
# ---------------------------------------------------------------------------

def test_closest_points_skew_known_values():
    # ray1: x-axis through origin; ray2: (3,2,5) + t*y
    # closest points: (3,0,0) on ray1 and (3,0,5) on ray2
    ray1 = _stack(_ray((0, 0, 0), (1, 0, 0)))
    ray2 = _stack(_ray((3, 2, 5), (0, 1, 0)))
    q1, q2 = hg.line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[0, :3], [3, 0, 0], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [3, 0, 5], atol=1e-12)


def test_closest_points_intersecting_lines_coincide_at_intersection():
    q1, q2 = hg.line_line_closest_points(_stack(X_AXIS_RAY), _stack(Y_AXIS_RAY))
    np.testing.assert_allclose(q1[0, :3], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [0, 0, 0], atol=1e-12)


def test_closest_points_output_is_homogeneous_points():
    rng = np.random.RandomState(3)
    a = _batch(rng.randn(6, 3), rng.randn(6, 3))
    b = _batch(rng.randn(6, 3), rng.randn(6, 3))
    q1, q2 = hg.line_line_closest_points(a, b)
    np.testing.assert_allclose(q1[:, 3], 1.0)
    np.testing.assert_allclose(q2[:, 3], 1.0)


def test_closest_points_lie_on_their_respective_lines():
    rng = np.random.RandomState(11)
    a = _batch(rng.randn(15, 3), rng.randn(15, 3))
    b = _batch(rng.randn(15, 3), rng.randn(15, 3))
    q1, q2 = hg.line_line_closest_points(a, b)
    # (Q1 - pt1) must be parallel to ax1 -> cross product is zero
    cross1 = np.cross(q1[:, :3] - a[:, :3, 0], a[:, :3, 1])
    cross2 = np.cross(q2[:, :3] - b[:, :3, 0], b[:, :3, 1])
    np.testing.assert_allclose(cross1, 0, atol=1e-9)
    np.testing.assert_allclose(cross2, 0, atol=1e-9)


def test_closest_points_connecting_segment_perpendicular_to_both_lines():
    rng = np.random.RandomState(12)
    a = _batch(rng.randn(15, 3), rng.randn(15, 3))
    b = _batch(rng.randn(15, 3), rng.randn(15, 3))
    q1, q2 = hg.line_line_closest_points(a, b)
    seg = q1[:, :3] - q2[:, :3]
    np.testing.assert_allclose(np.sum(seg * a[:, :3, 1], axis=-1), 0, atol=1e-9)
    np.testing.assert_allclose(np.sum(seg * b[:, :3, 1], axis=-1), 0, atol=1e-9)


def test_closest_points_distance_matches_line_line_distance():
    rng = np.random.RandomState(21)
    a = _batch(rng.randn(20, 3), rng.randn(20, 3))
    b = _batch(rng.randn(20, 3), rng.randn(20, 3))
    q1, q2 = hg.line_line_closest_points(a, b)
    seg_len = np.linalg.norm(q1[:, :3] - q2[:, :3], axis=-1)
    d = hg.line_line_distance(a, b)
    np.testing.assert_allclose(seg_len, d, atol=1e-9)


def test_closest_points_swapping_rays_swaps_outputs():
    rng = np.random.RandomState(5)
    a = _batch(rng.randn(8, 3), rng.randn(8, 3))
    b = _batch(rng.randn(8, 3), rng.randn(8, 3))
    q1, q2 = hg.line_line_closest_points(a, b)
    p1, p2 = hg.line_line_closest_points(b, a)
    np.testing.assert_allclose(p1, q2, atol=1e-10)
    np.testing.assert_allclose(p2, q1, atol=1e-10)


def test_closest_points_batch_output_shape():
    rng = np.random.RandomState(8)
    a = _batch(rng.randn(4, 3), rng.randn(4, 3))
    b = _batch(rng.randn(4, 3), rng.randn(4, 3))
    q1, q2 = hg.line_line_closest_points(a, b)
    assert q1.shape == (4, 4)
    assert q2.shape == (4, 4)


def test_closest_points_verbose_prints_intermediates(capsys):
    hg.line_line_closest_points(_stack(X_AXIS_RAY), _stack(SKEW_RAY), verbose=1)
    out = capsys.readouterr().out
    for label in ('C21', 'M', 'm2', 'R', 't1', 't2', 'Q1', 'Q2'):
        assert label in out


def test_closest_points_quiet_by_default(capsys):
    hg.line_line_closest_points(_stack(X_AXIS_RAY), _stack(SKEW_RAY))
    assert capsys.readouterr().out == ""
