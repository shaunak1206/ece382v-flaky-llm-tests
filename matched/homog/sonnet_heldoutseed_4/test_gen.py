import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    is_valid_rays,
    hnorm,
    hdot,
    rand_ray,
)


def _ray(origin, direction):
    """Build a batch-of-one ray with shape (1, 4, 2)."""
    return hray(np.array([origin], dtype=float), np.array([direction], dtype=float))


# ---------------------------------------------------------------- hray

def test_hray_single_shape_and_contents():
    r = hray([1, 2, 3], [0, 0, 2])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = hray([0, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(r[:3, 1], [0.6, 0.8, 0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_accepts_four_component_inputs():
    r = hray([1, 2, 3, 1], [0, 5, 0, 0])
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 1, 0, 0])


def test_hray_origin_w_forced_to_one():
    r = hray([1, 2, 3, 7], [1, 0, 0, 0])
    assert r[3, 0] == 1
    assert r[3, 1] == 0


def test_hray_is_valid_ray():
    r = hray([0.5, -1, 2], [1, 2, 3])
    assert is_valid_rays(r)


def test_hray_batch_origins_broadcast_direction():
    origins = np.random.RandomState(0).randn(5, 3)
    r = hray(origins, [0, 1, 0])
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins)
    np.testing.assert_allclose(r[:, :, 1], np.tile([0, 1, 0, 0], (5, 1)))
    assert is_valid_rays(r)


def test_hray_batch_both():
    rs = np.random.RandomState(1)
    origins = rs.randn(4, 3)
    dirs = rs.randn(4, 3)
    r = hray(origins, dirs)
    assert r.shape == (4, 4, 2)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1.0)
    # direction is parallel to the input direction
    expected = dirs / np.linalg.norm(dirs, axis=-1)[:, None]
    np.testing.assert_allclose(r[:, :3, 1], expected)


def test_hray_bad_origin_length_raises():
    with pytest.raises(ValueError):
        hray([1, 2], [0, 0, 1])


# ------------------------------------------------- line_line_distance

def test_distance_skew_axes():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 3], [0, 1, 0])
    d = line_line_distance(r1, r2)
    assert d.shape == (1,)
    assert np.isclose(d[0], 3.0)


def test_distance_intersecting_lines_is_zero():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, -5, 0], [0, 1, 0])
    d = line_line_distance(r1, r2)
    assert np.isclose(d[0], 0.0, atol=1e-8)


def test_distance_parallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([7, 3, 4], [1, 0, 0])
    d = line_line_distance(r1, r2)
    assert np.isclose(d[0], 5.0)


def test_distance_antiparallel_lines():
    r1 = _ray([0, 0, 0], [0, 0, 1])
    r2 = _ray([2, 0, 9], [0, 0, -1])
    d = line_line_distance(r1, r2)
    assert np.isclose(d[0], 2.0)


def test_distance_symmetric_and_nonnegative():
    np.random.seed(3)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    d12 = line_line_distance(r1, r2)
    d21 = line_line_distance(r2, r1)
    assert d12.shape == (10,)
    assert np.all(d12 >= 0)
    np.testing.assert_allclose(d12, d21, atol=1e-8)


def test_distance_invariant_to_moving_origin_along_line():
    np.random.seed(4)
    r1 = rand_ray(6)
    r2 = rand_ray(6)
    r1b = r1.copy()
    r1b[:, :3, 0] += 2.5 * r1[:, :3, 1]
    r2b = r2.copy()
    r2b[:, :3, 0] -= 1.7 * r2[:, :3, 1]
    np.testing.assert_allclose(line_line_distance(r1, r2),
                               line_line_distance(r1b, r2b), atol=1e-8)


# -------------------------------------------- line_line_closest_points

def test_closest_points_simple_skew():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([1, 0, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1[0], [1, 0, 0, 1], atol=1e-10)
    np.testing.assert_allclose(q2[0], [1, 0, 3, 1], atol=1e-10)


def test_closest_points_intersecting_lines_coincide():
    r1 = _ray([0, 0, 0], [1, 1, 0])
    r2 = _ray([4, 0, 0], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, q2, atol=1e-10)
    np.testing.assert_allclose(q1[0], [4, 4, 0, 1], atol=1e-10)


def test_closest_points_properties_random():
    np.random.seed(5)
    r1 = rand_ray(20)
    r2 = rand_ray(20)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (20, 4)
    assert q2.shape == (20, 4)
    ax1, ax2 = r1[:, :, 1], r2[:, :, 1]
    diff = q2 - q1
    # connecting segment is perpendicular to both lines
    np.testing.assert_allclose(hdot(diff, ax1), 0, atol=1e-8)
    np.testing.assert_allclose(hdot(diff, ax2), 0, atol=1e-8)
    # points lie on their lines: (Q - P) parallel to axis
    for q, r in ((q1, r1), (q2, r2)):
        v = q[:, :3] - r[:, :3, 0]
        cross = np.cross(v, r[:, :3, 1])
        np.testing.assert_allclose(cross, 0, atol=1e-8)
    # homogeneous coordinate preserved
    np.testing.assert_allclose(q1[:, 3], 1)
    np.testing.assert_allclose(q2[:, 3], 1)


def test_closest_points_distance_matches_line_line_distance():
    np.random.seed(6)
    r1 = rand_ray(15)
    r2 = rand_ray(15)
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(hnorm(q2 - q1),
                               line_line_distance(r1, r2), atol=1e-8)


def test_closest_points_parallel_lines_not_finite():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 1, 0], [1, 0, 0])
    with np.errstate(all='ignore'):
        q1, q2 = line_line_closest_points(r1, r2)
    assert not (np.all(np.isfinite(q1)) and np.all(np.isfinite(q2)))
