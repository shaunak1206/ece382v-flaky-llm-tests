import numpy as np
import pytest

from homog.homog import (
    hray,
    is_valid_rays,
    line_line_closest_points,
    line_line_distance,
)


def _ray(origin, direction):
    return hray(np.array(origin, dtype=float), np.array(direction, dtype=float))


def _batch(origin, direction):
    """Single ray with a leading batch axis, shape (1, 4, 2)."""
    return _ray(origin, direction)[np.newaxis]


def _dist(o1, d1, o2, d2):
    return float(line_line_distance(_batch(o1, d1), _batch(o2, d2))[0])


def _closest(o1, d1, o2, d2, verbose=0):
    return line_line_closest_points(
        _batch(o1, d1), _batch(o2, d2), verbose=verbose)


# ---------------------------------------------------------------------------
# hray
# ---------------------------------------------------------------------------

def test_hray_output_shape_single_ray():
    r = _ray((0, 0, 0), (1, 0, 0))
    assert r.shape == (4, 2)


def test_hray_origin_column_is_homogeneous_point():
    r = _ray((1, 2, 3), (0, 0, 1))
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])


def test_hray_direction_column_is_unit_vector_with_zero_w():
    r = _ray((0, 0, 0), (0, 0, 5))
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = _ray((0, 0, 0), (3, 4, 0))
    np.testing.assert_allclose(r[:, 1], [0.6, 0.8, 0, 0])
    np.testing.assert_allclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_forces_origin_w_to_one():
    # A homogeneous origin with w != 1 is still stored with w == 1.
    r = _ray((1, 2, 3, 2), (1, 0, 0))
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])


def test_hray_accepts_homogeneous_direction_4vec():
    r = _ray((0, 0, 0), (0, 0, 2, 0))
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_output_is_valid_ray():
    r = _ray((0.5, -1, 2), (1, 1, 1))
    assert is_valid_rays(r)


def test_hray_result_dtype_is_float64():
    r = hray(np.array([1, 2, 3]), np.array([0, 0, 1]))
    assert r.dtype == np.float64


def test_hray_does_not_modify_inputs():
    origin = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.0, 0.0, 5.0])
    origin_copy = origin.copy()
    direction_copy = direction.copy()
    hray(origin, direction)
    np.testing.assert_array_equal(origin, origin_copy)
    np.testing.assert_array_equal(direction, direction_copy)


def test_hray_batch_origins_with_single_direction_broadcasts():
    origins = np.arange(15, dtype=float).reshape(5, 3)
    r = hray(origins, np.array([3.0, 0.0, 0.0]))
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins)
    np.testing.assert_allclose(r[:, 3, 0], 1)
    np.testing.assert_allclose(r[:, :3, 1], np.tile([1.0, 0, 0], (5, 1)))


def test_hray_batch_origins_and_directions_paired():
    origins = np.array([[0.0, 0, 0], [1.0, 2, 3]])
    directions = np.array([[0.0, 2, 0], [0.0, 0, -4]])
    r = hray(origins, directions)
    assert r.shape == (2, 4, 2)
    np.testing.assert_allclose(r[0, :, 1], [0, 1, 0, 0])
    np.testing.assert_allclose(r[1, :, 1], [0, 0, -1, 0])
    assert is_valid_rays(r)


# ---------------------------------------------------------------------------
# line_line_distance
# ---------------------------------------------------------------------------

def test_distance_skew_lines_perpendicular_offset():
    # x-axis through origin; y-direction line through (0, 0, 1)
    assert _dist((0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0)) == pytest.approx(1.0)


def test_distance_general_skew_lines():
    # x-axis through origin; y-direction line through (3, 4, 5) -> distance 5
    assert _dist((0, 0, 0), (1, 0, 0), (3, 4, 5), (0, 1, 0)) == pytest.approx(5.0)


def test_distance_intersecting_lines_is_zero():
    # x-axis and y-axis both pass through the origin
    assert _dist((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(0.0)


def test_distance_parallel_lines():
    # two lines along x, separated by 2 in y
    assert _dist((0, 0, 0), (1, 0, 0), (0, 2, 0), (1, 0, 0)) == pytest.approx(2.0)


def test_distance_parallel_lines_offset_along_axis_is_unchanged():
    # sliding one parallel line along its own direction must not change distance
    assert _dist((0, 0, 0), (1, 0, 0), (7, 2, 0), (1, 0, 0)) == pytest.approx(2.0)


def test_distance_antiparallel_lines():
    assert _dist((0, 0, 0), (1, 0, 0), (0, 2, 0), (-1, 0, 0)) == pytest.approx(2.0)


def test_distance_is_symmetric():
    a = _batch((1, -2, 0.5), (0.3, 1, -0.2))
    b = _batch((-4, 3, 2), (1, 0.5, 2))
    np.testing.assert_allclose(line_line_distance(a, b),
                               line_line_distance(b, a))


def test_distance_invariant_to_direction_scale():
    d1 = _dist((0, 0, 0), (1, 0, 0), (3, 4, 5), (0, 1, 0))
    d2 = _dist((0, 0, 0), (1, 0, 0), (3, 4, 5), (0, 17, 0))
    assert d1 == pytest.approx(d2)


def test_distance_batched_returns_one_value_per_pair():
    r1 = np.stack([_ray((0, 0, 0), (1, 0, 0)),
                   _ray((0, 0, 0), (1, 0, 0)),
                   _ray((0, 0, 0), (1, 0, 0))])
    r2 = np.stack([_ray((0, 0, 1), (0, 1, 0)),     # skew, distance 1
                   _ray((0, 0, 0), (0, 1, 0)),     # intersecting, distance 0
                   _ray((0, 2, 0), (1, 0, 0))])    # parallel, distance 2
    d = line_line_distance(r1, r2)
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [1.0, 0.0, 2.0], atol=1e-12)


# ---------------------------------------------------------------------------
# line_line_closest_points
# ---------------------------------------------------------------------------

def test_closest_points_skew_hand_computed():
    # L1: x-axis through origin. L2: line x=1, z=3 (direction y) through (1,2,3).
    # Closest points: Q1=(1,0,0) on L1 and Q2=(1,0,3) on L2.
    q1, q2 = _closest((0, 0, 0), (1, 0, 0), (1, 2, 3), (0, 1, 0))
    np.testing.assert_allclose(q1[0, :3], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [1, 0, 3], atol=1e-12)


def test_closest_points_perpendicular_axes_at_origin():
    q1, q2 = _closest((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0))
    np.testing.assert_allclose(q1[0, :3], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [0, 0, 0], atol=1e-12)


def test_closest_points_separation_matches_distance():
    o1, d1 = (0, 0, 0), (1, 0, 0)
    o2, d2 = (3, 4, 5), (0, 1, 0)
    q1, q2 = _closest(o1, d1, o2, d2)
    sep = np.linalg.norm(q2[0, :3] - q1[0, :3])
    assert sep == pytest.approx(_dist(o1, d1, o2, d2))
    assert sep == pytest.approx(5.0)


def test_closest_points_verbose_prints_intermediates(capsys):
    _closest((0, 0, 0), (1, 0, 0), (1, 2, 3), (0, 1, 0), verbose=1)
    out = capsys.readouterr().out
    for label in ('C21', 'M', 'm2', 'R', 't1', 't2', 'Q1', 'Q2'):
        assert label in out


def test_closest_points_quiet_by_default(capsys):
    _closest((0, 0, 0), (1, 0, 0), (1, 2, 3), (0, 1, 0))
    assert capsys.readouterr().out == ""


def test_closest_points_random_lines_satisfy_geometry():
    rng = np.random.RandomState(0)
    n = 20
    rays1 = np.stack([_ray(rng.randn(3), rng.randn(3)) for _ in range(n)])
    rays2 = np.stack([_ray(rng.randn(3), rng.randn(3)) for _ in range(n)])
    q1, q2 = line_line_closest_points(rays1, rays2)
    assert q1.shape == (n, 4)
    assert q2.shape == (n, 4)

    pt1, ax1 = rays1[:, :3, 0], rays1[:, :3, 1]
    pt2, ax2 = rays2[:, :3, 0], rays2[:, :3, 1]
    q1 = q1[:, :3]
    q2 = q2[:, :3]

    # Q1 lies on line 1, Q2 lies on line 2.
    np.testing.assert_allclose(np.cross(q1 - pt1, ax1), 0, atol=1e-9)
    np.testing.assert_allclose(np.cross(q2 - pt2, ax2), 0, atol=1e-9)
    # The connecting segment is perpendicular to both axes.
    seg = q2 - q1
    np.testing.assert_allclose(np.sum(seg * ax1, axis=-1), 0, atol=1e-9)
    np.testing.assert_allclose(np.sum(seg * ax2, axis=-1), 0, atol=1e-9)


def test_closest_points_distance_consistency_batched():
    rng = np.random.RandomState(1)
    n = 10
    rays1 = np.stack([_ray(rng.randn(3), rng.randn(3)) for _ in range(n)])
    rays2 = np.stack([_ray(rng.randn(3), rng.randn(3)) for _ in range(n)])
    q1, q2 = line_line_closest_points(rays1, rays2)
    sep = np.linalg.norm(q2[:, :3] - q1[:, :3], axis=-1)
    np.testing.assert_allclose(sep, line_line_distance(rays1, rays2),
                               atol=1e-9)
