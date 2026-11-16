import numpy as np
import pytest

from homog.homog import (
    hray,
    is_valid_rays,
    line_line_closest_points,
    line_line_distance,
)


def _ray(origin, direction):
    """Build a batch of one ray so results are always arrays of shape (1, ...)."""
    return hray(np.array([origin], dtype=float),
                np.array([direction], dtype=float))


def _dist(ray1, ray2):
    return float(line_line_distance(ray1, ray2)[0])


# ---------------------------------------------------------------- hray

def test_hray_single_ray_shape():
    r = hray([1.0, 2.0, 3.0], [0.0, 0.0, 2.0])
    assert r.shape == (4, 2)


def test_hray_origin_column_is_homogeneous_point():
    r = hray([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(r[:, 0], [1.0, 2.0, 3.0, 1.0])


def test_hray_w_row_is_one_zero():
    r = hray([4.0, -1.0, 0.5], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(r[3, :], [1.0, 0.0])


def test_hray_normalizes_direction():
    r = hray([0.0, 0.0, 0.0], [0.0, 3.0, 4.0])
    np.testing.assert_allclose(r[:, 1], [0.0, 0.6, 0.8, 0.0])


def test_hray_direction_scale_invariant():
    r1 = hray([1.0, 1.0, 1.0], [2.0, 0.0, 0.0])
    r2 = hray([1.0, 1.0, 1.0], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(r1, r2)


def test_hray_ignores_w_component_of_direction():
    r = hray([0.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0])
    np.testing.assert_allclose(r[:, 1], [0.0, 0.0, 1.0, 0.0])


def test_hray_accepts_homogeneous_origin():
    r1 = hray([1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 1.0])
    r2 = hray([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(r1, r2)


def test_hray_result_is_valid_ray():
    r = hray([0.3, -2.0, 7.0], [1.0, 2.0, 2.0])
    assert is_valid_rays(r)


def test_hray_batch_shape_and_validity():
    r = hray(np.zeros((5, 3)), np.ones((5, 3)))
    assert r.shape == (5, 4, 2)
    assert is_valid_rays(r)


def test_hray_broadcasts_single_origin_over_directions():
    origin = np.array([1.0, 0.0, 0.0])
    dirs = np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 2.0],
                     [1.0, 1.0, 1.0]])
    r = hray(origin, dirs)
    assert r.shape == (4, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], np.tile(origin, (4, 1)))
    expected = dirs / np.linalg.norm(dirs, axis=1)[:, None]
    np.testing.assert_allclose(r[:, :3, 1], expected)


def test_hray_does_not_modify_inputs():
    origin = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.0, 0.0, 2.0])
    origin_copy = origin.copy()
    direction_copy = direction.copy()
    hray(origin, direction)
    np.testing.assert_array_equal(origin, origin_copy)
    np.testing.assert_array_equal(direction, direction_copy)


# ------------------------------------------------- line_line_distance

def test_distance_skew_lines_unit():
    ray1 = _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    assert _dist(ray1, ray2) == pytest.approx(1.0)


def test_distance_skew_lines_with_offset():
    # Common perpendicular is along z; z offset of 2 gives distance 2.
    ray1 = _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([3.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    assert _dist(ray1, ray2) == pytest.approx(2.0)


def test_distance_intersecting_lines_is_zero():
    # Coplanar lines in z=0 that cross at (5, 0, 0).
    ray1 = _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([5.0, 1.0, 0.0], [0.0, 1.0, 0.0])
    assert _dist(ray1, ray2) == pytest.approx(0.0, abs=1e-12)


def test_distance_parallel_lines():
    ray1 = _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([0.0, 2.0, 0.0], [1.0, 0.0, 0.0])
    assert _dist(ray1, ray2) == pytest.approx(2.0)


def test_distance_antiparallel_lines():
    ray1 = _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([5.0, 0.0, 2.0], [-1.0, 0.0, 0.0])
    assert _dist(ray1, ray2) == pytest.approx(2.0)


def test_distance_independent_of_position_along_lines():
    ray1 = _ray([7.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([3.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    assert _dist(ray1, ray2) == pytest.approx(2.0)


def test_distance_is_symmetric():
    ray1 = _ray([1.0, 2.0, 0.0], [1.0, 0.5, 0.0])
    ray2 = _ray([0.0, -1.0, 3.0], [0.0, 1.0, 2.0])
    assert _dist(ray1, ray2) == pytest.approx(_dist(ray2, ray1))


def test_distance_batch_mixed_cases():
    origins1 = np.zeros((3, 3))
    dirs1 = np.tile([1.0, 0.0, 0.0], (3, 1))
    origins2 = np.array([[0.0, 0.0, 1.0],   # skew, distance 1
                         [0.0, 2.0, 0.0],   # parallel, distance 2
                         [0.0, 0.0, 0.0]])  # intersecting, distance 0
    dirs2 = np.array([[0.0, 1.0, 0.0],
                      [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
    r1 = hray(origins1, dirs1)
    r2 = hray(origins2, dirs2)
    d = line_line_distance(r1, r2)
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [1.0, 2.0, 0.0], atol=1e-12)


def test_distance_matches_closest_point_separation():
    rng = np.random.RandomState(0)
    r1 = hray(rng.randn(20, 3), rng.randn(20, 3))
    r2 = hray(rng.randn(20, 3), rng.randn(20, 3))
    d = line_line_distance(r1, r2)
    q1, q2 = line_line_closest_points(r1, r2)
    sep = np.linalg.norm(q1[:, :3] - q2[:, :3], axis=-1)
    np.testing.assert_allclose(d, sep, rtol=1e-9, atol=1e-12)


# --------------------------------------------- line_line_closest_points

def test_closest_points_skew_known_values():
    ray1 = _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([3.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[0], [3.0, 0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(q2[0], [3.0, 0.0, 2.0, 1.0], atol=1e-12)


def test_closest_points_intersecting_lines_coincide():
    # Lines cross at (4, 2, 3).
    ray1 = _ray([1.0, 2.0, 3.0], [1.0, 0.0, 0.0])
    ray2 = _ray([4.0, 0.0, 3.0], [0.0, 1.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[0, :3], [4.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [4.0, 2.0, 3.0], atol=1e-12)


def test_closest_points_lie_on_their_lines():
    ray1 = _ray([0.0, 1.0, -1.0], [1.0, 2.0, 0.0])
    ray2 = _ray([3.0, 0.0, 2.0], [0.0, 1.0, 1.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    ax1 = ray1[0, :3, 1]
    ax2 = ray2[0, :3, 1]
    pt1 = ray1[0, :3, 0]
    pt2 = ray2[0, :3, 0]
    # Q1 - pt1 must be parallel to ax1 (cross product zero).
    np.testing.assert_allclose(
        np.cross(q1[0, :3] - pt1, ax1), np.zeros(3), atol=1e-10)
    np.testing.assert_allclose(
        np.cross(q2[0, :3] - pt2, ax2), np.zeros(3), atol=1e-10)


def test_closest_points_connector_perpendicular_to_both_axes():
    ray1 = _ray([0.0, 1.0, -1.0], [1.0, 2.0, 0.0])
    ray2 = _ray([3.0, 0.0, 2.0], [0.0, 1.0, 1.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    conn = q1[0, :3] - q2[0, :3]
    assert np.dot(conn, ray1[0, :3, 1]) == pytest.approx(0.0, abs=1e-10)
    assert np.dot(conn, ray2[0, :3, 1]) == pytest.approx(0.0, abs=1e-10)


def test_closest_points_swapping_arguments_swaps_outputs():
    ray1 = _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([3.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    s1, s2 = line_line_closest_points(ray2, ray1)
    np.testing.assert_allclose(s1, q2, atol=1e-12)
    np.testing.assert_allclose(s2, q1, atol=1e-12)


def test_closest_points_output_shapes_batch():
    rng = np.random.RandomState(1)
    r1 = hray(rng.randn(6, 3), rng.randn(6, 3))
    r2 = hray(rng.randn(6, 3), rng.randn(6, 3))
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (6, 4)
    assert q2.shape == (6, 4)


def test_closest_points_verbose_prints(capsys):
    ray1 = _ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = _ray([3.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    line_line_closest_points(ray1, ray2, verbose=1)
    out = capsys.readouterr().out
    for label in ('C21', 'M', 'm2', 'R', 't1', 't2', 'Q1', 'Q2'):
        assert label in out
