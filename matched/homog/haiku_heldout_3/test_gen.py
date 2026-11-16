import numpy as np
import pytest

from homog.homog import (hray, is_valid_rays, line_line_closest_points,
                         line_line_distance)


def _ray(origin, direction):
    return hray(np.asarray(origin, dtype=float),
                np.asarray(direction, dtype=float))


# ---------------------------------------------------------------------------
# hray
# ---------------------------------------------------------------------------

def test_hray_single_shape_is_4_by_2():
    r = _ray((1, 2, 3), (0, 0, 1))
    assert r.shape == (4, 2)


def test_hray_origin_stored_with_homogeneous_one():
    r = _ray((1, 2, 3), (0, 0, 1))
    np.testing.assert_allclose(r[:3, 0], [1, 2, 3])
    assert r[3, 0] == 1


def test_hray_direction_is_normalized():
    r = _ray((0, 0, 0), (0, 3, 4))
    np.testing.assert_allclose(r[:3, 1], [0, 0.6, 0.8])
    np.testing.assert_allclose(np.linalg.norm(r[:3, 1]), 1.0)
    assert r[3, 1] == 0


def test_hray_result_is_valid_ray():
    r = _ray((0.5, -1, 2), (1, 1, 1))
    assert is_valid_rays(r)


def test_hray_accepts_homogeneous_4vector_origin():
    r = hray(np.array([1.0, 2.0, 3.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(r[:3, 0], [1, 2, 3])
    assert r[3, 0] == 1


def test_hray_rejects_origin_of_bad_length():
    with pytest.raises(ValueError):
        hray(np.array([1.0, 2.0]), np.array([0.0, 0.0, 1.0]))


def test_hray_batch_shape():
    origins = np.random.RandomState(0).randn(5, 3)
    dirs = np.random.RandomState(1).randn(5, 3)
    r = hray(origins, dirs)
    assert r.shape == (5, 4, 2)


def test_hray_batch_matches_single_elementwise():
    rng = np.random.RandomState(2)
    origins = rng.randn(4, 3)
    dirs = rng.randn(4, 3)
    batch = hray(origins, dirs)
    for i in range(4):
        np.testing.assert_allclose(batch[i], _ray(origins[i], dirs[i]))


def test_hray_broadcasts_single_origin_over_many_directions():
    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float)
    r = hray(np.array([1.0, 1.0, 1.0]), dirs)
    assert r.shape == (3, 4, 2)
    for i in range(3):
        np.testing.assert_allclose(r[i, :3, 0], [1, 1, 1])
        assert r[i, 3, 0] == 1
        assert is_valid_rays(r[i])


# ---------------------------------------------------------------------------
# line_line_distance
# ---------------------------------------------------------------------------

def test_distance_skew_lines_unit_separation():
    # x-axis through origin, y-axis through (0, 0, 1)
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((0, 0, 1), (0, 1, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2), 1.0)


def test_distance_intersecting_lines_is_zero():
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((0, 0, 0), (0, 1, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2), 0.0,
                               atol=1e-12)


def test_distance_parallel_lines_offset_perpendicular():
    # perpendicular offset (0, 3, 4) has norm 5
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((5, 3, 4), (1, 0, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2), 5.0)


def test_distance_antiparallel_lines():
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((0, 4, 0), (-1, 0, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2), 4.0)


def test_distance_is_symmetric_in_arguments():
    rng = np.random.RandomState(3)
    ray1 = _ray(rng.randn(3), rng.randn(3))
    ray2 = _ray(rng.randn(3), rng.randn(3))
    np.testing.assert_allclose(line_line_distance(ray1, ray2),
                               line_line_distance(ray2, ray1))


def test_distance_matches_scalar_triple_product_formula():
    rng = np.random.RandomState(4)
    p1, d1 = rng.randn(3), rng.randn(3)
    p2, d2 = rng.randn(3), rng.randn(3)
    ray1, ray2 = _ray(p1, d1), _ray(p2, d2)
    u = ray1[:3, 1]
    v = ray2[:3, 1]
    cr = np.cross(u, v)
    expected = abs(np.dot(p2 - p1, cr)) / np.linalg.norm(cr)
    np.testing.assert_allclose(line_line_distance(ray1, ray2), expected,
                               rtol=1e-9)


def test_distance_batch_shape_and_values():
    rng = np.random.RandomState(5)
    p1, d1 = rng.randn(6, 3), rng.randn(6, 3)
    p2, d2 = rng.randn(6, 3), rng.randn(6, 3)
    ray1, ray2 = hray(p1, d1), hray(p2, d2)
    dist = line_line_distance(ray1, ray2)
    assert dist.shape == (6,)
    for i in range(6):
        single = line_line_distance(ray1[i], ray2[i])
        np.testing.assert_allclose(dist[i], single, rtol=1e-9)


# ---------------------------------------------------------------------------
# line_line_closest_points
# ---------------------------------------------------------------------------

def test_closest_points_skew_lines_through_origin_and_z1():
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((0, 0, 1), (0, 1, 0))
    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[:3], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(q2[:3], [0, 0, 1], atol=1e-12)


def test_closest_points_general_example():
    # line1: x-axis through origin; line2: direction y through (1, 0, 1)
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((1, 0, 1), (0, 1, 0))
    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[:3], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(q2[:3], [1, 0, 1], atol=1e-12)


def test_closest_points_are_homogeneous_points():
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((0, 0, 1), (0, 1, 0))
    q1, q2 = line_line_closest_points(ray1, ray2)
    assert q1.shape == (4,)
    assert q2.shape == (4,)
    assert q1[3] == 1
    assert q2[3] == 1


def test_closest_points_lie_on_their_respective_lines():
    rng = np.random.RandomState(6)
    ray1 = hray(rng.randn(5, 3), rng.randn(5, 3))
    ray2 = hray(rng.randn(5, 3), rng.randn(5, 3))
    q1, q2 = line_line_closest_points(ray1, ray2)
    p1, u = ray1[..., :3, 0], ray1[..., :3, 1]
    p2, v = ray2[..., :3, 0], ray2[..., :3, 1]
    # Q1 - p1 parallel to u  =>  cross product zero
    np.testing.assert_allclose(np.cross(q1[..., :3] - p1, u), 0, atol=1e-9)
    # Q2 - p2 parallel to v  =>  cross product zero
    np.testing.assert_allclose(np.cross(q2[..., :3] - p2, v), 0, atol=1e-9)


def test_closest_points_segment_is_perpendicular_to_both_lines():
    rng = np.random.RandomState(7)
    ray1 = hray(rng.randn(5, 3), rng.randn(5, 3))
    ray2 = hray(rng.randn(5, 3), rng.randn(5, 3))
    q1, q2 = line_line_closest_points(ray1, ray2)
    seg = q2[..., :3] - q1[..., :3]
    u = ray1[..., :3, 1]
    v = ray2[..., :3, 1]
    np.testing.assert_allclose(np.sum(seg * u, axis=-1), 0, atol=1e-9)
    np.testing.assert_allclose(np.sum(seg * v, axis=-1), 0, atol=1e-9)


def test_closest_points_separation_equals_line_line_distance():
    rng = np.random.RandomState(8)
    ray1 = hray(rng.randn(5, 3), rng.randn(5, 3))
    ray2 = hray(rng.randn(5, 3), rng.randn(5, 3))
    q1, q2 = line_line_closest_points(ray1, ray2)
    sep = np.linalg.norm(q2[..., :3] - q1[..., :3], axis=-1)
    dist = line_line_distance(ray1, ray2)
    np.testing.assert_allclose(sep, dist, rtol=1e-9)


def test_closest_points_batch_shape():
    rng = np.random.RandomState(9)
    ray1 = hray(rng.randn(7, 3), rng.randn(7, 3))
    ray2 = hray(rng.randn(7, 3), rng.randn(7, 3))
    q1, q2 = line_line_closest_points(ray1, ray2)
    assert q1.shape == (7, 4)
    assert q2.shape == (7, 4)


def test_closest_points_verbose_prints_intermediates(capsys):
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((0, 0, 1), (0, 1, 0))
    line_line_closest_points(ray1, ray2, verbose=1)
    out = capsys.readouterr().out
    for label in ('C21', 'M', 'm2', 'R', 't1', 't2', 'Q1', 'Q2'):
        assert label in out


def test_closest_points_silent_by_default(capsys):
    ray1 = _ray((0, 0, 0), (1, 0, 0))
    ray2 = _ray((0, 0, 1), (0, 1, 0))
    line_line_closest_points(ray1, ray2)
    assert capsys.readouterr().out == ''
