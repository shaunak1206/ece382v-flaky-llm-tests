import numpy as np
import pytest

from homog.homog import hray, line_line_distance, line_line_closest_points


def _line(origin, direction):
    """Build a batch of one ray with shape (1, 4, 2)."""
    return hray(np.array([origin], dtype=float),
                np.array([direction], dtype=float))


def _random_lines(rng, n):
    """Random 3D points and (unnormalized) directions, shape (n, 3)."""
    pts = rng.standard_normal((n, 3))
    axes = rng.standard_normal((n, 3))
    return pts, axes


def _ref_distance(p1, a1, p2, a2):
    """Reference skew-line distance, independent of the implementation."""
    n = np.cross(a1, a2)
    return np.abs(np.sum((p2 - p1) * n, axis=-1)) / np.linalg.norm(n, axis=-1)


# ---------------------------------------------------------------- hray

def test_hray_single_ray_structure():
    r = hray(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 2.0]))
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:3, 0], [1, 2, 3])
    assert r[3, 0] == 1
    np.testing.assert_allclose(r[:3, 1], [0, 0, 1])
    assert r[3, 1] == 0


def test_hray_direction_is_normalized():
    rng = np.random.RandomState(0)
    origins, dirs = _random_lines(rng, 20)
    dirs *= 10.0
    r = hray(origins, dirs)
    norms = np.linalg.norm(r[:, :3, 1], axis=-1)
    np.testing.assert_allclose(norms, np.ones(20), rtol=1e-12)
    np.testing.assert_allclose(r[:, 3, 0], 1)
    np.testing.assert_allclose(r[:, 3, 1], 0)


def test_hray_accepts_homogeneous_origin_and_direction():
    r3 = hray(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 2.0]))
    r4 = hray(np.array([1.0, 2.0, 3.0, 1.0]),
              np.array([0.0, 0.0, 2.0, 0.0]))
    np.testing.assert_allclose(r3, r4)


def test_hray_origin_batch_broadcasts_single_direction():
    origin = np.arange(15, dtype=float).reshape(5, 3)
    direction = np.array([0.0, 3.0, 4.0])
    r = hray(origin, direction)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origin)
    np.testing.assert_allclose(r[:, :3, 1], np.tile([0, 0.6, 0.8], (5, 1)))


def test_hray_direction_batch_broadcasts_single_origin():
    origin = np.array([7.0, -1.0, 2.0])
    dirs = np.eye(3)[[0, 1, 2, 0]]
    r = hray(origin, dirs)
    assert r.shape == (4, 4, 2)
    for i in range(4):
        np.testing.assert_allclose(r[i, :3, 0], origin)
        np.testing.assert_allclose(r[i, :3, 1], dirs[i])


def test_hray_elementwise_batch_matches_single_calls():
    rng = np.random.RandomState(1)
    origins, dirs = _random_lines(rng, 6)
    r = hray(origins, dirs)
    for i in range(6):
        np.testing.assert_allclose(r[i], hray(origins[i], dirs[i]))


def test_hray_incompatible_batch_shapes_raise():
    with pytest.raises(ValueError):
        hray(np.zeros((2, 3)), np.ones((3, 3)))


def test_hray_accepts_python_lists():
    r = hray([0, 0, 0], [1, 0, 0])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:3, 1], [1, 0, 0])
    assert r.dtype == np.float64


# ------------------------------------------------- line_line_distance

def test_line_line_distance_skew_lines():
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2 = _line((0, 0, 1), (0, 1, 0))
    d = line_line_distance(ray1, ray2)
    assert d.shape == (1,)
    np.testing.assert_allclose(d[0], 1.0, atol=1e-12)


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2 = _line((0, 0, 0), (0, 1, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2), [0.0],
                               atol=1e-12)


def test_line_line_distance_parallel_lines_offset():
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2 = _line((3, 2, 0), (1, 0, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2), [2.0],
                               atol=1e-12)


def test_line_line_distance_antiparallel_lines_offset():
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2 = _line((0, 2, 0), (-2, 0, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2), [2.0],
                               atol=1e-12)


def test_line_line_distance_independent_of_direction_scale():
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2a = _line((0, 0, 1), (0, 1, 0))
    ray2b = _line((0, 0, 1), (0, 7, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2a),
                               line_line_distance(ray1, ray2b), atol=1e-12)


def test_line_line_distance_symmetric():
    rng = np.random.RandomState(2)
    p1, a1 = _random_lines(rng, 15)
    p2, a2 = _random_lines(rng, 15)
    r1, r2 = hray(p1, a1), hray(p2, a2)
    np.testing.assert_allclose(line_line_distance(r1, r2),
                               line_line_distance(r2, r1), rtol=1e-12)


def test_line_line_distance_matches_reference_formula():
    rng = np.random.RandomState(3)
    p1, a1 = _random_lines(rng, 30)
    p2, a2 = _random_lines(rng, 30)
    d = line_line_distance(hray(p1, a1), hray(p2, a2))
    assert d.shape == (30,)
    np.testing.assert_allclose(d, _ref_distance(p1, a1, p2, a2),
                               rtol=1e-9, atol=1e-12)


def test_line_line_distance_invariant_to_sliding_origin_along_axis():
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2 = _line((0, 0, 1), (0, 1, 0))
    slid = _line((0, 0, 1) + np.array([0, 0, 0]) + np.array([0, 5, 0]),
                 (0, 1, 0))
    np.testing.assert_allclose(line_line_distance(ray1, ray2),
                               line_line_distance(ray1, slid), atol=1e-12)


def test_line_line_distance_same_line_is_zero():
    ray = _line((1, -2, 3), (0.3, 0.4, 0.5))
    np.testing.assert_allclose(line_line_distance(ray, ray), [0.0],
                               atol=1e-12)


def test_line_line_distance_mixed_parallel_and_skew_batch():
    origins1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    dirs1 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
    origins2 = np.array([[0, 0, 1], [0, 0, 0], [0, 4, 0]], dtype=float)
    dirs2 = np.array([[0, 1, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    d = line_line_distance(hray(origins1, dirs1), hray(origins2, dirs2))
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [1.0, 0.0, 4.0], atol=1e-12)


# --------------------------------------------- line_line_closest_points

def test_closest_points_skew_lines_known_answer():
    ray1 = _line((1, 2, 3), (0, 0, 1))
    ray2 = _line((4, 5, 6), (1, 0, 0))
    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[0, :3], [1, 2, 6], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [1, 5, 6], atol=1e-12)


def test_closest_points_intersecting_lines_coincide():
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2 = _line((2, -1, 0), (0, 1, 0))
    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[0, :3], [2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [2, 0, 0], atol=1e-12)


def test_closest_points_lie_on_their_lines():
    rng = np.random.RandomState(4)
    p1, a1 = _random_lines(rng, 12)
    p2, a2 = _random_lines(rng, 12)
    q1, q2 = line_line_closest_points(hray(p1, a1), hray(p2, a2))
    # Q1 - p1 must be parallel to a1, Q2 - p2 parallel to a2.
    np.testing.assert_allclose(
        np.cross(q1[:, :3] - p1, a1), np.zeros((12, 3)), atol=1e-9)
    np.testing.assert_allclose(
        np.cross(q2[:, :3] - p2, a2), np.zeros((12, 3)), atol=1e-9)


def test_closest_points_form_common_perpendicular():
    rng = np.random.RandomState(5)
    p1, a1 = _random_lines(rng, 12)
    p2, a2 = _random_lines(rng, 12)
    r1, r2 = hray(p1, a1), hray(p2, a2)
    q1, q2 = line_line_closest_points(r1, r2)
    seg = q2[:, :3] - q1[:, :3]
    an1 = a1 / np.linalg.norm(a1, axis=-1)[:, None]
    an2 = a2 / np.linalg.norm(a2, axis=-1)[:, None]
    np.testing.assert_allclose(np.sum(seg * an1, axis=-1), 0, atol=1e-9)
    np.testing.assert_allclose(np.sum(seg * an2, axis=-1), 0, atol=1e-9)
    # The segment length must equal the line-line distance.
    np.testing.assert_allclose(np.linalg.norm(seg, axis=-1),
                               line_line_distance(r1, r2), rtol=1e-9,
                               atol=1e-12)


def test_closest_points_verbose_prints(capsys):
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2 = _line((0, 0, 1), (0, 1, 0))
    line_line_closest_points(ray1, ray2, verbose=1)
    out = capsys.readouterr().out
    assert 'C21' in out
    assert 'Q1' in out and 'Q2' in out


def test_closest_points_quiet_by_default(capsys):
    ray1 = _line((0, 0, 0), (1, 0, 0))
    ray2 = _line((0, 0, 1), (0, 1, 0))
    line_line_closest_points(ray1, ray2)
    assert capsys.readouterr().out == ''


def test_closest_points_batch_shape_and_homogeneous_w():
    rng = np.random.RandomState(6)
    p1, a1 = _random_lines(rng, 7)
    p2, a2 = _random_lines(rng, 7)
    q1, q2 = line_line_closest_points(hray(p1, a1), hray(p2, a2))
    assert q1.shape == (7, 4)
    assert q2.shape == (7, 4)
    np.testing.assert_allclose(q1[:, 3], 1, atol=1e-12)
    np.testing.assert_allclose(q2[:, 3], 1, atol=1e-12)
