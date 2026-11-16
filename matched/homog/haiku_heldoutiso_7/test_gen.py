import numpy as np
import pytest

from homog import homog as hg


def _rays(origins, directions):
    """Build a batch of rays with shape (N, 4, 2) from (N, 3) arrays."""
    origins = np.atleast_2d(np.asarray(origins, dtype=float))
    directions = np.atleast_2d(np.asarray(directions, dtype=float))
    return hg.hray(origins, directions)


def _unit_rows(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1)[..., None]


# ---------------------------------------------------------------- hray

def test_hray_single_ray_content():
    r = hg.hray([1.0, 2.0, 3.0], [0.0, 0.0, 5.0])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = hg.hray([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])
    np.testing.assert_allclose(r[:3, 1], [0.6, 0.8, 0.0])
    assert np.linalg.norm(r[:3, 1]) == pytest.approx(1.0)


def test_hray_result_is_valid_ray():
    r = hg.hray([1.0, -2.0, 0.5], [1.0, 1.0, 1.0])
    assert hg.is_valid_rays(r)


def test_hray_accepts_homogeneous_inputs():
    r3 = hg.hray([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    r4 = hg.hray([1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 4.0, 0.0])
    np.testing.assert_allclose(r3, r4)


def test_hray_broadcasts_origins_against_single_direction():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
    r = hg.hray(origins, [0.0, 1.0, 0.0])
    assert r.shape == (3, 4, 2)
    for i in range(3):
        np.testing.assert_allclose(r[i, :3, 0], origins[i])
        np.testing.assert_allclose(r[i, 3, 0], 1)
        np.testing.assert_allclose(r[i, :, 1], [0, 1, 0, 0])


def test_hray_broadcasts_directions_against_single_origin():
    origin = np.array([5.0, 6.0, 7.0])
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]])
    r = hg.hray(origin, dirs)
    assert r.shape == (3, 4, 2)
    expected = _unit_rows(dirs)
    for i in range(3):
        np.testing.assert_allclose(r[i, :3, 0], origin)
        np.testing.assert_allclose(r[i, :3, 1], expected[i])


def test_hray_homogeneous_rows_are_constant_in_batch():
    rng = np.random.RandomState(42)
    r = hg.hray(rng.randn(4, 3), rng.randn(4, 3))
    assert r.shape == (4, 4, 2)
    np.testing.assert_allclose(r[:, 3, 0], 1)
    np.testing.assert_allclose(r[:, 3, 1], 0)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1)


# ------------------------------------------------- line_line_distance

def test_distance_skew_perpendicular_offset():
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[0.0, 1.0, 2.0]], [[0.0, 1.0, 0.0]])
    d = hg.line_line_distance(ray1, ray2)
    assert d.shape == (1,)
    assert d[0] == pytest.approx(2.0)


def test_distance_skew_oblique_axes():
    # Axis of ray2 is (0,1,1)/sqrt(2); distance from x-axis is 3/sqrt(2).
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[0.0, 0.0, 3.0]], [[0.0, 1.0, 1.0]])
    d = hg.line_line_distance(ray1, ray2)
    assert d[0] == pytest.approx(3.0 / np.sqrt(2.0))


def test_distance_intersecting_lines_is_zero():
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[2.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
    d = hg.line_line_distance(ray1, ray2)
    assert d[0] == pytest.approx(0.0, abs=1e-12)


def test_distance_parallel_lines_perpendicular_offset():
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[0.0, 3.0, 4.0]], [[1.0, 0.0, 0.0]])
    d = hg.line_line_distance(ray1, ray2)
    assert d[0] == pytest.approx(5.0)


def test_distance_parallel_ignores_offset_along_axis():
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[7.0, 2.0, 0.0]], [[1.0, 0.0, 0.0]])
    d = hg.line_line_distance(ray1, ray2)
    assert d[0] == pytest.approx(2.0)


def test_distance_antiparallel_lines():
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[0.0, 3.0, 4.0]], [[-1.0, 0.0, 0.0]])
    d = hg.line_line_distance(ray1, ray2)
    assert d[0] == pytest.approx(5.0)


def test_distance_is_symmetric():
    rng = np.random.RandomState(3)
    r1 = _rays(rng.randn(5, 3), rng.randn(5, 3))
    r2 = _rays(rng.randn(5, 3), rng.randn(5, 3))
    np.testing.assert_allclose(hg.line_line_distance(r1, r2),
                               hg.line_line_distance(r2, r1))


def test_distance_matches_reference_formula_random():
    rng = np.random.RandomState(1234)
    o1, o2 = rng.randn(8, 3), rng.randn(8, 3)
    d1, d2 = _unit_rows(rng.randn(8, 3)), _unit_rows(rng.randn(8, 3))
    cross = np.cross(d1, d2)
    expected = (np.abs(np.sum((o2 - o1) * cross, axis=-1))
                / np.linalg.norm(cross, axis=-1))
    got = hg.line_line_distance(_rays(o1, d1), _rays(o2, d2))
    assert got.shape == (8,)
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)


def test_distance_mixed_batch_skew_intersecting_parallel():
    ray1 = _rays([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                 [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ray2 = _rays([[0.0, 1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 3.0, 4.0]],
                 [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    d = hg.line_line_distance(ray1, ray2)
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [2.0, 0.0, 5.0], atol=1e-12)


# --------------------------------------------- line_line_closest_points

def test_closest_points_skew_known_values():
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[0.0, 1.0, 2.0]], [[0.0, 1.0, 0.0]])
    q1, q2 = hg.line_line_closest_points(ray1, ray2)
    assert q1.shape == (1, 4)
    assert q2.shape == (1, 4)
    np.testing.assert_allclose(q1[0, :3], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [0.0, 0.0, 2.0], atol=1e-12)
    assert q1[0, 3] == pytest.approx(1.0)
    assert q2[0, 3] == pytest.approx(1.0)


def test_closest_points_intersecting_lines_coincide():
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[2.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
    q1, q2 = hg.line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1[0, :3], [2.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q2[0, :3], [2.0, 0.0, 0.0], atol=1e-12)


def test_closest_points_lie_on_lines_and_are_perpendicular():
    rng = np.random.RandomState(7)
    o1, o2 = rng.randn(6, 3), rng.randn(6, 3)
    d1, d2 = _unit_rows(rng.randn(6, 3)), _unit_rows(rng.randn(6, 3))
    q1, q2 = hg.line_line_closest_points(_rays(o1, d1), _rays(o2, d2))
    q1, q2 = q1[:, :3], q2[:, :3]
    # Q1 lies on line 1 and Q2 lies on line 2.
    np.testing.assert_allclose(np.cross(q1 - o1, d1), 0, atol=1e-9)
    np.testing.assert_allclose(np.cross(q2 - o2, d2), 0, atol=1e-9)
    # The connecting segment is perpendicular to both axes.
    sep = q2 - q1
    np.testing.assert_allclose(np.sum(sep * d1, axis=-1), 0, atol=1e-9)
    np.testing.assert_allclose(np.sum(sep * d2, axis=-1), 0, atol=1e-9)


def test_closest_points_separation_equals_line_distance():
    rng = np.random.RandomState(99)
    r1 = _rays(rng.randn(6, 3), rng.randn(6, 3))
    r2 = _rays(rng.randn(6, 3), rng.randn(6, 3))
    q1, q2 = hg.line_line_closest_points(r1, r2)
    sep = np.linalg.norm(q2[:, :3] - q1[:, :3], axis=-1)
    np.testing.assert_allclose(sep, hg.line_line_distance(r1, r2),
                               rtol=1e-9, atol=1e-9)


def test_closest_points_swapping_rays_swaps_outputs():
    rng = np.random.RandomState(5)
    r1 = _rays(rng.randn(4, 3), rng.randn(4, 3))
    r2 = _rays(rng.randn(4, 3), rng.randn(4, 3))
    q1, q2 = hg.line_line_closest_points(r1, r2)
    p1, p2 = hg.line_line_closest_points(r2, r1)
    np.testing.assert_allclose(p1, q2, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(p2, q1, rtol=1e-9, atol=1e-9)


def test_closest_points_verbose_prints_only_when_requested(capsys):
    ray1 = _rays([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
    ray2 = _rays([[0.0, 1.0, 2.0]], [[0.0, 1.0, 0.0]])

    hg.line_line_closest_points(ray1, ray2)
    assert capsys.readouterr().out == ''

    hg.line_line_closest_points(ray1, ray2, verbose=1)
    out = capsys.readouterr().out
    assert 'C21' in out
    assert 'Q1' in out
    assert 'Q2' in out
