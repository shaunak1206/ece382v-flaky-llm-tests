import numpy as np

import homog.homog as hg


def _batch(*rays):
    """Stack individual (4, 2) rays into a (N, 4, 2) batch."""
    return np.stack(rays)


def test_hray_single_ray_layout():
    r = hg.hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:3, 0], [1, 2, 3])
    assert r[3, 0] == 1
    assert r[3, 1] == 0
    np.testing.assert_allclose(r[:3, 1], [0, 0, 1])


def test_hray_normalizes_direction():
    r = hg.hray([0, 0, 0], [3, 0, 4])
    np.testing.assert_allclose(r[:3, 1], [0.6, 0.0, 0.8])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_accepts_homogeneous_origin():
    r_3d = hg.hray([1, 2, 3], [0, 0, 1])
    r_4d = hg.hray([1, 2, 3, 1], [0, 0, 1])
    np.testing.assert_allclose(r_4d, r_3d)


def test_hray_homogeneous_direction_has_zero_w():
    # Direction given as 4-vectors with w=0; only the xyz part is normalized.
    r = hg.hray([[0, 0, 0], [0, 0, 0]], [[0, 2, 0, 0], [0, 0, 4, 0]])
    assert r.shape == (2, 4, 2)
    np.testing.assert_allclose(r[0, :3, 1], [0, 1, 0])
    np.testing.assert_allclose(r[1, :3, 1], [0, 0, 1])
    np.testing.assert_allclose(r[..., 3, 1], [0, 0])


def test_hray_batch_shape():
    np.random.seed(0)
    origins = np.random.randn(5, 3)
    directions = np.random.randn(5, 3)
    r = hg.hray(origins, directions)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins)
    assert np.all(r[:, 3, 0] == 1)
    assert np.all(r[:, 3, 1] == 0)
    assert hg.is_valid_rays(r)


def test_hray_broadcasts_single_origin_over_directions():
    origin = [1.0, -2.0, 0.5]
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    r = hg.hray(origin, directions)
    assert r.shape == (4, 4, 2)
    for i in range(4):
        np.testing.assert_allclose(r[i, :3, 0], origin)
    assert hg.is_valid_rays(r)


def test_hray_output_is_valid_rays():
    np.random.seed(1)
    pts = hg.rand_point((7,))
    vecs = hg.rand_vec((7,))
    r = hg.hray(pts, vecs)
    assert r.shape == (7, 4, 2)
    assert hg.is_valid_rays(r)


def test_hray_does_not_mutate_inputs():
    origin = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.0, 0.0, 5.0])
    origin_copy = origin.copy()
    direction_copy = direction.copy()
    hg.hray(origin, direction)
    np.testing.assert_array_equal(origin, origin_copy)
    np.testing.assert_array_equal(direction, direction_copy)


# Note: distance tests use batched (N, 4, 2) inputs. The unbatched
# (4, 2) path of line_line_distance indexes numpy scalars and is not
# exercised here.


def test_line_line_distance_skew_orthogonal():
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([0, 0, 3], [0, 1, 0]))
    d = hg.line_line_distance(r1, r2)
    np.testing.assert_allclose(d, [3.0])


def test_line_line_distance_intersecting_is_zero():
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([1, 0, 0], [0, 1, 0]))
    d = hg.line_line_distance(r1, r2)
    np.testing.assert_allclose(d, [0.0], atol=1e-12)


def test_line_line_distance_skew_general():
    # Lines (1,2,3)+s(1,1,0) and (4,0,-1)+t(0,1,1); expected 1/sqrt(3).
    r1 = _batch(hg.hray([1, 2, 3], [1, 1, 0]))
    r2 = _batch(hg.hray([4, 0, -1], [0, 1, 1]))
    d = hg.line_line_distance(r1, r2)
    np.testing.assert_allclose(d, [1.0 / np.sqrt(3.0)], rtol=1e-12)


def test_line_line_distance_parallel_offset():
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([5, 2, 0], [2, 0, 0]))
    d = hg.line_line_distance(r1, r2)
    np.testing.assert_allclose(d, [2.0])


def test_line_line_distance_antiparallel_offset():
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([5, 0, 2], [-1, 0, 0]))
    d = hg.line_line_distance(r1, r2)
    np.testing.assert_allclose(d, [2.0])


def test_line_line_distance_is_symmetric():
    np.random.seed(2)
    r1 = hg.rand_ray((10,))
    r2 = hg.rand_ray((10,))
    d12 = hg.line_line_distance(r1, r2)
    d21 = hg.line_line_distance(r2, r1)
    np.testing.assert_allclose(d12, d21, rtol=1e-12)


def test_line_line_distance_invariant_to_sliding_origin():
    r2 = _batch(hg.hray([0, 2, 0], [0, 0, 1]))
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r1_slid = _batch(hg.hray([5, 0, 0], [1, 0, 0]))
    d = hg.line_line_distance(r1, r2)
    d_slid = hg.line_line_distance(r1_slid, r2)
    np.testing.assert_allclose(d, [2.0])
    np.testing.assert_allclose(d_slid, d, rtol=1e-12)


def test_line_line_distance_batch_shape():
    np.random.seed(3)
    r1 = hg.rand_ray((6,))
    r2 = hg.rand_ray((6,))
    d = hg.line_line_distance(r1, r2)
    assert d.shape == (6,)
    assert np.all(d >= 0)


def test_line_line_distance_matches_closest_point_gap():
    np.random.seed(4)
    r1 = hg.rand_ray((8,))
    r2 = hg.rand_ray((8,))
    d = hg.line_line_distance(r1, r2)
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    gap = np.linalg.norm((Q2 - Q1)[..., :3], axis=-1)
    np.testing.assert_allclose(gap, d, rtol=1e-9, atol=1e-12)


def test_line_line_closest_points_skew_orthogonal():
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([0, 0, 3], [0, 1, 0]))
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    np.testing.assert_allclose(Q1[0], [0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(Q2[0], [0, 0, 3, 1], atol=1e-12)


def test_line_line_closest_points_general_example():
    # Line 1: (0,0,0) + s*(1,0,0); Line 2: (1,-1,2) + t*(0,1,0).
    # Closest points: (1,0,0) on line 1 and (1,0,2) on line 2.
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([1, -1, 2], [0, 1, 0]))
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    np.testing.assert_allclose(Q1[0], [1, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(Q2[0], [1, 0, 2, 1], atol=1e-12)


def test_line_line_closest_points_intersecting_lines_meet():
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([1, 0, 0], [0, 1, 0]))
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    np.testing.assert_allclose(Q1, Q2, atol=1e-12)
    np.testing.assert_allclose(Q1[0], [1, 0, 0, 1], atol=1e-12)


def test_line_line_closest_points_lie_on_their_lines():
    np.random.seed(5)
    r1 = hg.rand_ray((10,))
    r2 = hg.rand_ray((10,))
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    assert Q1.shape == (10, 4)
    assert Q2.shape == (10, 4)
    assert np.all(Q1[..., 3] == 1)
    assert np.all(Q2[..., 3] == 1)
    # Each closest point must be collinear with its own line.
    cross1 = np.cross(Q1[..., :3] - r1[..., :3, 0], r1[..., :3, 1])
    cross2 = np.cross(Q2[..., :3] - r2[..., :3, 0], r2[..., :3, 1])
    np.testing.assert_allclose(cross1, 0, atol=1e-9)
    np.testing.assert_allclose(cross2, 0, atol=1e-9)


def test_line_line_closest_points_gap_perpendicular_to_both_axes():
    np.random.seed(6)
    r1 = hg.rand_ray((10,))
    r2 = hg.rand_ray((10,))
    Q1, Q2 = hg.line_line_closest_points(r1, r2)
    gap = (Q2 - Q1)[..., :3]
    dot1 = np.sum(gap * r1[..., :3, 1], axis=-1)
    dot2 = np.sum(gap * r2[..., :3, 1], axis=-1)
    np.testing.assert_allclose(dot1, 0, atol=1e-9)
    np.testing.assert_allclose(dot2, 0, atol=1e-9)


def test_line_line_closest_points_batch_matches_single():
    np.random.seed(7)
    r1 = hg.rand_ray((4,))
    r2 = hg.rand_ray((4,))
    Q1_all, Q2_all = hg.line_line_closest_points(r1, r2)
    for i in range(4):
        Q1, Q2 = hg.line_line_closest_points(r1[i:i + 1], r2[i:i + 1])
        np.testing.assert_allclose(Q1[0], Q1_all[i], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(Q2[0], Q2_all[i], rtol=1e-12, atol=1e-12)


def test_line_line_closest_points_verbose_prints(capsys):
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([1, -1, 2], [0, 1, 0]))
    hg.line_line_closest_points(r1, r2, verbose=1)
    out = capsys.readouterr().out
    for label in ['C21', 'M', 'm2', 'R', 't1', 't2', 'Q1', 'Q2']:
        assert label in out


def test_line_line_closest_points_silent_by_default(capsys):
    r1 = _batch(hg.hray([0, 0, 0], [1, 0, 0]))
    r2 = _batch(hg.hray([1, -1, 2], [0, 1, 0]))
    hg.line_line_closest_points(r1, r2)
    assert capsys.readouterr().out == ""
