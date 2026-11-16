import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    hnorm,
    hdot,
    hcross,
    is_valid_rays,
)


def _ray_batch(origins, dirs):
    return hray(np.array(origins, dtype=float), np.array(dirs, dtype=float))


# ---------------------------------------------------------------- hray


def test_hray_single_shape():
    r = hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)


def test_hray_origin_column_is_homogeneous_point():
    r = hray([1, 2, 3], [0, 0, 1])
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])


def test_hray_direction_column_is_unit_vector_with_zero_w():
    r = hray([1, 2, 3], [0, 0, 2])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_arbitrary_direction():
    r = hray([0, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(r[:, 1], [0.6, 0.8, 0, 0])
    assert abs(np.linalg.norm(r[:3, 1]) - 1) < 1e-12


def test_hray_accepts_four_component_inputs():
    r = hray([1, 2, 3, 1], [0, 5, 0, 0])
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 1, 0, 0])


def test_hray_batch_shape_and_values():
    np.random.seed(0)
    o = np.random.randn(5, 3)
    d = np.random.randn(5, 3)
    r = hray(o, d)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], o)
    np.testing.assert_allclose(r[:, 3, 0], 1)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1)
    np.testing.assert_allclose(r[:, 3, 1], 0)
    assert is_valid_rays(r)


def test_hray_broadcasts_single_origin_over_directions():
    d = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    r = hray([1, 1, 1], d)
    assert r.shape == (3, 4, 2)
    for i in range(3):
        np.testing.assert_allclose(r[i, :, 0], [1, 1, 1, 1])
    np.testing.assert_allclose(r[:, :3, 1], np.eye(3))


def test_hray_result_is_valid_ray():
    assert is_valid_rays(hray([0.5, -2, 7], [1, 2, 3]))


# ------------------------------------------------- line_line_distance


def test_distance_skew_orthogonal_lines():
    r1 = _ray_batch([[0, 0, 0]], [[1, 0, 0]])
    r2 = _ray_batch([[0, 0, 3]], [[0, 1, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [3.0], atol=1e-10)


def test_distance_intersecting_lines_is_zero():
    r1 = _ray_batch([[1, 0, 0]], [[1, 0, 0]])
    r2 = _ray_batch([[0, 0, -2]], [[0, 0, 1]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [0.0], atol=1e-10)


def test_distance_parallel_lines():
    r1 = _ray_batch([[0, 0, 0]], [[1, 0, 0]])
    r2 = _ray_batch([[7, 3, 4]], [[1, 0, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0], atol=1e-10)


def test_distance_antiparallel_lines():
    r1 = _ray_batch([[0, 0, 0]], [[0, 0, 1]])
    r2 = _ray_batch([[3, 4, 10]], [[0, 0, -1]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [5.0], atol=1e-10)


def test_distance_symmetric_and_invariant_to_origin_shift_and_direction_flip():
    np.random.seed(1)
    o1, o2 = np.random.randn(8, 3), np.random.randn(8, 3)
    d1, d2 = np.random.randn(8, 3), np.random.randn(8, 3)
    r1, r2 = hray(o1, d1), hray(o2, d2)
    base = line_line_distance(r1, r2)
    assert base.shape == (8,)
    assert np.all(base >= 0)
    np.testing.assert_allclose(line_line_distance(r2, r1), base, atol=1e-9)

    # slide origins along their own lines
    n1 = r1[:, :3, 1]
    n2 = r2[:, :3, 1]
    r1s = hray(o1 + 2.5 * n1, d1)
    r2s = hray(o2 - 1.5 * n2, d2)
    np.testing.assert_allclose(line_line_distance(r1s, r2s), base, atol=1e-9)

    # flipping a direction leaves the line unchanged
    r1f = hray(o1, -d1)
    np.testing.assert_allclose(line_line_distance(r1f, r2), base, atol=1e-9)


def test_distance_batch_mixed_cases():
    r1 = _ray_batch([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    r2 = _ray_batch([[0, 0, 3], [0, 2, 0], [1, 0, 0]],
                    [[0, 1, 0], [1, 0, 0], [0, 1, 0]])
    d = line_line_distance(r1, r2)
    np.testing.assert_allclose(d, [3.0, 2.0, 0.0], atol=1e-10)


# -------------------------------------------- line_line_closest_points


def test_closest_points_simple_skew():
    r1 = _ray_batch([[0, 0, 0]], [[1, 0, 0]])
    r2 = _ray_batch([[0, 0, 3]], [[0, 1, 0]])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[0, 0, 0, 1]], atol=1e-10)
    np.testing.assert_allclose(q2, [[0, 0, 3, 1]], atol=1e-10)


def test_closest_points_with_offset_origins():
    r1 = _ray_batch([[5, 0, 0]], [[1, 0, 0]])
    r2 = _ray_batch([[0, 2, 3]], [[0, 1, 0]])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[0, 0, 0, 1]], atol=1e-10)
    np.testing.assert_allclose(q2, [[0, 0, 3, 1]], atol=1e-10)


def test_closest_points_single_ray_pair_unbatched():
    r1 = hray([5, 0, 0], [1, 0, 0])
    r2 = hray([0, 2, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [0, 0, 0, 1], atol=1e-10)
    np.testing.assert_allclose(q2, [0, 0, 3, 1], atol=1e-10)


def test_closest_points_geometric_properties_random():
    np.random.seed(2)
    n = 12
    r1 = hray(np.random.randn(n, 3), np.random.randn(n, 3))
    r2 = hray(np.random.randn(n, 3), np.random.randn(n, 3))
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (n, 4)
    assert q2.shape == (n, 4)
    pt1, ax1 = r1[:, :, 0], r1[:, :, 1]
    pt2, ax2 = r2[:, :, 0], r2[:, :, 1]

    # points are homogeneous points
    np.testing.assert_allclose(q1[:, 3], 1)
    np.testing.assert_allclose(q2[:, 3], 1)

    # each point lies on its own line
    np.testing.assert_allclose(hcross(q1 - pt1, ax1)[:, :3], 0, atol=1e-8)
    np.testing.assert_allclose(hcross(q2 - pt2, ax2)[:, :3], 0, atol=1e-8)

    # connecting segment is perpendicular to both lines
    seg = q2 - q1
    np.testing.assert_allclose(hdot(seg, ax1), 0, atol=1e-8)
    np.testing.assert_allclose(hdot(seg, ax2), 0, atol=1e-8)

    # segment length equals the line-line distance
    np.testing.assert_allclose(hnorm(seg), line_line_distance(r1, r2),
                               atol=1e-8)


def test_closest_points_coincide_for_intersecting_lines():
    r1 = _ray_batch([[1, 0, 0]], [[1, 0, 0]])
    r2 = _ray_batch([[0, 0, -2]], [[0, 0, 1]])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, q2, atol=1e-10)
    np.testing.assert_allclose(q1, [[0, 0, 0, 1]], atol=1e-10)
