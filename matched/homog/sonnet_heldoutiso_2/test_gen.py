import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
    is_valid_rays,
    rand_ray,
    hdot,
    hnorm,
    hcross,
)


def _ray(origin, direction):
    """Build a batch-of-one ray with shape (1, 4, 2)."""
    return hray([origin], [direction])


# ---------------------------------------------------------------- hray


def test_hray_shape_and_values_single():
    r = hray([1, 2, 3], [0, 0, 1])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction():
    r = hray([0, 0, 0], [0, 0, 5])
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0], atol=1e-12)
    r = hray([0, 0, 0], [3, 4, 0])
    np.testing.assert_allclose(r[:, 1], [0.6, 0.8, 0, 0], atol=1e-12)


def test_hray_accepts_four_component_inputs():
    r = hray([1, 2, 3, 1], [0, 3, 0, 0])
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1])
    np.testing.assert_allclose(r[:, 1], [0, 1, 0, 0], atol=1e-12)


def test_hray_forces_origin_w_to_one():
    r = hray([1, 2, 3, 7], [1, 0, 0, 0])
    assert r[3, 0] == 1
    np.testing.assert_allclose(r[:3, 0], [1, 2, 3])


def test_hray_batched():
    np.random.seed(1)
    origin = np.random.randn(5, 3)
    direction = np.random.randn(5, 3)
    r = hray(origin, direction)
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origin)
    np.testing.assert_allclose(r[:, 3, 0], 1)
    np.testing.assert_allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1)
    np.testing.assert_allclose(r[:, 3, 1], 0)
    # direction is parallel to the input direction
    expected = direction / np.linalg.norm(direction, axis=-1)[:, None]
    np.testing.assert_allclose(r[:, :3, 1], expected, atol=1e-12)


def test_hray_broadcasts_single_origin_over_directions():
    np.random.seed(2)
    direction = np.random.randn(4, 3)
    r = hray([1, 2, 3], direction)
    assert r.shape == (4, 4, 2)
    for i in range(4):
        np.testing.assert_allclose(r[i, :, 0], [1, 2, 3, 1])


def test_hray_output_is_valid_ray():
    np.random.seed(3)
    r = hray(np.random.randn(6, 3), np.random.randn(6, 3))
    assert is_valid_rays(r)


# --------------------------------------------------- line_line_distance


def test_distance_skew_axes_aligned():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 3], [0, 1, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [3.0], atol=1e-10)


def test_distance_intersecting_lines_is_zero():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, 5, 0], [0, 1, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [0.0], atol=1e-10)


def test_distance_parallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([5, 2, 0], [1, 0, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [2.0], atol=1e-10)


def test_distance_antiparallel_lines():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([-3, 0, 2], [-1, 0, 0])
    np.testing.assert_allclose(line_line_distance(r1, r2), [2.0], atol=1e-10)


def test_distance_parallel_oblique_direction():
    d = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    perp = np.array([0.0, 0.0, 1.0]) * 2.5
    origin = np.array([1.0, -2.0, 0.5])
    r1 = _ray(origin, d)
    r2 = _ray(origin + perp + 7 * d, d)
    np.testing.assert_allclose(line_line_distance(r1, r2), [2.5], atol=1e-9)


def test_distance_is_symmetric():
    np.random.seed(4)
    r1 = rand_ray(10)
    r2 = rand_ray(10)
    np.testing.assert_allclose(
        line_line_distance(r1, r2), line_line_distance(r2, r1), atol=1e-10)


def test_distance_invariant_to_sliding_origin_along_line():
    np.random.seed(5)
    r1 = rand_ray(6)
    r2 = rand_ray(6)
    d0 = line_line_distance(r1, r2)
    r1s = r1.copy()
    r1s[:, :3, 0] += 3.7 * r1[:, :3, 1]
    r2s = r2.copy()
    r2s[:, :3, 0] -= 2.1 * r2[:, :3, 1]
    np.testing.assert_allclose(line_line_distance(r1s, r2s), d0, atol=1e-8)


def test_distance_nonnegative_and_bounded_by_origin_separation():
    np.random.seed(6)
    r1 = rand_ray(20)
    r2 = rand_ray(20)
    d = line_line_distance(r1, r2)
    assert d.shape == (20,)
    assert np.all(d >= 0)
    sep = np.linalg.norm(r2[:, :3, 0] - r1[:, :3, 0], axis=-1)
    assert np.all(d <= sep + 1e-9)


def test_distance_batch_mixes_parallel_and_skew():
    r1 = hray([[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0]])
    r2 = hray([[0, 0, 3], [0, 4, 0]], [[0, 1, 0], [1, 0, 0]])
    np.testing.assert_allclose(line_line_distance(r1, r2), [3.0, 4.0],
                               atol=1e-10)


# ----------------------------------------- line_line_closest_points


def test_closest_points_simple_skew():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([0, 0, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[0, 0, 0, 1]], atol=1e-10)
    np.testing.assert_allclose(q2, [[0, 0, 3, 1]], atol=1e-10)


def test_closest_points_offset_skew():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, 0, 3], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[2, 0, 0, 1]], atol=1e-10)
    np.testing.assert_allclose(q2, [[2, 0, 3, 1]], atol=1e-10)


def test_closest_points_intersecting_lines_coincide():
    r1 = _ray([0, 0, 0], [1, 0, 0])
    r2 = _ray([2, 5, 0], [0, 1, 0])
    q1, q2 = line_line_closest_points(r1, r2)
    np.testing.assert_allclose(q1, [[2, 0, 0, 1]], atol=1e-10)
    np.testing.assert_allclose(q2, [[2, 0, 0, 1]], atol=1e-10)


def test_closest_points_random_properties():
    np.random.seed(7)
    r1 = rand_ray(12)
    r2 = rand_ray(12)
    q1, q2 = line_line_closest_points(r1, r2)
    assert q1.shape == (12, 4)
    assert q2.shape == (12, 4)
    ax1, ax2 = r1[:, :, 1], r2[:, :, 1]
    # points have w == 1
    np.testing.assert_allclose(q1[:, 3], 1, atol=1e-10)
    np.testing.assert_allclose(q2[:, 3], 1, atol=1e-10)
    # q1 lies on line 1, q2 lies on line 2
    np.testing.assert_allclose(
        hnorm(hcross(q1 - r1[:, :, 0], ax1)), 0, atol=1e-8)
    np.testing.assert_allclose(
        hnorm(hcross(q2 - r2[:, :, 0], ax2)), 0, atol=1e-8)
    # connecting segment is perpendicular to both lines
    diff = q2 - q1
    np.testing.assert_allclose(hdot(diff, ax1), 0, atol=1e-8)
    np.testing.assert_allclose(hdot(diff, ax2), 0, atol=1e-8)
    # segment length equals the line-line distance
    np.testing.assert_allclose(
        hnorm(diff), line_line_distance(r1, r2), atol=1e-8)
