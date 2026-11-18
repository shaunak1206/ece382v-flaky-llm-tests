import numpy as np
import pytest

from homog.homog import (
    hray,
    line_line_distance,
    line_line_closest_points,
)


def make_ray(origin, direction):
    return hray(np.array(origin, dtype=float), np.array(direction, dtype=float))


def test_hray_converts_3d_origin_and_direction_to_homogeneous_ray():
    r = hray([1.0, 2.0, 3.0], [0.0, 0.0, 5.0])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(r[3, 0], 1.0)
    np.testing.assert_allclose(r[3, 1], 0.0)
    np.testing.assert_allclose(r[:3, 1], [0.0, 0.0, 1.0])


def test_hray_passes_through_4d_origin_unchanged_and_normalizes_direction():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([0.0, 4.0, 0.0, 0.0])
    r = hray(origin, direction)
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], origin)
    np.testing.assert_allclose(r[:3, 1], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(r[3, 1], 0.0)


def test_hray_broadcasts_origin_and_direction_arrays():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -2.0]])
    rays = hray(origins, directions)
    assert rays.shape == (2, 4, 2)
    np.testing.assert_allclose(rays[0, :3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(rays[1, :3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(rays[0, :3, 1], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rays[1, :3, 1], [0.0, 0.0, -1.0])


def test_hray_raises_for_invalid_origin_length():
    with pytest.raises(ValueError):
        hray([1.0, 2.0], [1.0, 0.0, 0.0])


def test_hray_raises_for_invalid_direction_length():
    with pytest.raises(ValueError):
        hray([1.0, 2.0, 3.0], [1.0, 0.0])


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    d = line_line_distance(ray1, ray2)
    assert d == pytest.approx(0.0, abs=1e-12)


def test_line_line_distance_skew_lines_matches_expected_distance():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 1.0, 1.0], [0.0, 0.0, 1.0])
    d = line_line_distance(ray1, ray2)
    assert d == pytest.approx(1.0, abs=1e-12)


def test_line_line_distance_parallel_lines_uses_perpendicular_offset():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 2.0, 0.0], [2.0, 0.0, 0.0])
    d = line_line_distance(ray1, ray2)
    assert d == pytest.approx(2.0, abs=1e-12)


def test_line_line_distance_is_symmetric():
    ray1 = make_ray([1.0, 2.0, 3.0], [0.0, 1.0, 0.0])
    ray2 = make_ray([-2.0, 5.0, -1.0], [1.0, 0.0, 1.0])
    d12 = line_line_distance(ray1, ray2)
    d21 = line_line_distance(ray2, ray1)
    assert d12 == pytest.approx(d21, abs=1e-12)


def test_line_line_closest_points_intersecting_lines_return_same_point():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(q2, [0.0, 0.0, 0.0])


def test_line_line_closest_points_skew_lines_returns_expected_points():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 1.0, 1.0], [0.0, 0.0, 1.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(q1, [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(q2, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(q1 - q2), 1.0, atol=1e-12)


def test_line_line_closest_points_points_lie_on_their_respective_lines():
    ray1 = make_ray([1.0, 2.0, 3.0], [1.0, 2.0, 0.0])
    ray2 = make_ray([-2.0, 1.0, 4.0], [0.0, 1.0, 1.0])
    q1, q2 = line_line_closest_points(ray1, ray2)

    p1 = ray1[:3, 0]
    a1 = ray1[:3, 1]
    p2 = ray2[:3, 0]
    a2 = ray2[:3, 1]

    v1 = q1 - p1
    v2 = q2 - p2
    np.testing.assert_allclose(np.cross(v1, a1), [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.cross(v2, a2), [0.0, 0.0, 0.0], atol=1e-12)


def test_line_line_closest_points_segment_between_points_is_perpendicular_to_both_lines():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    ray2 = make_ray([0.0, 0.0, 1.0], [0.0, 1.0, 1.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    diff = q2 - q1
    np.testing.assert_allclose(np.dot(diff, ray1[:3, 1]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.dot(diff, ray2[:3, 1]), 0.0, atol=1e-12)


def test_line_line_closest_points_is_symmetric_up_to_order():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    ray2 = make_ray([0.0, 1.0, 1.0], [0.0, 0.0, 1.0])
    q1, q2 = line_line_closest_points(ray1, ray2)
    p2, p1 = line_line_closest_points(ray2, ray1)
    np.testing.assert_allclose(q1, p1, atol=1e-12)
    np.testing.assert_allclose(q2, p2, atol=1e-12)


def test_line_line_closest_points_parallel_lines_raise_or_fail_predictably():
    ray1 = make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    ray2 = make_ray([0.0, 1.0, 0.0], [2.0, 0.0, 0.0])

    with np.errstate(divide="ignore", invalid="ignore"):
        q1, q2 = line_line_closest_points(ray1, ray2)

    assert q1.shape == (4,)
    assert q2.shape == (4,)
    assert np.any(~np.isfinite(q1[:3])) or np.any(~np.isfinite(q2[:3]))


def test_line_line_distance_vectorized_over_multiple_ray_pairs():
    ray1 = np.stack(
        [
            make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            make_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        ],
        axis=0,
    )
    ray2 = np.stack(
        [
            make_ray([0.0, 1.0, 1.0], [0.0, 0.0, 1.0]),
            make_ray([0.0, 2.0, 0.0], [0.0, 1.0, 0.0]),
        ],
        axis=0,
    )
    d = line_line_distance(ray1, ray2)
    np.testing.assert_allclose(d, [1.0, 0.0], atol=1e-12)
