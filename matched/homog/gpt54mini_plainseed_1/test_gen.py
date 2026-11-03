import numpy as np
import pytest

from homog.homog import hray, line_line_distance, line_line_closest_points


def make_ray(origin, direction):
    return hray(np.array(origin, dtype=float), np.array(direction, dtype=float))


def test_hray_from_3d_point_and_direction_shapes_and_homogeneous_components():
    r = hray([1, 2, 3], [0, 0, 5])

    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:3, 0], [1, 2, 3])
    np.testing.assert_allclose(r[3, 0], 1.0)
    np.testing.assert_allclose(r[3, 1], 0.0)
    np.testing.assert_allclose(np.linalg.norm(r[:3, 1]), 1.0)
    np.testing.assert_allclose(r[:3, 1], [0, 0, 1])


def test_hray_preserves_four_component_origin_and_direction_inputs():
    origin = np.array([1.5, -2.0, 3.25, 1.0])
    direction = np.array([0.0, 4.0, 0.0, 0.0])

    r = hray(origin, direction)

    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], origin)
    np.testing.assert_allclose(r[:, 1], np.array([0.0, 1.0, 0.0, 0.0]))


def test_hray_broadcasts_scalar_ray_inputs_to_batch():
    origin = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    direction = np.array([0.0, 0.0, 2.0])

    r = hray(origin, direction)

    assert r.shape == (2, 4, 2)
    np.testing.assert_allclose(r[0, :3, 0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(r[1, :3, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(r[..., 3, 0], 1.0)
    np.testing.assert_allclose(r[..., 3, 1], 0.0)
    np.testing.assert_allclose(r[..., :3, 1], np.array([[0.0, 0.0, 1.0],
                                                       [0.0, 0.0, 1.0]]))


def test_hray_raises_when_origin_has_invalid_last_dimension():
    with pytest.raises((ValueError, IndexError)):
        hray([1, 2], [0, 0, 1])


def test_hray_raises_when_direction_has_invalid_last_dimension():
    with pytest.raises((ValueError, IndexError)):
        hray([1, 2, 3], [0, 1])


def test_line_line_distance_intersecting_orthogonal_lines_is_zero():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 0, 0], [0, 1, 0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 0.0)


def test_line_line_distance_skew_lines_matches_known_value():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 1, 1], [0, 1, 0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 1.0)


def test_line_line_distance_parallel_offset_lines_uses_perpendicular_separation():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 2, 0], [1, 0, 0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 2.0)


def test_line_line_distance_antiparallel_lines_are_treated_as_parallel():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 3, 4], [-2, 0, 0])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 5.0)


def test_line_line_distance_same_line_is_zero():
    ray1 = make_ray([1, 2, 3], [0, 0, 1])
    ray2 = make_ray([10, 20, 30], [0, 0, -5])

    dist = line_line_distance(ray1, ray2)

    assert np.isclose(dist, 0.0)


def test_line_line_distance_is_symmetric():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 1, 1], [0, 1, 0])

    d12 = line_line_distance(ray1, ray2)
    d21 = line_line_distance(ray2, ray1)

    assert np.isclose(d12, d21)
    assert np.isclose(d12, 1.0)


def test_line_line_distance_matches_broadcasted_batch_values():
    ray1 = np.stack(
        [make_ray([0, 0, 0], [1, 0, 0]),
         make_ray([0, 0, 0], [1, 0, 0])],
        axis=0,
    )
    ray2 = np.stack(
        [make_ray([0, 1, 1], [0, 1, 0]),
         make_ray([0, 2, 0], [1, 0, 0])],
        axis=0,
    )

    d = line_line_distance(ray1, ray2)

    np.testing.assert_allclose(d, np.array([1.0, 2.0]))


def test_line_line_closest_points_intersecting_lines_return_same_point():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 0, 0], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1[:3], [0, 0, 0])
    np.testing.assert_allclose(q2[:3], [0, 0, 0])
    np.testing.assert_allclose(q1, q2)


def test_line_line_closest_points_skew_lines_have_expected_pair():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 1, 1], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1[:3], [0, 0, 0])
    np.testing.assert_allclose(q2[:3], [0, 0, 1])
    np.testing.assert_allclose(q2[:3] - q1[:3], [0, 0, 1])
    assert np.isclose(np.linalg.norm(q2[:3] - q1[:3]), line_line_distance(ray1, ray2))


def test_line_line_closest_points_parallel_lines_returns_one_valid_minimizer():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 2, 0], [1, 0, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1[1:], [0, 0, 0])
    np.testing.assert_allclose(q2[1:], [2, 0, 0])
    assert np.isclose(np.linalg.norm(q2[:3] - q1[:3]), 2.0)


def test_line_line_closest_points_are_symmetric_up_to_order():
    ray1 = make_ray([0, 0, 0], [1, 0, 0])
    ray2 = make_ray([0, 1, 1], [0, 1, 0])

    q1a, q2a = line_line_closest_points(ray1, ray2)
    q2b, q1b = line_line_closest_points(ray2, ray1)

    np.testing.assert_allclose(q1a, q1b)
    np.testing.assert_allclose(q2a, q2b)


def test_line_line_closest_points_batch_shape_matches_inputs():
    ray1 = np.stack(
        [make_ray([0, 0, 0], [1, 0, 0]),
         make_ray([0, 0, 0], [1, 0, 0])],
        axis=0,
    )
    ray2 = np.stack(
        [make_ray([0, 0, 0], [0, 1, 0]),
         make_ray([0, 1, 1], [0, 1, 0])],
        axis=0,
    )

    q1, q2 = line_line_closest_points(ray1, ray2)

    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    np.testing.assert_allclose(q1[0], [0, 0, 0, 1])
    np.testing.assert_allclose(q2[0], [0, 0, 0, 1])
    np.testing.assert_allclose(q1[1, :3], [0, 0, 0])
    np.testing.assert_allclose(q2[1, :3], [0, 0, 1])
