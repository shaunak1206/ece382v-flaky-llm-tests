import numpy as np
import pytest

from homog.homog import hray, hpoint, hvec, hnormalized
from homog.homog import line_line_distance, line_line_closest_points


def test_hray_from_3d_inputs_builds_valid_ray():
    origin = [1.0, 2.0, 3.0]
    direction = [0.0, 0.0, 2.0]
    ray = hray(origin, direction)

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:, 0], [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(ray[:, 1], [0.0, 0.0, 1.0, 0.0])


def test_hray_accepts_homogeneous_origin_and_direction():
    origin = np.array([4.0, 5.0, 6.0, 1.0])
    direction = np.array([1.0, 2.0, 2.0, 0.0])
    ray = hray(origin, direction)

    assert ray.shape == (4, 2)
    np.testing.assert_allclose(ray[:, 0], origin)
    np.testing.assert_allclose(ray[:, 1], direction / np.linalg.norm(direction[:3]))


def test_hray_broadcasts_origin_and_direction():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    direction = np.array([0.0, 1.0, 0.0])
    rays = hray(origins, direction)

    assert rays.shape == (2, 4, 2)
    np.testing.assert_allclose(rays[0, :, 0], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(rays[1, :, 0], [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(rays[:, :, 1], np.array([[0.0, 1.0, 0.0, 0.0],
                                                        [0.0, 1.0, 0.0, 0.0]]))


def test_hray_normalizes_non_unit_direction():
    ray = hray([0, 0, 0], [0, 3, 4])

    np.testing.assert_allclose(ray[:, 1], [0.0, 0.6, 0.8, 0.0])


def test_hpoint_and_hvec_support_hray_inputs_roundtrip():
    pt = hpoint([1, 2, 3])
    vec = hvec([4, 5, 6])

    np.testing.assert_allclose(pt, [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(vec, [4.0, 5.0, 6.0, 0.0])


def test_hray_invalid_direction_shape_raises():
    with pytest.raises(ValueError):
        hray([0, 0, 0], [1, 2])


def test_line_line_distance_intersecting_lines_is_zero():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 0, 0], [0, 1, 0])

    d = line_line_distance(ray1, ray2)

    assert d == pytest.approx(0.0)


def test_line_line_distance_skew_lines_matches_expected_value():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    d = line_line_distance(ray1, ray2)

    assert d == pytest.approx(1.0)


def test_line_line_distance_parallel_lines_uses_point_to_line_distance():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 0], [2, 0, 0])

    d = line_line_distance(ray1, ray2)

    assert d == pytest.approx(1.0)


def test_line_line_distance_antiparallel_lines_same_result_as_parallel():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 2, 0], [-3, 0, 0])

    d = line_line_distance(ray1, ray2)

    assert d == pytest.approx(2.0)


def test_line_line_distance_symmetry():
    ray1 = hray([1, 2, 3], [1, 0, 0])
    ray2 = hray([0, 4, 7], [0, 1, 1])

    d12 = line_line_distance(ray1, ray2)
    d21 = line_line_distance(ray2, ray1)

    assert d12 == pytest.approx(d21)


def test_line_line_closest_points_intersecting_lines_return_same_point():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 0, 0], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(q2, [0.0, 0.0, 0.0, 1.0])


def test_line_line_closest_points_skew_lines_are_correct():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 1], [0, 1, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    np.testing.assert_allclose(q1, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(q2, [0.0, 0.0, 1.0, 1.0])


def test_line_line_closest_points_is_symmetric_in_the_set_of_points():
    ray1 = hray([1, 2, 3], [1, 0, 0])
    ray2 = hray([4, 5, 6], [0, 1, 0])

    q1a, q2a = line_line_closest_points(ray1, ray2)
    q2b, q1b = line_line_closest_points(ray2, ray1)

    np.testing.assert_allclose(q1a, q1b)
    np.testing.assert_allclose(q2a, q2b)


def test_line_line_closest_points_parallel_lines_produce_non_finite_values():
    ray1 = hray([0, 0, 0], [1, 0, 0])
    ray2 = hray([0, 1, 0], [2, 0, 0])

    q1, q2 = line_line_closest_points(ray1, ray2)

    assert not np.all(np.isfinite(q1))
    assert not np.all(np.isfinite(q2))


def test_line_line_distance_with_batch_inputs():
    ray1 = np.stack(
        [hray([0, 0, 0], [1, 0, 0]),
         hray([0, 0, 0], [1, 0, 0])],
        axis=0,
    )
    ray2 = np.stack(
        [hray([0, 0, 0], [0, 1, 0]),
         hray([0, 3, 0], [0, 1, 0])],
        axis=0,
    )

    d = line_line_distance(ray1, ray2)

    np.testing.assert_allclose(d, [0.0, 3.0])
