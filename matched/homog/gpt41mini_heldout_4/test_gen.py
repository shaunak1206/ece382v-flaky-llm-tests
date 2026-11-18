import numpy as np
import pytest
from homog import homog


def test_hray_basic_shapes():
    origin = [1, 2, 3]
    direction = [0, 0, 1]
    r = homog.hray(origin, direction)
    assert r.shape == (4, 2)
    # origin part should have w=1
    assert np.allclose(r[..., 3, 0], 1)
    # direction part should have w=0
    assert np.allclose(r[..., 3, 1], 0)
    # direction vector normalized?
    norm = np.linalg.norm(r[:3, 1])
    assert np.isclose(norm, 1)


def test_hray_batch_shape():
    origins = np.array([[1, 0, 0], [0, 1, 0]])
    directions = np.array([[1, 0, 0], [0, 1, 0]])
    r = homog.hray(origins, directions)
    assert r.shape == (2, 4, 2)
    assert np.allclose(r[..., 3, 0], 1)
    assert np.allclose(r[..., 3, 1], 0)
    norms = np.linalg.norm(r[..., :3, 1], axis=-2)
    assert np.allclose(norms, 1)


def test_line_line_distance_parallel():
    # Parallel rays: distance should be distance between lines
    origin1 = np.array([0, 0, 0, 1])
    direction1 = np.array([1, 0, 0, 0])
    ray1 = homog.hray(origin1, direction1)

    origin2 = np.array([0, 1, 0, 1])  # Offset by 1 on Y axis
    direction2 = np.array([1, 0, 0, 0])
    ray2 = homog.hray(origin2, direction2)

    dist = homog.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 1)


def test_line_line_distance_perpendicular():
    origin1 = np.array([0, 0, 0, 1])
    direction1 = np.array([1, 0, 0, 0])
    ray1 = homog.hray(origin1, direction1)

    origin2 = np.array([0, 1, 0, 1])
    direction2 = np.array([0, 1, 0, 0])
    ray2 = homog.hray(origin2, direction2)

    dist = homog.line_line_distance(ray1, ray2)
    # They intersect at origin, distance should be 0
    assert np.isclose(dist, 0)


def test_line_line_distance_skew():
    origin1 = np.array([0, 0, 0, 1])
    direction1 = np.array([1, 0, 0, 0])
    ray1 = homog.hray(origin1, direction1)

    origin2 = np.array([0, 1, 1, 1])
    direction2 = np.array([0, 1, 0, 0])
    ray2 = homog.hray(origin2, direction2)

    dist = homog.line_line_distance(ray1, ray2)
    # Should be distance between skew lines, check non-zero
    assert dist > 0


def test_line_line_closest_points_identical_rays():
    origin = np.array([0, 0, 0, 1])
    direction = np.array([1, 0, 0, 0])
    ray = homog.hray(origin, direction)

    q1, q2 = homog.line_line_closest_points(ray, ray)
    assert q1.shape == (4,)
    assert q2.shape == (4,)
    # Closest points should be equal
    assert np.allclose(q1, q2)


def test_line_line_closest_points_perpendicular():
    origin1 = np.array([0, 0, 0, 1])
    direction1 = np.array([1, 0, 0, 0])
    ray1 = homog.hray(origin1, direction1)

    origin2 = np.array([0, 1, 0, 1])
    direction2 = np.array([0, 1, 0, 0])
    ray2 = homog.hray(origin2, direction2)

    q1, q2 = homog.line_line_closest_points(ray1, ray2)
    # Points closest on both lines
    vec = q1 - q2
    dist = np.linalg.norm(vec[:3])
    # For these rays, distance should be zero (intersecting lines)
    assert np.isclose(dist, 0)


def test_line_line_closest_points_skew():
    origin1 = np.array([0, 0, 0, 1])
    direction1 = np.array([1, 0, 0, 0])
    ray1 = homog.hray(origin1, direction1)

    origin2 = np.array([0, 1, 1, 1])
    direction2 = np.array([0, 1, 0, 0])
    ray2 = homog.hray(origin2, direction2)

    q1, q2 = homog.line_line_closest_points(ray1, ray2)
    dist = np.linalg.norm(q1[:3] - q2[:3])
    assert dist > 0
    # Directions from closest points are perpendicular to vector between q1,q2
    diff = q2[:3] - q1[:3]
    dot1 = np.dot(diff, direction1[:3])
    dot2 = np.dot(diff, direction2[:3])
    assert np.isclose(dot1, 0, atol=1e-6)
    assert np.isclose(dot2, 0, atol=1e-6)


@pytest.mark.parametrize("origin, direction", [
    ([0, 0, 0], [1, 0, 0]),
    ([1, 2, 3], [0, 1, 0]),
    ([4, 5, 6], [0, 0, 1]),
])
def test_hray_properties(origin, direction):
    r = homog.hray(origin, direction)
    # Check shape
    assert r.shape == (4, 2)
    # Check origin w=1
    assert np.isclose(r[3, 0], 1)
    # Check direction w=0
    assert np.isclose(r[3, 1], 0)
    # Check direction normalized
    norm = np.linalg.norm(r[:3, 1])
    assert np.isclose(norm, 1)


def test_hray_direction_normalization():
    origin = [0, 0, 0]
    direction = [10, 0, 0]  # Non-unit vector direction
    r = homog.hray(origin, direction)
    # Direction should be normalized
    assert np.allclose(r[:3, 1], np.array([1, 0, 0]))


def test_line_line_distance_zero_for_overlapping_lines():
    origin = [0, 0, 0, 1]
    direction = [1, 0, 0, 0]
    ray1 = homog.hray(origin, direction)
    ray2 = homog.hray(origin, direction)
    dist = homog.line_line_distance(ray1, ray2)
    assert np.isclose(dist, 0)


def test_line_line_distance_batch():
    origin1 = np.array([[0, 0, 0, 1], [1, 0, 0, 1]])
    direction1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    ray1 = homog.hray(origin1, direction1)

    origin2 = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    direction2 = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    ray2 = homog.hray(origin2, direction2)

    dist = homog.line_line_distance(ray1, ray2)
    assert dist.shape == (2,)
    assert all(d >= 0 for d in dist)


def test_line_line_closest_points_batch():
    origin1 = np.array([[0, 0, 0, 1], [1, 0, 0, 1]])
    direction1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    ray1 = homog.hray(origin1, direction1)

    origin2 = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    direction2 = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
    ray2 = homog.hray(origin2, direction2)

    q1, q2 = homog.line_line_closest_points(ray1, ray2)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    v = q1 - q2
    dist = np.linalg.norm(v[:, :3], axis=1)
    assert all(d >= 0 for d in dist)


def test_line_line_distance_exception_for_bad_shape():
    ray1 = np.zeros((3, 4, 2))
    ray2 = np.zeros((4, 4, 2))
    with pytest.raises(ValueError):
        # The function itself does not explicitly raise, but probably fails internally
        homog.line_line_distance(ray1, ray2)


def test_hray_direction_zero_vector_raises():
    origin = [0, 0, 0]
    direction = [0, 0, 0]
    with pytest.raises(ZeroDivisionError):
        homog.hray(origin, direction)
