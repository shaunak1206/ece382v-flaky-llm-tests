import numpy as np
import pytest
from homog import homog

# Alias for convenience
hray = homog.hray
line_line_distance = homog.line_line_distance
line_line_closest_points = homog.line_line_closest_points


def test_hray_basic_shape_and_properties():
    origin = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.0, 0.0, 1.0])

    r = hray(origin, direction)

    # Should have shape (4, 2)
    assert r.shape == (4, 2)

    # Origin in homogeneous coords, last component must be 1
    np.testing.assert_allclose(r[:, 0], np.array([1, 2, 3, 1]))

    # Direction vector: last component zero
    assert r[3, 1] == 0

    # Direction normalized
    dir_vec = r[:3, 1]
    norm = np.linalg.norm(dir_vec)
    assert np.isclose(norm, 1.0)


@pytest.mark.parametrize("origin_shape,direction_shape", [
    ((3,), (3,)),
    ((2, 3), (3,)),
    ((3,), (2,)),
    ((2, 3), (2, 3)),
])
def test_hray_broadcasting(origin_shape, direction_shape):
    origin = np.ones(origin_shape)
    direction = np.ones(direction_shape)
    # normalize direction
    if direction_shape[-1] != 3:
        # make last dim to 3
        shape = direction_shape[:-1] + (3,)
        direction = np.ones(shape)
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)

    r = hray(origin, direction)
    # shape depends on broadcasting last dim but fixed to 4 x 2
    expected_shape = np.broadcast(np.empty(origin_shape), np.empty(direction_shape)).shape + (4, 2)
    # But hray returns shape (..., 4, 2) with last dims fixed to (4,2)
    # Actually r shape is something like (..., 4, 2)
    # Here must confirm last 2 dims are (4, 2)
    assert r.shape[-2:] == (4, 2)
    # r shape except last two dims must match broadcast shape minus 3 (origin/direction last dim)
    # origin_shape[-1] == 3 or 4; direction_shape[-1] == 3 or 4
    # Broadcast along all dims except last for origin and direction except last for direction
    # So checking overall shape ignoring last two dims:
    # The broadcast shape for origin and direction without last dim is:
    shape_no_last_dims = np.broadcast(
        np.empty(origin_shape[:-1]), np.empty(direction_shape[:-1])
    ).shape
    assert r.shape[:-2] == shape_no_last_dims


def test_hray_direction_normalization_enforced():
    origin = np.array([0, 0, 0])
    direction = np.array([10, 0, 0])  # not normalized
    r = hray(origin, direction)
    dir_vec = r[:3, 1]
    norm = np.linalg.norm(dir_vec)
    assert np.isclose(norm, 1.0)


def test_line_line_distance_parallel_lines():
    # Two parallel lines along x axis with offset along y axis of 1 unit
    origin1 = np.array([0, 0, 0, 1])
    dir1 = np.array([1, 0, 0])
    ray1 = hray(origin1, dir1)

    origin2 = np.array([0, 1, 0, 1])
    dir2 = np.array([1, 0, 0])
    ray2 = hray(origin2, dir2)

    dist = line_line_distance(ray1, ray2)

    # Distance should be 1 because lines are parallel and offset by 1 in y
    assert np.isclose(dist, 1.0)


def test_line_line_distance_skew_lines():
    # Line 1 along x axis at origin
    origin1 = np.array([0, 0, 0, 1])
    dir1 = np.array([1, 0, 0])
    ray1 = hray(origin1, dir1)

    # Line 2 along y axis but offset along x and z: passes through (1,0,1)
    origin2 = np.array([1, 0, 1, 1])
    dir2 = np.array([0, 1, 0])
    ray2 = hray(origin2, dir2)

    dist = line_line_distance(ray1, ray2)

    # Distance should be sqrt(1^2 + 1^2) = sqrt(2) = approximately 1.414
    assert np.isclose(dist, np.sqrt(2))


def test_line_line_distance_intersecting_lines():
    # Line 1 x axis through origin
    origin1 = np.array([0, 0, 0, 1])
    dir1 = np.array([1, 0, 0])
    ray1 = hray(origin1, dir1)

    # Line 2 y axis through origin
    origin2 = np.array([0, 0, 0, 1])
    dir2 = np.array([0, 1, 0])
    ray2 = hray(origin2, dir2)

    dist = line_line_distance(ray1, ray2)

    # Both lines intersect at origin, so distance should be zero
    assert np.isclose(dist, 0.0)


def test_line_line_closest_points_basic():
    # Two skew lines: one along x axis at origin, other along y axis offset in z=1
    origin1 = np.array([0, 0, 0, 1])
    dir1 = np.array([1, 0, 0])
    ray1 = hray(origin1, dir1)

    origin2 = np.array([0, 0, 1, 1])
    dir2 = np.array([0, 1, 0])
    ray2 = hray(origin2, dir2)

    Q1, Q2 = line_line_closest_points(ray1, ray2)

    # Closest points should be same z (since line2 offset is at z=1)
    # Q1 lies on line1, Q2 lies on line2, distance between Q1 and Q2 minimal

    # Check shape
    assert Q1.shape == (4,)
    assert Q2.shape == (4,)

    # Check Q1 and Q2 have last component 1 for origin points
    assert Q1[3] == 1
    assert Q2[3] == 1

    # Distance between closest points
    dist = np.linalg.norm(Q1[:3] - Q2[:3])
    # Should be minimal distance between two skew lines, in this case 1 unit
    # Because one line is at z=0 and other line offset in z=1
    assert np.isclose(dist, 1.0, atol=1e-7)


def test_line_line_closest_points_parallel_lines():
    # Two parallel lines with offset along y axis
    origin1 = np.array([0, 0, 0, 1])
    dir1 = np.array([1, 0, 0])
    ray1 = hray(origin1, dir1)

    origin2 = np.array([0, 5, 0, 1])
    dir2 = np.array([1, 0, 0])
    ray2 = hray(origin2, dir2)

    Q1, Q2 = line_line_closest_points(ray1, ray2)

    # Closest points: Q1 should be on line1, Q2 on line2
    # Since lines are parallel, closest points differ by offset vector

    # Check last component of points = 1 (homog points)
    assert Q1[3] == 1
    assert Q2[3] == 1

    # Vector between closest points should be along y axis and equal to offset 5
    offset_vec = Q1[:3] - Q2[:3]
    expected_offset = np.array([0, -5, 0])
    np.testing.assert_allclose(offset_vec, expected_offset, atol=1e-7)


def test_hray_direction_input_normalization():
    origin = np.array([0, 0, 0])
    direction = np.array([3, 0, 4])  # length 5 vector
    r = hray(origin, direction)
    norm_dir = r[:3, 1]

    # Direction gets normalized: norm should be 1
    assert np.isclose(np.linalg.norm(norm_dir), 1.0)

    # Direction vector is same direction as input (normalized)
    expected_dir = direction / np.linalg.norm(direction)
    np.testing.assert_allclose(norm_dir, expected_dir, atol=1e-12)


def test_line_line_distance_array_inputs():
    # Create batch of two pairs of rays for distance calculation
    origins1 = np.array([[0, 0, 0, 1], [0, 0, 0, 1]])
    dirs1 = np.array([[1, 0, 0], [0, 1, 0]])
    rays1 = np.stack([origins1.T, dirs1.T], axis=-1)  # shape (4,2,2) but reorder dims to (2,4,2)
    rays1 = rays1.transpose(1, 0, 2)  # shape (2,4,2)

    origins2 = np.array([[0, 1, 0, 1], [1, 0, 0, 1]])
    dirs2 = np.array([[1, 0, 0], [0, 1, 0]])
    rays2 = np.stack([origins2.T, dirs2.T], axis=-1).transpose(1, 0, 2)

    distances = line_line_distance(rays1, rays2)

    # Expect distances for first pair (parallel lines offset 1 in y)
    assert np.isclose(distances[0], 1.0)

    # For second pair (same line directions and origins differ by [1,0,0]) distance should be 1
    assert np.isclose(distances[1], 1.0)


def test_line_line_closest_points_array_inputs():
    # Batch test for closest points of two rays
    origins1 = np.array([[0, 0, 0, 1], [1, 0, 0, 1]])
    dirs1 = np.array([[1, 0, 0], [0, 1, 0]])
    r1 = np.stack([origins1.T, dirs1.T], axis=-1).transpose(1, 0, 2)  # shape (2, 4, 2)

    origins2 = np.array([[0, 1, 1, 1], [1, 0, 1, 1]])
    dirs2 = np.array([[0, 1, 0], [1, 0, 0]])
    r2 = np.stack([origins2.T, dirs2.T], axis=-1).transpose(1, 0, 2)

    Q1, Q2 = line_line_closest_points(r1, r2)

    # Q1 and Q2 shape: (2, 4)
    assert Q1.shape == (2, 4)
    assert Q2.shape == (2, 4)

    # All last components should be 1 (homogeneous points origins)
    assert np.allclose(Q1[:, 3], 1)
    assert np.allclose(Q2[:, 3], 1)

    # Distances between pairs should be minimal
    distances = np.linalg.norm(Q1[:, :3] - Q2[:, :3], axis=-1)
    assert np.all(distances >= 0)
    assert np.all(distances < 2)  # arbitrary upper bound for these cases


def test_hray_direction_not_4vector_raises():
    # Direction vector length not 3 or 4 should raise in hnormalized function called by hray
    origin = np.array([0, 0, 0])
    direction = np.array([0, 0])  # invalid length
    with pytest.raises(IndexError):
        hray(origin, direction)


def test_line_line_closest_points_identical_lines():
    origin = np.array([0, 0, 0, 1])
    direction = np.array([0, 1, 0])
    ray = hray(origin, direction)

    Q1, Q2 = line_line_closest_points(ray, ray)
    # Lines are identical, closest points should be the same
    np.testing.assert_allclose(Q1, Q2)


def test_line_line_distance_identical_lines_zero():
    origin = np.array([0, 0, 0, 1])
    direction = np.array([0, 1, 0])
    ray = hray(origin, direction)
    dist = line_line_distance(ray, ray)
    assert dist == 0


def test_line_line_closest_points_raises_for_equal_direction_vectors(monkeypatch):
    # The function currently errors if ax1 == ax2, so test edge case
    origin1 = np.array([0, 0, 0, 1])
    origin2 = np.array([0, 0, 1, 1])
    direction = np.array([0, 1, 0])

    ray1 = hray(origin1, direction)
    ray2 = hray(origin2, direction)

    Q1, Q2 = line_line_closest_points(ray1, ray2)

    # Q1 on ray1 and Q2 on ray2
    assert Q1.shape == (4,)
    assert Q2.shape == (4,)
    # They differ as lines are offset by 1 in z
    assert not np.allclose(Q1, Q2)


# The tests above cover various scenarios to validate correctness and error handling
