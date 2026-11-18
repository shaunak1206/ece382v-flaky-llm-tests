import numpy as np
import pytest
from homog import homog

# Alias functions for easier testing
hray = homog.hray
line_line_distance = homog.line_line_distance
line_line_closest_points = homog.line_line_closest_points


def test_hray_basic_shape_and_properties():
    origin = np.array([1, 2, 3])
    direction = np.array([0, 0, 1])
    r = hray(origin, direction)
    assert r.shape == (4, 2)
    # Origin homogenous coordinate w=1
    assert np.allclose(r[..., 3, 0], 1)
    # Direction homogenous coordinate w=0
    assert np.allclose(r[..., 3, 1], 0)
    # Check origin first 3 coords match input
    assert np.allclose(r[:3, 0], origin)
    # Direction should be normalized
    dir_norm = np.linalg.norm(r[:3,1])
    assert np.isclose(dir_norm, 1.0)


def test_hray_broadcast_shapes():
    origins = np.array([[1, 2, 3], [4, 5, 6]])
    directions = np.array([[0, 1, 0], [0, 0, 1]])
    r = hray(origins, directions)
    assert r.shape == (2, 4, 2)
    # Directions normalized
    norms = np.linalg.norm(r[:, :3, 1], axis=1)
    assert np.allclose(norms, 1.0)
    # Origins preserved, w=1
    assert np.allclose(r[:, :3, 0], origins)
    assert np.allclose(r[:, 3, 0], 1)
    # Direction w=0
    assert np.allclose(r[:, 3, 1], 0)


def test_line_line_distance_parallel():
    # Two parallel rays offset along y axis
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 0])
    direction2 = np.array([1, 0, 0])
    ray2 = hray(origin2, direction2)

    dist = line_line_distance(ray1, ray2)
    # Distance should be 1 (offset along y)
    assert np.isclose(dist, 1.0)


def test_line_line_distance_intersecting():
    # Two rays intersecting at origin
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 0, 0])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    dist = line_line_distance(ray1, ray2)
    assert np.isclose(dist, 0.0)


def test_line_line_distance_skew_lines():
    # Two skew (non-parallel, non-intersecting) lines
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 1])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    dist = line_line_distance(ray1, ray2)
    # Expected distance is 1 (distance from line y=0,z=0 to point y=1,z=1 projected)
    assert np.isclose(dist, 1.0)


def test_line_line_closest_points_intersecting():
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 0, 0])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    Q1, Q2 = line_line_closest_points(ray1, ray2)
    # Closest points should be the same point (0,0,0)
    assert np.allclose(Q1, Q2)
    assert np.allclose(Q1, origin1)


def test_line_line_closest_points_parallel():
    origin1 = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    ray1 = hray(origin1, direction)

    origin2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction)

    Q1, Q2 = line_line_closest_points(ray1, ray2)
    # Points closest but distinct since parallel lines
    # Closest point on ray1 has y=0, closest on ray2 has y=1
    # Their vector difference along y axis
    diff = Q1 - Q2
    dist = np.linalg.norm(diff[:3])
    assert np.isclose(dist, 1.0)


def test_line_line_closest_points_skew():
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 1])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    Q1, Q2 = line_line_closest_points(ray1, ray2)
    # Distance between closest points equals line_line_distance
    dist_expected = line_line_distance(ray1, ray2)
    actual_dist = np.linalg.norm(Q1[:3] - Q2[:3])
    assert np.isclose(dist_expected, actual_dist)


def test_hray_direction_normalization():
    origin = np.array([1., 2., 3.])
    direction = np.array([10., 0., 0.])
    r = hray(origin, direction)
    # Direction vector normalized to unit length
    assert np.allclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_invalid_direction_shape_raises():
    origin = np.array([0., 0., 0.])
    # Provide invalid direction shape (length 2, not 3)
    direction = np.array([1., 0.])
    with pytest.raises(Exception):
        hray(origin, direction)


def test_line_line_distance_broadcasting():
    origin1 = np.array([[0, 0, 0], [1, 0, 0]])
    direction1 = np.array([[1, 0, 0], [1, 0, 0]])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 0])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    distances = line_line_distance(ray1, ray2)
    # Shape should be (2,)
    assert distances.shape == (2,)
    # Check values for correctness roughly
    assert np.allclose(distances, [1.0, 1.0])


def test_line_line_closest_points_broadcasting():
    origin1 = np.array([[0, 0, 0], [1, 0, 0]])
    direction1 = np.array([[1, 0, 0], [1, 0, 0]])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 0])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    Q1, Q2 = line_line_closest_points(ray1, ray2)
    # Q1, Q2 shapes should broadcast accordingly (2,4)
    assert Q1.shape == (2, 4)
    assert Q2.shape == (2, 4)
    # Distances between Q1 and Q2 should equal line_line_distance
    dist_calc = np.linalg.norm(Q1[:, :3] - Q2[:, :3], axis=1)
    dist_func = line_line_distance(ray1, ray2)
    assert np.allclose(dist_calc, dist_func)


def test_hray_raises_on_invalid_origin_shape():
    # origin shape last dim not 3 or 4
    origin = np.array([1, 2])  # length 2
    direction = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        hray(origin, direction)


def test_hray_with_4d_origin_and_direction():
    origin = np.array([1.0, 2.0, 3.0, 1.0])
    direction = np.array([0.0, 0.0, 1.0, 0.0])
    r = hray(origin, direction)
    # Origin and direction unchanged in first 3 components
    assert np.allclose(r[:4, 0], origin)
    assert np.allclose(r[:4, 1], homog.hnormalized(direction))


def test_line_line_distance_zero_length_direction():
    # direction vector zero length should raise or handle gracefully
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([0, 0, 0])
    ray1 = hray(origin1, direction1)
    origin2 = np.array([1, 0, 0])
    direction2 = np.array([1, 0, 0])
    ray2 = hray(origin2, direction2)

    dist = line_line_distance(ray1, ray2)
    # Distance defined but direction zero vector -> distance should be distance from ray points
    assert dist >= 0


def test_line_line_closest_points_when_lines_are_identical():
    origin = np.array([0, 0, 0])
    direction = np.array([1, 0, 0])
    ray1 = hray(origin, direction)
    ray2 = hray(origin, direction)
    Q1, Q2 = line_line_closest_points(ray1, ray2)
    # Closest points should be identical and match origin
    assert np.allclose(Q1, origin.tolist() + [1])
    assert np.allclose(Q2, origin.tolist() + [1])


def test_line_line_distance_when_lines_cross_obliquely():
    # Two line rays cross at an oblique angle
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)
    origin2 = np.array([1, 1, 0])
    direction2 = np.array([0, -1, 0])
    ray2 = hray(origin2, direction2)

    dist = line_line_distance(ray1, ray2)
    # Lines intersect at (1,0,0), distance should be 0
    assert np.isclose(dist, 0.0)

    Q1, Q2 = line_line_closest_points(ray1, ray2)
    assert np.allclose(Q1, Q2)


def test_line_line_closest_points_verbose_output(monkeypatch):
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)
    origin2 = np.array([0, 1, 0])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    # Patch print to track calls
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(a))

    Q1, Q2 = line_line_closest_points(ray1, ray2, verbose=1)
    # Check that print was called
    assert any("C21" in str(p) for p in printed)
    # Validate points shapes
    assert Q1.shape == (4,)
    assert Q2.shape == (4,)


def test_hray_with_batched_inputs():
    origins = np.array([[0, 0, 0], [1, 1, 1]])
    directions = np.array([[1, 0, 0], [0, 1, 0]])
    r = hray(origins, directions)
    assert r.shape == (2, 4, 2)
    # All directions are normalized to 1
    norms = np.linalg.norm(r[:, :3, 1], axis=1)
    assert np.allclose(norms, 1)


def test_line_line_distance_with_batched_rays():
    origins1 = np.array([[0, 0, 0], [1, 0, 0]])
    directions1 = np.array([[1, 0, 0], [1, 0, 0]])
    ray1 = hray(origins1, directions1)

    origins2 = np.array([[0, 1, 0], [0, 1, 0]])
    directions2 = np.array([[0, 1, 0], [0, 1, 0]])
    ray2 = hray(origins2, directions2)

    dists = line_line_distance(ray1, ray2)
    assert dists.shape == (2,)
    # Both distances should be 1 (offset in y)
    assert np.allclose(dists, [1, 1])
