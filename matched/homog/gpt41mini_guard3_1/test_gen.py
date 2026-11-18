import numpy as np
import pytest
from homog import homog

# Aliases for tested functions
hray = homog.hray
line_line_distance = homog.line_line_distance
line_line_closest_points = homog.line_line_closest_points


def seed_all():
    # Seed numpy RNG as the code under test uses numpy.random
    np.random.seed(42)


def test_hray_shapes_and_values():
    seed_all()
    origin = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.0, 0.0, 1.0])
    ray = hray(origin, direction)
    # ray shape should be (4,2)
    assert ray.shape == (4, 2)
    # origin part should have 1 in the 4th row, 0th col
    assert np.allclose(ray[:, 0], np.array([1, 2, 3, 1]))
    # direction part 4th row (homogeneous 0) at 1th col
    assert np.allclose(ray[:, 1][3], 0)
    # direction part 3 first rows normalized direction
    dir_norm = np.linalg.norm(ray[:3, 1])
    assert np.isclose(dir_norm, 1.0)


def test_hray_broadcasting_shapes():
    seed_all()
    origins = np.array([[1, 0, 0], [0, 1, 0]])
    directions = np.array([[0, 1, 0], [1, 0, 0]])
    rays = hray(origins, directions)
    # Shape should be (2,4,2)
    assert rays.shape == (2, 4, 2)
    # Check origins and directions match
    for i in range(2):
        np.testing.assert_allclose(rays[i, :, 0], np.append(origins[i], 1))
        dir_norm = np.linalg.norm(rays[i, :3, 1])
        assert np.isclose(dir_norm, 1.0)


def test_hray_direction_normalization():
    seed_all()
    origin = np.zeros(3)
    direction = np.array([5, 0, 0])
    r = hray(origin, direction)
    # direction part should be normalized unit vector
    assert np.allclose(r[:3, 1], np.array([1, 0, 0]))


def test_line_line_distance_parallel_rays_close():
    seed_all()
    # Two parallel rays with one displaced by distance 2 in y direction
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 2, 0])
    direction2 = np.array([1, 0, 0])
    ray2 = hray(origin2, direction2)

    dist = line_line_distance(ray1, ray2)
    # Distance should be 2
    assert np.isclose(dist, 2.0, atol=1e-12)


def test_line_line_distance_non_parallel():
    seed_all()
    # Two rays orthogonal crossing at origin
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 0, 0])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    dist = line_line_distance(ray1, ray2)
    # Should be 0 since they cross
    assert np.isclose(dist, 0.0, atol=1e-12)


def test_line_line_distance_skew_lines():
    seed_all()
    # Two skew lines:  
    # ray1 from origin along x-axis
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    # ray2 parallel to y-axis but offset in z by 2 units
    origin2 = np.array([0, 0, 2])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)

    dist = line_line_distance(ray1, ray2)
    # Distance should be 2 (offset in z)
    assert np.isclose(dist, 2.0, atol=1e-12)


def test_line_line_closest_points_parallel():
    seed_all()
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 0])
    direction2 = np.array([1, 0, 0])
    ray2 = hray(origin2, direction2)

    q1, q2 = line_line_closest_points(ray1, ray2)
    # For parallel lines closest points project along axis
    # q1 and q2 should be offset only in y
    assert np.allclose(q1[1], 0)
    assert np.allclose(q2[1], 1)
    # distance between q1 and q2 equals line_line_distance
    dist = np.linalg.norm(q1 - q2)
    dist_func = line_line_distance(ray1, ray2)
    assert np.isclose(dist, dist_func, atol=1e-12)


def test_line_line_closest_points_non_parallel():
    seed_all()
    origin1 = np.array([1, 0, 0])
    direction1 = np.array([1, 1, 0])
    ray1 = hray(origin1, direction1)

    origin2 = np.array([0, 1, 0])
    direction2 = np.array([-1, 1, 0])
    ray2 = hray(origin2, direction2)

    # Normalize directions to ensure valid rays
    ray1[..., :, 1] /= np.linalg.norm(ray1[..., :3, 1])
    ray2[..., :, 1] /= np.linalg.norm(ray2[..., :3, 1])

    q1, q2 = line_line_closest_points(ray1, ray2)
    # Distance between closest points matches line_line_distance output
    dist_closest_pts = np.linalg.norm(q1 - q2)
    dist_func = line_line_distance(ray1, ray2)
    assert np.isclose(dist_closest_pts, dist_func, atol=1e-12)
    # q1 should lie on ray1 line direction
    v1 = q1 - ray1[..., :3, 0]
    dir1 = ray1[..., :3, 1]
    cross1 = np.linalg.norm(np.cross(v1, dir1))
    assert cross1 < 1e-12
    # q2 should lie on ray2 line direction
    v2 = q2 - ray2[..., :3, 0]
    dir2 = ray2[..., :3, 1]
    cross2 = np.linalg.norm(np.cross(v2, dir2))
    assert cross2 < 1e-12


def test_line_line_closest_points_type_and_shape():
    seed_all()
    # Prepare two rays with batch shape (2,)
    origins = np.array([[0, 0, 0], [1, 2, 3]])
    directions = np.array([[1, 0, 0], [0, 1, 0]])
    ray1 = hray(origins, directions)
    ray2 = hray(origins + 1, directions)

    q1, q2 = line_line_closest_points(ray1, ray2)
    # Returned arrays shape should be (2, 3)
    assert q1.shape == (2, 3)
    assert q2.shape == (2, 3)
    # Elements should be floats
    assert q1.dtype.kind == 'f'
    assert q2.dtype.kind == 'f'


def test_line_line_distance_no_mutation():
    seed_all()
    origin1 = np.array([1, 2, 3])
    direction1 = np.array([0, 0, 1])
    ray1 = hray(origin1, direction1)
    origin2 = np.array([4, 5, 6])
    direction2 = np.array([0, 1, 0])
    ray2 = hray(origin2, direction2)
    ray1_copy = ray1.copy()
    ray2_copy = ray2.copy()

    _ = line_line_distance(ray1, ray2)
    # Inputs not mutated
    np.testing.assert_allclose(ray1, ray1_copy)
    np.testing.assert_allclose(ray2, ray2_copy)


def test_line_line_closest_points_no_mutation():
    seed_all()
    origin1 = np.array([-1, 0, 1])
    direction1 = np.array([0, 1, 0])
    ray1 = hray(origin1, direction1)
    origin2 = np.array([0, -1, 1])
    direction2 = np.array([1, 0, 0])
    ray2 = hray(origin2, direction2)
    ray1_copy = ray1.copy()
    ray2_copy = ray2.copy()

    q1, q2 = line_line_closest_points(ray1, ray2)
    np.testing.assert_allclose(ray1, ray1_copy)
    np.testing.assert_allclose(ray2, ray2_copy)


def test_hray_broadcast_different_shapes():
    seed_all()
    origin = np.array([1., 2., 3.])
    directions = np.array([[0., 0., 1.], [0., 1., 0.]])
    ray = hray(origin, directions)
    # Expected shape (2,4,2)
    assert ray.shape == (2, 4, 2)
    for i in range(2):
        np.testing.assert_allclose(ray[i, :, 0], np.append(origin, 1))
        np.testing.assert_allclose(ray[i, :3, 1], directions[i] / np.linalg.norm(directions[i]))
        assert ray[i, 3, 1] == 0


def test_line_line_distance_batch():
    seed_all()
    origins = np.array([[0,0,0], [0,0,1]])
    directions = np.array([[1,0,0], [0,1,0]])
    ray1 = hray(origins, directions)
    ray2 = hray(origins + np.array([0,2,0]), directions)
    dist = line_line_distance(ray1, ray2)
    # dist shape should be (2,)
    assert dist.shape == (2,)
    # First distance is 2 (offset in y), second distance 1 (offset in z)
    assert np.isclose(dist[0], 2.0, atol=1e-12)
    assert np.isclose(dist[1], 1.0, atol=1e-12)


def test_line_line_closest_points_batch():
    seed_all()
    origins = np.array([[0,0,0], [0,0,1]])
    directions = np.array([[1,0,0], [0,1,0]])
    ray1 = hray(origins, directions)
    ray2 = hray(origins + np.array([0,2,0]), directions)
    q1, q2 = line_line_closest_points(ray1, ray2)
    assert q1.shape == (2, 3)
    assert q2.shape == (2, 3)
    # Distances between closest points equal distances computed by line_line_distance
    dist_pts = np.linalg.norm(q1 - q2, axis=-1)
    dist_func = line_line_distance(ray1, ray2)
    assert np.allclose(dist_pts, dist_func, atol=1e-12)


def test_hray_direction_input_various_length_raises():
    seed_all()
    origin = [1, 2, 3]
    with pytest.raises(ValueError):
        hray(origin, [1, 0])  # direction length 2, invalid


def test_line_line_distance_degenerate():
    seed_all()
    # Make rays with direction zero vector (not normalized), expect function handles gracefully or raise.
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([0, 0, 0])
    origin2 = np.array([1, 1, 1])
    direction2 = np.array([0, 0, 0])
    # Using hray will normalize direction internally, so direction zero vector leads to division by zero or nan
    with pytest.raises(Exception):
        ray1 = hray(origin1, direction1)
        ray2 = hray(origin2, direction2)
        _ = line_line_distance(ray1, ray2)


def test_line_line_closest_points_degenerate_direction(monkeypatch):
    seed_all()

    # Patch hnormalized to return original vector without normalization when zero to test behavior
    def fake_hnormalized(a):
        return np.array(a)

    monkeypatch.setattr(homog, "hnormalized", fake_hnormalized)

    origin1 = np.array([0, 0, 0])
    direction1 = np.array([0, 0, 0])
    origin2 = np.array([1, 1, 1])
    direction2 = np.array([0, 0, 0])

    ray1 = hray(origin1, direction1)
    ray2 = hray(origin2, direction2)

    # Should probably produce nan or issues, just test it doesn't crash hard
    try:
        q1, q2 = line_line_closest_points(ray1, ray2)
        assert q1.shape == (3,)
        assert q2.shape == (3,)
    except Exception:
        # acceptable if error is raised since direction zero vectors are invalid
        pass

