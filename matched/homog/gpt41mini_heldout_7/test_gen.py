import numpy as np
import pytest
from homog import homog


# Helper to create a valid hray for testing
def make_ray(origin=None, direction=None):
    if origin is None:
        origin = np.array([1, 2, 3])
    if direction is None:
        direction = np.array([0, 0, 1])
    return homog.hray(origin, direction)


def test_hray_basic_shape_and_values():
    origin = np.array([1, 2, 3])
    direction = np.array([0, 0, 1])
    r = homog.hray(origin, direction)
    assert r.shape == (4, 2)
    # Origin column
    np.testing.assert_allclose(r[:3, 0], origin)
    assert r[3, 0] == 1
    # Direction column normalized
    dir_norm = np.linalg.norm(direction)
    np.testing.assert_allclose(r[:3, 1], direction / dir_norm)
    assert r[3, 1] == 0


def test_hray_broadcasting():
    origins = np.array([[1, 0, 0], [0, 1, 0]])
    directions = np.array([0, 0, 1])
    r = homog.hray(origins, directions)
    assert r.shape == (2, 4, 2)
    for i in range(2):
        np.testing.assert_allclose(r[i, :3, 0], origins[i])
        np.testing.assert_allclose(r[i, :3, 1], np.array([0, 0, 1]))
        assert r[i, 3, 0] == 1
        assert r[i, 3, 1] == 0


def test_line_line_distance_parallel_lines():
    # Create two parallel rays offset by 1 unit in y direction
    origin1 = np.array([0, 0, 0])
    direction1 = np.array([1, 0, 0])
    ray1 = homog.hray(origin1, direction1)

    origin2 = np.array([0, 1, 0])  # offset by 1 in y-axis
    direction2 = np.array([1, 0, 0])
    ray2 = homog.hray(origin2, direction2)

    dist = homog.line_line_distance(ray1, ray2)
    # Distance should be 1 as lines are parallel and 1 unit apart
    assert np.isclose(dist, 1)


def test_line_line_distance_skew_lines():
    # Two skew lines at known distance
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([0, 1, 1], [0, 1, 0])

    dist = homog.line_line_distance(ray1, ray2)
    # The distance between these skew lines is sqrt(1^2 + 1^2) = sqrt(2)
    assert np.isclose(dist, np.sqrt(2))


def test_line_line_distance_identical_lines():
    ray = homog.hray([0, 0, 0], [1, 0, 0])
    dist = homog.line_line_distance(ray, ray)
    assert np.isclose(dist, 0)


@pytest.mark.parametrize("pt1, ax1, pt2, ax2", [
    # simple crossing lines
    ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]),
    # skew lines
    ([1, 0, 0], [0, 1, 0], [0, 2, 1], [1, 0, 0]),
])
def test_line_line_closest_points_pa_basic(pt1, ax1, pt2, ax2):
    pt1 = homog.hpoint(pt1)
    pt2 = homog.hpoint(pt2)
    ax1 = homog.hnormalized(ax1)
    ax2 = homog.hnormalized(ax2)
    q1, q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2)
    # q1 and q2 must be on the respective lines: q - pt along axis direction
    vec1 = q1 - pt1
    vec2 = q2 - pt2
    # check q1 to pt1 perpendicular to ax1
    dot1 = np.sum(vec1[..., :3] * ax1[..., :3], axis=-1)
    assert np.allclose(dot1, 0, atol=1e-7)
    dot2 = np.sum(vec2[..., :3] * ax2[..., :3], axis=-1)
    assert np.allclose(dot2, 0, atol=1e-7)
    # check distance between q1 and q2 = line-line distance
    dist = np.linalg.norm(q1[..., :3] - q2[..., :3], axis=-1)
    dist_ref = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    assert np.allclose(dist, dist_ref, atol=1e-7)


def test_line_line_closest_points_basic():
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([0, 1, 0], [0, 1, 0])
    q1, q2 = homog.line_line_closest_points(ray1, ray2)
    # Points closest between two lines should be on respective lines
    for qi, ray, label in zip((q1, q2), (ray1, ray2), ('q1', 'q2')):
        vec = qi - ray[..., :, 0]
        dot = np.sum(vec[..., :3] * ray[..., :, 1], axis=-1)
        assert np.allclose(dot, 0, atol=1e-7), f"{label} not perpendicular"


def test_line_line_closest_points_parallel_lines_raise():
    # Expect exception because lines are parallel and method doesn't handle this gracefully
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    ray2 = homog.hray([1, 0, 0], [1, 0, 0])
    with pytest.raises(ZeroDivisionError):
        # The code will attempt division by zero in cross product magnitude
        homog.line_line_closest_points(ray1, ray2)


def test_hray_direction_normalization():
    direction = np.array([5, 0, 0])
    r = homog.hray([0, 0, 0], direction)
    norm_dir = direction / np.linalg.norm(direction)
    np.testing.assert_allclose(r[:3, 1], norm_dir)


def test_line_line_distance_pa_handles_identical_axes():
    pt1 = homog.hpoint([0, 0, 0])
    pt2 = homog.hpoint([0, 0, 1])
    ax1 = homog.hnormalized([1, 0, 0])
    ax2 = homog.hnormalized([1, 0, 0])
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    # Since axes are identical, distance = norm of projection perpendicular of pt2-pt1 onto ax1
    dp = homog.hnorm(homog.proj_perp(ax1, pt2 - pt1))
    assert np.isclose(dist, dp)


def test_line_line_distance_pa_handles_orthogonal_axes():
    pt1 = homog.hpoint([0, 0, 0])
    pt2 = homog.hpoint([1, 0, 0])
    ax1 = homog.hnormalized([0, 1, 0])
    ax2 = homog.hnormalized([1, 0, 0])
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    # Manually compute expected distance between skew lines
    cross = homog.hcross(ax1, ax2)
    n = abs(homog.hdot(pt2 - pt1, cross))
    d = homog.hnorm(cross)
    expected = n / d
    assert np.isclose(dist, expected)


def test_hray_output_dtype_and_broadcast():
    origins = np.array([[1., 2., 3.], [4., 5., 6.]])
    directions = np.array([[0., 0., 1.], [1., 0., 0.]])
    rays = homog.hray(origins, directions)
    assert rays.shape == (2, 4, 2)
    assert rays.dtype == origins.dtype


def test_line_line_closest_points_pa_verbose_output(capsys):
    pt1 = homog.hpoint([0, 0, 0])
    pt2 = homog.hpoint([0, 1, 0])
    ax1 = homog.hnormalized([1, 0, 0])
    ax2 = homog.hnormalized([0, 1, 0])
    q1, q2 = homog.line_line_closest_points_pa(pt1, ax1, pt2, ax2, verbose=1)
    captured = capsys.readouterr()
    assert "C21" in captured.out
    assert "M" in captured.out


def test_line_line_distance_with_batched_inputs():
    origins1 = np.array([[0, 0, 0], [1, 1, 1]])
    directions1 = np.array([[1, 0, 0], [0, 1, 0]])
    origins2 = np.array([[0, 1, 0], [2, 1, 1]])
    directions2 = np.array([[0, 1, 0], [1, 0, 0]])

    ray1 = homog.hray(origins1, directions1)
    ray2 = homog.hray(origins2, directions2)
    dist = homog.line_line_distance(ray1, ray2)
    assert dist.shape == (2,)
    assert np.all(dist >= 0)


def test_line_line_closest_points_with_batched_inputs():
    origins1 = np.array([[0, 0, 0], [1, 1, 1]])
    directions1 = np.array([[1, 0, 0], [0, 1, 0]])
    origins2 = np.array([[0, 1, 0], [2, 1, 1]])
    directions2 = np.array([[0, 1, 0], [1, 0, 0]])

    ray1 = homog.hray(origins1, directions1)
    ray2 = homog.hray(origins2, directions2)
    q1, q2 = homog.line_line_closest_points(ray1, ray2)
    assert q1.shape == (2, 4)
    assert q2.shape == (2, 4)
    # Check q1 and q2 points are valid homogeneous points
    assert np.allclose(q1[:, 3], 1)
    assert np.allclose(q2[:, 3], 1)


def test_hray_invalid_direction_normalization():
    # Zero vector direction triggers divide by zero in normalization
    origin = np.array([0, 0, 0])
    direction = np.array([0, 0, 0])
    # Should produce NaNs or infs but handle gracefully (normalization returns nan or inf)
    r = homog.hray(origin, direction)
    # The direction part should be nan or raise warning; we test that origin column is correct
    np.testing.assert_allclose(r[:3, 0], origin)
    assert r[3, 0] == 1
    # Direction column 3rd component might be nan
    assert r[3, 1] == 0


def test_line_line_distance_handles_zero_cross_product():
    pt1 = homog.hpoint([0, 0, 0])
    pt2 = homog.hpoint([1, 0, 0])
    ax1 = homog.hnormalized([1, 0, 0])
    ax2 = homog.hnormalized([1, 0, 0])  # same axis
    dist = homog.line_line_distance_pa(pt1, ax1, pt2, ax2)
    proj = homog.hnorm(homog.proj_perp(ax1, pt2 - pt1))
    assert np.isclose(dist, proj)


def test_hray_output_type_is_numpy_array():
    origin = [1, 2, 3]
    direction = [0, 1, 0]
    r = homog.hray(origin, direction)
    assert isinstance(r, np.ndarray)
    assert r.dtype == np.float64


def test_hray_direction_is_normalized_and_length_one():
    direction = [3, 4, 0]
    r = homog.hray([0, 0, 0], direction)
    dir_vec = r[:3, 1]
    length = np.linalg.norm(dir_vec)
    assert np.isclose(length, 1)


def test_line_line_distance_special_case_near_parallel():
    # Lines almost parallel but not exactly
    ray1 = homog.hray([0, 0, 0], [1, 0, 0])
    # Slightly off-axis direction
    ray2 = homog.hray([0, 1, 0], [1, 0.00001, 0])
    dist = homog.line_line_distance(ray1, ray2)
    assert dist > 0
    assert dist < 1  # Should be close to 1 but less because of tilt


def test_line_line_distance_vectorized_scalar_angle_axes():
    # Create arrays with scalar angles with vector axes
    origins1 = np.array([[0, 0, 0]])
    directions1 = np.array([[1, 0, 0]])
    origins2 = np.array([[0, 1, 0]])
    directions2 = np.array([[0, 1, 0]])
    ray1 = homog.hray(origins1, directions1)
    ray2 = homog.hray(origins2, directions2)
    dist = homog.line_line_distance(ray1, ray2)
    assert np.isscalar(dist) or dist.shape == (1,)
    assert dist >= 0

