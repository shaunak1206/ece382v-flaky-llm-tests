import numpy as np
import pytest

from homog.homog import (
    hray,
    is_valid_rays,
    line_line_closest_points,
    line_line_distance,
)


def _stack(*rays):
    """Stack single (4, 2) rays into a batch of shape (N, 4, 2)."""
    return np.stack(rays)


# ---------------------------------------------------------------- hray

def test_hray_returns_shape_4_2():
    r = hray((1.0, 2.0, 3.0), (0.0, 0.0, 2.0))
    assert r.shape == (4, 2)


def test_hray_column0_is_homogeneous_origin():
    r = hray((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))
    np.testing.assert_allclose(r[:, 0], [1.0, 2.0, 3.0, 1.0], atol=1e-12)


def test_hray_direction_is_unit_with_zero_w():
    r = hray((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    np.testing.assert_allclose(r[:3, 1], [0.0, 0.0, 1.0], atol=1e-12)
    assert r[3, 1] == 0.0
    np.testing.assert_allclose(np.linalg.norm(r[:3, 1]), 1.0, atol=1e-12)


def test_hray_normalizes_non_unit_direction():
    # |(1, 2, 2)| = 3, so the unit direction is (1/3, 2/3, 2/3)
    r = hray((0.0, 0.0, 0.0), (1.0, 2.0, 2.0))
    np.testing.assert_allclose(r[:3, 1], [1 / 3, 2 / 3, 2 / 3], atol=1e-12)
    assert r[3, 1] == 0.0


def test_hray_accepts_homogeneous_4_vector_origin():
    r3 = hray((1.0, 2.0, 3.0), (0.0, 1.0, 0.0))
    r4 = hray((1.0, 2.0, 3.0, 1.0), (0.0, 1.0, 0.0))
    np.testing.assert_allclose(r3, r4, atol=1e-12)


def test_hray_output_is_valid_rays():
    r = hray((-1.0, 0.5, 4.0), (3.0, -4.0, 12.0))
    assert is_valid_rays(r)


def test_hray_batched_pairs_shape_and_values():
    origins = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-5.0, 0.0, 1.0]])
    dirs = np.array([[0.0, 0.0, 4.0], [2.0, 0.0, 0.0], [0.0, -3.0, 4.0]])
    r = hray(origins, dirs)
    assert r.shape == (3, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins, atol=1e-12)
    np.testing.assert_allclose(r[:, 3, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(r[0, :3, 1], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(r[1, :3, 1], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(r[2, :3, 1], [0, -0.6, 0.8], atol=1e-12)
    np.testing.assert_allclose(r[:, 3, 1], 0.0, atol=1e-12)


def test_hray_broadcasts_single_origin_over_directions():
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0], [0.0, 0.0, -2.0]])
    r = hray((7.0, 8.0, 9.0), dirs)
    assert r.shape == (4, 4, 2)
    for i in range(4):
        np.testing.assert_allclose(r[i, :, 0], [7.0, 8.0, 9.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(r[3, :3, 1], [0, 0, -1], atol=1e-12)


# ---------------------------------------------------- line_line_distance

def test_line_line_distance_skew_lines():
    # x-axis through origin, y-direction line at z=2
    r1 = hray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    r2 = hray((0.0, 0.0, 2.0), (0.0, 1.0, 0.0))
    d = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(d, [2.0], atol=1e-12)


def test_line_line_distance_intersecting_lines_is_zero():
    r1 = hray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    r2 = hray((2.0, -1.0, 0.0), (0.0, 1.0, 0.0))
    d = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(d, [0.0], atol=1e-12)


def test_line_line_distance_parallel_lines():
    r1 = hray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    r2 = hray((5.0, 3.0, 0.0), (1.0, 0.0, 0.0))
    d = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(d, [3.0], atol=1e-12)


def test_line_line_distance_antiparallel_lines():
    r1 = hray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    r2 = hray((5.0, 0.0, -4.0), (-1.0, 0.0, 0.0))
    d = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(d, [4.0], atol=1e-12)


def test_line_line_distance_is_symmetric():
    r1 = hray((1.0, 2.0, 3.0), (1.0, 1.0, 0.0))
    r2 = hray((-2.0, 0.0, 1.0), (0.0, 1.0, 2.0))
    d12 = line_line_distance(_stack(r1), _stack(r2))
    d21 = line_line_distance(_stack(r2), _stack(r1))
    np.testing.assert_allclose(d12, d21, atol=1e-12)


def test_line_line_distance_invariant_to_sliding_along_axis():
    r1 = hray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    r2 = hray((0.0, 0.0, 2.0), (0.0, 1.0, 0.0))
    r2_slid = hray((0.0, 3.0, 2.0), (0.0, 1.0, 0.0))  # moved along its axis
    d = line_line_distance(_stack(r1), _stack(r2))
    d_slid = line_line_distance(_stack(r1), _stack(r2_slid))
    np.testing.assert_allclose(d, d_slid, atol=1e-12)


def test_line_line_distance_batch_values():
    r1 = hray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    rays1 = _stack(r1, r1, r1)
    rays2 = _stack(
        hray((0.0, 0.0, 2.0), (0.0, 1.0, 0.0)),   # skew, distance 2
        hray((2.0, -1.0, 0.0), (0.0, 1.0, 0.0)),  # intersecting, distance 0
        hray((5.0, 3.0, 0.0), (1.0, 0.0, 0.0)),   # parallel, distance 3
    )
    d = line_line_distance(rays1, rays2)
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [2.0, 0.0, 3.0], atol=1e-12)


def test_line_line_distance_matches_independent_formula():
    p1 = np.array([1.0, 2.0, 3.0])
    a = np.array([1.0, 1.0, 0.0])
    p2 = np.array([-2.0, 0.0, 1.0])
    b = np.array([0.0, 1.0, 2.0])
    r1 = hray(p1, a)
    r2 = hray(p2, b)
    an = a / np.linalg.norm(a)
    bn = b / np.linalg.norm(b)
    c = np.cross(an, bn)
    expected = abs(np.dot(p2 - p1, c)) / np.linalg.norm(c)
    d = line_line_distance(_stack(r1), _stack(r2))
    assert d[0] >= 0.0
    np.testing.assert_allclose(d, [expected], atol=1e-10)


# -------------------------------------------------- line_line_closest_points

def test_closest_points_explicit_skew_case():
    # L1: (1, t, 0); L2: (s, 5, 1). Closest pair is (1,5,0) and (1,5,1).
    r1 = hray((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    r2 = hray((0.0, 5.0, 1.0), (1.0, 0.0, 0.0))
    q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    np.testing.assert_allclose(q1[0], [1.0, 5.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(q2[0], [1.0, 5.0, 1.0, 1.0], atol=1e-12)


def test_closest_points_intersecting_lines_coincide():
    r1 = hray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    r2 = hray((2.0, -1.0, 0.0), (0.0, 1.0, 0.0))
    q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    np.testing.assert_allclose(q1[0], [2.0, 0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(q2[0], [2.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_closest_points_lie_on_their_own_lines():
    r1 = hray((1.0, 2.0, 3.0), (1.0, 1.0, 0.0))
    r2 = hray((-2.0, 0.0, 1.0), (0.0, 1.0, 2.0))
    q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    # Q1 - pt1 must be parallel to ax1, Q2 - pt2 parallel to ax2
    cross1 = np.cross(q1[0, :3] - r1[:3, 0], r1[:3, 1])
    cross2 = np.cross(q2[0, :3] - r2[:3, 0], r2[:3, 1])
    np.testing.assert_allclose(cross1, 0.0, atol=1e-10)
    np.testing.assert_allclose(cross2, 0.0, atol=1e-10)


def test_closest_points_segment_is_common_perpendicular():
    r1 = hray((1.0, 2.0, 3.0), (1.0, 1.0, 0.0))
    r2 = hray((-2.0, 0.0, 1.0), (0.0, 1.0, 2.0))
    q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    seg = q2[0, :3] - q1[0, :3]
    np.testing.assert_allclose(np.dot(seg, r1[:3, 1]), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.dot(seg, r2[:3, 1]), 0.0, atol=1e-10)
    dist = line_line_distance(_stack(r1), _stack(r2))
    np.testing.assert_allclose(np.linalg.norm(seg), dist[0], atol=1e-10)


def test_closest_points_swapping_rays_swaps_outputs():
    r1 = hray((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    r2 = hray((0.0, 5.0, 1.0), (1.0, 0.0, 0.0))
    q1, q2 = line_line_closest_points(_stack(r1), _stack(r2))
    p1, p2 = line_line_closest_points(_stack(r2), _stack(r1))
    np.testing.assert_allclose(q1, p2, atol=1e-12)
    np.testing.assert_allclose(q2, p1, atol=1e-12)


def test_closest_points_batch_matches_single_pairs():
    pairs = [
        (hray((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
         hray((0.0, 5.0, 1.0), (1.0, 0.0, 0.0))),
        (hray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
         hray((2.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        (hray((1.0, 2.0, 3.0), (1.0, 1.0, 0.0)),
         hray((-2.0, 0.0, 1.0), (0.0, 1.0, 2.0))),
    ]
    rays1 = _stack(*(p[0] for p in pairs))
    rays2 = _stack(*(p[1] for p in pairs))
    q1, q2 = line_line_closest_points(rays1, rays2)
    assert q1.shape == (3, 4)
    assert q2.shape == (3, 4)
    for i, (a, b) in enumerate(pairs):
        s1, s2 = line_line_closest_points(_stack(a), _stack(b))
        np.testing.assert_allclose(q1[i], s1[0], atol=1e-12)
        np.testing.assert_allclose(q2[i], s2[0], atol=1e-12)
