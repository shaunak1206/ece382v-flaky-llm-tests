import numpy as np
import pytest

from homog import homog as hg

ATOL = 1e-12


def _batch(*rays):
    """Stack single (4, 2) rays into a batch of shape (len(rays), 4, 2)."""
    return np.stack(rays)


# ---------------------------------------------------------------- hray ----

def test_hray_builds_point_and_direction_columns():
    r = hg.hray([1.0, 2.0, 3.0], [0.0, 0.0, 1.0])
    assert r.shape == (4, 2)
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1], atol=ATOL)
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0], atol=ATOL)


def test_hray_normalizes_direction():
    r = hg.hray([0.0, 0.0, 0.0], [0.0, 3.0, 4.0])
    np.testing.assert_allclose(r[:, 1], [0, 0.6, 0.8, 0], atol=ATOL)


def test_hray_accepts_homogeneous_inputs():
    r = hg.hray([1.0, 2.0, 3.0, 1.0], [0.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(r[:, 0], [1, 2, 3, 1], atol=ATOL)
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0], atol=ATOL)


@pytest.mark.parametrize("direction", [
    (1.0, 2.0, 2.0),
    (0.0, -5.0, 0.0),
    (-1.0, 1.0, 1.0),
    (0.5, 0.5, 0.5),
])
def test_hray_result_is_valid_ray_with_unit_direction(direction):
    r = hg.hray([0.0, 0.0, 0.0], direction)
    assert hg.is_valid_rays(r)
    np.testing.assert_allclose(np.linalg.norm(r[:3, 1]), 1.0, atol=ATOL)


def test_hray_batch_of_origins_shares_direction():
    origins = np.arange(15, dtype=float).reshape(5, 3)
    r = hg.hray(origins, [0.0, 0.0, 2.0])
    assert r.shape == (5, 4, 2)
    np.testing.assert_allclose(r[:, :3, 0], origins, atol=ATOL)
    np.testing.assert_allclose(r[:, 3, 0], np.ones(5), atol=ATOL)
    np.testing.assert_allclose(r[:, :3, 1], np.tile([0, 0, 1.0], (5, 1)),
                               atol=ATOL)
    np.testing.assert_allclose(r[:, 3, 1], np.zeros(5), atol=ATOL)


def test_hray_broadcasts_single_origin_over_batch_of_directions():
    dirs = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [0, 0, -3.0]])
    r = hg.hray([1.0, 1.0, 1.0], dirs)
    assert r.shape == (4, 4, 2)
    for i in range(4):
        np.testing.assert_allclose(r[i, :3, 0], [1, 1, 1], atol=ATOL)
    np.testing.assert_allclose(r[:, :3, 1],
                               [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
                               atol=ATOL)


def test_hray_does_not_modify_inputs():
    origin = np.array([1.0, 2.0, 3.0])
    direction = np.array([0.0, 0.0, 2.0])
    origin_before = origin.copy()
    direction_before = direction.copy()
    hg.hray(origin, direction)
    np.testing.assert_array_equal(origin, origin_before)
    np.testing.assert_array_equal(direction, direction_before)


def test_hray_rejects_origin_of_wrong_length():
    with pytest.raises(ValueError):
        hg.hray([1.0, 2.0], [0.0, 0.0, 1.0])


def test_hray_returns_float64_for_integer_inputs():
    r = hg.hray(np.array([1, 2, 3]), np.array([0, 0, 2]))
    assert r.dtype == np.float64
    np.testing.assert_allclose(r[:, 1], [0, 0, 1, 0], atol=ATOL)


# ------------------------------------------------ line_line_distance ----

def test_distance_of_perpendicular_skew_lines():
    r1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = hg.hray([0.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    d = hg.line_line_distance(_batch(r1), _batch(r2))
    assert d.shape == (1,)
    np.testing.assert_allclose(d[0], 2.0, atol=ATOL)


def test_distance_of_intersecting_lines_is_zero():
    r1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = hg.hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    d = hg.line_line_distance(_batch(r1), _batch(r2))
    np.testing.assert_allclose(d[0], 0.0, atol=ATOL)


def test_distance_of_parallel_lines_is_perpendicular_offset():
    r1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = hg.hray([0.0, 3.0, 4.0], [1.0, 0.0, 0.0])
    d = hg.line_line_distance(_batch(r1), _batch(r2))
    np.testing.assert_allclose(d[0], 5.0, atol=ATOL)


def test_distance_of_antiparallel_lines_is_perpendicular_offset():
    r1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = hg.hray([0.0, 3.0, 4.0], [-1.0, 0.0, 0.0])
    d = hg.line_line_distance(_batch(r1), _batch(r2))
    np.testing.assert_allclose(d[0], 5.0, atol=ATOL)


def test_distance_of_general_skew_lines_matches_formula():
    # Lines (1,2,3)+t(1,1,0) and (-1,0,4)+s(0,1,2).
    # cross = (2,-2,1), |cross| = 3, (p2-p1).cross = 1, so distance = 1/3.
    r1 = hg.hray([1.0, 2.0, 3.0], [1.0, 1.0, 0.0])
    r2 = hg.hray([-1.0, 0.0, 4.0], [0.0, 1.0, 2.0])
    d = hg.line_line_distance(_batch(r1), _batch(r2))
    np.testing.assert_allclose(d[0], 1.0 / 3.0, atol=ATOL)


def test_distance_is_unchanged_by_sliding_origins_along_lines():
    r1 = hg.hray([6.0, 7.0, 3.0], [1.0, 1.0, 0.0])    # (1,2,3) + 5*(1,1,0)
    r2 = hg.hray([-1.0, -3.0, -2.0], [0.0, 1.0, 2.0])  # (-1,0,4) - 3*(0,1,2)
    d = hg.line_line_distance(_batch(r1), _batch(r2))
    np.testing.assert_allclose(d[0], 1.0 / 3.0, atol=ATOL)


def test_distance_is_symmetric_in_its_arguments():
    r1 = hg.hray([1.0, 2.0, 3.0], [1.0, 1.0, 0.0])
    r2 = hg.hray([-1.0, 0.0, 4.0], [0.0, 1.0, 2.0])
    d12 = hg.line_line_distance(_batch(r1), _batch(r2))
    d21 = hg.line_line_distance(_batch(r2), _batch(r1))
    np.testing.assert_allclose(d12, d21, atol=ATOL)


def test_distance_batch_gives_one_value_per_pair():
    # parallel -> 5, skew -> 2, intersecting -> 0
    a1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    b1 = hg.hray([0.0, 3.0, 4.0], [1.0, 0.0, 0.0])
    a2 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    b2 = hg.hray([0.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    a3 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    b3 = hg.hray([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    d = hg.line_line_distance(_batch(a1, a2, a3), _batch(b1, b2, b3))
    assert d.shape == (3,)
    np.testing.assert_allclose(d, [5.0, 2.0, 0.0], atol=ATOL)


# ------------------------------------------- line_line_closest_points ----

def test_closest_points_of_skew_lines_have_expected_values():
    r1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = hg.hray([1.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    q1, q2 = hg.line_line_closest_points(_batch(r1), _batch(r2))
    assert q1.shape == (1, 4) and q2.shape == (1, 4)
    np.testing.assert_allclose(q1[0], [1, 0, 0, 1], atol=ATOL)
    np.testing.assert_allclose(q2[0], [1, 0, 2, 1], atol=ATOL)


def test_closest_points_of_intersecting_lines_coincide():
    r1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = hg.hray([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    q1, q2 = hg.line_line_closest_points(_batch(r1), _batch(r2))
    np.testing.assert_allclose(q1[0], [1, 0, 0, 1], atol=ATOL)
    np.testing.assert_allclose(q2[0], [1, 0, 0, 1], atol=ATOL)


LINE_CASES = [
    pytest.param(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                 ([1.0, 0.0, 2.0], [0.0, 1.0, 0.0]), id="skew_axes"),
    pytest.param(([1.0, 2.0, 3.0], [1.0, 1.0, 0.0]),
                 ([-1.0, 0.0, 4.0], [0.0, 1.0, 2.0]), id="general_skew"),
    pytest.param(([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
                 ([3.0, 0.0, 5.0], [1.0, 1.0, 0.0]), id="z_axis_vs_diag"),
    pytest.param(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                 ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]), id="intersecting"),
]


@pytest.mark.parametrize("line1, line2", LINE_CASES)
def test_closest_points_lie_on_lines_and_are_perpendicular(line1, line2):
    r1 = hg.hray(*line1)
    r2 = hg.hray(*line2)
    q1, q2 = hg.line_line_closest_points(_batch(r1), _batch(r2))
    q1, q2 = q1[0], q2[0]
    pt1, ax1 = r1[:3, 0], r1[:3, 1]
    pt2, ax2 = r2[:3, 0], r2[:3, 1]

    # homogeneous w stays 1
    np.testing.assert_allclose(q1[3], 1.0, atol=ATOL)
    np.testing.assert_allclose(q2[3], 1.0, atol=ATOL)

    # q1 lies on line 1 and q2 lies on line 2
    np.testing.assert_allclose(np.cross(q1[:3] - pt1, ax1), 0, atol=1e-9)
    np.testing.assert_allclose(np.cross(q2[:3] - pt2, ax2), 0, atol=1e-9)

    # the connecting segment is perpendicular to both lines
    seg = q2[:3] - q1[:3]
    np.testing.assert_allclose(np.dot(seg, ax1), 0, atol=1e-9)
    np.testing.assert_allclose(np.dot(seg, ax2), 0, atol=1e-9)

    # the segment length equals the line-line distance
    d = hg.line_line_distance(_batch(r1), _batch(r2))[0]
    np.testing.assert_allclose(np.linalg.norm(seg), d, atol=1e-9)


def test_closest_points_swap_of_arguments_swaps_outputs():
    r1 = hg.hray([1.0, 2.0, 3.0], [1.0, 1.0, 0.0])
    r2 = hg.hray([-1.0, 0.0, 4.0], [0.0, 1.0, 2.0])
    q1, q2 = hg.line_line_closest_points(_batch(r1), _batch(r2))
    p1, p2 = hg.line_line_closest_points(_batch(r2), _batch(r1))
    np.testing.assert_allclose(q1, p2, atol=1e-9)
    np.testing.assert_allclose(q2, p1, atol=1e-9)


def test_closest_points_batch_matches_single_pairs():
    s1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    s2 = hg.hray([1.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    g1 = hg.hray([1.0, 2.0, 3.0], [1.0, 1.0, 0.0])
    g2 = hg.hray([-1.0, 0.0, 4.0], [0.0, 1.0, 2.0])

    qa, qb = hg.line_line_closest_points(_batch(s1, g1), _batch(s2, g2))
    assert qa.shape == (2, 4) and qb.shape == (2, 4)
    np.testing.assert_allclose(qa[0], [1, 0, 0, 1], atol=ATOL)
    np.testing.assert_allclose(qb[0], [1, 0, 2, 1], atol=ATOL)

    single_a, single_b = hg.line_line_closest_points(_batch(g1), _batch(g2))
    np.testing.assert_allclose(qa[1], single_a[0], atol=1e-12)
    np.testing.assert_allclose(qb[1], single_b[0], atol=1e-12)


def test_closest_points_verbose_prints_but_does_not_change_result(capsys):
    r1 = hg.hray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    r2 = hg.hray([1.0, 0.0, 2.0], [0.0, 1.0, 0.0])
    quiet_a, quiet_b = hg.line_line_closest_points(_batch(r1), _batch(r2))
    capsys.readouterr()  # discard anything printed so far

    loud_a, loud_b = hg.line_line_closest_points(_batch(r1), _batch(r2),
                                                 verbose=1)
    out = capsys.readouterr().out
    assert 'C21' in out and 'Q1' in out and 'Q2' in out
    np.testing.assert_allclose(loud_a, quiet_a, atol=ATOL)
    np.testing.assert_allclose(loud_b, quiet_b, atol=ATOL)
