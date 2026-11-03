import numpy as np
import pytest

from homog.homog import (
    hray,
    is_valid_rays,
    line_line_closest_points,
    line_line_distance,
)


def _random_pair(rs):
    """Two random rays of shape (4, 2) drawn from a RandomState."""
    r1 = hray(rs.randn(3), rs.randn(3))
    r2 = hray(rs.randn(3), rs.randn(3))
    return r1, r2


# ---------------------------------------------------------------- hray


def test_hray_builds_point_and_unit_direction_columns():
    r = hray((1, 2, 3), (0, 0, 2))
    assert r.shape == (4, 2)
    assert np.allclose(r[:, 0], [1, 2, 3, 1])
    assert np.allclose(r[:, 1], [0, 0, 1, 0])


def test_hray_normalizes_direction_to_unit_length():
    r = hray((0, 0, 0), (3, 4, 0))
    assert np.allclose(r[:, 1], [0.6, 0.8, 0, 0])
    assert np.isclose(np.linalg.norm(r[:3, 1]), 1.0)


def test_hray_output_passes_is_valid_rays():
    rs = np.random.RandomState(0)
    for _ in range(5):
        r = hray(rs.randn(3), rs.randn(3))
        assert is_valid_rays(r)


def test_hray_forces_origin_homogeneous_w_to_one():
    r = hray((1, 2, 3, 5), (0, 1, 0))
    assert np.allclose(r[:, 0], [1, 2, 3, 1])


def test_hray_batched_inputs_give_stacked_rays():
    origins = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 1, 1]], dtype=float)
    dirs = np.array(
        [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, -2], [1, 1, 1]], dtype=float)
    r = hray(origins, dirs)
    assert r.shape == (5, 4, 2)
    for i in range(5):
        assert np.allclose(r[i, :3, 0], origins[i])
        assert r[i, 3, 0] == 1
        unit = dirs[i] / np.linalg.norm(dirs[i])
        assert np.allclose(r[i, :, 1], np.append(unit, 0))


def test_hray_broadcasts_single_origin_over_many_directions():
    rs = np.random.RandomState(1)
    origin = np.array([1.0, -2.0, 0.5])
    dirs = rs.randn(7, 3)
    r = hray(origin, dirs)
    assert r.shape == (7, 4, 2)
    assert np.allclose(r[:, :3, 0], origin)
    assert np.allclose(r[:, 3, 0], 1)
    assert np.allclose(np.linalg.norm(r[:, :3, 1], axis=-1), 1)
    assert np.allclose(r[:, :3, 1],
                       dirs / np.linalg.norm(dirs, axis=-1, keepdims=True))


def test_hray_rejects_two_component_origin():
    with pytest.raises(ValueError):
        hray([1.0, 2.0], [0, 0, 1])


# ------------------------------------------------- line_line_distance


def test_distance_skew_lines_equals_perpendicular_separation():
    r1 = hray((0, 0, 0), (1, 0, 0))
    r2 = hray((3, 4, 5), (0, 1, 0))
    assert float(line_line_distance(r1, r2)) == pytest.approx(5.0)


def test_distance_intersecting_lines_is_zero():
    r1 = hray((0, 0, 0), (1, 0, 0))
    r2 = hray((2, 0, 0), (0, 1, 0))
    assert float(line_line_distance(r1, r2)) == pytest.approx(0.0, abs=1e-12)


def test_distance_parallel_lines_is_perpendicular_offset():
    r1 = hray((0, 0, 0), (0, 0, 1))
    r2 = hray((3, 4, 7), (0, 0, 1))
    assert float(line_line_distance(r1, r2)) == pytest.approx(5.0)


def test_distance_antiparallel_lines_uses_parallel_branch():
    r1 = hray((0, 0, 0), (0, 0, 1))
    r2 = hray((3, 4, 7), (0, 0, -1))
    assert float(line_line_distance(r1, r2)) == pytest.approx(5.0)


def test_distance_coincident_lines_is_zero():
    r1 = hray((0, 0, 0), (1, 1, 0))
    r2 = hray((2, 2, 0), (1, 1, 0))
    assert float(line_line_distance(r1, r2)) == pytest.approx(0.0, abs=1e-9)


def test_distance_is_symmetric_in_arguments():
    rs = np.random.RandomState(2)
    for _ in range(5):
        r1, r2 = _random_pair(rs)
        d12 = float(line_line_distance(r1, r2))
        d21 = float(line_line_distance(r2, r1))
        assert d12 == pytest.approx(d21)


def test_distance_matches_closest_point_gap():
    rs = np.random.RandomState(3)
    for _ in range(5):
        r1, r2 = _random_pair(rs)
        q1, q2 = line_line_closest_points(r1, r2)
        gap = np.linalg.norm(q2[:3] - q1[:3])
        assert gap == pytest.approx(float(line_line_distance(r1, r2)),
                                    rel=1e-9)


def test_distance_batched_matches_scalar_calls():
    rs = np.random.RandomState(4)
    pairs = [_random_pair(rs) for _ in range(4)]
    b1 = np.stack([p[0] for p in pairs])
    b2 = np.stack([p[1] for p in pairs])
    batched = line_line_distance(b1, b2)
    assert batched.shape == (4,)
    for i, (r1, r2) in enumerate(pairs):
        assert batched[i] == pytest.approx(float(line_line_distance(r1, r2)))


def test_distance_invariant_to_direction_scaling():
    r1 = hray((0, 0, 0), (1, 0, 0))
    r2 = hray((3, 4, 5), (0, 1, 0))
    r1[..., :3, 1] *= 2.0
    r2[..., :3, 1] *= 3.0
    assert float(line_line_distance(r1, r2)) == pytest.approx(5.0)


# ------------------------------------------- line_line_closest_points


def test_closest_points_known_skew_configuration():
    r1 = hray((0, 0, 0), (1, 0, 0))
    r2 = hray((3, 4, 5), (0, 1, 0))
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1, [3, 0, 0, 1])
    assert np.allclose(q2, [3, 0, 5, 1])


def test_closest_points_crossing_lines_meet_at_crossing():
    r1 = hray((0, 0, 0), (1, 0, 0))
    r2 = hray((2, -1, 0), (0, 1, 0))
    q1, q2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1, [2, 0, 0, 1])
    assert np.allclose(q2, [2, 0, 0, 1])


def test_closest_points_lie_on_their_respective_lines():
    rs = np.random.RandomState(5)
    for _ in range(5):
        r1, r2 = _random_pair(rs)
        q1, q2 = line_line_closest_points(r1, r2)
        assert np.allclose(np.cross(q1[:3] - r1[:3, 0], r1[:3, 1]), 0,
                           atol=1e-9)
        assert np.allclose(np.cross(q2[:3] - r2[:3, 0], r2[:3, 1]), 0,
                           atol=1e-9)
        assert q1[3] == pytest.approx(1.0)
        assert q2[3] == pytest.approx(1.0)


def test_closest_points_connecting_segment_perpendicular_to_both():
    rs = np.random.RandomState(6)
    for _ in range(5):
        r1, r2 = _random_pair(rs)
        q1, q2 = line_line_closest_points(r1, r2)
        seg = q2[:3] - q1[:3]
        assert abs(np.dot(seg, r1[:3, 1])) < 1e-9
        assert abs(np.dot(seg, r2[:3, 1])) < 1e-9


def test_closest_points_batched_matches_scalar_calls():
    rs = np.random.RandomState(7)
    pairs = [_random_pair(rs) for _ in range(5)]
    b1 = np.stack([p[0] for p in pairs])
    b2 = np.stack([p[1] for p in pairs])
    q1, q2 = line_line_closest_points(b1, b2)
    assert q1.shape == (5, 4)
    assert q2.shape == (5, 4)
    for i, (r1, r2) in enumerate(pairs):
        s1, s2 = line_line_closest_points(r1, r2)
        assert np.allclose(q1[i], s1)
        assert np.allclose(q2[i], s2)


def test_closest_points_verbose_prints_intermediates(capsys):
    r1 = hray((0, 0, 0), (1, 0, 0))
    r2 = hray((3, 4, 5), (0, 1, 0))
    q1, q2 = line_line_closest_points(r1, r2, verbose=1)
    out = capsys.readouterr().out
    for label in ('C21', 'M', 'm2', 'R', 't1', 't2', 'Q1', 'Q2'):
        assert label in out
    quiet1, quiet2 = line_line_closest_points(r1, r2)
    assert np.allclose(q1, quiet1)
    assert np.allclose(q2, quiet2)


def test_closest_points_parallel_lines_not_finite():
    # The docstring notes this path is unsupported (ax1 == ax2); the cross
    # product is zero, so the division yields NaN rather than a real answer.
    r1 = hray((0, 0, 0), (0, 0, 1))
    r2 = hray((3, 4, 7), (0, 0, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        q1, q2 = line_line_closest_points(r1, r2)
    assert not np.isfinite(q1).all()
    assert not np.isfinite(q2).all()
