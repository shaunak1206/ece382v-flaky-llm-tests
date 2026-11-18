import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import (
    Interval,
    IntervalConstants,
    IntervalExceptions,
)
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin,
    cos,
    abs as interval_abs,
    exp,
    sqrt,
    log,
    split,
    bisect,
    constrain,
    shrink,
    hausdorff_distance,
)


def test_interval_init_and_str_repr():
    i = Interval(1.0, 2.0)
    assert str(i) == "[1.0; 2.0]"
    assert repr(i) == "[1.0; 2.0]"
    # check lb and ub properties
    assert i.lb == 1.0
    assert i.ub == 2.0


def test_interval_wrong_bounds_exception():
    with pytest.raises(IntervalExceptions.WrongBoundsException):
        Interval(2.0, 1.0)

    # Should fix bounds when fix=True
    i = Interval(2.0, 1.0, fix=True)
    assert i.lb == 1.0
    assert i.ub == 2.0


def test_interval_copy_and_try_to_reduce():
    i = Interval(1.0, 1.0 + IntervalConstants._reduction_width / 2)
    reduced = i._try_to_reduce()
    assert isinstance(reduced, float)
    assert math.isclose(reduced, 0.5 * (i.lb + i.ub))
    # Larger than reduction_width returns Interval copy
    i2 = Interval(1.0, 1.0 + 1e-3)
    no_reduce = i2._try_to_reduce()
    assert isinstance(no_reduce, Interval)
    assert no_reduce == i2


def test_interval_eq_and_ne_with_float_and_interval():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(1.0 + 1e-8, 2.0 + 1e-8)
    f = 1.5
    assert (i1 == i2) is True  # within admissible error
    assert (i1 != i2) is False
    assert (i1 == f) is False
    with pytest.raises(NotImplementedError):
        i1 == "string"
    # Using __ne__ returns opposite
    assert (i1 != f) is True


def test_interval_arithmetic_add_neg_sub_mul_div():
    i1 = Interval(1.0, 2.0)
    i2 = Interval(0.5, 0.6)

    # Addition
    res_add = i1 + i2
    assert isinstance(res_add, (Interval, float))
    assert math.isclose(res_add.lb, 1.5)
    assert math.isclose(res_add.ub, 2.6)

    res_add_float = i1 + 1.0
    assert isinstance(res_add_float, (Interval, float))
    assert res_add_float.lb == 2.0 and res_add_float.ub == 3.0

    # Negation
    neg_i1 = -i1
    assert neg_i1.lb == -2.0 and neg_i1.ub == -1.0

    # Subtraction
    res_sub = i1 - i2
    assert res_sub.lb == pytest.approx(0.4)
    assert res_sub.ub == pytest.approx(1.5)

    # Reverse subtraction
    res_rsub = 5.0 - i1
    assert isinstance(res_rsub, (Interval, float))
    assert res_rsub.lb == pytest.approx(3.0)
    assert res_rsub.ub == pytest.approx(4.0)

    # Multiplication with interval
    res_mul = i1 * i2
    assert res_mul.lb == pytest.approx(0.5)
    assert res_mul.ub == pytest.approx(1.2)

    # Multiplication with float
    res_mul_float = i1 * 2
    assert res_mul_float.lb == 2.0
    assert res_mul_float.ub == 4.0

    # Division (by interval)
    res_div = i1 / i2
    assert isinstance(res_div, (Interval, float))

    # Division (by float)
    res_div_float = i1 / 2
    assert res_div_float.lb == 0.5
    assert res_div_float.ub == 1.0


def test_interval_inversion():
    i_pos = Interval(1.0, 2.0)
    inv_pos = ~i_pos
    assert math.isclose(inv_pos.lb, 1/2.0)
    assert math.isclose(inv_pos.ub, 1/1.0)

    i_neg = Interval(-2.0, -1.0)
    inv_neg = ~i_neg
    assert math.isclose(inv_neg.lb, 1/(-1.0))
    assert math.isclose(inv_neg.ub, 1/(-2.0))

    i_cross = Interval(-1.0, 1.0)
    inv_cross = ~i_cross
    assert inv_cross.lb == -math.inf and inv_cross.ub == math.inf

    i_zero_lb = Interval(0.0, 2.0)
    inv_zero_lb = ~i_zero_lb
    assert inv_zero_lb.lb == pytest.approx(1/2.0)
    assert inv_zero_lb.ub == math.inf

    i_zero_ub = Interval(-2.0, 0.0)
    inv_zero_ub = ~i_zero_ub
    assert inv_zero_ub.lb == -math.inf
    assert math.isclose(inv_zero_ub.ub, 1/(-2.0))


def test_box_creation_and_properties():
    i1 = Interval(0, 1)
    i2 = Interval(2, 3)
    box = Box(i1, i2)
    assert isinstance(box, Box)
    assert box.lb[0] == i1.lb and box.lb[1] == i2.lb
    assert box.ub[0] == i1.ub and box.ub[1] == i2.ub
    assert np.allclose(box.middle, [i1.middle, i2.middle])

    # test __str__ and __repr__
    s = str(box)
    r = repr(box)
    assert isinstance(s, str)
    assert isinstance(r, str)
    assert s == r

    # test __eq__ and __ne__
    box2 = Box(i1, i2)
    assert box == box2
    assert not (box != box2)
    box3 = Box(i2, i1)
    assert box != box3

    with pytest.raises(NotImplementedError):
        box == "string"

    # test copy
    box_copy = box.__copy__()
    assert isinstance(box_copy, Box)
    assert box_copy == box
    assert box_copy is not box


def test_box_width_and_try_to_reduce():
    i1 = Interval(1.0, 1.0)  # width 0
    i2 = Interval(2.0, 3.0)  # width 1
    box = Box(i1, i2)
    idxs, w = box.width
    assert w == 1.0
    assert idxs == [1]

    # Width zero box
    box_zero = Box(i1, Interval(5.5, 5.5))
    idxs_zero, w_zero = box_zero.width
    assert w_zero == 0.0
    val = box_zero._try_to_reduce()
    # should return array of middles (float numpy array)
    assert isinstance(val, np.ndarray)
    assert np.allclose(val, [i1.middle, 5.5])

    # Non zero width box returns copy
    val2 = box._try_to_reduce()
    assert isinstance(val2, Box)
    assert val2 == box


def test_rs_subinclusion_function_behavior(monkeypatch):
    # monkeypatch np.random.uniform to return fixed points for determinism
    fixed_points = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    calls = []

    def fake_uniform(low, high):
        calls.append((low, high))
        return fixed_points[len(calls) - 1]

    monkeypatch.setattr("numpy.random.uniform", fake_uniform)

    def f(x):
        return sum(x)

    rsf = RS_SubinclusionFunction(f, number_of_samples=2)

    b = Box(Interval(0, 1), Interval(0, 1))

    res = rsf(b)
    assert isinstance(res, Interval)
    expected_min = min(fixed_points[0].sum(), fixed_points[1].sum())
    expected_max = max(fixed_points[0].sum(), fixed_points[1].sum())
    assert res.lb == expected_min
    assert res.ub == expected_max
    # check uniform was called with correct bounds
    for low, high in calls:
        assert np.allclose(low, b.lb)
        assert np.allclose(high, b.ub)


@pytest.mark.parametrize(
    "func, val, expected_lb_range, expected_ub_range",
    [
        (sin, Interval(0, math.pi / 2), 0.0, 1.0),
        (cos, Interval(0, math.pi / 2), math.cos(math.pi / 2), 1.0),
        (interval_abs, Interval(-1.0, 1.0), 0.0, 1.0),
        (exp, Interval(0, 1), math.exp(0), math.exp(1)),
    ],
)
def test_monotonic_functions_basic(func, val, expected_lb_range, expected_ub_range):
    res = func(val)
    # Because of reduction, result could be float if width close to 0
    if isinstance(res, Interval):
        assert expected_lb_range <= res.lb <= res.ub <= expected_ub_range * 1.1
    else:
        # Reduced to float
        assert isinstance(res, float)


def test_sqrt_and_log_exceptions():
    i_neg = Interval(-2.0, -1.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        _ = sqrt(i_neg)

    i_log_neg = Interval(-1.0, 0.0)
    with pytest.raises(IntervalExceptions.OperationIsNotDefined):
        _ = log(i_neg)

    # sqrt with crossing zero
    i_cross = Interval(-1.0, 2.0)
    res = sqrt(i_cross)
    assert isinstance(res, (Interval, float))
    assert res.lb == 0.0
    assert res.ub == 2.0 ** 0.5 or isinstance(res, float)


def test_split_and_bisect_functions():
    i = Interval(0, 10)
    parts = split(i, [1, 1, 2])
    assert len(parts) == 3
    total_length = sum((p.ub - p.lb if isinstance(p, Interval) else 0) for p in parts)
    assert abs(total_length - i.width) < 1e-6
    # bisect is equal split with [1,1]
    bisected = bisect(i)
    assert len(bisected) == 2
    assert abs(bisected[0].width - bisected[1].width) < 1e-10

    # splitting a float returns the float repeated
    flt = 5.0
    parts_flt = split(flt, [1, 2, 3])
    assert parts_flt == [5.0, 5.0, 5.0]


def test_constrain_and_shrink():
    i = Interval(1.0, 5.0)
    constrained = constrain(i, 2.0, 4.0)
    assert constrained.lb >= 2.0 and constrained.ub <= 4.0

    constrained_float = constrain(10.0, 0, 5)
    assert constrained_float == 5.0

    shrunk = shrink(i, 0.5)
    mid = i.middle
    width = i.width
    assert math.isclose(shrunk.lb, mid - 0.25 * width)
    assert math.isclose(shrunk.ub, mid + 0.25 * width)

    shrunk_float = shrink(5.0, 0.1)
    assert shrunk_float == 5.0


@pytest.mark.parametrize(
    "i1, i2, expected",
    [
        (Interval(0, 2), Interval(1, 3), 1.0),
        (Interval(1, 3), Interval(0, 2), 1.0),
        (Interval(0, 5), Interval(1, 4), 0.0),
        (1.0, Interval(0.5, 1.5), 0.0),
        (Interval(0.5, 1.5), 1.0, 0.0),
        (2.0, 1.0, 1.0),
    ],
)
def test_hausdorff_distance(i1, i2, expected):
    d = hausdorff_distance(i1, i2)
    assert isinstance(d, float)
    assert math.isclose(d, expected, rel_tol=1e-7)


def test_interval_pow_even_and_odd():
    i = Interval(-2.0, 1.0)

    # Even power (2), includes zero in interval
    res_even = i ** 2
    assert isinstance(res_even, (Interval, float))
    if isinstance(res_even, Interval):
        assert res_even.lb == 0.0
        assert res_even.ub >= max(i.lb ** 2, i.ub ** 2)

    # Odd power (3)
    res_odd = i ** 3
    assert isinstance(res_odd, (Interval, float))
    if isinstance(res_odd, Interval):
        assert res_odd.lb == pytest.approx(i.lb ** 3)
        assert res_odd.ub == pytest.approx(i.ub ** 3)
