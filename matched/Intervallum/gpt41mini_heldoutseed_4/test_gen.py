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


class TestBox:
    def test_creation_and_str_repr(self):
        b = Box(Interval(1, 2), Interval(3, 4))
        assert isinstance(b, Box)
        s = str(b)
        r = repr(b)
        assert s == r
        assert "[1; 2]" in s and "[3; 4]" in s

    def test_equality_and_inequality(self):
        b1 = Box(Interval(0, 1), Interval(2, 3))
        b2 = Box(Interval(0, 1), Interval(2, 3))
        b3 = Box(Interval(0, 1), Interval(2, 4))
        assert b1 == b2
        assert b1 != b3
        with pytest.raises(NotImplementedError):
            _ = (b1 == [1, 2])

    def test_copy_and_properties(self):
        b = Box(Interval(1, 3), Interval(2, 4))
        c = b.__copy__()
        assert c == b
        assert np.allclose(b.lb, [1, 2])
        assert np.allclose(b.ub, [3, 4])
        assert np.allclose(b.middle, [2, 3])
        w_list, w_max = b.width
        assert w_max == max(b[0].width, b[1].width)
        # If max width >0, copy is returned
        r = b._try_to_reduce()
        assert isinstance(r, Box)
        # Test zero width box returns middle
        single_pt_box = Box(Interval(1, 1), Interval(2, 2))
        assert np.allclose(single_pt_box._try_to_reduce(), single_pt_box.middle)


class TestInterval:
    def test_init_invalid_bounds(self):
        with pytest.raises(IntervalExceptions.WrongBoundsException):
            Interval(5, 4)
        # fix=True flips bounds
        i = Interval(5, 4, fix=True)
        assert i.lb == 4 and i.ub == 5

    def test_str_repr_and_properties(self):
        i = Interval(1, 3)
        assert str(i) == "[1; 3]"
        assert i.lb == 1
        assert i.ub == 3
        assert i.middle == 2
        assert i.width == 2

    def test_lb_ub_setters_and_exceptions(self):
        i = Interval(1, 4)
        i.lb = 2
        assert i.lb == 2
        with pytest.raises(IntervalExceptions.WrongBoundsException):
            i.lb = 5
        i.ub = 6
        assert i.ub == 6
        with pytest.raises(IntervalExceptions.WrongBoundsException):
            i.ub = 1

    def test_try_to_reduce(self):
        i = Interval(1, 1 + IntervalConstants._reduction_width / 2)
        r = i._try_to_reduce()
        assert isinstance(r, float)
        # wider interval returns Interval
        i2 = Interval(1, 2)
        r2 = i2._try_to_reduce()
        assert isinstance(r2, Interval)

    def test_inner_subtraction_and_shift_operator(self):
        i1 = Interval(2, 5)
        i2 = Interval(1, 3)
        diff = Interval.inner_subtraction(i1, i2)
        assert diff.lb == 1 and diff.ub == 4
        assert (i1 << i2) == diff
        # Also test scalar case
        diff2 = Interval.inner_subtraction(i1, 2.0)
        assert diff2.lb == 0 and diff2.ub == 3

    def test_eq_ne_comparisons(self):
        i1 = Interval(1, 1.00000001)
        i2 = Interval(1, 1.00000002)
        assert i1 == i2
        i3 = Interval(1, 1.0001)
        assert i1 != i3
        assert i1 == 1.0
        assert i1 != 1.0001
        with pytest.raises(NotImplementedError):
            _ = (i1 == "string")

    def test_ordering_operations(self):
        i1 = Interval(1, 2)
        i2 = Interval(2, 3)
        assert i1 < i2
        assert i1 <= i2
        assert i2 > i1
        assert i2 >= i1
        assert i1 < 3.0
        assert i1 <= 2.0
        assert i1 > 0.0
        assert i1 >= 1.0

    def test_add_neg_sub_mul(self):
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        assert i1 + i2 == Interval(4, 6)
        assert i1 + 3 == Interval(4, 5)
        assert 3 + i1 == Interval(4, 5)
        assert -i1 == Interval(-2, -1)
        assert i1 - i2 == Interval(-3, -1)
        assert i1 - 1 == Interval(0, 1)
        assert 5 - i1 == Interval(3, 4)
        assert i1 * i2 == Interval(3, 8)
        assert i1 * 2 == Interval(2, 4)
        assert 2 * i1 == Interval(2, 4)

    def test_invert_div(self):
        i1 = Interval(1, 2)
        inv = ~i1
        assert inv.lb == 1 / 2
        assert inv.ub == 1 / 1
        i2 = Interval(-1, 1)
        inv2 = ~i2
        assert inv2.lb == -math.inf and inv2.ub == math.inf
        i3 = Interval(0, 1)
        inv3 = ~i3
        assert inv3.lb == 1 / 1 and inv3.ub == math.inf
        i4 = Interval(-1, 0)
        inv4 = ~i4
        assert inv4.lb == -math.inf and inv4.ub == 1 / -1

        # Division tests
        i5 = Interval(2, 3)
        i6 = Interval(1, 2)
        div = i5 / i6
        assert isinstance(div, Interval)
        div2 = i5 / 2
        assert isinstance(div2, Interval)
        div3 = 6 / i6
        assert isinstance(div3, Interval)

    def test_pow_even_odd(self):
        i1 = Interval(-1, 2)
        sq = i1 ** 2
        assert isinstance(sq, Interval)
        cb = i1 ** 3
        assert isinstance(cb, Interval)
        # Reduced for power if width small
        i2 = Interval(2, 2 + IntervalConstants._reduction_width / 2)
        sq2 = i2 ** 2
        assert isinstance(sq2, float) or isinstance(sq2, Interval)


class TestRS_SubinclusionFunction:
    def test_call_returns_interval(self):
        def f(x):
            return np.sum(x)

        box = Box(Interval(1, 2), Interval(3, 4))
        rsf = RS_SubinclusionFunction(f, 100)
        val = rsf(box)
        assert isinstance(val, Interval)
        assert val.lb <= val.ub

    def test_call_with_scalar_output(self):
        def f_scalar(x):
            return float(np.prod(x))

        box = Box(Interval(0, 1), Interval(2, 3))
        rsf = RS_SubinclusionFunction(f_scalar, 50)
        val = rsf(box)
        assert isinstance(val, Interval)


class TestIntervalFunctions:
    def test_sin_cos_with_float_and_interval(self):
        f, pts = sin(0.5)
        assert callable(f) and isinstance(pts, list)
        i = Interval(0, 2 * math.pi)
        s = sin(i)
        c = cos(i)
        assert isinstance(s, Interval)
        assert isinstance(c, Interval)
        # full period returns [-1,1]
        assert s.lb >= -1 and s.ub <= 1
        assert c.lb >= -1 and c.ub <= 1

    def test_abs_exp_with_float_and_interval(self):
        a1 = interval_abs(-0.5)
        a2 = interval_abs(Interval(-1, 1))
        assert isinstance(a1, float)
        assert isinstance(a2, Interval)
        e1 = exp(1.0)
        e2 = exp(Interval(0, 1))
        assert isinstance(e1, float)
        assert isinstance(e2, Interval)

    def test_sqrt_and_log_valid_and_invalid(self):
        sq1 = sqrt(4.0)
        assert sq1 == 2.0
        sq2 = sqrt(Interval(4, 9))
        assert isinstance(sq2, Interval)
        sq3 = sqrt(Interval(-1, 1))
        # should include 0
        assert 0.0 in [round(x, 7) for x in [sq3.lb, sq3.ub]] or sq3.lb <= 0 <= sq3.ub

        with pytest.raises(IntervalExceptions.OperationIsNotDefined):
            sqrt(Interval(-2, -1))

        lg1 = log(1.0)
        assert pytest.approx(lg1, abs=1e-8) == 0.0
        lg2 = log(Interval(1, 10))
        assert isinstance(lg2, Interval)
        lg3 = log(Interval(-1, 1))
        assert lg3.lb == -math.inf
        with pytest.raises(IntervalExceptions.OperationIsNotDefined):
            log(Interval(-5, 0))

    def test_split_and_bisect(self):
        i = Interval(0, 10)
        parts = split(i, [1, 1, 2])
        assert len(parts) == 3
        total_width = sum(part.width if isinstance(part, Interval) else 0 for part in parts)
        assert abs(total_width - i.width) < 1e-12

        # bisect returns two intervals of half width
        bparts = bisect(i)
        assert len(bparts) == 2
        half_width = i.width / 2
        for part in bparts:
            assert abs(part.width - half_width) < 1e-12

    def test_constrain(self):
        i = Interval(-5, 15)
        constrained = constrain(i, 0, 10)
        assert constrained.lb == 0
        assert constrained.ub == 10

        # constrain scalar
        cs = constrain(12.0, 0, 10)
        assert cs == 10

    def test_shrink(self):
        i = Interval(0, 10)
        shrunk = shrink(i, 0.5)
        mid = i.middle
        expected_width = 0.5 * i.width * 0.5
        assert shrunk.lb == mid - expected_width
        assert shrunk.ub == mid + expected_width

        # shrink scalar returns same scalar
        s = shrink(5.0, 0.5)
        assert s == 5.0

    def test_hausdorff_distance(self):
        i1 = Interval(0, 5)
        i2 = Interval(3, 10)
        dist = hausdorff_distance(i1, i2)
        # Distance should be positive
        assert dist >= 0
        i3 = Interval(0, 2)
        dist2 = hausdorff_distance(i2, i3)
        assert dist2 >= 0
        # Distance symmetric
        assert hausdorff_distance(i1, i2) == hausdorff_distance(i2, i1)

        # scalar cases
        assert hausdorff_distance(1.0, Interval(0.5, 1.5)) >= 0
        assert hausdorff_distance(2.0, 2.0) == 0


@pytest.mark.parametrize("power", [2, 4, 6])
def test_interval_pow_even(power):
    i = Interval(-2, 3)
    p = i ** power
    assert isinstance(p, Interval)
    # Should include 0 if lb * ub < 0
    if i.lb * i.ub < 0:
        assert p.lb == 0 or p.lb < 0  # for even powers lower bound can be zero or minimum


@pytest.mark.parametrize("power", [1, 3, 5])
def test_interval_pow_odd(power):
    i = Interval(-2, 3)
    p = i ** power
    assert isinstance(p, Interval)
    # For odd powers min is lb^power and max is ub^power
    expected_lb = min(i.lb ** power, i.ub ** power)
    expected_ub = max(i.lb ** power, i.ub ** power)
    assert math.isclose(p.lb, expected_lb, rel_tol=1e-12)
    assert math.isclose(p.ub, expected_ub, rel_tol=1e-12)
