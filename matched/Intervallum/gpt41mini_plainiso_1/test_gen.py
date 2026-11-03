import math
import numpy as np
import pytest

from intervallum.box import Box
from intervallum.interval import Interval, IntervalExceptions, IntervalConstants
from intervallum.inclusion_functions import RS_SubinclusionFunction
from intervallum.interval_functions import (
    sin, cos, abs as interval_abs, exp, sqrt, log,
    split, bisect, constrain, shrink,
    _hausdorff_distance_part, hausdorff_distance,
)


class TestBox:
    def test_box_creation_and_str_repr(self):
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        b = Box(i1, i2)
        assert isinstance(b, Box)
        assert str(b) == "[1; 2] x [3; 4]"
        assert repr(b) == "[1; 2] x [3; 4]"

    def test_box_equality_and_inequality(self):
        i1 = Interval(1, 2)
        b1 = Box(i1, i1)
        b2 = Box(i1, i1)
        b3 = Box(i1, Interval(2, 3))

        assert b1 == b2
        assert not (b1 != b2)
        assert b1 != b3
        assert not (b1 == b3)
        with pytest.raises(NotImplementedError):
            b1 == 123  # unsupported type

    def test_box_copy(self):
        i1 = Interval(1, 2)
        b = Box(i1)
        b_copy = b.__copy__()
        assert b == b_copy
        assert b is not b_copy
        assert all(isinstance(x, Interval) for x in b_copy)

    def test_box_properties_lb_ub_middle(self):
        i1 = Interval(1, 3)
        i2 = Interval(2, 5)
        b = Box(i1, 7)
        lb = b.lb
        ub = b.ub
        middle = b.middle
        # lb and ub of Box elements
        assert np.allclose(lb, [1, 7])
        assert np.allclose(ub, [3, 7])
        # middle for Interval and for scalar (unchanged)
        assert np.allclose(middle, [2, 7])

    def test_box_width(self):
        b = Box(Interval(1, 4), Interval(2, 3), 5)
        idxs, w = b.width
        # widths: 3,1,0 => max is 3 with index 0
        assert w == 3
        assert idxs == [0]

        # multiple with same max width
        b2 = Box(Interval(2, 5), Interval(1, 4), Interval(10, 13))
        idxs2, w2 = b2.width
        # widths: 3, 3, 3 all tied
        assert w2 == 3
        assert set(idxs2) == {0, 1, 2}

    def test_box_try_to_reduce_returns_middle_if_width_zero(self):
        b = Box(Interval(1, 1), Interval(2, 2))
        reduced = b._try_to_reduce()
        # Both intervals have width 0 so returns array of middles (points), not Box
        assert isinstance(reduced, np.ndarray)
        assert np.allclose(reduced, [1, 2])

    def test_box_try_to_reduce_returns_copy_if_width_nonzero(self):
        b = Box(Interval(1, 3), Interval(4, 5))
        reduced = b._try_to_reduce()
        assert isinstance(reduced, Box)
        assert np.allclose(reduced.lb, b.lb)


class TestInterval:
    def test_interval_init_valid(self):
        i = Interval(1, 2)
        assert i.lb == 1
        assert i.ub == 2
        assert isinstance(i, Interval)

    def test_interval_init_invalid_raises(self):
        with pytest.raises(IntervalExceptions.WrongBoundsException):
            Interval(3, 2)

    def test_interval_init_invalid_fix(self):
        i = Interval(3, 2, fix=True)
        assert i.lb == 2
        assert i.ub == 3

    def test_interval_str_repr(self):
        i = Interval(1, 2)
        s = str(i)
        r = repr(i)
        assert s == "[1; 2]"
        assert r == s

    def test_interval_property_lb_ub_setters(self):
        i = Interval(0, 5)
        i.lb = 1
        assert i.lb == 1
        i.ub = 10
        assert i.ub == 10

        with pytest.raises(IntervalExceptions.WrongBoundsException):
            i.lb = 11
        with pytest.raises(IntervalExceptions.WrongBoundsException):
            i.ub = 0

    def test_interval_middle_and_width(self):
        i = Interval(1, 3)
        assert i.middle == 2
        assert i.width == 2

    def test_interval_copy(self):
        i = Interval(1, 4)
        i_copy = i.__copy__()
        assert i == i_copy
        assert i is not i_copy

    def test_try_to_reduce_returns_float_when_small_width(self):
        i = Interval(1, 1 + IntervalConstants._reduction_width / 2)
        result = i._try_to_reduce()
        assert isinstance(result, float)
        expected = 0.5 * (i.lb + i.ub)
        assert math.isclose(result, expected)

    def test_try_to_reduce_returns_interval_when_width_large(self):
        i = Interval(1, 2)
        result = i._try_to_reduce()
        assert isinstance(result, Interval)
        assert result == i
        assert result is not i

    def test_inner_subtraction(self):
        a = Interval(3, 5)
        b = Interval(1, 2)
        result = Interval.inner_subtraction(a, b)
        assert isinstance(result, (float, Interval))
        # [3-1;5-2] = [2;3], possibly reduced to float if width is small
        if isinstance(result, Interval):
            assert result.lb == pytest.approx(2)
            assert result.ub == pytest.approx(3)
        else:
            assert 2 <= result <= 3

    def test_eq_and_ne_and_not_implemented(self):
        i1 = Interval(1, 3)
        i2 = Interval(2, 4)
        v = 2.0

        # two intervals close enough should be equal (within admissible_error)
        i3 = Interval(1, 3)
        assert (i1 == i3) is True
        assert (i1 != i3) is False

        # interval vs float close enough
        approx_i = Interval(1, 1.00000000005)
        assert approx_i == 1.00000000005
        assert not (approx_i != 1.00000000005)

        # interval vs float not equal
        i4 = Interval(1, 2)
        assert (i4 == 1.5) is False

        # unsupported other type
        with pytest.raises(NotImplementedError):
            i1 == [1, 2]

    def test_comparison_operators(self):
        i1 = Interval(1, 2)
        i2 = Interval(2, 3)
        assert (i1 < i2) is True
        assert (i1 <= i2) is True
        assert (i2 > i1) is True
        assert (i2 >= i1) is True
        # With float
        assert (i1 < 2.0) is True
        assert (i1 <= 1.0) is False
        assert (i1 > 0.5) is True
        assert (i1 >= 1.0) is True

    def test_add_neg_sub_mul_operations(self):
        a = Interval(1, 3)
        b = Interval(2, 4)
        # addition
        c = a + b
        assert isinstance(c, (float, Interval))
        if isinstance(c, Interval):
            assert c.lb == pytest.approx(3)
            assert c.ub == pytest.approx(7)

        # reverse add
        r = 4 + a
        assert isinstance(r, (float, Interval))

        # negation
        neg = -a
        if isinstance(neg, Interval):
            assert neg.lb == pytest.approx(-3)
            assert neg.ub == pytest.approx(-1)

        # subtraction
        sub = a - b
        # a - b = a + (-b)
        assert isinstance(sub, (float, Interval))

        # reverse subtraction
        rsub = 5 - a
        assert isinstance(rsub, (float, Interval))

        # multiplication with interval
        mul = a * b
        assert isinstance(mul, (float, Interval))

        # multiplication with float
        mul2 = a * 2
        assert isinstance(mul2, (float, Interval))

        # reverse multiplication
        rmul = 2 * a
        assert isinstance(rmul, (float, Interval))

    def test_invert_and_division(self):
        i1 = Interval(1, 2)
        off = ~i1
        assert isinstance(off, (float, Interval))

        # divisor includes zero
        i_zero = Interval(-1, 1)
        inv_zero = ~i_zero
        assert isinstance(inv_zero, (float, Interval))
        # division by interval
        div = i1 / i1
        assert isinstance(div, (float, Interval))

        # division by float
        div2 = i1 / 2.0
        assert isinstance(div2, (float, Interval))

        # reverse division
        div3 = 2.0 / i1
        assert isinstance(div3, (float, Interval))

    def test_power_even_and_odd(self):
        i1 = Interval(-2, 3)
        # even power includes 0 crossing
        even_pow = i1 ** 2
        assert isinstance(even_pow, (float, Interval))
        # odd power
        odd_pow = i1 ** 3
        assert isinstance(odd_pow, (float, Interval))


class TestRS_SubinclusionFunction:
    def test_call_returns_interval_enclosing_values(self):
        def f(x):
            return np.sum(x)

        low = Interval(0, 1)
        high = Interval(1, 2)
        b = Box(low, high)
        rsub = RS_SubinclusionFunction(f, number_of_samples=100)
        result = rsub(b)
        assert isinstance(result, Interval)
        # result lb <= any sampled sum <= result ub
        # Since f sums over two intervals: sum in [0+1,1+2] = [1,3]
        assert result.lb <= 1
        assert result.ub >= 3
        assert result.lb <= result.ub


class TestIntervalFunctions:
    def test_sin_with_float_and_interval(self):
        val = 1.0
        res1 = sin(val)
        assert isinstance(res1, (float, Interval))

        i = Interval(0, math.pi / 2)
        res2 = sin(i)
        assert isinstance(res2, (float, Interval))

        # large interval covering period >= 2pi returns [-1,1]
        big_i = Interval(0, 10 * math.pi)
        res3 = sin(big_i)
        assert isinstance(res3, (float, Interval))
        if isinstance(res3, Interval):
            assert res3.lb <= -1.0 and res3.ub >= 1.0

    def test_cos_with_float_and_interval(self):
        val = 0.0
        res1 = cos(val)
        assert isinstance(res1, (float, Interval))

        i = Interval(0, math.pi)
        res2 = cos(i)
        assert isinstance(res2, (float, Interval))

        big_i = Interval(0, 10 * math.pi)
        res3 = cos(big_i)
        assert isinstance(res3, (float, Interval))
        if isinstance(res3, Interval):
            assert res3.lb <= -1.0 and res3.ub >= 1.0

    def test_abs_with_float_and_interval(self):
        val = -3.0
        res1 = interval_abs(val)
        assert isinstance(res1, (float, Interval))

        i = Interval(-2, 3)
        res2 = interval_abs(i)
        assert isinstance(res2, (float, Interval))
        # Should include zero because interval crosses 0
        if isinstance(res2, Interval):
            assert res2.lb == 0.0

    def test_exp_with_float_and_interval(self):
        val = 1.0
        res1 = exp(val)
        assert isinstance(res1, (float, Interval))

        i = Interval(1, 2)
        res2 = exp(i)
        assert isinstance(res2, (float, Interval))
        if isinstance(res2, Interval):
            assert res2.lb <= math.exp(1) and res2.ub >= math.exp(2)

    def test_sqrt_with_float_and_interval_and_errors(self):
        val = 4.0
        res1 = sqrt(val)
        assert isinstance(res1, (float, Interval))

        i = Interval(0, 4)
        res2 = sqrt(i)
        assert isinstance(res2, (float, Interval))
        if isinstance(res2, Interval):
            assert res2.lb == 0
            assert res2.ub <= 2

        # Interval with negative ub raises
        neg_i = Interval(-3, -1)
        with pytest.raises(IntervalExceptions.OperationIsNotDefined):
            sqrt(neg_i)

        # Interval overlapping zero lb < 0
        cross_i = Interval(-1, 4)
        res3 = sqrt(cross_i)
        assert isinstance(res3, (float, Interval))
        if isinstance(res3, Interval):
            assert res3.lb == 0

    def test_log_with_float_and_interval_and_errors(self):
        val = math.e
        res1 = log(val)
        assert isinstance(res1, (float, Interval))

        i = Interval(1, 10)
        res2 = log(i)
        assert isinstance(res2, (float, Interval))

        # Interval with ub <= 0 raises
        neg_i = Interval(-3, 0)
        with pytest.raises(IntervalExceptions.OperationIsNotDefined):
            log(neg_i)

        # Interval overlapping zero lb < 0
        cross_i = Interval(-1, 4)
        res3 = log(cross_i)
        assert isinstance(res3, (float, Interval))
        if isinstance(res3, Interval):
            assert math.isfinite(res3.lb) or res3.lb == -math.inf

    def test_split_and_bisect_intervals_and_floats(self):
        i = Interval(0, 10)
        parts = split(i, [1, 2, 1])
        assert len(parts) == 3
        total_width = sum(p.width if isinstance(p, Interval) else 0 for p in parts)
        # sum widths should match original interval width
        if all(isinstance(p, Interval) for p in parts):
            assert math.isclose(total_width, i.width)

        # split for float returns repeated floats
        f = 5.0
        partsf = split(f, [1, 3])
        assert partsf == [5.0, 5.0]

        # bisect shortcut
        bisected = bisect(i)
        assert len(bisected) == 2

    def test_constrain_intervals_and_floats(self):
        i = Interval(-10, 20)
        constrained = constrain(i, 0, 15)
        assert isinstance(constrained, Interval)
        assert 0 <= constrained.lb <= 15
        assert 0 <= constrained.ub <= 15

        f = 100.0
        constrained_f = constrain(f, 0, 50)
        assert constrained_f == 50

        constrained_f2 = constrain(f, 0, 150)
        assert constrained_f2 == 100

    def test_shrink_interval_and_float(self):
        i = Interval(0, 10)
        shrunk = shrink(i, alpha=0.5)
        assert isinstance(shrunk, Interval)
        mid = i.middle
        expected_width = i.width * 0.5 * 0.5 * 2
        # shrunk width is alpha * original width
        assert math.isclose(shrunk.width, i.width * 0.5)

        f = 3.0
        shrunk_f = shrink(f, 0.5)
        assert shrunk_f == 3.0

    def test_hausdorff_distance_part_and_hausdorff_distance(self):
        i1 = Interval(0, 10)
        i2 = Interval(5, 15)
        d1 = _hausdorff_distance_part(i1, i2)
        d2 = _hausdorff_distance_part(i2, i1)
        assert d1 >= 0
        assert d2 >= 0
        hd = hausdorff_distance(i1, i2)
        assert hd == max(d1, d2)

        # test with floats
        hd_f = hausdorff_distance(3, 7.5)
        assert hd_f >= 0
