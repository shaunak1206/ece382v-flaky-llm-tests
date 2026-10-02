import pytest
from toolz.functoolz import (
    identity, apply, thread_first, thread_last, compose, compose_left,
    pipe, complement, juxt, do, curry, flip, excepts, memoize
)


class TestIdentity:
    def test_identity_returns_same(self):
        assert identity(3) == 3

    def test_identity_with_string(self):
        assert identity('hello') == 'hello'

    def test_identity_with_list(self):
        lst = [1, 2, 3]
        assert identity(lst) is lst

    def test_identity_with_none(self):
        assert identity(None) is None


class TestApply:
    def test_apply_double(self):
        double = lambda x: 2 * x
        assert apply(double, 5) == 10

    def test_apply_add(self):
        def add(x, y):
            return x + y
        assert apply(add, 3, 4) == 7

    def test_apply_with_kwargs(self):
        def power(base, exp=2):
            return base ** exp
        assert apply(power, 3, exp=3) == 27


class TestThreadFirst:
    def test_thread_first_simple(self):
        inc = lambda x: x + 1
        double = lambda x: 2 * x
        result = thread_first(1, inc, double)
        assert result == 4

    def test_thread_first_with_extra_args(self):
        def add(x, y):
            return x + y
        def pow(x, y):
            return x ** y
        result = thread_first(1, (add, 4), (pow, 2))
        assert result == 25

    def test_thread_first_multiple_steps(self):
        inc = lambda x: x + 1
        triple = lambda x: x * 3
        result = thread_first(2, inc, triple, inc)
        assert result == 10


class TestThreadLast:
    def test_thread_last_simple(self):
        inc = lambda x: x + 1
        double = lambda x: 2 * x
        result = thread_last(1, inc, double)
        assert result == 4

    def test_thread_last_with_extra_args(self):
        def add(x, y):
            return x + y
        def pow(x, y):
            return x ** y
        result = thread_last(1, (add, 4), (pow, 2))
        assert result == 32

    def test_thread_last_with_filter_map(self):
        inc = lambda x: x + 1
        iseven = lambda x: x % 2 == 0
        result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
        assert result == [2, 4]


class TestCompose:
    def test_compose_basic(self):
        inc = lambda i: i + 1
        result = compose(str, inc)(3)
        assert result == '4'

    def test_compose_multiple(self):
        double = lambda x: x * 2
        inc = lambda x: x + 1
        result = compose(double, inc)(5)  # double(inc(5))
        assert result == 12

    def test_compose_empty(self):
        result = compose()
        assert result(5) == 5

    def test_compose_single(self):
        inc = lambda x: x + 1
        result = compose(inc)
        assert result(5) == 6


class TestComposeLeft:
    def test_compose_left_basic(self):
        inc = lambda i: i + 1
        result = compose_left(inc, str)(3)
        assert result == '4'

    def test_compose_left_multiple(self):
        inc = lambda x: x + 1
        double = lambda x: x * 2
        result = compose_left(inc, double)(5)  # double(inc(5))
        assert result == 12


class TestPipe:
    def test_pipe_basic(self):
        double = lambda i: 2 * i
        result = pipe(3, double, str)
        assert result == '6'

    def test_pipe_multiple(self):
        inc = lambda x: x + 1
        double = lambda x: x * 2
        square = lambda x: x * x
        result = pipe(2, inc, double, square)
        assert result == 36  # (2+1)*2 = 6, 6*6 = 36

    def test_pipe_single(self):
        inc = lambda x: x + 1
        result = pipe(5, inc)
        assert result == 6


class TestComplement:
    def test_complement_iseven(self):
        def iseven(n):
            return n % 2 == 0
        isodd = complement(iseven)
        assert iseven(2) is True
        assert isodd(2) is False
        assert isodd(3) is True

    def test_complement_returns_bool(self):
        is_positive = lambda x: x > 0
        is_not_positive = complement(is_positive)
        assert is_not_positive(-5) is True
        assert is_not_positive(5) is False


class TestJuxt:
    def test_juxt_basic(self):
        inc = lambda x: x + 1
        double = lambda x: x * 2
        result = juxt(inc, double)(10)
        assert result == (11, 20)

    def test_juxt_with_list(self):
        inc = lambda x: x + 1
        double = lambda x: x * 2
        result = juxt([inc, double])(10)
        assert result == (11, 20)

    def test_juxt_single_function(self):
        inc = lambda x: x + 1
        result = juxt(inc)(5)
        assert result == (6,)

    def test_juxt_multiple_args(self):
        def add(x, y):
            return x + y
        def mul(x, y):
            return x * y
        result = juxt(add, mul)(3, 4)
        assert result == (7, 12)


class TestDo:
    def test_do_function(self):
        log = []
        inc = lambda x: x + 1
        inc = compose(inc, do(log.append))
        result1 = inc(1)
        result2 = inc(11)
        assert result1 == 2
        assert result2 == 12
        assert log == [1, 11]

    def test_do_returns_original(self):
        side_effect = []
        inc = lambda x: x + 1
        result = do(side_effect.append, 5)
        assert result == 5
        assert side_effect == [5]


class TestCurry:
    def test_curry_basic(self):
        def mul(x, y):
            return x * y
        mul_curry = curry(mul)
        double = mul_curry(2)
        assert double(10) == 20

    def test_curry_as_decorator(self):
        @curry
        def f(x, y, a=10):
            return a * (x + y)
        add = f(a=1)
        assert add(2, 3) == 5

    def test_curry_with_kwargs(self):
        @curry
        def add(a, b, c=0):
            return a + b + c
        add_3 = add(1, b=2)
        assert add_3(c=3) == 6

    def test_curry_equality(self):
        def add(x, y):
            return x + y
        c1 = curry(add)(1)
        c2 = curry(add)(1)
        assert c1 == c2

    def test_curry_hash(self):
        def mul(x, y):
            return x * y
        c = curry(mul)(2)
        # Should be hashable
        s = {c}
        assert c in s


class TestFlip:
    def test_flip_basic(self):
        def div(a, b):
            return a // b
        result = flip(div, 2, 6)
        assert result == 3

    def test_flip_partial(self):
        def div(a, b):
            return a // b
        div_by_two = flip(div, 2)
        assert div_by_two(4) == 2

    def test_flip_with_isinstance(self):
        data = [1, 'a', 'b', 2, 1.5, object(), 3]
        only_ints = list(filter(flip(isinstance, int), data))
        assert only_ints == [1, 2, 3]


class TestExcepts:
    def test_excepts_basic(self):
        excepting = excepts(
            ValueError,
            lambda a: [1, 2].index(a),
            lambda _: -1,
        )
        assert excepting(1) == 0
        assert excepting(3) == -1

    def test_excepts_multiple_exceptions(self):
        excepting = excepts((IndexError, KeyError), lambda a: a[0])
        assert excepting([]) is None
        assert excepting([1]) == 1
        assert excepting({}) is None
        assert excepting({0: 1}) == 1

    def test_excepts_no_exception(self):
        excepting = excepts(ValueError, lambda x: int(x))
        assert excepting('5') == 5


class TestMemoize:
    def test_memoize_basic(self):
        call_count = {'count': 0}
        def add(x, y):
            call_count['count'] += 1
            return x + y
        add_memo = memoize(add)
        assert add_memo(1, 2) == 3
        assert add_memo(1, 2) == 3
        assert call_count['count'] == 1

    def test_memoize_as_decorator(self):
        call_count = {'count': 0}
        @memoize
        def add(x, y):
            call_count['count'] += 1
            return x + y
        assert add(2, 3) == 5
        assert add(2, 3) == 5
        assert call_count['count'] == 1

    def test_memoize_with_cache(self):
        @memoize(cache={(1, 2): 3})
        def add(x, y):
            return x + y
        assert add(1, 2) == 3

    def test_memoize_with_key_function(self):
        call_count = {'count': 0}
        @memoize(key=lambda args, kwargs: args)
        def add(x, y, verbose=False):
            call_count['count'] += 1
            return x + y
        assert add(1, 2, verbose=True) == 3
        assert add(1, 2, verbose=False) == 3
        assert call_count['count'] == 1

    def test_memoize_preserves_name(self):
        @memoize
        def my_func(x):
            return x
        assert my_func.__name__ == 'my_func'


class TestCurryAdvanced:
    def test_curry_with_varargs(self):
        @curry
        def func(*args):
            return sum(args)
        partial = func(1, 2)
        assert partial(3, 4) == 10

    def test_curry_call_method(self):
        @curry
        def add(x, y):
            return x + y
        curried_add = add(1)
        result = curried_add.call(2)
        assert result == 3


class TestComposeEquality:
    def test_compose_equality(self):
        inc = lambda x: x + 1
        double = lambda x: x * 2
        c1 = compose(double, inc)
        c2 = compose(double, inc)
        assert c1 == c2

    def test_compose_inequality(self):
        inc = lambda x: x + 1
        double = lambda x: x * 2
        triple = lambda x: x * 3
        c1 = compose(double, inc)
        c2 = compose(triple, inc)
        assert c1 != c2


class TestJuxtEquality:
    def test_juxt_equality(self):
        inc = lambda x: x + 1
        double = lambda x: x * 2
        j1 = juxt(inc, double)
        j2 = juxt(inc, double)
        # juxt instances should be equal if created with same functions
        assert j1.funcs == j2.funcs
