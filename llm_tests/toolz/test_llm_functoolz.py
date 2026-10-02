"""LLM-generated unit tests for toolz.functoolz, based on source/docstrings only."""
import pytest

from toolz.functoolz import (
    curry, compose, compose_left, pipe, thread_first, thread_last, memoize,
    juxt, do, flip, excepts, complement, apply, identity,
)


def inc(x):
    return x + 1


def double(x):
    return x * 2


def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply():
    assert apply(double, 5) == 10
    result = tuple(map(apply, [double, inc, double], [10, 500, 8000]))
    assert result == (20, 501, 16000)


def test_curry_basic_partial_application():
    def mul(x, y):
        return x * y
    curried_mul = curry(mul)
    doubler = curried_mul(2)
    assert doubler(10) == 20
    assert curried_mul(2, 10) == 20


def test_curry_as_decorator_with_kwargs():
    @curry
    def f(x, y, a=10):
        return a * (x + y)
    add = f(a=1)
    assert add(2, 3) == 5
    assert f(2, 3) == 50


def test_curry_equality_and_hash():
    def mul(x, y):
        return x * y
    c1 = curry(mul, 2)
    c2 = curry(mul, 2)
    assert c1 == c2
    assert hash(c1) == hash(c2)
    assert c1(5) == 10


def test_compose_right_to_left():
    assert compose(str, inc)(3) == '4'


def test_compose_no_args_is_identity():
    f = compose()
    assert f(42) == 42


def test_compose_left_to_right():
    assert compose_left(inc, str)(3) == '4'


def test_pipe():
    assert pipe(3, double, str) == '6'


def test_thread_first():
    assert thread_first(1, inc, double) == 4


def test_thread_first_with_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y
    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_last():
    assert thread_last(1, inc, double) == 4


def test_thread_last_with_tuple_forms():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y
    assert thread_last(1, (add, 4), (pow_, 2)) == 32


def test_thread_last_map_filter_pipeline():
    iseven = lambda x: x % 2 == 0
    result = list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
    assert result == [2, 4]


def test_memoize_caches_result():
    calls = []

    def add(x, y):
        calls.append((x, y))
        return x + y

    memoized = memoize(add)
    assert memoized(1, 2) == 3
    assert memoized(1, 2) == 3
    assert calls == [(1, 2)]


def test_memoize_as_decorator_unary():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert square(4) == 16
    assert square(4) == 16
    assert len(calls) == 1


def test_juxt_multiple_funcs_and_from_list():
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_do_returns_input_with_side_effect():
    log = []
    result = do(log.append, 5)
    assert result == 5
    assert log == [5]


def test_do_composed_with_compose():
    from toolz import curry
    log = []
    curried_do = curry(do)
    logging_inc = compose(inc, curried_do(log.append))
    assert logging_inc(1) == 2
    assert logging_inc(11) == 12
    assert log == [1, 11]


def test_flip():
    def div(a, b):
        return a // b
    assert flip(div, 2, 6) == 3


def test_flip_with_filter_isinstance():
    data = [1, 'a', 'b', 2, 1.5, object(), 3]
    only_ints = list(filter(flip(isinstance, int), data))
    assert only_ints == [1, 2, 3]


def test_excepts_catches_and_handles():
    excepting = excepts(
        ValueError,
        lambda a: [1, 2].index(a),
        lambda _: -1,
    )
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler_returns_none():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1


def test_complement():
    def iseven(n):
        return n % 2 == 0
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True
