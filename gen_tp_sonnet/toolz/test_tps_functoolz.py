import pytest

from toolz import (
    complement, compose, compose_left, curry, excepts, flip, identity, juxt,
    memoize, pipe, thread_first, thread_last,
)
from toolz.functoolz import apply, do


def inc(x):
    return x + 1


def double(x):
    return 2 * x


def test_identity():
    assert identity(3) == 3
    obj = object()
    assert identity(obj) is obj


def test_apply():
    assert apply(double, 5) == 10
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (20, 501, 16000)


def test_thread_first_simple():
    assert thread_first(1, inc, double) == 4


def test_thread_first_with_args():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_first(1, (add, 4), (pow_, 2)) == 25


def test_thread_last_simple():
    assert thread_last(1, inc, double) == 4


def test_thread_last_with_args():
    def add(x, y):
        return x + y

    def pow_(x, y):
        return x ** y

    assert thread_last(1, (add, 4), (pow_, 2)) == 32
    assert list(thread_last([1, 2, 3], (map, inc), (filter, lambda x: x % 2 == 0))) == [2, 4]


def test_curry_partial_application():
    @curry
    def mul(x, y):
        return x * y

    times2 = mul(2)
    assert times2(10) == 20
    assert mul(3, 4) == 12
    assert mul(3)(4) == 12


def test_curry_keyword_arguments():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add = f(a=1)
    assert add(2, 3) == 5
    assert f(1, 2) == 30


def test_memoize_caches_results():
    calls = []

    @memoize
    def add(x, y):
        calls.append((x, y))
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert add(2, 3) == 5
    assert calls == [(1, 2), (2, 3)]


def test_memoize_initial_cache():
    @memoize(cache={(1, 2): 3})
    def add(x, y):
        return x + y + 1000

    assert add(1, 2) == 3
    assert add(2, 2) == 1004


def test_memoize_custom_key():
    calls = []

    def f(x):
        calls.append(x)
        return x * 2

    g = memoize(f, key=lambda args, kwargs: str(args[0]).lower())
    assert g('A') == 'AA'
    assert g('a') == 'AA'
    assert calls == ['A']


def test_compose_right_to_left():
    assert compose(str, inc)(3) == '4'
    assert compose(double, inc)(3) == 8
    assert compose(inc, double)(3) == 7


def test_compose_empty_is_identity():
    assert compose()(5) == 5


def test_compose_left_left_to_right():
    assert compose_left(inc, str)(3) == '4'
    assert compose_left(inc, double)(3) == 8
    assert compose_left(double, inc)(3) == 7


def test_pipe():
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3
    assert pipe(3, inc, double, inc) == 9


def test_complement():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True


def test_juxt():
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)


def test_do_returns_input_and_runs_side_effect():
    log = []
    assert do(log.append, 5) == 5
    assert log == [5]


def test_do_composed_logging():
    log = []
    f = compose(inc, do(log.append))
    assert f(1) == 2
    assert f(11) == 12
    assert log == [1, 11]


def test_flip():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2


def test_excepts_with_handler():
    excepting = excepts(ValueError, lambda a: [1, 2].index(a), lambda _: -1)
    assert excepting(1) == 0
    assert excepting(3) == -1


def test_excepts_default_handler_returns_none():
    excepting = excepts((IndexError, KeyError), lambda a: a[0])
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 1}) == 1
