from functools import partial
from operator import add, sub

import pytest

from toolz import (
    apply,
    complement,
    compose,
    compose_left,
    curry,
    do,
    excepts,
    flip,
    identity,
    juxt,
    memoize,
    pipe,
    thread_first,
    thread_last,
)


def inc(x):
    return x + 1


def double(x):
    return 2 * x


def iseven(n):
    return n % 2 == 0


def test_curry_partial_application_and_keywords():
    @curry
    def mul(x, y):
        return x * y

    double_it = mul(2)
    assert double_it(10) == 20
    assert mul(2, 10) == 20

    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add_it = f(a=1)
    assert add_it(2, 3) == 5
    assert f(1, 2) == 30

    @curry
    def volume(a, b, c):
        return a * b * c

    assert volume(1)(2)(3) == 6
    assert volume(1, 2)(3) == 6
    assert volume(1)(2, 3) == 6


def test_curry_preserves_name_and_doc():
    def documented(x, y):
        """Docstring here."""
        return x + y

    c = curry(documented)
    assert c.__name__ == 'documented'
    assert c.__doc__ == 'Docstring here.'
    assert c.func is documented


def test_curry_errors():
    c = curry(lambda x: x)
    with pytest.raises(TypeError):
        c(1, 2)
    with pytest.raises(TypeError):
        curry(5)


def test_curry_equality_and_hash():
    def f(x, y):
        return x + y

    assert curry(f) == curry(f)
    assert curry(f)(1) == curry(f)(1)
    assert curry(f)(1) != curry(f)(2)
    assert hash(curry(f)(1)) == hash(curry(f)(1))
    assert len({curry(f), curry(f), curry(f)(1)}) == 2


def test_curry_merges_with_partial():
    def f(a, b, c):
        return (a, b, c)

    c = curry(partial(f, 1))
    assert c.args == (1,)
    assert c(2)(3) == (1, 2, 3)


def test_compose_right_to_left():
    assert compose(str, inc)(3) == '4'
    assert compose(inc, double)(5) == 11
    assert compose(double, inc)(5) == 12
    assert compose(inc, add)(1, 2) == 4


def test_compose_empty_single_and_left():
    assert compose()(7) == 7
    assert compose(inc) is inc
    assert compose_left(inc, str)(3) == '4'
    assert compose_left(inc, double)(5) == 12
    assert compose_left()(9) == 9


def test_compose_name_and_equality():
    f = compose(double, inc)
    assert f.__name__ == 'double_of_inc'
    assert f == compose(double, inc)
    assert f != compose(inc, double)


def test_pipe():
    assert pipe(3, double, str) == '6'
    assert pipe(1, inc, inc, double) == 6
    assert pipe([1, 2, 3]) == [1, 2, 3]


def test_thread_first_and_thread_last():
    assert thread_first(1, inc, double) == 4
    assert thread_first(1, (add, 4), (pow, 2)) == 25
    assert thread_first(10, (sub, 3)) == 7
    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4), (pow, 2)) == 32
    assert thread_last(10, (sub, 3)) == -7
    result = thread_last([1, 2, 3], (map, inc), (filter, iseven))
    assert list(result) == [2, 4]


def test_memoize_caches_results():
    calls = []

    @memoize
    def slow_add(x, y):
        calls.append((x, y))
        return x + y

    assert slow_add(1, 2) == 3
    assert slow_add(1, 2) == 3
    assert slow_add(2, 3) == 5
    assert calls == [(1, 2), (2, 3)]


def test_memoize_unary_function_and_keywords():
    calls = []

    @memoize
    def square(x):
        calls.append(x)
        return x * x

    assert [square(3), square(3), square(4)] == [9, 9, 16]
    assert calls == [3, 4]

    kw_calls = []

    @memoize
    def scaled(x, factor=1):
        kw_calls.append((x, factor))
        return x * factor

    assert scaled(2, factor=3) == 6
    assert scaled(2, factor=3) == 6
    assert scaled(2) == 2
    assert kw_calls == [(2, 3), (2, 1)]


def test_memoize_initial_cache_and_key_function():
    @memoize(cache={(1, 2): 99})
    def add_pair(x, y):
        return x + y

    assert add_pair(1, 2) == 99
    assert add_pair(2, 2) == 4

    calls = []

    @memoize(key=lambda args, kwargs: args)
    def add_verbose(x, y, verbose=False):
        calls.append(1)
        return x + y

    assert add_verbose(1, 2) == 3
    assert add_verbose(1, 2, verbose=True) == 3
    assert len(calls) == 1


def test_memoize_unhashable_argument_and_metadata():
    @memoize
    def head(seq):
        return seq[0]

    with pytest.raises(TypeError):
        head([1, 2, 3])

    def original(x):
        """doc"""
        return x

    memoized = memoize(original)
    assert memoized.__name__ == 'original'
    assert memoized.__doc__ == 'doc'
    assert memoized.__wrapped__ is original


def test_juxt_and_do():
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)
    assert juxt(add, sub)(5, 3) == (8, 2)

    log = []
    assert do(log.append, 5) == 5
    assert log == [5]
    logged_inc = compose(inc, partial(do, log.append))
    assert logged_inc(1) == 2
    assert logged_inc(11) == 12
    assert log == [5, 1, 11]


def test_flip():
    assert flip(sub, 1, 10) == 9
    assert flip(sub)(1)(10) == 9
    assert flip(sub, 1)(10) == 9
    data = [1, 'a', 'b', 2, 1.5, 3]
    assert list(filter(flip(isinstance, int), data)) == [1, 2, 3]


def test_excepts_handlers():
    excepting = excepts(ValueError, lambda a: [1, 2].index(a), lambda _: -1)
    assert excepting(1) == 0
    assert excepting(3) == -1

    multi = excepts((IndexError, KeyError), lambda a: a[0])
    assert multi([]) is None
    assert multi([1]) == 1
    assert multi({}) is None
    assert multi({0: 1}) == 1


def test_excepts_uncaught_and_name():
    excepting = excepts(KeyError, lambda a: a[0])
    with pytest.raises(TypeError):
        excepting(5)

    def lookup(d):
        return d['k']

    assert excepts(KeyError, lookup).__name__ == 'lookup_excepting_KeyError'


def test_complement_apply_identity():
    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True
    assert list(filter(complement(bool), [0, 1, '', 'a', None])) == [0, '', None]

    assert apply(double, 5) == 10
    assert apply(add, 2, 3) == 5
    assert apply(sorted, [3, 1, 2], reverse=True) == [3, 2, 1]
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == (
        20, 501, 16000)
    with pytest.raises(TypeError):
        apply()

    obj = object()
    assert identity(obj) is obj
    assert identity(3) == 3
    assert identity(None) is None
