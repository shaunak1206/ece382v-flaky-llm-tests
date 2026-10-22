from functools import partial
from operator import add

import pytest

from toolz.functoolz import (
    Compose, apply, complement, compose, compose_left, curry, do, excepts,
    flip, has_keywords, has_varargs, identity, is_arity, is_partial_args,
    is_valid_args, juxt, memoize, num_required_args, pipe, thread_first,
    thread_last,
)


def inc(x):
    return x + 1


def double(x):
    return 2 * x


def test_identity_returns_argument_unchanged():
    sentinel = object()
    assert identity(3) == 3
    assert identity(sentinel) is sentinel


def test_apply_calls_function_with_args_and_kwargs():
    def f(a, b=0):
        return a * 10 + b

    assert apply(double, 5) == 10
    assert apply(f, 1, b=2) == 12
    assert tuple(map(apply, [double, inc, double], [10, 500, 8000])) == \
        (20, 501, 16000)
    with pytest.raises(TypeError):
        apply()


def test_thread_first_and_thread_last():
    def power(x, y):
        return x ** y

    assert thread_first(1, inc, double) == 4
    assert thread_first(1, (add, 4), (power, 2)) == 25
    assert thread_last(1, inc, double) == 4
    assert thread_last(1, (add, 4), (power, 2)) == 32
    assert thread_first(7) == 7
    assert list(thread_last([1, 2, 3], (map, inc),
                            (filter, lambda x: x % 2 == 0))) == [2, 4]


def test_pipe_applies_functions_left_to_right():
    assert pipe(3, double, str) == '6'
    assert pipe(3) == 3
    assert pipe(1, inc, inc, double) == 6


def test_compose_applies_right_to_left():
    assert compose(str, inc)(3) == '4'
    assert compose(double, inc)(3) == 8
    assert compose(inc, double)(3) == 7
    assert compose(inc, inc, inc)(0) == 3


def test_compose_edge_cases():
    assert compose() is identity
    assert compose(inc) is inc
    assert compose(inc, double) == compose(inc, double)
    assert compose(inc, double) != compose(double, inc)
    # the right-most function may take several arguments and keywords
    assert compose(double, add)(1, 2) == 6

    def kw(a, b=1):
        return a - b

    assert compose(inc, kw)(5, b=2) == 4


def test_compose_metadata():
    composed = compose(double, inc)
    assert isinstance(composed, Compose)
    assert composed.__name__ == 'double_of_inc'
    assert composed.__doc__ == 'lambda *args, **kwargs: double(inc(*args, **kwargs))'
    assert repr(composed).startswith('Compose(')


def test_compose_left_applies_left_to_right():
    assert compose_left(inc, str)(3) == '4'
    assert compose_left(inc, double)(3) == 8
    assert compose_left(double, inc)(3) == 7
    assert compose_left() is identity
    assert compose_left(inc) is inc


def test_complement_negates_predicate():
    def iseven(n):
        return n % 2 == 0

    isodd = complement(iseven)
    assert iseven(2) is True
    assert isodd(2) is False
    assert isodd(3) is True
    assert complement(bool)([]) is True


def test_juxt_returns_tuple_of_results():
    assert juxt(inc, double)(10) == (11, 20)
    assert juxt([inc, double])(10) == (11, 20)
    assert juxt(add, max)(1, 2) == (3, 2)
    assert juxt()(5) == ()


def test_do_runs_side_effect_and_returns_input():
    log = []
    assert do(log.append, 5) == 5
    assert do(log.append, 'x') == 'x'
    assert log == [5, 'x']
    # return value of func is discarded
    assert do(lambda x: 'ignored', 3) == 3


def test_flip_swaps_last_two_arguments():
    def div(a, b):
        return a // b

    assert flip(div, 2, 6) == 3
    div_by_two = flip(div, 2)
    assert div_by_two(4) == 2
    data = [1, 'a', 'b', 2, 1.5, 3]
    assert list(filter(flip(isinstance, int), data)) == [1, 2, 3]


def test_excepts_handles_matching_exceptions():
    def index_of(a):
        return [1, 2].index(a)

    def minus_one(_):
        return -1

    excepting = excepts(ValueError, index_of, minus_one)
    assert excepting(1) == 0
    assert excepting(3) == -1
    assert excepting.__name__ == 'index_of_excepting_ValueError'


def test_excepts_default_handler_tuple_and_propagation():
    def head(a):
        return a[0]

    excepting = excepts((IndexError, KeyError), head)
    assert excepting([]) is None
    assert excepting([1]) == 1
    assert excepting({}) is None
    assert excepting({0: 'v'}) == 'v'
    assert excepting.__name__ == 'head_excepting_IndexError_or_KeyError'
    # exceptions that are not listed propagate
    with pytest.raises(TypeError):
        excepting(5)

    seen = []

    def boom():
        raise ValueError('bad')

    def handler(exc):
        seen.append(exc)
        return 'handled'

    assert excepts(ValueError, boom, handler)() == 'handled'
    assert isinstance(seen[0], ValueError)
    assert str(seen[0]) == 'bad'


def test_curry_partial_application():
    @curry
    def mul3(x, y, z):
        return x * y * z

    assert mul3(2)(3)(4) == 24
    assert mul3(2, 3)(4) == 24
    assert mul3(2)(3, 4) == 24
    assert mul3(2, 3, 4) == 24
    double_it = curry(lambda x, y: x * y)(2)
    assert double_it(10) == 20


def test_curry_keyword_arguments():
    @curry
    def f(x, y, a=10):
        return a * (x + y)

    add_one_scale = f(a=1)
    assert add_one_scale(2, 3) == 5
    assert f(2, 3) == 50
    assert f(2)(3, a=2) == 10


def test_curry_errors_and_introspection():
    with pytest.raises(TypeError):
        curry(5)
    with pytest.raises(TypeError):
        curry()

    @curry
    def add2(x, y):
        return x + y

    with pytest.raises(TypeError):
        add2(1, 2, 3)
    assert add2.func_name == 'add2'
    assert add2.__name__ == 'add2'
    assert add2(1).args == (1,)
    assert add2(1).func is add2.func


def test_curry_equality_hash_and_partial_unpacking():
    def f(x, y):
        return x + y

    assert curry(f) == curry(f)
    assert hash(curry(f)) == hash(curry(f))
    assert curry(f)(1) == curry(f)(1)
    assert curry(f)(1) != curry(f)(2)
    unpacked = curry(partial(f, 1))
    assert unpacked.func is f
    assert unpacked.args == (1,)
    assert unpacked(2) == 3


def test_curry_bind_call_and_methods():
    @curry
    def f(x, y):
        return x - y

    assert f.bind(10)(3) == 7
    assert f.call(10, 3) == 7

    class A:
        def __init__(self, base):
            self.base = base

        @curry
        def plus(self, x, y):
            return self.base + x + y

    a = A(100)
    assert a.plus(1, 2) == 103
    assert a.plus(1)(2) == 103


def test_memoize_caches_results():
    calls = []

    def f(x):
        calls.append(x)
        return x * 2

    mf = memoize(f)
    assert mf(2) == 4
    assert mf(2) == 4
    assert mf(3) == 6
    assert calls == [2, 3]
    assert mf.__name__ == 'f'
    assert mf.__wrapped__ is f


def test_memoize_with_multiple_args_kwargs_and_decorator_usage():
    calls = []

    @memoize
    def g(x, y=1):
        calls.append((x, y))
        return x + y

    assert g(1) == 2
    assert g(1) == 2
    assert g(1, y=2) == 3
    assert g(1, y=2) == 3
    assert g(2, 3) == 5
    assert len(calls) == 3


def test_memoize_with_custom_cache_and_key():
    @memoize(cache={(1, 2): 99})
    def add_pair(x, y):
        return x + y

    assert add_pair(1, 2) == 99
    assert add_pair(2, 2) == 4

    calls = []

    @memoize(key=lambda args, kwargs: args)
    def h(x, y, verbose=False):
        calls.append(1)
        return x + y

    assert h(1, 2) == 3
    assert h(1, 2, verbose=True) == 3
    assert len(calls) == 1


def test_memoize_unhashable_argument_raises_type_error():
    mf = memoize(lambda x: x)
    with pytest.raises(TypeError):
        mf([1, 2])


def test_signature_introspection_helpers():
    def f(x, y, z=3):
        return x + y + z

    def varargs(*args, **kwargs):
        return args

    def unary(x):
        return x

    assert num_required_args(f) == 2
    assert num_required_args(varargs) == 0
    assert has_varargs(varargs) is True
    assert has_varargs(f) is False
    assert has_keywords(f) is True
    assert has_keywords(unary) is False
    assert is_arity(1, unary) is True
    assert is_arity(1, f) is False
    assert is_arity(2, varargs) is False


def test_argument_validity_helpers():
    def two(x, y):
        return x + y

    assert is_valid_args(two, (1,), {}) is False
    assert is_valid_args(two, (1, 2), {}) is True
    assert is_valid_args(two, (1,), {'y': 2}) is True
    assert is_partial_args(two, (1,), {}) is True
    assert is_partial_args(two, (1, 2), {}) is True
    assert is_partial_args(two, (1, 2, 3), {}) is False
