import warnings
from datetime import timedelta

import pytest

from tenacity import (
    RetryCallState,
    Retrying,
    _utils,
    wait_chain,
    wait_combine,
    wait_exception,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
    wait_incrementing,
    wait_none,
    wait_random,
    wait_random_exponential,
)


def make_state(attempt_number=1, exc=None, with_outcome=True):
    rs = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    rs.attempt_number = attempt_number
    if with_outcome:
        if exc is not None:
            rs.set_exception((type(exc), exc, None))
        else:
            rs.set_result("value")
    return rs


def test_wait_fixed_and_none_return_constant_seconds():
    assert wait_fixed(3)(make_state(1)) == 3.0
    assert wait_fixed(3)(make_state(9)) == 3.0
    assert wait_fixed(timedelta(seconds=2, milliseconds=500))(make_state(1)) == 2.5
    assert isinstance(wait_fixed(3)(make_state(1)), float)
    assert wait_none()(make_state(4)) == 0.0


def test_wait_random_uses_random_fraction_between_bounds(monkeypatch):
    monkeypatch.setattr("tenacity.wait.random.random", lambda: 0.5)
    assert wait_random(1, 3)(make_state()) == 2.0
    monkeypatch.setattr("tenacity.wait.random.random", lambda: 0.0)
    assert wait_random(1, 3)(make_state()) == 1.0
    # min == max is always that value regardless of the random draw
    monkeypatch.setattr("tenacity.wait.random.random", lambda: 0.75)
    assert wait_random(2, 2)(make_state()) == 2.0


def test_wait_incrementing_grows_linearly_and_is_clamped():
    w = wait_incrementing(start=1, increment=2, max=6)
    assert [w(make_state(n)) for n in (1, 2, 3, 4, 5, 10)] == [1, 3, 5, 6, 6, 6]
    # a negative increment can never produce a negative wait
    down = wait_incrementing(start=1, increment=-5)
    assert down(make_state(1)) == 1
    assert down(make_state(3)) == 0


def test_wait_exponential_doubles_by_default_and_honours_parameters():
    w = wait_exponential()
    assert [w(make_state(n)) for n in range(1, 6)] == [1, 2, 4, 8, 16]
    assert wait_exponential(multiplier=0.5)(make_state(3)) == 2.0
    assert wait_exponential(exp_base=3)(make_state(3)) == 9
    assert wait_exponential(multiplier=2, exp_base=3)(make_state(3)) == 18
    assert wait_exponential(multiplier=0)(make_state(5)) == 0


def test_wait_exponential_respects_min_and_max():
    capped = wait_exponential(max=10)
    assert [capped(make_state(n)) for n in range(1, 7)] == [1, 2, 4, 8, 10, 10]
    floored = wait_exponential(min=3, max=10)
    assert [floored(make_state(n)) for n in (1, 2, 3)] == [3, 3, 4]
    td = wait_exponential(max=timedelta(seconds=5))
    assert td(make_state(20)) == 5


def test_wait_exponential_handles_huge_attempt_numbers():
    assert wait_exponential(max=60)(make_state(5000)) == 60
    assert wait_exponential()(make_state(5000)) == _utils.MAX_WAIT


def test_wait_random_exponential_draws_uniformly_up_to_exponential_bound(monkeypatch):
    calls = []

    def fake_uniform(low, high):
        calls.append((low, high))
        return high / 2

    monkeypatch.setattr("tenacity.wait.random.uniform", fake_uniform)
    w = wait_random_exponential(multiplier=1, max=100)
    assert w(make_state(3)) == 2.0
    assert calls == [(0, 4)]
    capped = wait_random_exponential(multiplier=1, max=5)
    assert capped(make_state(10)) == 2.5
    assert calls[-1] == (0, 5)


def test_wait_exponential_jitter_without_jitter_is_deterministic():
    w = wait_exponential_jitter(jitter=0)
    assert [w(make_state(n)) for n in (1, 2, 3, 4)] == [1, 2, 4, 8]
    assert wait_exponential_jitter(jitter=0, multiplier=3)(make_state(3)) == 12
    assert wait_exponential_jitter(jitter=0, max=5)(make_state(10)) == 5
    assert wait_exponential_jitter(jitter=0, min=3)(make_state(1)) == 3


def test_wait_exponential_jitter_initial_parameter_handling():
    with pytest.raises(ValueError):
        wait_exponential_jitter(initial=2, multiplier=3)
    with pytest.warns(DeprecationWarning):
        w = wait_exponential_jitter(initial=2, jitter=0)
    assert w.multiplier == 2
    assert w(make_state(1)) == 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wait_exponential_jitter(multiplier=2, jitter=0)


def test_wait_exponential_jitter_adds_bounded_jitter_and_survives_overflow():
    w = wait_exponential_jitter(jitter=1)
    for _ in range(50):
        assert 1 <= w(make_state(1)) <= 2
    assert wait_exponential_jitter(max=60, jitter=1)(make_state(5000)) == 60


def test_wait_chain_uses_each_strategy_then_repeats_last():
    w = wait_chain(wait_fixed(1), wait_fixed(2), wait_fixed(3))
    assert [w(make_state(n)) for n in range(1, 6)] == [1, 2, 3, 3, 3]
    with pytest.raises(ValueError):
        wait_chain()


def test_wait_combine_and_add_operator_sum_waits():
    combined = wait_combine(wait_fixed(1), wait_fixed(2), wait_fixed(4))
    assert combined(make_state()) == 7.0
    added = wait_fixed(1) + wait_fixed(2)
    assert isinstance(added, wait_combine)
    assert added(make_state()) == 3.0
    # plain callables are accepted by wait_combine
    assert wait_combine(wait_fixed(1), lambda rs: 0.5)(make_state()) == 1.5


def test_wait_radd_supports_sum_and_plain_callables():
    total = sum([wait_fixed(1), wait_fixed(2)])
    assert total(make_state()) == 3.0
    single = wait_fixed(5)
    assert sum([single]) is single
    mixed = (lambda rs: 2) + wait_fixed(1)
    assert mixed(make_state()) == 3.0
    with pytest.raises(TypeError):
        5 + wait_fixed(1)


def test_wait_exception_passes_exception_to_predicate():
    def predicate(exc):
        return 7 if isinstance(exc, KeyError) else 1.5

    w = wait_exception(predicate)
    assert w(make_state(exc=KeyError("k"))) == 7
    assert w(make_state(exc=ValueError("v"))) == 1.5


def test_wait_exception_requires_a_failed_outcome():
    w = wait_exception(lambda exc: 1)
    with pytest.raises(RuntimeError):
        w(make_state(with_outcome=False))
    with pytest.raises(RuntimeError):
        w(make_state())  # outcome holds a result, not an exception
