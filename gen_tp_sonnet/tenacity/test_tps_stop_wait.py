import sys
import threading
from datetime import timedelta

import pytest

import tenacity
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    retry,
    stop_after_attempt,
    stop_after_delay,
    stop_before_delay,
    stop_never,
    stop_when_event_set,
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


def make_state(attempt=1, exc=None, result=None):
    rs = RetryCallState(Retrying(), None, (), {})
    rs.attempt_number = attempt
    if exc is not None:
        try:
            raise exc
        except BaseException:
            rs.set_exception(sys.exc_info())
    else:
        rs.set_result(result)
    return rs


def failing_function(**retry_kwargs):
    calls = []
    sleeps = []

    @retry(sleep=sleeps.append, **retry_kwargs)
    def f():
        calls.append(1)
        raise ValueError("boom")

    return f, calls, sleeps


# ---------------------------------------------------------------- stop


def test_stop_after_attempt_not_reached():
    assert not stop_after_attempt(3)(make_state(attempt=2))


def test_stop_after_attempt_reached_and_exceeded():
    strategy = stop_after_attempt(3)
    assert strategy(make_state(attempt=3))
    assert strategy(make_state(attempt=4))


def test_stop_never_always_false():
    assert not stop_never(make_state(attempt=1))
    assert not stop_never(make_state(attempt=1000))


def test_stop_any_true_if_one_matches():
    strategy = tenacity.stop_any(stop_after_attempt(2), stop_after_attempt(10))
    assert strategy(make_state(attempt=2))
    assert not strategy(make_state(attempt=1))


def test_stop_all_requires_every_condition():
    strategy = tenacity.stop_all(stop_after_attempt(2), stop_after_attempt(5))
    assert not strategy(make_state(attempt=2))
    assert strategy(make_state(attempt=5))


def test_stop_or_and_operators_build_combined_strategies():
    either = stop_after_attempt(2) | stop_after_attempt(5)
    both = stop_after_attempt(2) & stop_after_attempt(5)
    assert isinstance(either, tenacity.stop_any)
    assert isinstance(both, tenacity.stop_all)
    assert either(make_state(attempt=2))
    assert not both(make_state(attempt=2))


def test_stop_when_event_set_follows_event():
    event = threading.Event()
    strategy = stop_when_event_set(event)
    state = make_state()
    assert not strategy(state)
    event.set()
    assert strategy(state)


def test_stop_after_delay_zero_stops_after_first_attempt():
    f, calls, _ = failing_function(stop=stop_after_delay(0))
    with pytest.raises(RetryError):
        f()
    assert len(calls) == 1


def test_stop_after_delay_accepts_timedelta_and_combines_with_attempts():
    f, calls, _ = failing_function(
        stop=stop_after_delay(timedelta(hours=1)) | stop_after_attempt(2)
    )
    with pytest.raises(RetryError):
        f()
    assert len(calls) == 2


def test_stop_before_delay_stops_before_a_sleep_that_would_exceed_limit():
    f, calls, sleeps = failing_function(
        stop=stop_before_delay(1), wait=wait_fixed(5)
    )
    with pytest.raises(RetryError):
        f()
    assert len(calls) == 1
    assert sleeps == []


# ---------------------------------------------------------------- wait


def test_wait_fixed_returns_constant():
    strategy = wait_fixed(3)
    assert strategy(make_state(attempt=1)) == 3
    assert strategy(make_state(attempt=7)) == 3


def test_wait_fixed_accepts_timedelta():
    assert wait_fixed(timedelta(seconds=2))(make_state()) == 2


def test_wait_none_is_zero():
    assert wait_none()(make_state(attempt=4)) == 0


def test_wait_incrementing_grows_linearly():
    strategy = wait_incrementing(start=1, increment=2)
    assert strategy(make_state(attempt=1)) == 1
    assert strategy(make_state(attempt=2)) == 3
    assert strategy(make_state(attempt=3)) == 5


def test_wait_incrementing_respects_max():
    strategy = wait_incrementing(start=0, increment=10, max=25)
    assert strategy(make_state(attempt=2)) == 10
    assert strategy(make_state(attempt=5)) == 25


def test_wait_exponential_default_doubles():
    strategy = wait_exponential()
    assert [strategy(make_state(attempt=n)) for n in (1, 2, 3, 4)] == [1, 2, 4, 8]


def test_wait_exponential_multiplier_and_base():
    assert wait_exponential(multiplier=2)(make_state(attempt=3)) == 8
    assert wait_exponential(exp_base=3)(make_state(attempt=3)) == 9


def test_wait_exponential_min_and_max_clamp():
    strategy = wait_exponential(multiplier=1, min=4, max=10)
    assert strategy(make_state(attempt=1)) == 4
    assert strategy(make_state(attempt=4)) == 8
    assert strategy(make_state(attempt=5)) == 10
    assert strategy(make_state(attempt=20)) == 10


def test_wait_random_stays_within_bounds():
    strategy = wait_random(min=1, max=2)
    for _ in range(100):
        assert 1 <= strategy(make_state()) <= 2


def test_wait_random_exponential_window_grows_and_is_capped():
    strategy = wait_random_exponential(multiplier=1, max=4)
    for _ in range(100):
        assert 0 <= strategy(make_state(attempt=1)) <= 1
        assert 0 <= strategy(make_state(attempt=10)) <= 4


def test_wait_chain_uses_each_strategy_then_last_forever():
    strategy = wait_chain(wait_fixed(1), wait_fixed(2), wait_fixed(5))
    values = [strategy(make_state(attempt=n)) for n in (1, 2, 3, 4, 10)]
    assert values == [1, 2, 5, 5, 5]


def test_wait_combine_sums_strategies():
    strategy = wait_combine(wait_fixed(1), wait_fixed(2.5))
    assert strategy(make_state()) == 3.5


def test_wait_plus_operator_and_sum():
    added = wait_fixed(1) + wait_fixed(2)
    summed = sum([wait_fixed(1), wait_fixed(2), wait_fixed(3)])
    assert isinstance(added, wait_combine)
    assert added(make_state()) == 3
    assert summed(make_state()) == 6


def test_wait_fixed_plus_random_is_between_bounds():
    strategy = wait_fixed(3) + wait_random(0, 2)
    for _ in range(100):
        assert 3 <= strategy(make_state()) <= 5


def test_wait_exception_uses_predicate_on_exception():
    def predicate(exc):
        return 5.0 if isinstance(exc, KeyError) else 1.0

    strategy = wait_exception(predicate)
    assert strategy(make_state(exc=KeyError("k"))) == 5.0
    assert strategy(make_state(exc=ValueError("v"))) == 1.0


def test_wait_exponential_jitter_without_jitter_is_deterministic():
    strategy = wait_exponential_jitter(initial=1, jitter=0)
    assert [strategy(make_state(attempt=n)) for n in (1, 2, 3, 4)] == [1, 2, 4, 8]


def test_wait_exponential_jitter_respects_max_and_jitter_range():
    capped = wait_exponential_jitter(jitter=0, max=3)
    assert capped(make_state(attempt=10)) == 3
    jittery = wait_exponential_jitter(jitter=2)
    for _ in range(100):
        assert 1 <= jittery(make_state(attempt=1)) <= 3


def test_retry_sleeps_for_fixed_wait_between_attempts():
    f, calls, sleeps = failing_function(
        stop=stop_after_attempt(3), wait=wait_fixed(2)
    )
    with pytest.raises(RetryError):
        f()
    assert len(calls) == 3
    assert sleeps == [2, 2]


def test_retry_sleeps_follow_exponential_wait():
    f, calls, sleeps = failing_function(
        stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1)
    )
    with pytest.raises(RetryError):
        f()
    assert sleeps == [1, 2, 4]


def test_retry_sleeps_follow_wait_chain_and_combined_wait():
    f, _, sleeps = failing_function(
        stop=stop_after_attempt(4),
        wait=wait_chain(wait_fixed(1), wait_fixed(2), wait_fixed(3)),
    )
    with pytest.raises(RetryError):
        f()
    assert sleeps == [1, 2, 3]

    g, _, sleeps2 = failing_function(
        stop=stop_after_attempt(3), wait=wait_fixed(1) + wait_fixed(2)
    )
    with pytest.raises(RetryError):
        g()
    assert sleeps2 == [3, 3]
