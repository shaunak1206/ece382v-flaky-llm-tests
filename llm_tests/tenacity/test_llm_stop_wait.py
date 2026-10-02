"""Tests for stop and wait strategies."""
import threading
import time

import pytest

from tenacity import (
    RetryCallState,
    Retrying,
    RetryError,
    retry,
    stop_after_attempt,
    stop_after_delay,
    stop_all,
    stop_any,
    stop_before_delay,
    stop_never,
    stop_when_event_set,
    wait_chain,
    wait_combine,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
    wait_incrementing,
    wait_none,
    wait_random,
    wait_random_exponential,
)


def make_state(attempt=1, elapsed=0.0, upcoming=0.0):
    rs = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    rs.attempt_number = attempt
    rs.outcome_timestamp = rs.start_time + elapsed
    rs.upcoming_sleep = upcoming
    return rs


# ---- stop strategies -------------------------------------------------------

@pytest.mark.parametrize(
    "attempt,expected",
    [(1, False), (2, False), (3, True), (4, True)],
)
def test_stop_after_attempt_threshold(attempt, expected):
    assert stop_after_attempt(3)(make_state(attempt=attempt)) is expected




@pytest.mark.parametrize(
    "elapsed,expected", [(0.0, False), (4.9, False), (5.0, True), (10, True)]
)
def test_stop_after_delay_threshold(elapsed, expected):
    assert stop_after_delay(5)(make_state(elapsed=elapsed)) is expected






def test_stop_never_never_stops():
    assert stop_never(make_state(attempt=10**6, elapsed=10**6)) is False




def test_stop_any_stops_when_one_condition_true():
    stop = stop_any(stop_after_attempt(5), stop_after_delay(10))
    assert stop(make_state(attempt=5, elapsed=0)) is True
    assert stop(make_state(attempt=1, elapsed=11)) is True
    assert stop(make_state(attempt=1, elapsed=1)) is False


def test_stop_all_requires_every_condition():
    stop = stop_all(stop_after_attempt(3), stop_after_delay(10))
    assert stop(make_state(attempt=3, elapsed=1)) is False
    assert stop(make_state(attempt=1, elapsed=11)) is False
    assert stop(make_state(attempt=3, elapsed=11)) is True






def test_stop_after_delay_with_real_time():
    @retry(stop=stop_after_delay(0.3), wait=wait_fixed(0.1))
    def fn():
        raise ValueError

    start = time.time()
    with pytest.raises(RetryError):
        fn()
    elapsed = time.time() - start
    assert 0.3 <= elapsed < 0.8




# ---- wait strategies -------------------------------------------------------







def test_wait_random_within_bounds():
    w = wait_random(1, 2)
    values = [w(make_state()) for _ in range(500)]
    assert all(1 <= v <= 2 for v in values)


def test_wait_random_has_jitter():
    w = wait_random(0, 10)
    values = {w(make_state()) for _ in range(50)}
    assert len(values) > 1


def test_wait_random_average_near_midpoint():
    w = wait_random(0, 1)
    values = [w(make_state()) for _ in range(2000)]
    mean = sum(values) / len(values)
    assert 0.45 < mean < 0.55




def test_wait_random_actual_sleep_duration_in_range():
    flaky_calls = []

    @retry(wait=wait_random(0.05, 0.15), stop=stop_after_attempt(3))
    def fn():
        flaky_calls.append(time.time())
        raise ValueError

    with pytest.raises(RetryError):
        fn()
    gaps = [b - a for a, b in zip(flaky_calls, flaky_calls[1:])]
    assert len(gaps) == 2
    for g in gaps:
        assert 0.05 <= g < 0.3


def test_wait_incrementing_values():
    w = wait_incrementing(start=1, increment=2)
    assert [w(make_state(attempt=n)) for n in (1, 2, 3, 4)] == [1, 3, 5, 7]




def test_wait_exponential_default_multiplier():
    w = wait_exponential()
    assert [w(make_state(attempt=n)) for n in (1, 2, 3, 4)] == [1, 2, 4, 8]




def test_wait_exponential_min_and_max():
    w = wait_exponential(multiplier=1, min=3, max=10)
    assert w(make_state(attempt=1)) == 3
    assert w(make_state(attempt=3)) == 4
    assert w(make_state(attempt=20)) == 10




def test_wait_random_exponential_within_window():
    w = wait_random_exponential(multiplier=1, max=8)
    for attempt, high in [(1, 1), (2, 2), (3, 4), (4, 8), (10, 8)]:
        for _ in range(100):
            v = w(make_state(attempt=attempt))
            assert 0 <= v <= high




def test_wait_random_exponential_mean_is_half_of_window():
    w = wait_random_exponential(multiplier=1, max=100)
    values = [w(make_state(attempt=4)) for _ in range(3000)]
    mean = sum(values) / len(values)
    assert 3.5 < mean < 4.5


def test_wait_exponential_jitter_bounds():
    w = wait_exponential_jitter(max=5, jitter=1)
    for attempt in range(1, 8):
        v = w(make_state(attempt=attempt))
        assert 0 <= v <= 5




def test_wait_chain_progression():
    w = wait_chain(wait_fixed(1), wait_fixed(2), wait_fixed(5))
    assert [w(make_state(attempt=n)) for n in (1, 2, 3, 4, 10)] == [1, 2, 5, 5, 5]




def test_wait_combine_with_plus():
    w = wait_fixed(1) + wait_fixed(2)
    assert isinstance(w, wait_combine)
    assert w(make_state()) == 3


def test_wait_combine_fixed_plus_random_bounds():
    w = wait_fixed(3) + wait_random(0, 2)
    values = [w(make_state()) for _ in range(300)]
    assert all(3 <= v <= 5 for v in values)
    assert len(set(values)) > 1




def test_wait_exponential_drives_sleep_calls():
    slept = []

    @retry(wait=wait_exponential(multiplier=1), stop=stop_after_attempt(5), sleep=slept.append)
    def fn():
        raise ValueError

    with pytest.raises(RetryError):
        fn()
    assert slept == [1, 2, 4, 8]


