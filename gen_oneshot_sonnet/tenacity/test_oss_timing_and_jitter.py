"""Tests that check real elapsed time and the randomness of the jitter strategies."""

import random
import threading
import time

import pytest

import tenacity
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry,
    sleep_using_event,
    stop_after_attempt,
    stop_after_delay,
    stop_before_delay,
    stop_when_event_set,
    wait_chain,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
    wait_incrementing,
    wait_random,
    wait_random_exponential,
)


class Flaky:
    def __init__(self, fail_times, result="ok"):
        self.fail_times = fail_times
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError("failure %d" % self.calls)
        return self.result


class AsyncFlaky:
    def __init__(self, fail_times, result="ok"):
        self.fail_times = fail_times
        self.result = result
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ValueError("failure %d" % self.calls)
        return self.result


class RealSleepRecorder:
    """Sleeps for real but remembers every requested duration."""

    def __init__(self):
        self.sleeps = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)
        time.sleep(seconds)


def always_fail():
    raise ValueError("always failing")


def state_at(attempt):
    state = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    state.attempt_number = attempt
    state.set_exception((ValueError, ValueError("x"), None))
    return state


def test_fixed_wait_takes_real_time_between_attempts():
    flaky = Flaky(2)
    retrying = Retrying(wait=wait_fixed(0.1), stop=stop_after_attempt(5))
    start = time.monotonic()
    assert retrying(flaky) == "ok"
    elapsed = time.monotonic() - start
    assert flaky.calls == 3
    assert elapsed >= 0.19
    assert elapsed < 1.0


@pytest.mark.parametrize(
    "wait, expected",
    [
        (wait_exponential(multiplier=0.05), [0.05, 0.1, 0.2]),
        (wait_chain(wait_fixed(0.05), wait_fixed(0.1)), [0.05, 0.1, 0.1]),
        (wait_incrementing(start=0.05, increment=0.05), [0.05, 0.1, 0.15]),
    ],
)
def test_wait_strategies_sleep_for_the_expected_durations(wait, expected):
    recorder = RealSleepRecorder()
    retrying = Retrying(wait=wait, stop=stop_after_attempt(4), sleep=recorder)
    start = time.monotonic()
    assert retrying(Flaky(3)) == "ok"
    elapsed = time.monotonic() - start
    assert recorder.sleeps == pytest.approx(expected)
    assert elapsed >= sum(expected) - 0.01
    assert elapsed < sum(expected) + 1.0


def test_stop_after_delay_gives_up_after_the_time_limit():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("nope")

    retrying = Retrying(wait=wait_fixed(0.1), stop=stop_after_delay(0.3))
    start = time.monotonic()
    with pytest.raises(RetryError):
        retrying(fn)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3
    assert elapsed < 1.5
    assert 3 <= len(calls) <= 5


def test_stop_before_delay_does_not_overshoot_the_limit():
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("nope")

    retrying = Retrying(wait=wait_fixed(0.2), stop=stop_before_delay(0.5))
    start = time.monotonic()
    with pytest.raises(RetryError):
        retrying(fn)
    elapsed = time.monotonic() - start
    assert len(calls) == 3
    assert elapsed >= 0.4
    assert elapsed < 0.5


def test_wait_random_sleeps_stay_within_bounds():
    recorder = RealSleepRecorder()
    retrying = Retrying(
        wait=wait_random(0.05, 0.15), stop=stop_after_attempt(4), sleep=recorder
    )
    start = time.monotonic()
    with pytest.raises(RetryError):
        retrying(always_fail)
    elapsed = time.monotonic() - start
    assert len(recorder.sleeps) == 3
    assert all(0.05 <= s <= 0.15 for s in recorder.sleeps)
    assert elapsed >= sum(recorder.sleeps) - 0.01
    assert elapsed < 0.45 + 1.0


def test_random_waits_actually_vary_between_calls():
    state = state_at(3)
    waits = [
        wait_random(0, 1),
        wait_random_exponential(multiplier=1, max=10),
        wait_exponential_jitter(jitter=1),
    ]
    for wait in waits:
        samples = {wait(state) for _ in range(30)}
        assert len(samples) > 1

    uniform = wait_random(0, 1)
    values = [uniform(state) for _ in range(500)]
    mean = sum(values) / len(values)
    assert 0.35 < mean < 0.65


def test_random_waits_are_reproducible_when_seeded():
    state = state_at(3)
    factories = [
        lambda: wait_random(0, 1),
        lambda: wait_random_exponential(multiplier=1, max=10),
        lambda: wait_exponential_jitter(),
    ]
    for factory in factories:
        random.seed(99)
        first = [factory()(state) for _ in range(5)]
        random.seed(99)
        second = [factory()(state) for _ in range(5)]
        assert first == second


def test_sleep_helpers_block_for_about_the_requested_time():
    start = time.monotonic()
    tenacity.sleep(0.1)
    assert time.monotonic() - start >= 0.09

    already_set = threading.Event()
    already_set.set()
    start = time.monotonic()
    sleep_using_event(already_set)(5)
    assert time.monotonic() - start < 1.0

    never_set = threading.Event()
    start = time.monotonic()
    sleep_using_event(never_set)(0.1)
    elapsed = time.monotonic() - start
    assert elapsed >= 0.09
    assert elapsed < 1.0


def test_event_can_interrupt_a_long_wait():
    event = threading.Event()
    calls = []

    def fn():
        calls.append(1)
        raise ValueError("failing")

    retrying = Retrying(
        wait=wait_fixed(5),
        stop=stop_when_event_set(event),
        sleep=sleep_using_event(event),
    )
    timer = threading.Timer(0.1, event.set)
    start = time.monotonic()
    timer.start()
    try:
        with pytest.raises(RetryError):
            retrying(fn)
    finally:
        timer.cancel()
    elapsed = time.monotonic() - start
    assert elapsed < 2.0
    assert len(calls) == 2


def test_seconds_since_start_grows_across_attempts():
    seen = []

    def after(state):
        seen.append(state.seconds_since_start)

    retrying = Retrying(wait=wait_fixed(0.1), stop=stop_after_attempt(3), after=after)
    with pytest.raises(RetryError):
        retrying(always_fail)
    assert len(seen) == 3
    assert seen == sorted(seen)
    assert seen[1] >= 0.09
    assert seen[2] >= 0.19
    assert retrying.statistics["idle_for"] == pytest.approx(0.2)
    assert retrying.statistics["delay_since_first_attempt"] >= 0.19


def test_decorated_function_retries_with_real_waits():
    flaky = Flaky(3)

    @retry(wait=wait_fixed(0.05), stop=stop_after_attempt(4))
    def fn():
        return flaky()

    start = time.monotonic()
    assert fn() == "ok"
    elapsed = time.monotonic() - start
    assert flaky.calls == 4
    assert elapsed >= 0.14
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_async_retry_waits_for_real_time():
    flaky = AsyncFlaky(2)
    retrying = AsyncRetrying(wait=wait_fixed(0.1), stop=stop_after_attempt(5))

    async def fn():
        return await flaky()

    start = time.monotonic()
    assert await retrying(fn) == "ok"
    elapsed = time.monotonic() - start
    assert flaky.calls == 3
    assert elapsed >= 0.19
    assert elapsed < 1.0
