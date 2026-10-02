"""Tests for the stop_* and wait_* strategies."""

import threading
from datetime import timedelta

import pytest

import tenacity
from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    _utils,
    stop_after_attempt,
    stop_after_delay,
    stop_all,
    stop_any,
    stop_before_delay,
    stop_never,
    stop_when_event_set,
    wait_chain,
    wait_combine,
    wait_exception,
    wait_exponential,
    wait_exponential_jitter,
    wait_fixed,
    wait_full_jitter,
    wait_incrementing,
    wait_none,
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


class SleepRecorder:
    def __init__(self):
        self.sleeps = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)


def make_state(
    attempt=1,
    exc=None,
    result=None,
    seconds=None,
    upcoming_sleep=0.0,
    with_outcome=True,
):
    """Build a RetryCallState with a controlled outcome and elapsed time."""
    state = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    state.attempt_number = attempt
    if with_outcome:
        if exc is not None:
            state.set_exception((type(exc), exc, None))
        else:
            state.set_result(result)
        if seconds is not None:
            state.start_time = 0.0
            state.outcome_timestamp = float(seconds)
    state.upcoming_sleep = upcoming_sleep
    return state


# ---------------------------------------------------------------------------
# stop strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attempt, expected", [(1, False), (2, False), (3, True), (4, True)]
)
def test_stop_after_attempt(attempt, expected):
    stop = stop_after_attempt(3)
    assert stop(make_state(attempt=attempt, exc=ValueError())) is expected


def test_stop_after_delay():
    stop = stop_after_delay(5)
    assert stop(make_state(seconds=4.9)) is False
    assert stop(make_state(seconds=5)) is True
    assert stop(make_state(seconds=6)) is True
    assert stop_after_delay(timedelta(seconds=2))(make_state(seconds=2)) is True
    assert stop_after_delay(timedelta(seconds=2))(make_state(seconds=1.5)) is False
    with pytest.raises(RuntimeError):
        stop(make_state(with_outcome=False))


def test_stop_before_delay():
    stop = stop_before_delay(5)
    assert stop(make_state(seconds=3, upcoming_sleep=1.9)) is False
    assert stop(make_state(seconds=3, upcoming_sleep=2)) is True
    assert stop(make_state(seconds=6, upcoming_sleep=0)) is True
    with pytest.raises(RuntimeError):
        stop(make_state(with_outcome=False))


def test_stop_any_all_and_operators():
    state = make_state(attempt=2, seconds=1)
    attempts = stop_after_attempt(2)  # True for this state
    delay = stop_after_delay(10)  # False for this state
    assert stop_any(attempts, delay)(state) is True
    assert stop_all(attempts, delay)(state) is False
    assert stop_all(attempts, lambda rs: True)(state) is True
    assert stop_any(lambda rs: False)(state) is False
    assert isinstance(attempts | delay, stop_any)
    assert isinstance(attempts & delay, stop_all)
    assert (attempts | delay)(state) is True
    assert (attempts & delay)(state) is False


def test_stop_never_and_stop_when_event_set():
    assert stop_never(make_state(attempt=10**6)) is False

    event = threading.Event()
    stop = stop_when_event_set(event)
    assert stop(make_state()) is False
    event.set()
    assert stop(make_state()) is True


def test_stop_strategies_drive_retrying():
    flaky = Flaky(100)
    retrying = Retrying(
        stop=stop_after_attempt(10) | stop_after_attempt(3), sleep=SleepRecorder()
    )
    with pytest.raises(RetryError):
        retrying(flaky)
    assert flaky.calls == 3

    flaky = Flaky(100)
    retrying = Retrying(
        stop=stop_after_attempt(2) & stop_after_attempt(4), sleep=SleepRecorder()
    )
    with pytest.raises(RetryError):
        retrying(flaky)
    assert flaky.calls == 4

    flaky = Flaky(10)
    assert Retrying(stop=stop_never, sleep=SleepRecorder())(flaky) == "ok"
    assert flaky.calls == 11


# ---------------------------------------------------------------------------
# wait strategies
# ---------------------------------------------------------------------------


def test_wait_fixed_and_wait_none():
    state = make_state(exc=ValueError())
    assert wait_fixed(3)(state) == 3.0
    assert wait_fixed(timedelta(seconds=1.5))(state) == 1.5
    assert wait_none()(state) == 0
    assert isinstance(wait_none(), wait_fixed)


def test_wait_random_stays_in_range():
    state = make_state(exc=ValueError())
    wait = wait_random(1, 2)
    samples = [wait(state) for _ in range(200)]
    assert all(1 <= sample <= 2 for sample in samples)
    assert len(set(samples)) > 1

    defaults = [wait_random()(state) for _ in range(50)]
    assert all(0 <= sample <= 1 for sample in defaults)

    assert wait_random(3, 3)(state) == 3

    delta = wait_random(timedelta(seconds=1), timedelta(seconds=2))
    assert all(1 <= delta(state) <= 2 for _ in range(50))


def test_wait_incrementing():
    wait = wait_incrementing(start=1, increment=2, max=6)
    got = [wait(make_state(attempt=n)) for n in range(1, 6)]
    assert got == [1, 3, 5, 6, 6]
    assert wait_incrementing(start=-5, increment=1)(make_state(attempt=1)) == 0
    assert wait_incrementing()(make_state(attempt=3)) == 200


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, [1, 2, 4, 8]),
        ({"multiplier": 2}, [2, 4, 8, 16]),
        ({"multiplier": 0.5}, [0.5, 1, 2, 4]),
        ({"exp_base": 3}, [1, 3, 9, 27]),
    ],
)
def test_wait_exponential_sequence(kwargs, expected):
    wait = wait_exponential(**kwargs)
    got = [wait(make_state(attempt=n)) for n in range(1, 5)]
    assert got == pytest.approx(expected)


def test_wait_exponential_min_max_and_overflow():
    capped = wait_exponential(max=5)
    got = [capped(make_state(attempt=n)) for n in range(1, 7)]
    assert got == pytest.approx([1, 2, 4, 5, 5, 5])
    assert wait_exponential(min=3)(make_state(attempt=1)) == 3
    assert wait_exponential(min=3, max=10)(make_state(attempt=10)) == 10
    assert wait_exponential(max=timedelta(seconds=5))(make_state(attempt=10)) == 5
    assert wait_exponential()(make_state(attempt=5000)) == _utils.MAX_WAIT


def test_wait_random_exponential_window_grows_and_is_capped():
    wait = wait_random_exponential(multiplier=1, max=10)
    for attempt, high in [(1, 1), (2, 2), (3, 4), (10, 10)]:
        samples = [wait(make_state(attempt=attempt)) for _ in range(100)]
        assert all(0 <= sample <= high for sample in samples)
        assert len(set(samples)) > 1

    floor = wait_random_exponential(multiplier=1, min=2, max=10)
    assert floor(make_state(attempt=1)) == 2
    assert wait_full_jitter is wait_random_exponential


def test_wait_exponential_jitter_bounds():
    wait = wait_exponential_jitter()
    for attempt in range(1, 5):
        base = 2 ** (attempt - 1)
        samples = [wait(make_state(attempt=attempt)) for _ in range(100)]
        assert all(base <= sample <= base + 1 for sample in samples)
        assert len(set(samples)) > 1

    assert wait_exponential_jitter(jitter=0)(make_state(attempt=4)) == 8
    assert wait_exponential_jitter(max=3)(make_state(attempt=10)) == 3
    assert wait_exponential_jitter(min=5, jitter=0)(make_state(attempt=1)) == 5
    assert wait_exponential_jitter(multiplier=0.5, jitter=0)(make_state(attempt=3)) == 2


def test_wait_exponential_jitter_initial_is_deprecated():
    with pytest.warns(DeprecationWarning):
        wait = wait_exponential_jitter(initial=2, jitter=0)
    assert wait(make_state(attempt=1)) == 2
    with pytest.raises(ValueError):
        wait_exponential_jitter(initial=2, multiplier=3)


def test_wait_chain():
    chain = wait_chain(wait_fixed(1), wait_fixed(2), wait_fixed(5))
    got = [chain(make_state(attempt=n)) for n in range(1, 7)]
    assert got == [1, 2, 5, 5, 5, 5]
    with pytest.raises(ValueError):
        wait_chain()


def test_wait_combination_and_addition():
    state = make_state(exc=ValueError())
    combined = wait_fixed(1) + wait_fixed(2)
    assert isinstance(combined, wait_combine)
    assert combined(state) == 3
    assert (wait_fixed(1) + wait_fixed(2) + wait_fixed(3))(state) == 6

    jittered = wait_fixed(1) + wait_random(0, 1)
    assert all(1 <= jittered(state) <= 2 for _ in range(50))

    assert (wait_fixed(1) + (lambda rs: 2))(state) == 3
    assert ((lambda rs: 2) + wait_fixed(1))(state) == 3
    assert sum([wait_fixed(1), wait_fixed(2), wait_fixed(4)])(state) == 7
    assert (0 + wait_fixed(2))(state) == 2
    with pytest.raises(TypeError):
        5 + wait_fixed(1)


def test_wait_exception_uses_predicate():
    wait = wait_exception(lambda exc: 5.0 if isinstance(exc, ValueError) else 1.0)
    assert wait(make_state(exc=ValueError())) == 5.0
    assert wait(make_state(exc=KeyError())) == 1.0
    with pytest.raises(RuntimeError):
        wait(make_state(with_outcome=False))
    with pytest.raises(RuntimeError):
        wait(make_state(result=1))


def test_wait_strategies_drive_retrying_sleeps():
    recorder = SleepRecorder()
    chained = Retrying(
        stop=stop_after_attempt(5),
        wait=wait_chain(wait_fixed(1), wait_fixed(2), wait_fixed(3)),
        sleep=recorder,
    )
    assert chained(Flaky(4)) == "ok"
    assert recorder.sleeps == [1, 2, 3, 3]

    recorder = SleepRecorder()
    exponential = Retrying(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, max=5),
        sleep=recorder,
    )
    assert exponential(Flaky(4)) == "ok"
    assert recorder.sleeps == pytest.approx([1, 2, 4, 5])

    recorder = SleepRecorder()
    combined = Retrying(
        stop=stop_after_attempt(4),
        wait=wait_incrementing(start=1, increment=2) + wait_fixed(0.5),
        sleep=recorder,
    )
    assert combined(Flaky(3)) == "ok"
    assert recorder.sleeps == pytest.approx([1.5, 3.5, 5.5])
    assert tenacity.wait_full_jitter is wait_random_exponential
