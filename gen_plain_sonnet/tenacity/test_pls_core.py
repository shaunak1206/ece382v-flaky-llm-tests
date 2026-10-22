import threading

import pytest

import tenacity
from tenacity import (
    AttemptManager,
    Future,
    RetryCallState,
    RetryError,
    Retrying,
    TryAgain,
    retry_always,
    retry_if_exception_type,
    retry_if_result,
    retry_never,
    stop_after_attempt,
    stop_after_delay,
    stop_before_delay,
    stop_when_event_set,
    wait_exponential,
    wait_fixed,
)


def no_sleep(seconds):
    return None


class Flaky:
    """Callable that raises `failures` times and then returns `result`."""

    def __init__(self, failures, exc=ValueError, result="ok"):
        self.failures = failures
        self.exc = exc
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc(f"fail {self.calls}")
        return self.result


def test_success_on_first_attempt_does_not_sleep():
    sleeps = []
    fn = Flaky(0)
    r = Retrying(stop=stop_after_attempt(3), wait=wait_fixed(1), sleep=sleeps.append)
    assert r(fn) == "ok"
    assert fn.calls == 1
    assert sleeps == []


def test_retries_until_success_and_returns_value():
    fn = Flaky(2, result=42)
    r = Retrying(stop=stop_after_attempt(5), sleep=no_sleep)
    assert r(fn) == 42
    assert fn.calls == 3


def test_retry_error_raised_after_stop_with_last_attempt_and_cause():
    fn = Flaky(10)
    r = Retrying(stop=stop_after_attempt(3), sleep=no_sleep)
    with pytest.raises(RetryError) as info:
        r(fn)
    err = info.value
    assert fn.calls == 3
    assert err.last_attempt.attempt_number == 3
    assert err.last_attempt.failed
    assert isinstance(err.last_attempt.exception(), ValueError)
    assert isinstance(err.__cause__, ValueError)
    assert str(err).startswith("RetryError[")


def test_reraise_raises_original_exception_instead_of_retry_error():
    fn = Flaky(10)
    r = Retrying(stop=stop_after_attempt(3), sleep=no_sleep, reraise=True)
    with pytest.raises(ValueError, match="fail 3"):
        r(fn)
    assert fn.calls == 3


def test_retry_error_callback_replaces_exception():
    r = Retrying(
        stop=stop_after_attempt(2),
        sleep=no_sleep,
        retry_error_callback=lambda rs: ("gave up", rs.attempt_number),
    )
    assert r(Flaky(10)) == ("gave up", 2)


def test_unmatched_exception_propagates_immediately():
    fn = Flaky(5, exc=KeyError)
    r = Retrying(
        stop=stop_after_attempt(5),
        sleep=no_sleep,
        retry=retry_if_exception_type(ValueError),
    )
    with pytest.raises(KeyError):
        r(fn)
    assert fn.calls == 1


def test_retry_if_result_loops_until_predicate_is_false():
    values = iter([1, 2, 3, 10, 11])
    r = Retrying(
        stop=stop_after_attempt(10),
        sleep=no_sleep,
        retry=retry_if_result(lambda v: v < 5),
    )
    assert r(lambda: next(values)) == 10


def test_try_again_forces_retry_even_when_retry_is_never():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise TryAgain
        return "done"

    r = Retrying(stop=stop_after_attempt(5), sleep=no_sleep, retry=retry_never)
    assert r(fn) == "done"
    assert len(calls) == 3


def test_try_again_with_reraise_surfaces_underlying_exception():
    def fn():
        raise TryAgain

    r = Retrying(stop=stop_after_attempt(2), sleep=no_sleep, reraise=True)
    # no underlying cause or context: the TryAgain itself is re-raised
    with pytest.raises(TryAgain):
        r(fn)

    def fn_with_context():
        try:
            raise KeyError("underlying")
        except KeyError:
            raise TryAgain

    with pytest.raises(KeyError):
        r(fn_with_context)


def test_sleep_receives_wait_values_between_attempts():
    sleeps = []
    r = Retrying(stop=stop_after_attempt(4), wait=wait_fixed(2), sleep=sleeps.append)
    with pytest.raises(RetryError):
        r(Flaky(10))
    assert sleeps == [2.0, 2.0, 2.0]

    sleeps2 = []
    r2 = Retrying(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1),
        sleep=sleeps2.append,
    )
    with pytest.raises(RetryError):
        r2(Flaky(10))
    assert sleeps2 == [1.0, 2.0, 4.0]


def test_statistics_are_recorded_for_a_call():
    r = Retrying(stop=stop_after_attempt(5), wait=wait_fixed(1.5), sleep=no_sleep)
    assert r.statistics == {}
    r(Flaky(2))
    assert r.statistics["attempt_number"] == 3
    assert r.statistics["idle_for"] == 3.0
    assert r.statistics["delay_since_first_attempt"] >= 0
    assert "start_time" in r.statistics


def test_time_based_stops_end_retrying():
    fn = Flaky(10)
    r = Retrying(stop=stop_after_delay(0), sleep=no_sleep)
    with pytest.raises(RetryError):
        r(fn)
    assert fn.calls == 1

    fn2 = Flaky(10)
    r2 = Retrying(stop=stop_before_delay(5), wait=wait_fixed(10), sleep=no_sleep)
    with pytest.raises(RetryError):
        r2(fn2)
    assert fn2.calls == 1

    fn3 = Flaky(2)
    r3 = Retrying(stop=stop_before_delay(1000), wait=wait_fixed(1), sleep=no_sleep)
    assert r3(fn3) == "ok"
    assert fn3.calls == 3


def test_falsy_and_plain_callable_wait_and_stop():
    sleeps = []
    r = Retrying(stop=stop_after_attempt(3), wait=None, sleep=sleeps.append)
    with pytest.raises(RetryError):
        r(Flaky(10))
    assert sleeps == [0.0, 0.0]

    sleeps2 = []
    fn = Flaky(10)
    r2 = Retrying(
        stop=lambda rs: rs.attempt_number >= 3,
        wait=lambda rs: 0.25,
        sleep=sleeps2.append,
    )
    with pytest.raises(RetryError):
        r2(fn)
    assert fn.calls == 3
    assert sleeps2 == [0.25, 0.25]


def test_before_after_and_before_sleep_hooks_fire_in_order():
    events = []
    r = Retrying(
        stop=stop_after_attempt(5),
        sleep=no_sleep,
        before=lambda rs: events.append(("before", rs.attempt_number)),
        after=lambda rs: events.append(("after", rs.attempt_number)),
        before_sleep=lambda rs: events.append(("sleep", rs.attempt_number)),
    )
    r(Flaky(2))
    assert events == [
        ("before", 1),
        ("after", 1),
        ("sleep", 1),
        ("before", 2),
        ("after", 2),
        ("sleep", 2),
        ("before", 3),
    ]

    counts = {"after": 0, "sleep": 0}
    r2 = Retrying(
        stop=stop_after_attempt(2),
        sleep=no_sleep,
        after=lambda rs: counts.__setitem__("after", counts["after"] + 1),
        before_sleep=lambda rs: counts.__setitem__("sleep", counts["sleep"] + 1),
    )
    with pytest.raises(RetryError):
        r2(Flaky(10))
    assert counts == {"after": 2, "sleep": 1}


def test_disabled_retrying_calls_function_once():
    fn = Flaky(1)
    r = Retrying(enabled=False, stop=stop_after_attempt(5), sleep=no_sleep)
    with pytest.raises(ValueError):
        r(fn)
    assert fn.calls == 1
    assert Retrying(enabled=False)(lambda: "direct") == "direct"


def test_context_manager_iteration_retries_exhausts_and_can_be_disabled():
    count = 0
    for attempt in Retrying(stop=stop_after_attempt(5), sleep=no_sleep):
        with attempt:
            count += 1
            if count < 3:
                raise ValueError("again")
    assert count == 3

    count = 0
    with pytest.raises(RetryError):
        for attempt in Retrying(stop=stop_after_attempt(2), sleep=no_sleep):
            with attempt:
                count += 1
                raise ValueError("always")
    assert count == 2

    with pytest.raises(ValueError, match="direct"):
        for attempt in Retrying(enabled=False):
            with attempt:
                raise ValueError("direct")


def test_copy_overrides_selected_parameters_only():
    def hook(rs):
        return None

    base = Retrying(
        stop=stop_after_attempt(3), reraise=True, name="job", before_sleep=hook
    )
    clone = base.copy(stop=stop_after_attempt(7))
    assert clone is not base
    assert isinstance(clone, Retrying)
    assert clone.stop.max_attempt_number == 7
    assert base.stop.max_attempt_number == 3
    assert clone.reraise is True
    assert clone.wait is base.wait
    assert str(clone) == "job"
    assert base.copy().before_sleep is hook
    assert base.copy(before_sleep=None).before_sleep is None


def test_getstate_drops_thread_local_and_str_uses_name():
    r = Retrying(stop=stop_after_attempt(3), name="pickled")
    state = r.__getstate__()
    assert "_local" not in state
    restored = Retrying.__new__(Retrying)
    restored.__setstate__(state)
    assert restored.stop is r.stop
    assert restored.statistics == {}
    assert str(restored) == "pickled"
    assert str(Retrying()) == "<unknown>"


def test_retry_call_state_basics():
    r = Retrying(name="job")
    rs = RetryCallState(r, fn=None, args=(), kwargs={})
    assert rs.attempt_number == 1
    assert rs.outcome is None
    assert rs.seconds_since_start is None
    assert rs.get_fn_name() == "job"
    assert "none yet" in repr(rs)

    rs.set_exception((ValueError, ValueError("boom"), None))
    assert rs.outcome.failed
    assert rs.seconds_since_start is not None and rs.seconds_since_start >= 0
    assert "failed (ValueError boom)" in repr(rs)

    rs.next_action = tenacity.RetryAction(1)
    rs.prepare_for_next_attempt()
    assert rs.attempt_number == 2
    assert rs.outcome is None
    assert rs.outcome_timestamp is None
    assert rs.next_action is None

    rs.set_result(5)
    assert "returned 5" in repr(rs)


def test_attempt_manager_swallows_exceptions_and_records_outcome():
    rs = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    manager = AttemptManager(rs)
    exc = ValueError("inner")
    manager.__enter__()
    assert manager.__exit__(ValueError, exc, None) is True
    assert rs.outcome.failed
    assert rs.outcome.exception() is exc

    rs2 = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    assert AttemptManager(rs2).__exit__(None, None, None) is None
    assert rs2.outcome.failed is False
    assert rs2.outcome.result() is None


def test_retry_error_reraise_behaviour():
    failed = Future.construct(1, KeyError("k"), True)
    with pytest.raises(KeyError):
        RetryError(failed).reraise()

    succeeded = Future.construct(1, "fine", False)
    err = RetryError(succeeded)
    with pytest.raises(RetryError) as info:
        err.reraise()
    assert info.value is err
