"""Core behaviour of @retry, Retrying, RetryError, statistics, retry_with and callbacks."""

import threading

import pytest

import tenacity
from tenacity import (
    Future,
    RetryCallState,
    RetryError,
    Retrying,
    TryAgain,
    retry,
    retry_if_exception_type,
    retry_if_result,
    retry_never,
    stop_after_attempt,
    stop_when_event_set,
    wait_fixed,
)


class Flaky:
    """Callable that fails `fail_times` times and then returns `result`."""

    def __init__(self, fail_times, exc=ValueError, result="ok"):
        self.fail_times = fail_times
        self.exc = exc
        self.result = result
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc("failure %d" % self.calls)
        return self.result


class SleepRecorder:
    """Sleep replacement that records requested durations without sleeping."""

    def __init__(self):
        self.sleeps = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)


def test_retry_decorator_forms_and_metadata():
    flaky = Flaky(2)

    @retry
    def bare():
        """Bare docstring."""
        return flaky()

    assert bare() == "ok"
    assert flaky.calls == 3
    assert bare.__name__ == "bare"
    assert bare.__doc__ == "Bare docstring."
    assert isinstance(bare.retry, Retrying)
    assert callable(bare.retry_with)

    other = Flaky(1)

    @retry()
    def with_parens():
        return other()

    assert with_parens() == "ok"
    assert other.calls == 2

    @retry(stop=stop_after_attempt(2))
    def add(a, b=1):
        return a + b

    assert add(1, b=2) == 3


def test_gives_up_with_retry_error_describing_last_attempt():
    flaky = Flaky(10)
    recorder = SleepRecorder()
    retrying = Retrying(stop=stop_after_attempt(3), sleep=recorder)
    with pytest.raises(RetryError) as excinfo:
        retrying(flaky)
    err = excinfo.value
    assert flaky.calls == 3
    assert err.last_attempt.attempt_number == 3
    assert err.last_attempt.failed
    assert isinstance(err.last_attempt.exception(), ValueError)
    assert err.__cause__ is err.last_attempt.exception()
    assert str(err).startswith("RetryError[")
    assert len(recorder.sleeps) == 2  # no sleep after the final attempt


def test_reraise_raises_the_original_exception():
    retrying = Retrying(stop=stop_after_attempt(3), reraise=True, sleep=SleepRecorder())
    with pytest.raises(ValueError, match="failure 3"):
        retrying(Flaky(10))


def test_custom_retry_error_class_and_error_callback():
    class MyRetryError(RetryError):
        pass

    retrying = Retrying(
        stop=stop_after_attempt(2),
        retry_error_cls=MyRetryError,
        sleep=SleepRecorder(),
    )
    with pytest.raises(MyRetryError):
        retrying(Flaky(5))

    with_callback = Retrying(
        stop=stop_after_attempt(2),
        retry_error_callback=lambda state: ("gave up", state.attempt_number),
        sleep=SleepRecorder(),
    )
    assert with_callback(Flaky(5)) == ("gave up", 2)


def test_result_based_failure_raises_retry_error_and_reraise_returns_it():
    retrying = Retrying(
        stop=stop_after_attempt(2),
        retry=retry_if_result(lambda value: value < 5),
        sleep=SleepRecorder(),
    )
    with pytest.raises(RetryError) as excinfo:
        retrying(lambda: 1)
    assert excinfo.value.last_attempt.result() == 1
    with pytest.raises(RetryError):
        excinfo.value.reraise()

    reraising = retrying.copy(reraise=True)
    with pytest.raises(RetryError):
        reraising(lambda: 1)


def test_non_matching_exception_propagates_immediately():
    flaky = Flaky(5, exc=TypeError)
    retrying = Retrying(
        retry=retry_if_exception_type(ValueError),
        stop=stop_after_attempt(5),
        sleep=SleepRecorder(),
    )
    with pytest.raises(TypeError):
        retrying(flaky)
    assert flaky.calls == 1


def test_statistics_after_retried_call():
    flaky = Flaky(2)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), sleep=SleepRecorder())
    def fn():
        return flaky()

    assert fn() == "ok"
    assert fn.statistics["attempt_number"] == 3
    assert fn.statistics["idle_for"] == 2
    assert "start_time" in fn.statistics
    assert "delay_since_first_attempt" in fn.statistics


def test_statistics_are_reset_between_calls():
    state = {"failures": 2}

    @retry(sleep=SleepRecorder())
    def fn():
        if state["failures"] > 0:
            state["failures"] -= 1
            raise ValueError("again")
        return 1

    fn()
    assert fn.statistics["attempt_number"] == 3
    fn()
    assert fn.statistics["attempt_number"] == 1
    assert fn.statistics["idle_for"] == 0


def test_retry_with_overrides_settings_without_touching_original():
    recorder = SleepRecorder()
    flaky = Flaky(3)

    @retry(stop=stop_after_attempt(2), sleep=recorder)
    def fn():
        return flaky()

    with pytest.raises(RetryError):
        fn()
    patient = fn.retry_with(stop=stop_after_attempt(5))
    assert patient() == "ok"
    assert fn.retry.stop.max_attempt_number == 2
    assert patient.retry.stop.max_attempt_number == 5

    recorder.sleeps.clear()
    flaky_again = Flaky(1)

    @retry(stop=stop_after_attempt(3), sleep=recorder)
    def quick():
        return flaky_again()

    slower = quick.retry_with(wait=wait_fixed(3))
    assert slower() == "ok"
    assert recorder.sleeps == [3.0]


def test_retrying_object_forwards_arguments_and_sleeps_between_attempts():
    recorder = SleepRecorder()
    retrying = Retrying(stop=stop_after_attempt(4), wait=wait_fixed(2), sleep=recorder)
    calls = []

    def fn(a, b=0):
        calls.append((a, b))
        if len(calls) < 3:
            raise OSError("not yet")
        return a + b

    assert retrying(fn, 1, b=2) == 3
    assert calls == [(1, 2)] * 3
    assert recorder.sleeps == [2.0, 2.0]


def test_copy_str_and_repr():
    retrying = Retrying(stop=stop_after_attempt(2), name="job")
    clone = retrying.copy(stop=stop_after_attempt(7))
    assert clone is not retrying
    assert type(clone) is Retrying
    assert clone.stop.max_attempt_number == 7
    assert retrying.stop.max_attempt_number == 2
    assert clone.wait is retrying.wait
    assert clone.retry is retrying.retry
    assert retrying.copy(reraise=True).reraise is True
    assert retrying.reraise is False
    assert str(retrying) == "job"
    assert str(Retrying()) == "<unknown>"
    assert repr(retrying).startswith("<Retrying object at 0x")


def test_try_again_forces_a_retry():
    state = {"n": 0}

    @retry(retry=retry_never, stop=stop_after_attempt(5), sleep=SleepRecorder())
    def fn():
        state["n"] += 1
        if state["n"] < 3:
            raise TryAgain
        return "done"

    assert fn() == "done"
    assert state["n"] == 3

    @retry(stop=stop_after_attempt(3), sleep=SleepRecorder())
    def always():
        raise TryAgain

    with pytest.raises(RetryError) as excinfo:
        always()
    assert isinstance(excinfo.value.last_attempt.exception(), TryAgain)

    @retry(stop=stop_after_attempt(3), reraise=True, sleep=SleepRecorder())
    def always_reraise():
        raise TryAgain

    with pytest.raises(TryAgain):
        always_reraise()


def test_disabled_retrying_calls_function_once():
    flaky = Flaky(5)
    retrying = Retrying(enabled=False, stop=stop_after_attempt(5), sleep=SleepRecorder())
    with pytest.raises(ValueError):
        retrying(flaky)
    assert flaky.calls == 1

    other = Flaky(5)

    @retry(enabled=False)
    def fn():
        return other()

    with pytest.raises(ValueError):
        fn()
    assert other.calls == 1


def test_retrying_as_iterator_context_manager():
    calls = 0
    for attempt in Retrying(stop=stop_after_attempt(5), sleep=SleepRecorder()):
        with attempt:
            calls += 1
            if calls < 3:
                raise ValueError("again")
    assert calls == 3

    attempts = 0
    with pytest.raises(RetryError):
        for attempt in Retrying(stop=stop_after_attempt(2), sleep=SleepRecorder()):
            with attempt:
                attempts += 1
                raise ValueError("always")
    assert attempts == 2


def test_retry_on_methods_and_with_stop_event():
    class Service:
        def __init__(self):
            self.calls = 0

        @retry(stop=stop_after_attempt(3), sleep=SleepRecorder())
        def go(self):
            self.calls += 1
            if self.calls < 3:
                raise OSError("down")
            return self.calls

    assert Service().go() == 3

    event = threading.Event()
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 3:
            event.set()
        raise ValueError("always")

    retrying = Retrying(stop=stop_when_event_set(event), sleep=SleepRecorder())
    with pytest.raises(RetryError):
        retrying(fn)
    assert len(calls) == 3


def test_before_after_and_before_sleep_ordering():
    events = []

    def before(state):
        events.append(("before", state.attempt_number))

    def after(state):
        events.append(("after", state.attempt_number))

    def before_sleep(state):
        events.append(("before_sleep", state.attempt_number))

    retrying = Retrying(
        stop=stop_after_attempt(3),
        before=before,
        after=after,
        before_sleep=before_sleep,
        sleep=SleepRecorder(),
    )
    with pytest.raises(RetryError):
        retrying(Flaky(10))
    assert events == [
        ("before", 1),
        ("after", 1),
        ("before_sleep", 1),
        ("before", 2),
        ("after", 2),
        ("before_sleep", 2),
        ("before", 3),
        ("after", 3),
    ]

    events.clear()
    assert retrying(Flaky(2)) == "ok"
    assert events == [
        ("before", 1),
        ("after", 1),
        ("before_sleep", 1),
        ("before", 2),
        ("after", 2),
        ("before_sleep", 2),
        ("before", 3),
    ]


def test_retry_call_state_and_future_helpers():
    state = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    assert state.attempt_number == 1
    assert state.seconds_since_start is None
    assert "attempt #1" in repr(state)
    assert "none yet" in repr(state)
    state.set_result(5)
    assert state.seconds_since_start >= 0
    assert "returned 5" in repr(state)
    state.prepare_for_next_attempt()
    assert state.attempt_number == 2
    assert state.outcome is None

    failed = Future.construct(1, ValueError("x"), True)
    assert failed.failed
    succeeded = Future.construct(2, 5, False)
    assert not succeeded.failed
    assert succeeded.result() == 5
    assert succeeded.attempt_number == 2
    assert float(tenacity.DoSleep(1.5)) == 1.5
