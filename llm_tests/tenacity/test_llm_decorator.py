"""Tests for the @retry decorator, Retrying, RetryError and statistics."""
import time

import pytest

from tenacity import (
    RetryError,
    Retrying,
    TryAgain,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_fixed,
    wait_none,
)


class Flaky:
    """Callable that fails a given number of times before succeeding."""

    def __init__(self, failures, exc=ValueError, result="ok"):
        self.failures = failures
        self.exc = exc
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc("failure %d" % self.calls)
        return self.result


def test_retry_bare_decorator_succeeds_after_failures():
    flaky = Flaky(3)

    @retry
    def fn():
        return flaky()

    assert fn() == "ok"
    assert flaky.calls == 4




def test_retry_passes_args_and_kwargs():
    calls = []

    @retry(stop=stop_after_attempt(3))
    def fn(a, b, c=0):
        calls.append((a, b, c))
        if len(calls) < 2:
            raise RuntimeError("boom")
        return a + b + c

    assert fn(1, 2, c=3) == 6
    assert calls == [(1, 2, 3), (1, 2, 3)]




def test_stop_after_attempt_raises_retry_error():
    flaky = Flaky(100)

    @retry(stop=stop_after_attempt(3))
    def fn():
        return flaky()

    with pytest.raises(RetryError) as exc_info:
        fn()
    assert flaky.calls == 3
    assert exc_info.value.last_attempt.attempt_number == 3
    assert exc_info.value.last_attempt.failed


def test_retry_error_exposes_last_exception():
    flaky = Flaky(100, exc=KeyError)

    @retry(stop=stop_after_attempt(2))
    def fn():
        return flaky()

    with pytest.raises(RetryError) as exc_info:
        fn()
    last = exc_info.value.last_attempt.exception()
    assert isinstance(last, KeyError)
    assert isinstance(exc_info.value.__cause__, KeyError)




def test_reraise_true_raises_original_exception():
    flaky = Flaky(100, exc=ValueError)

    @retry(stop=stop_after_attempt(3), reraise=True)
    def fn():
        return flaky()

    with pytest.raises(ValueError, match="failure 3"):
        fn()
    assert flaky.calls == 3




def test_non_matching_exception_is_not_retried():
    flaky = Flaky(5, exc=KeyError)

    @retry(retry=retry_if_exception_type(ValueError), stop=stop_after_attempt(5))
    def fn():
        return flaky()

    with pytest.raises(KeyError):
        fn()
    assert flaky.calls == 1


def test_retry_error_callback_return_value_is_returned():
    @retry(
        stop=stop_after_attempt(2),
        retry_error_callback=lambda rs: "fallback",
    )
    def fn():
        raise ValueError("x")

    assert fn() == "fallback"






def test_retry_on_result_until_value_is_acceptable():
    values = iter([None, None, 42])

    @retry(retry=retry_if_result(lambda r: r is None))
    def fn():
        return next(values)

    assert fn() == 42


def test_try_again_forces_retry():
    state = {"n": 0}

    @retry(stop=stop_after_attempt(5), retry=retry_if_result(lambda r: False))
    def fn():
        state["n"] += 1
        if state["n"] < 3:
            raise TryAgain
        return "done"

    assert fn() == "done"
    assert state["n"] == 3




def test_retrying_object_call_directly():
    flaky = Flaky(2)
    r = Retrying(stop=stop_after_attempt(5), sleep=lambda s: None)
    assert r(flaky) == "ok"
    assert flaky.calls == 3




def test_retrying_context_manager_iteration():
    attempts = 0
    flaky = Flaky(2)
    result = None
    for attempt in Retrying(stop=stop_after_attempt(5), sleep=lambda s: None):
        with attempt:
            attempts += 1
            result = flaky()
    assert result == "ok"
    assert attempts == 3


def test_retrying_context_manager_gives_up():
    flaky = Flaky(100)
    with pytest.raises(RetryError):
        for attempt in Retrying(stop=stop_after_attempt(2), sleep=lambda s: None):
            with attempt:
                flaky()
    assert flaky.calls == 2




def test_custom_sleep_receives_wait_values():
    slept = []
    flaky = Flaky(3)
    r = Retrying(wait=wait_fixed(2), sleep=slept.append)
    r(flaky)
    assert slept == [2, 2, 2]




def test_statistics_after_success():
    flaky = Flaky(2)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(0.01))
    def fn():
        return flaky()

    fn()
    stats = fn.statistics
    assert stats["attempt_number"] == 3
    assert stats["idle_for"] == pytest.approx(0.02)






def test_retry_with_overrides_settings():
    flaky = Flaky(100)

    @retry(stop=stop_after_attempt(10))
    def fn():
        return flaky()

    with pytest.raises(RetryError):
        fn.retry_with(stop=stop_after_attempt(2))()
    assert flaky.calls == 2








def test_retry_decorates_methods():
    class Service:
        def __init__(self):
            self.n = 0

        @retry(stop=stop_after_attempt(4))
        def call(self, x):
            self.n += 1
            if self.n < 3:
                raise RuntimeError
            return x * 2

    s = Service()
    assert s.call(21) == 42
    assert s.n == 3




def test_real_wait_fixed_timing():
    flaky = Flaky(2)

    @retry(wait=wait_fixed(0.1), stop=stop_after_attempt(5))
    def fn():
        return flaky()

    start = time.time()
    fn()
    elapsed = time.time() - start
    assert elapsed >= 0.2
    assert elapsed < 0.5


def test_no_wait_by_default_is_fast():
    flaky = Flaky(5)

    @retry(wait=wait_none())
    def fn():
        return flaky()

    start = time.time()
    fn()
    assert time.time() - start < 0.1


