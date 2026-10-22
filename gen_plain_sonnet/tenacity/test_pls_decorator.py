import pytest

import tenacity
from tenacity import (
    RetryError,
    Retrying,
    retry,
    retry_always,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)


def no_sleep(seconds):
    return None


def test_decorator_with_arguments_retries_and_exposes_attributes():
    calls = []

    @retry(stop=stop_after_attempt(4), sleep=no_sleep)
    def sometimes_fails():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "success"

    assert sometimes_fails() == "success"
    assert len(calls) == 3
    assert isinstance(sometimes_fails.retry, Retrying)
    assert callable(sometimes_fails.retry_with)
    assert isinstance(sometimes_fails.statistics, dict)


def test_bare_and_empty_call_decorator_forms_use_defaults():
    calls = {"bare": 0, "empty": 0}

    @retry
    def bare():
        calls["bare"] += 1
        if calls["bare"] < 3:
            raise RuntimeError("again")
        return "bare ok"

    @retry()
    def empty():
        calls["empty"] += 1
        if calls["empty"] < 2:
            raise RuntimeError("again")
        return "empty ok"

    assert bare() == "bare ok"
    assert empty() == "empty ok"
    assert calls == {"bare": 3, "empty": 2}


def test_decorator_preserves_function_metadata_and_defaults():
    def original(a, b=5, *, c=7):
        """Original docstring."""
        return a + b + c

    wrapped = retry(stop=stop_after_attempt(2), sleep=no_sleep)(original)
    assert wrapped.__name__ == "original"
    assert wrapped.__doc__ == "Original docstring."
    assert wrapped.__wrapped__ is original
    assert wrapped.__defaults__ == (5,)
    assert wrapped.__kwdefaults__ == {"c": 7}
    assert wrapped(1) == 13
    assert wrapped(1, 2, c=3) == 6


def test_retry_with_returns_new_wrapper_with_overridden_settings():
    calls = []

    @retry(stop=stop_after_attempt(5), sleep=no_sleep)
    def always_fails():
        calls.append(1)
        raise ValueError("nope")

    quick = always_fails.retry_with(stop=stop_after_attempt(2))
    with pytest.raises(RetryError):
        quick()
    assert len(calls) == 2

    calls.clear()
    with pytest.raises(ValueError):
        always_fails.retry_with(reraise=True)()
    assert len(calls) == 5
    # the original wrapper keeps its own configuration
    assert always_fails.retry.stop.max_attempt_number == 5
    assert always_fails.retry.reraise is False


def test_statistics_reset_on_each_call_to_decorated_function():
    outcomes = iter([ValueError, ValueError, None, None])

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), sleep=no_sleep)
    def task():
        exc = next(outcomes)
        if exc is not None:
            raise exc("fail")
        return "done"

    assert task() == "done"
    assert task.statistics["attempt_number"] == 3
    assert task.statistics["idle_for"] == 2.0
    assert task() == "done"
    assert task.statistics["attempt_number"] == 1
    assert task.statistics["idle_for"] == 0


def test_decorator_works_on_methods_and_only_retries_listed_exceptions():
    class Service:
        def __init__(self):
            self.calls = 0

        @retry(
            stop=stop_after_attempt(3),
            sleep=no_sleep,
            retry=retry_if_exception_type(ConnectionError),
        )
        def fetch(self, kind):
            self.calls += 1
            if kind == "flaky" and self.calls < 3:
                raise ConnectionError("down")
            if kind == "fatal":
                raise KeyError("missing")
            return f"{kind}:{self.calls}"

    flaky = Service()
    assert flaky.fetch("flaky") == "flaky:3"
    fatal = Service()
    with pytest.raises(KeyError):
        fatal.fetch("fatal")
    assert fatal.calls == 1


def test_decorating_a_retry_strategy_instance_emits_a_warning():
    with pytest.warns(UserWarning, match="retry_base instance"):
        retry(retry_always)


def test_top_level_package_exports_resolve():
    for name in tenacity.__all__:
        assert hasattr(tenacity, name), name
    assert tenacity.wait_full_jitter is tenacity.wait_random_exponential
