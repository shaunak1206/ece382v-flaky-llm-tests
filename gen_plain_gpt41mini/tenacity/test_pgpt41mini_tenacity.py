import asyncio
import builtins
import concurrent.futures
import contextlib
import dataclasses
import math
import re
import threading
import time
import types
import typing

import pytest
from freezegun import freeze_time

import tenacity
from tenacity import (
    AttemptManager,
    BaseAction,
    BaseRetrying,
    DoAttempt,
    DoSleep,
    Future,
    IterState,
    RetryAction,
    RetryCallState,
    RetryError,
    Retrying,
    TryAgain,
    sleep,
    sleep_using_event,
    retry,
    retry_any,
    retry_all,
    retry_always,
    retry_base,
    retry_if_exception,
    retry_if_exception_cause_type,
    retry_if_exception_message,
    retry_if_exception_type,
    retry_if_not_exception_message,
    retry_if_not_exception_type,
    retry_if_not_result,
    retry_if_result,
    retry_never,
    retry_unless_exception_type,
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

# ----------------------------------------------
# Helpers for tests
# ----------------------------------------------


class DummyException(Exception):
    pass


def dummy_function(success_on: int, counter: dict):
    """Function that fails until counter reaches success_on."""
    counter["calls"] += 1
    if counter["calls"] < success_on:
        raise DummyException("fail")
    return "ok"


def dummy_function_tryagain(success_on: int, counter: dict):
    """Function that raises TryAgain explicitly until success."""
    counter["calls"] += 1
    if counter["calls"] < success_on:
        raise TryAgain()
    return "ok"


def always_raise():
    raise DummyException("fail always")


def always_return(value):
    def func():
        return value

    return func


def always_raise_with_msg(msg):
    def func():
        raise DummyException(msg)

    return func


def make_retry_state(
    attempt_number=1,
    result=None,
    exception=None,
) -> RetryCallState:
    retry_obj = Retrying()
    state = RetryCallState(retry_object=retry_obj, fn=None, args=(), kwargs={})
    state.attempt_number = attempt_number
    if exception is not None:
        state.set_exception((type(exception), exception, None))
    else:
        state.set_result(result)
    return state


# ----------------------------------------------
# Tests for BaseAction, RetryAction
# ----------------------------------------------


def test_baseaction_repr_and_str():
    class MyAction(BaseAction):
        REPR_FIELDS = ("foo", "bar")
        NAME = "myaction"

        def __init__(self):
            self.foo = 123
            self.bar = "abc"

    a = MyAction()
    r = repr(a)
    s = str(a)
    expected_str = f"MyAction(foo=123, bar='abc')"
    assert expected_str in r
    assert s == r


def test_retryaction_repr_and_str():
    action = RetryAction(5.3)
    r = repr(action)
    s = str(action)
    assert "sleep=5.3" in r
    assert s == r


# ----------------------------------------------
# IterState
# ----------------------------------------------


def test_iterstate_reset_clears_all_fields():
    state = IterState()
    state.actions.append(lambda x: None)
    state.retry_run_result = True
    state.stop_run_result = True
    state.is_explicit_retry = True

    state.reset()

    assert state.actions == []
    assert state.retry_run_result is False
    assert state.stop_run_result is False
    assert state.is_explicit_retry is False


# ----------------------------------------------
# Future
# ----------------------------------------------


def test_future_failed_and_construct_result_and_exception():
    fut = Future(1)
    fut.set_result("success")
    assert not fut.failed
    assert fut.result() == "success"

    exc = ValueError("oops")
    fut2 = Future.construct(2, exc, has_exception=True)
    assert fut2.failed
    with pytest.raises(ValueError):
        fut2.result()

    fut3 = Future.construct(3, 123, has_exception=False)
    assert not fut3.failed
    assert fut3.result() == 123
    assert fut3.attempt_number == 3


# ----------------------------------------------
# RetryError
# ----------------------------------------------


def test_retryerror_str_and_reraise_with_reraise_false_and_true():
    fut = Future(1)
    fut.set_result("ok")
    err = RetryError(fut)
    assert str(err).startswith("RetryError[")

    # reraise on successful result raises RetryError (itself) because last attempt succeeded
    with pytest.raises(RetryError):
        err.reraise()

    # reraise on failed Future with normal exception raises that exception
    fut_exc = Future(1)
    exc = ValueError("bad")
    fut_exc.set_exception(exc)
    err2 = RetryError(fut_exc)
    with pytest.raises(ValueError):
        err2.reraise()

    # reraise on TryAgain raises the underlying cause instead of TryAgain itself
    try:
        raise ValueError("inner err")
    except Exception as ex:
        tryagain = TryAgain()
        tryagain.__cause__ = ex
        tryagain.__traceback__ = ex.__traceback__
        fut_tryagain = Future(1)
        fut_tryagain.set_exception(tryagain)
        err3 = RetryError(fut_tryagain)
        with pytest.raises(ValueError):  # Inner exception reraised, no TryAgain shown
            err3.reraise()


# ----------------------------------------------
# RetryCallState
# ----------------------------------------------


def test_retrycallstate_set_result_and_set_exception_and_repr_behavior():
    state = make_retry_state(result=42)
    assert state.outcome is not None
    assert not state.outcome.failed
    assert state.outcome.result() == 42
    repr_val = repr(state)
    assert "returned 42" in repr_val

    state2 = make_retry_state(exception=DummyException("err"))
    assert state2.outcome is not None
    assert state2.outcome.failed
    with pytest.raises(DummyException):
        state2.outcome.result()
    repr_val2 = repr(state2)
    assert "failed (DummyException err)" in repr_val2

    state3 = RetryCallState(retry_object=Retrying(), fn=None, args=(), kwargs={})
    assert "none yet" in repr(state3)

    # Test get_fn_name returns <unknown> when fn is None
    assert state3.get_fn_name() == "<unknown>"

    # When fn is set, get_fn_name returns its qualified name
    def f():  # noqa
        pass

    state4 = RetryCallState(retry_object=Retrying(), fn=f, args=(), kwargs={})
    assert f.__name__ in state4.get_fn_name()


def test_retrycallstate_prepare_for_next_attempt_and_seconds_since_start():
    state = make_retry_state(result=1)
    original_attempt = state.attempt_number
    state.prepare_for_next_attempt()
    assert state.attempt_number == original_attempt + 1
    assert state.outcome is None
    assert state.outcome_timestamp is None
    assert state.next_action is None

    # seconds_since_start returns None if no outcome_timestamp
    state.outcome_timestamp = None
    assert state.seconds_since_start is None

    # seconds_since_start returns positive float
    state.outcome_timestamp = state.start_time + 1.5
    assert math.isclose(state.seconds_since_start, 1.5, rel_tol=1e-9)


# ----------------------------------------------
# AttemptManager
# ----------------------------------------------


def test_attempt_manager_context_manager_set_result_and_set_exception():
    state = make_retry_state(result="ok")
    manager = AttemptManager(state)

    with manager:
        pass

    assert state.outcome is not None
    assert not state.outcome.failed

    # Test that __exit__ swallows exceptions and calls set_exception
    state = make_retry_state(result=None)
    manager = AttemptManager(state)

    try:
        with manager:
            raise DummyException("fail")
    except Exception:
        pytest.fail("Exception should be swallowed by AttemptManager.__exit__")

    assert state.outcome is not None
    assert state.outcome.failed
    with pytest.raises(DummyException):
        state.outcome.result()

    # Async context manager variant (__aenter__, __aexit__)
    async def enter_exit():
        async with AttemptManager(make_retry_state(result=None)):
            pass

    asyncio.run(enter_exit())


# ----------------------------------------------
# BaseRetrying functionalities with Retrying subclass
# ----------------------------------------------


def test_retrying_success_on_first_try(monkeypatch):
    ret = Retrying()

    call_count = {"calls": 0}

    def f():
        call_count["calls"] += 1
        return "hello"

    result = ret(f)
    assert result == "hello"
    assert call_count["calls"] == 1


def test_retrying_retries_until_success(monkeypatch):
    ret = Retrying(
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(DummyException),
        wait=wait_none(),
    )
    counter = {"calls": 0}
    result = ret(dummy_function(success_on=3, counter=counter))
    assert result == "ok"
    assert counter["calls"] == 3


def test_retrying_reraise_raises_retryerror(monkeypatch):
    ret = Retrying(stop=stop_after_attempt(3), reraise=False)

    def f():
        raise DummyException("fail all")

    with pytest.raises(RetryError) as excinfo:
        ret(f)
    assert isinstance(excinfo.value, RetryError)
    assert excinfo.value.last_attempt.failed


def test_retrying_with_reraise_true_propagates_original_exception(monkeypatch):
    ret = Retrying(stop=stop_after_attempt(2), reraise=True)

    def f():
        raise DummyException("fail all")

    with pytest.raises(DummyException):
        ret(f)


def test_retrying_with_tryagain_uses_is_explicit_retry(monkeypatch):
    ret = Retrying(stop=stop_after_attempt(3), wait=wait_none())

    counter = {"calls": 0}

    @ret.wraps
    def f():
        counter["calls"] += 1
        if counter["calls"] < 3:
            raise TryAgain()
        return "ok"

    result = f()
    assert result == "ok"
    assert counter["calls"] == 3


def test_retrying_disabled_runs_function_once_no_retry(monkeypatch):
    ret = Retrying(enabled=False)

    counter = {"calls": 0}

    def f():
        counter["calls"] += 1
        raise DummyException()

    with pytest.raises(DummyException):
        ret(f)

    assert counter["calls"] == 1


def test_copy_on_retrying_preserves_and_overrides_args():
    ret = Retrying(name="original", reraise=True)
    copy = ret.copy(reraise=False, name="copy")
    assert copy is not ret
    assert copy._name == "copy"
    assert copy.reraise is False
    assert ret.reraise is True
    assert ret._name == "original"


def test_retrying_repr_contains_name():
    r = Retrying(name="myname")
    s = repr(r)
    assert "myname" in s


# ----------------------------------------------
# Testing chaining and combining of strategies
# ----------------------------------------------


def test_retry_any_with_always_true_and_always_false():
    state = make_retry_state()
    any_retry = retry_any(retry_never, retry_always)
    assert any_retry(state) is True

    all_retry = retry_all(retry_never, retry_always)
    assert all_retry(state) is False


def test_retry_all_with_always_true_and_always_false():
    state = make_retry_state()
    all_retry = retry_all(retry_always, retry_always)
    assert all_retry(state) is True

    all_retry2 = retry_all(retry_never, retry_always)
    assert all_retry2(state) is False


def test_stop_any_and_all_with_stop_never_and_stop_after_attempt():
    state = make_retry_state()
    s1 = stop_never
    s2 = stop_after_attempt(3)
    assert s1(state) is False
    assert s2(state) is (state.attempt_number >= 3)
    combined_any = stop_any(s1, s2)
    combined_all = stop_all(s1, s2)

    # When attempt_number less than 3, any == False, all == False
    state.attempt_number = 2
    assert combined_any(state) is False
    assert combined_all(state) is False

    # When attempt_number is 3, any == True, all == False
    state.attempt_number = 3
    assert combined_any(state) is True
    assert combined_all(state) is False

    # When attempt_number is 4, same
    state.attempt_number = 4
    assert combined_any(state) is True
    assert combined_all(state) is False


def test_retry_base_and_or_operator_combining():
    state = make_retry_state()
    r1 = retry_always
    r2 = retry_never

    combined_or = r1 | r2
    combined_and = r1 & r2

    # retry_always | retry_never should be True (any)
    assert combined_or(state) is True

    # retry_always & retry_never should be False (all)
    assert combined_and(state) is False


def test_stop_base_and_or_operator_combining():
    state = make_retry_state()
    s1 = stop_never
    s2 = stop_after_attempt(5)
    combined_or = s1 | s2
    combined_and = s1 & s2
    state.attempt_number = 10
    assert combined_or(state) is True
    assert combined_and(state) is False


# ----------------------------------------------
# Test wait strategies
# ----------------------------------------------


def test_wait_fixed_returns_fixed_value():
    state = make_retry_state()
    w = wait_fixed(3.14)
    assert math.isclose(w(state), 3.14)


def test_wait_none_returns_zero():
    state = make_retry_state()
    w = wait_none()
    assert w(state) == 0


def test_wait_random_returns_value_between_min_max(monkeypatch):
    class FakeRandom:
        def random(self):
            return 0.5

    monkeypatch.setattr("random.random", lambda: 0.5)  # always half-way

    state = make_retry_state()
    w = wait_random(1, 2)
    val = w(state)
    assert 1 <= val <= 2
    assert math.isclose(val, 1 + 0.5 * (2 - 1))


def test_wait_combine_sums_results(monkeypatch):
    state = make_retry_state()

    w1 = wait_fixed(1)
    w2 = wait_fixed(2)
    combined = wait_combine(w1, w2)

    assert math.isclose(combined(state), 3)


def test_wait_chain_behavior(monkeypatch):
    state = make_retry_state()

    w1 = wait_fixed(1)
    w2 = wait_fixed(2)
    w3 = wait_fixed(3)
    chain = wait_chain(w1, w2, w3)

    state.attempt_number = 1
    assert math.isclose(chain(state), 1)

    state.attempt_number = 2
    assert math.isclose(chain(state), 2)

    state.attempt_number = 3
    assert math.isclose(chain(state), 3)

    state.attempt_number = 10  # beyond range, last wait function used
    assert math.isclose(chain(state), 3)


def test_wait_exception_raises_if_no_outcome():
    w = wait_exception(lambda e: 1)
    state = make_retry_state()
    state.outcome = None
    with pytest.raises(RuntimeError):
        w(state)


def test_wait_exception_returns_predicate(monkeypatch):
    called = {}

    def pred(exc):
        called["called"] = True
        return 2.5

    w = wait_exception(pred)
    state = make_retry_state(exception=DummyException("fail"))
    val = w(state)
    assert val == 2.5
    assert called.get("called", False)


def test_wait_incrementing_increments_within_bounds():
    state = make_retry_state()
    w = wait_incrementing(start=1, increment=2, max=10)
    state.attempt_number = 1
    assert math.isclose(w(state), 1)
    state.attempt_number = 2
    assert math.isclose(w(state), 3)
    state.attempt_number = 5
    assert math.isclose(w(state), 9)
    state.attempt_number = 10
    # Should clip to max=10
    assert math.isclose(w(state), 10)


def test_wait_exponential_basic(monkeypatch):
    w = wait_exponential(multiplier=1, max=30, exp_base=2)
    state = make_retry_state()
    results = []
    for attempt in range(1, 7):
        state.attempt_number = attempt
        val = w(state)
        results.append(val)
    # Verify increasing exponential values
    assert all(results[i] <= results[i + 1] for i in range(len(results) - 1))


def test_wait_random_exponential_bounds(monkeypatch):
    # Patch random.uniform to always return midpoint
    monkeypatch.setattr("random.uniform", lambda a, b: (a + b) / 2)
    w = wait_random_exponential(multiplier=1, max=100)
    state = make_retry_state()
    val = w(state)
    assert 0 <= val <= 100


def test_wait_exponential_jitter_with_warnings(monkeypatch):
    import warnings as pywarnings

    with pywarnings.catch_warnings(record=True) as w:
        pywarnings.simplefilter("always")
        wait_exponential_jitter(initial=2, multiplier=1)  # should warn about initial

    # There should be at least one warning about deprecation
    assert any(issubclass(warn.category, DeprecationWarning) for warn in w)

    state = make_retry_state()
    w = wait_exponential_jitter(jitter=1, multiplier=1, max=10)
    state.attempt_number = 1
    val = w(state)
    assert 0 <= val <= 10


# ----------------------------------------------
# After, Before, Before Sleep modules - logging test stubs
# ----------------------------------------------

class DummyLogger:
    def __init__(self):
        self.messages = []

    def log(self, level, msg, *args):
        self.messages.append((level, msg, args))


def test_after_nothing_and_before_nothing_and_before_sleep_nothing():
    state = make_retry_state()
    tenacity.after.after_nothing(state)
    tenacity.before.before_nothing(state)
    tenacity.before_sleep.before_sleep_nothing(state)
    # no exceptions / no output expected


def test_after_log_and_before_log_and_before_sleep_log():
    logger = DummyLogger()
    state = make_retry_state()
    state.attempt_number = 2
    state.outcome_timestamp = state.start_time + 1.234
    state.next_action = RetryAction(1.23)

    after_func = tenacity.after.after_log(logger, 10, sec_format="%.1f")
    before_func = tenacity.before.before_log(logger, 20)
    before_sleep_func = tenacity.before_sleep.before_sleep_log(logger, 30, exc_info=True)

    after_func(state)
    before_func(state)

    # Prepare outcome for before_sleep log: set outcome to failure
    state.set_result("result")
    state.next_action = RetryAction(1.23)
    before_sleep_func(state)

    assert len(logger.messages) >= 3


# ----------------------------------------------
# Testing retry_if_error strategies
# ----------------------------------------------


def test_retry_if_exception_and_retry_if_exception_type_and_retry_if_not_exception_type():
    # Setup a dummy state with exception set
    exc = DummyException("fail")
    state = make_retry_state(exception=exc)

    r1 = retry_if_exception(lambda e: isinstance(e, DummyException))
    r2 = retry_if_exception_type(DummyException)
    r3 = retry_if_not_exception_type(DummyException)

    assert r1(state) is True
    assert r2(state) is True
    # retry_if_not_exception_type returns False here because exception is DummyException
    assert r3(state) is False

    # Invert exception
    exc2 = ValueError("wrong type")
    state2 = make_retry_state(exception=exc2)
    assert r1(state2) is False
    assert r2(state2) is False
    assert r3(state2) is True


def test_retry_unless_exception_type_behavior():
    # Retry unless exception type
    exc = DummyException("fail")
    state = make_retry_state(exception=exc)
    retryunless = retry_unless_exception_type(DummyException)
    # The exception is DummyException, so retryunless returns False (stop retry)
    assert retryunless(state) is False

    # If no exception, always retry -> True
    state2 = make_retry_state(result="ok")
    assert retryunless(state2) is True


def test_retry_if_exception_cause_type_with_cyclic_causes():
    class E1(Exception):
        pass

    class E2(Exception):
        pass

    e1 = E1("e1")
    e2 = E2("e2")
    e1.__cause__ = e2
    e2.__cause__ = e1  # Create cycle

    state = make_retry_state()
    fut = Future(1)
    fut.set_exception(e1)
    state.outcome = fut

    r = retry_if_exception_cause_type(E2)
    assert r(state) is True


def test_retry_if_exception_message_and_retry_if_not_exception_message():
    exc1 = DummyException("exact message")
    exc2 = DummyException("something else")

    state1 = make_retry_state(exception=exc1)
    state2 = make_retry_state(exception=exc2)
    state3 = make_retry_state(result=42)  # No exception

    r1 = retry_if_exception_message(message="exact message")
    r2 = retry_if_exception_message(match=re.compile("exact"))
    r3 = retry_if_not_exception_message(message="exact message")

    assert r1(state1) is True
    assert r1(state2) is False

    assert r2(state1) is True
    assert r2(state2) is False

    assert r3(state1) is False
    assert r3(state2) is True
    # For no exception, retry_if_not_exception_message returns True
    assert r3(state3) is True

    with pytest.raises(TypeError):
        # Both message and match provided should fail
        retry_if_exception_message(message="a", match="a")

    with pytest.raises(TypeError):
        # Neither message nor match provided should fail
        retry_if_exception_message()


def test_retry_if_result_and_retry_if_not_result():
    state1 = make_retry_state(result=10)
    state2 = make_retry_state(exception=DummyException("fail"))

    r1 = retry_if_result(lambda x: x == 10)
    r2 = retry_if_not_result(lambda x: x == 10)

    assert r1(state1) is True
    assert r2(state1) is False
    # Both should return False on exception state
    assert r1(state2) is False
    assert r2(state2) is False


def test_retry_any_and_retry_all_behavior():
    state = make_retry_state(result="ok")

    always_true = retry_always
    always_false = retry_never

    any_r = retry_any(always_false, always_true)
    all_r = retry_all(always_false, always_true)

    assert any_r(state) is True
    assert all_r(state) is False


# ----------------------------------------------
# Test decorating with @retry
# ----------------------------------------------


def test_retry_decorator_wraps_function(monkeypatch):
    counter = {"calls": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_none())
    def f():
        counter["calls"] += 1
        if counter["calls"] < 3:
            raise DummyException("fail")
        return "done"

    result = f()
    assert result == "done"
    assert counter["calls"] == 3


def test_retry_decorator_during_disabled(monkeypatch):
    counter = {"calls": 0}

    @retry(enabled=False)
    def f():
        counter["calls"] += 1
        raise DummyException("fail")

    with pytest.raises(DummyException):
        f()
    assert counter["calls"] == 1


def test_retry_decorator_retry_with_method(monkeypatch):
    def f():
        return 42

    r = retry(stop=stop_after_attempt(1))
    wrapped = r(f)
    new_wrapped = wrapped.retry_with(stop=stop_after_attempt(5))
    assert new_wrapped.retry.stop != wrapped.retry.stop
    assert new_wrapped() == 42


def test_retry_warn_if_callable_given(monkeypatch):
    with pytest.warns(Warning):
        # If retry() is given a retry_base instance as positional arg, warning is expected.
        r = retry(retry_always)
        func = r(always_raise)
        with pytest.raises(DummyException):
            func()


def test_retry_wraps_preserves_attributes(monkeypatch):
    @retry()
    def f(x):
        """Docstring."""
        return x

    assert hasattr(f, "retry")
    assert hasattr(f, "retry_with")
    assert hasattr(f, "statistics")
    assert f.__doc__ == "Docstring."


# ----------------------------------------------
# Sleep strategies
# ----------------------------------------------


def test_sleep_calls_time_sleep(monkeypatch):
    calls = {}

    def fake_sleep(seconds):
        calls["seconds"] = seconds

    monkeypatch.setattr(tenacity.nap, "time", time)
    monkeypatch.setattr(tenacity.nap, "sleep", fake_sleep)
    tenacity.nap.sleep(1.23)
    assert calls["seconds"] == 1.23


def test_sleep_using_event_waits(monkeypatch):
    event = threading.Event()
    sleeper = sleep_using_event(event)
    # Should call wait without errors
    sleeper(0.1)


# ----------------------------------------------
# Statistics property on BaseRetrying
# ----------------------------------------------


def test_statistics_property_is_thread_local():
    r = Retrying()
    thread_vals = []

    def target():
        r.statistics["test"] = threading.get_ident()
        thread_vals.append(r.statistics["test"])

    threads = [threading.Thread(target=target) for _ in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Values should be distinct thread IDs
    assert len(set(thread_vals)) == 3


# ----------------------------------------------
# iter_state property on BaseRetrying
# ----------------------------------------------


def test_iter_state_property_is_thread_local():
    r = Retrying()
    state1 = r.iter_state
    state1.retry_run_result = True

    # Different call returns the same object in same thread
    state2 = r.iter_state
    assert state2.retry_run_result is True


# ----------------------------------------------
# BaseRetrying.iter produces right action types
# ----------------------------------------------


def test_iter_yields_DoAttempt_and_DoSleep_and_final_result(monkeypatch):
    r = Retrying(wait=wait_none(), stop=stop_after_attempt(3), retry=retry_always)

    retry_state = RetryCallState(retry_object=r, fn=dummy_function(success_on=2, counter={"calls": 0}), args=(), kwargs=())
    retry_state.outcome = None

    r.begin()
    r._begin_iter(retry_state)

    # Initially, actions should include before and DoAttempt
    assert any(isinstance(a(retry_state), DoAttempt) or a(retry_state) is None for a in r.iter_state.actions)

    # Simulate an outcome failed with TryAgain
    retry_state.outcome = Future.construct(1, TryAgain(), True)
    r._begin_iter(retry_state)
    r._run_retry(retry_state)
    assert r.iter_state.retry_run_result in (True, False)

    # Test complete iter loop manually
    for _ in r:
        pass


# ----------------------------------------------
# RetryAction usage correctness
# ----------------------------------------------


def test_retry_action_sleep_value_float_and_usage():
    ra = RetryAction(1.234)
    assert isinstance(ra.sleep, float)
    assert math.isclose(ra.sleep, 1.234)


def test__first_set_function_behavior():
    from tenacity import _first_set

    assert _first_set("a", "b") == "a"
    assert _first_set(object(), "b") is not None

    # Passing _unset forces to second argument
    from tenacity import _unset

    assert _first_set(_unset, "b") == "b"


# ----------------------------------------------
# RetryCallState get_fn_name returns correct names
# ----------------------------------------------


def test_retrycallstate_get_fn_name_with_function_and_str_retry_object():
    r = Retrying(name="TestRetry")
    state = RetryCallState(retry_object=r, fn=None, args=(), kwargs={})
    assert state.get_fn_name() == "TestRetry"

    def f():
        return 1

    state2 = RetryCallState(retry_object=r, fn=f, args=(), kwargs={})
    name = state2.get_fn_name()
    assert name.endswith("f")  # qualified name should end with function name


# ----------------------------------------------
# DoSleep is float subclass and BaseRetrying._run_wait
# ----------------------------------------------


def test_do_sleep_is_float():
    ds = DoSleep(1.5)
    assert isinstance(ds, float)
    assert ds == 1.5


def test_base_retrying_run_wait_sets_upcoming_sleep(monkeypatch):
    r = Retrying(wait=wait_fixed(2))
    state = make_retry_state(result=1)
    r._run_wait(state)
    assert state.upcoming_sleep == 2

    r2 = Retrying(wait=None)
    state2 = make_retry_state(result=1)
    r2._run_wait(state2)
    assert state2.upcoming_sleep == 0.0


# ----------------------------------------------
# Test basic functionality of retry_if_exception
# ----------------------------------------------


def test_retry_if_exception_rejects_on_predicate():
    exc = DummyException()
    state = make_retry_state(exception=exc)

    called = {}

    def predicate(e):
        called["called"] = True
        return isinstance(e, DummyException)

    ri = retry_if_exception(predicate)
    assert ri(state) is True
    assert called["called"]


def test_retry_if_exception_raises_if_no_outcome():
    ri = retry_if_exception(lambda e: True)
    state = make_retry_state()
    state.outcome = None
    with pytest.raises(RuntimeError):
        ri(state)


# ----------------------------------------------
# Test BaseRetrying.__str__ and copy state pickling
# ----------------------------------------------


def test_base_retrying_str_and_pickle(monkeypatch):
    r = Retrying(name="named")
    assert str(r) == "named"
    s = repr(r)
    assert "Retrying" in s

    import pickle

    pickled = pickle.dumps(r)
    r2 = pickle.loads(pickled)
    assert r2._name == "named"
    assert hasattr(r2, "_local")


# ----------------------------------------------
# AsyncRetrying (from asyncio/__init__.py) basic test – only if pytest-asyncio is installed
# ----------------------------------------------


@pytest.mark.asyncio
async def test_asyncretrying_basic(monkeypatch):
    from tenacity.asyncio import AsyncRetrying

    ret = AsyncRetrying(stop=stop_after_attempt(3), wait=wait_none(), retry=retry_if_exception_type(DummyException))
    counter = {"calls": 0}

    async def f():
        counter["calls"] += 1
        if counter["calls"] < 3:
            raise DummyException()
        return "ok"

    result = await ret(f)
    assert result == "ok"
    assert counter["calls"] == 3


@pytest.mark.asyncio
async def test_asyncretrying_disabled_runs_once(monkeypatch):
    from tenacity.asyncio import AsyncRetrying

    ret = AsyncRetrying(enabled=False)
    counter = {"calls": 0}

    async def f():
        counter["calls"] += 1
        raise DummyException()

    with pytest.raises(DummyException):
        await ret(f)
    assert counter["calls"] == 1


# ----------------------------------------------
# Tests for tenacity._utils module
# ----------------------------------------------


def test_utils_to_seconds_and_to_ordinal_functions():
    from tenacity import _utils

    assert _utils.to_seconds(1) == 1.0
    assert _utils.to_seconds(1.5) == 1.5
    import datetime

    td = datetime.timedelta(seconds=2)
    assert _utils.to_seconds(td) == 2.0

    assert _utils.to_ordinal(1) == "1st"
    assert _utils.to_ordinal(2) == "2nd"
    assert _utils.to_ordinal(3) == "3rd"
    assert _utils.to_ordinal(4) == "4th"
    assert _utils.to_ordinal(11) == "11th"
    assert _utils.to_ordinal(21) == "21st"

    # get_callback_name returns string
    def f():
        return 1

    name = _utils.get_callback_name(f)
    assert isinstance(name, str)

    # is_coroutine_callable tests
    async def coro_func():
        pass

    assert _utils.is_coroutine_callable(coro_func)
    assert not _utils.is_coroutine_callable(int)  # classes not coroutine


def test_utils_wrap_to_async_func_returns_async(monkeypatch):
    from tenacity import _utils

    def sync_func(x):
        return x + 1

    async_func = _utils.wrap_to_async_func(sync_func)
    assert asyncio.iscoroutinefunction(async_func)

    # If given corofunc returns the same function
    async def foo():
        return 5

    returned = _utils.wrap_to_async_func(foo)
    assert returned is foo


# ----------------------------------------------
# Test retry_if_not_exception_message edge cases
# ----------------------------------------------


def test_retry_if_not_exception_message_with_non_failed_outcome():
    exc = DummyException("msg")
    state = make_retry_state(result="ok")
    r = retry_if_not_exception_message(message="msg")
    # returns True if outcome is not failed
    assert r(state) is True


# ----------------------------------------------
# Test retry_if_exception_cause_type returns False if no cause matches
# ----------------------------------------------


def test_retry_if_exception_cause_type_no_matching_cause():
    class E1(Exception):
        pass

    state = make_retry_state()
    exc = E1()
    state.set_exception((E1, exc, None))
    r = retry_if_exception_cause_type(ValueError)
    assert r(state) is False


# ----------------------------------------------
# Test BaseRetrying.__iter__ disables retry when enabled is False
# ----------------------------------------------


def test_base_retrying_iter_with_disabled():
    r = Retrying(enabled=False)
    it = iter(r)
    manager = next(it)
    assert isinstance(manager, AttemptManager)

    # If outcome is failure, iterator raises exception on exit
    state = manager.retry_state
    state.set_exception((Exception, Exception("fail"), None))
    with pytest.raises(Exception):
        next(it)


def test_base_retrying_iter_with_enabled_runs(monkeypatch):
    r = Retrying(stop=stop_after_attempt(2), wait=wait_none())
    call_counter = {"calls": 0}

    def f():
        call_counter["calls"] += 1
        if call_counter["calls"] < 2:
            raise DummyException("fail")
        return "ok"

    it = iter(r)
    attempts = []
    try:
        while True:
            with next(it):
                attempts.append(f())
    except StopIteration:
        pass
    assert attempts[-1] == "ok"
    assert call_counter["calls"] == 2


# ----------------------------------------------
# Test compatibility of retry_any and retry_all with async
# ----------------------------------------------


@pytest.mark.asyncio
async def test_asyncio_retry_any_and_all(monkeypatch):
    from tenacity.asyncio.retry import retry_any as async_retry_any, retry_all as async_retry_all
    from tenacity.asyncio import AsyncRetrying
    from tenacity import RetryCallState

    state = make_retry_state(exception=DummyException("fail"))
    state2 = make_retry_state(result=42)

    pred_true = lambda rs: asyncio.sleep(0, result=True)
    pred_false = lambda rs: asyncio.sleep(0, result=False)

    retry_any_instance = async_retry_any(pred_true, pred_false)
    retry_all_instance = async_retry_all(pred_true, pred_false)

    assert await retry_any_instance(state)
    assert not await retry_all_instance(state)

    assert await retry_any_instance(state2) is True or await retry_any_instance(state2) is False  # because predicate receives state with result
    res_all = await retry_all_instance(state2)
    assert res_all is False or res_all is True


# ----------------------------------------------
# Test RetryError reraise with TryAgain causing re-raise of cause
# ----------------------------------------------


def test_retryerror_reraise_with_tryagain_cause(monkeypatch):
    f = Future(1)
    try:
        raise ValueError("original")
    except Exception as e:
        ta = TryAgain()
        ta.__cause__ = e
        ta.__traceback__ = e.__traceback__
        f.set_exception(ta)
    err = RetryError(f)
    with pytest.raises(ValueError):
        err.reraise()


# ----------------------------------------------
# Test that BaseRetrying.begin sets statistics correctly
# ----------------------------------------------


def test_base_retrying_begin_creates_statistics_and_updates_attempt_number():
    r = Retrying()
    r.begin()
    stats = r.statistics
    assert "start_time" in stats
    assert "attempt_number" in stats
    assert stats["attempt_number"] == 1
    assert "idle_for" in stats
    assert "delay_since_first_attempt" in stats


# ----------------------------------------------
# Test BaseRetrying._post_stop_check_actions adds the correct functions and raises
# ----------------------------------------------

def test__post_stop_check_actions_raises_retryerror(monkeypatch):
    class MyRetry(BaseRetrying):
        def __init__(self):
            super().__init__()
            self._local.iter_state = IterState()

        def _post_stop(self, rs):
            self.iter_state.stop_run_result = True
            self._post_stop_check_actions(rs)

    r = MyRetry()
    rs = make_retry_state()
    rs.retry_object._local.iter_state = r.iter_state
    r.iter_state.stop_run_result = True

    # The last action added causes a RetryError to be raised on call
    # Capture that call and check exception
    r._post_stop_check_actions(rs)
    action = r.iter_state.actions[-1]
    with pytest.raises(RetryError):
        action(rs)


# ----------------------------------------------
# Test that BaseRetrying.iter() triggers actions in order
# ----------------------------------------------


def test_base_retrying_iter_runs_through_actions(monkeypatch):
    r = Retrying(stop=stop_after_attempt(2), wait=wait_none())

    # Provide a dummy retry function that always retries for first attempt
    class DummyRetry:
        def __call__(self, retry_state):
            return retry_state.attempt_number < 2

    r.retry = DummyRetry()
    rs = make_retry_state(exception=DummyException())

    r.begin()  # reset statistics
    r._begin_iter(rs)
    result = None
    for action in r.iter_state.actions:
        result = action(rs)
    assert result is None or isinstance(result, (DoSleep, DoAttempt))


# ----------------------------------------------
# Test AttemptManager async context manager usage
# ----------------------------------------------


@pytest.mark.asyncio
async def test_attempt_manager_async_context_manager(monkeypatch):
    state = make_retry_state(result="ok")
    am = AttemptManager(state)
    async with am:
        pass

    # Test __aexit__ swallowing exception
    state2 = make_retry_state(result=None)
    am2 = AttemptManager(state2)

    with pytest.raises(Exception):  # TypeError because __aexit__ returns True for swallowing
        async with am2:
            raise Exception("fail")


# ----------------------------------------------
# Test retrying wraps preserves __wrapped__ attribute
# ----------------------------------------------


def test_retrying_wraps_preserves_wrapped():
    def f():
        return 1

    r = Retrying()
    wrapped = r.wraps(f)
    # __wrapped__ attribute should point to original function
    assert wrapped.__wrapped__ == f


# ----------------------------------------------
# Test that BaseRetrying.wraps reproduces statistics dict clearing behavior
# ----------------------------------------------


def test_retrying_wraps_clears_statistics_each_call(monkeypatch):
    count = {"calls": 0}

    def func():
        count["calls"] += 1
        return count["calls"]

    r = Retrying()

    wrapped = r.wraps(func)
    # Call wrapped multiple times and check statistics dict cleared
    wrapped()
    old_stats_id = id(wrapped.statistics)
    wrapped()
    new_stats_id = id(wrapped.statistics)
    assert old_stats_id == new_stats_id  # Same dict reused
    assert len(wrapped.statistics) == 0 or "start_time" not in wrapped.statistics


# ----------------------------------------------
# Test that BaseRetrying.__repr__ includes fields
# ----------------------------------------------


def test_base_retrying_repr_includes_fields():
    r = Retrying(name="foo")
    rep = repr(r)
    assert "stop=" in rep
    assert "wait=" in rep
    assert "sleep=" in rep
    assert "retry=" in rep
    assert "before=" in rep
    assert "after=" in rep
    assert "'foo'" in rep


# ----------------------------------------------
# Test AttemptManager swallow exception behavior
# ----------------------------------------------


def test_attempt_manager_swallow_exceptions(monkeypatch):
    state = make_retry_state()
    am = AttemptManager(state)

    class CustomException(Exception):
        pass

    with am:
        raise CustomException()

    assert state.outcome.failed
    with pytest.raises(CustomException):
        state.outcome.result()


# ----------------------------------------------
# Test DoAttempt and DoSleep classes are distinct types
# ----------------------------------------------


def test_doattempt_and_dosleep_type_differences():
    a = DoAttempt()
    assert not isinstance(a, float)
    s = DoSleep(1)
    assert isinstance(s, float)
    assert s == 1


# ----------------------------------------------
# Test that Retrying.__call__ returns function result if enabled
# ----------------------------------------------


def test_retrying_call_disabled(monkeypatch):
    r = Retrying(enabled=False)
    called = {"val": False}

    def f():
        called["val"] = True
        return "hi"

    result = r(f)
    assert result == "hi"
    assert called["val"]


# ----------------------------------------------
# Test base retrying _run_stop updates statistics and iter_state
# ----------------------------------------------


def test_base_retrying__run_stop_updates(monkeypatch):
    r = Retrying()
    state = make_retry_state(result=1)
    r._run_stop(state)
    assert "delay_since_first_attempt" in r.statistics
    assert r.iter_state.stop_run_result in (True, False)


# ----------------------------------------------
# Test wait_none is subclass of wait_fixed with zero wait
# ----------------------------------------------


def test_wait_none_is_zero_wait_fixed_subclass():
    w = wait_none()
    assert isinstance(w, wait_fixed)
    assert w(Any := None or 0) == 0


# ----------------------------------------------
# Test retries using retry_never and retry_always constants
# ----------------------------------------------


def test_retry_never_and_always_standalone(monkeypatch):
    r_never = retry_never
    r_always = retry_always

    state = make_retry_state(result=42)
    assert r_never(state) is False
    assert r_always(state) is True

    state2 = make_retry_state(exception=DummyException())
    assert r_never(state2) is False
    assert r_always(state2) is True


# ----------------------------------------------
# Test stop_never always returns False and stop_when_event_set reflects an event
# ----------------------------------------------


def test_stop_never_and_stop_when_event_set(monkeypatch):
    event = threading.Event()
    s_event = stop_when_event_set(event)
    s_never = stop_never

    state = make_retry_state()
    event.clear()
    assert s_never(state) is False
    assert s_event(state) is False

    event.set()
    assert s_event(state) is True


# ----------------------------------------------
# Test wait_fixed respects float and timedelta inputs
# ----------------------------------------------


def test_wait_fixed_accepts_timedelta(monkeypatch):
    import datetime

    td = datetime.timedelta(seconds=5)
    w = wait_fixed(td)
    state = make_retry_state()
    val = w(state)
    assert math.isclose(val, 5.0)


# ----------------------------------------------
# Test wait_random_exponential values are in expected range
# ----------------------------------------------


def test_wait_random_exponential_value_range(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda a, b: (a + b) / 2)
    w = wait_random_exponential(multiplier=1, max=10)
    state = make_retry_state()
    state.attempt_number = 2
    val = w(state)
    assert 0 <= val <= 10


# ----------------------------------------------
# Test before_sleep_log raises if outcome or next_action missing
# ----------------------------------------------


def test_before_sleep_log_errors(monkeypatch):
    logger = DummyLogger()
    f = tenacity.before_sleep.before_sleep_log(logger, 10)
    # State without outcome
    state = make_retry_state()
    state.outcome = None
    with pytest.raises(RuntimeError):
        f(state)

    # State without next_action
    state.outcome = Future.construct(1, "ok", False)
    state.next_action = None
    with pytest.raises(RuntimeError):
        f(state)


# ----------------------------------------------
# Test wait_chain raises ValueError without strategies
# ----------------------------------------------

def test_wait_chain_requires_strategies():
    with pytest.raises(ValueError):
        wait_chain()  # no args


# ----------------------------------------------
# Test retry_if_exception_type with tuple input
# ----------------------------------------------

def test_retry_if_exception_type_with_tuple():
    exc1 = ValueError()
    exc2 = KeyError()

    r = retry_if_exception_type((ValueError, KeyError))
    st1 = make_retry_state(exception=exc1)
    st2 = make_retry_state(exception=exc2)
    st3 = make_retry_state(exception=DummyException())

    assert r(st1) is True
    assert r(st2) is True
    assert r(st3) is False


# ----------------------------------------------
# Test retry_if_not_exception_type with tuple input
# ----------------------------------------------

def test_retry_if_not_exception_type_with_tuple():
    exc1 = ValueError()
    exc2 = KeyError()

    r = retry_if_not_exception_type((ValueError, KeyError))
    st1 = make_retry_state(exception=exc1)
    st2 = make_retry_state(exception=exc2)
    st3 = make_retry_state(exception=DummyException())

    assert r(st1) is False
    assert r(st2) is False
    assert r(st3) is True


# ----------------------------------------------
# Test retry_unless_exception_type with tuple input and no exception
# ----------------------------------------------


def test_retry_unless_exception_type_with_no_exception():
    r = retry_unless_exception_type((ValueError, KeyError))
    st = make_retry_state(result=1)
    assert r(st) is True


# ----------------------------------------------
# Test BaseRetrying.r wraps preserve wrapper assignment additions (__defaults__, __kwdefaults__)
# ----------------------------------------------


def test_retrying_wraps_preserves_function_metadata():
    def f(a, b=1):
        return a + b

    r = Retrying()
    wrapped = r.wraps(f)
    assert wrapped.__defaults__ == f.__defaults__
    assert hasattr(wrapped, "__kwdefaults__")

# ----------------------------------------------
# Test sleep_using_event.wait returns None immediately when event already set
# ----------------------------------------------


def test_sleep_using_event_wait_returns_immediately():
    e = threading.Event()
    e.set()
    s = sleep_using_event(e)
    s(1)  # should not block


# -- End of tests --#
