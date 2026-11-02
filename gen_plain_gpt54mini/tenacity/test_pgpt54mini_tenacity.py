import asyncio
import functools
import inspect
import threading
import types
import warnings
from datetime import timedelta

import pytest
from freezegun import freeze_time

import tenacity
from tenacity import (
    AttemptManager,
    BaseRetrying,
    DoAttempt,
    DoSleep,
    Future,
    NO_RESULT,
    RetryAction,
    RetryCallState,
    RetryError,
    Retrying,
    TryAgain,
    after_log,
    after_nothing,
    before_log,
    before_nothing,
    before_sleep_log,
    before_sleep_nothing,
    retry,
    retry_all,
    retry_always,
    retry_any,
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
    sleep_using_event,
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
    wait_incrementing,
    wait_none,
    wait_random,
    wait_random_exponential,
)
from tenacity import _utils


class Logger:
    def __init__(self):
        self.calls = []

    def log(self, level, msg, *args):
        self.calls.append((level, msg, args))


class DummyRetrying(Retrying):
    pass


def make_retry_state(retry_object=None, outcome=None, attempt_number=1):
    retry_object = retry_object or Retrying()
    rs = RetryCallState(retry_object=retry_object, fn=None, args=(), kwargs={})
    rs.attempt_number = attempt_number
    rs.outcome = outcome
    return rs


def test_no_result_is_object():
    assert NO_RESULT is not None
    assert NO_RESULT != object()


def test_first_set_private_helper():
    assert tenacity._first_set(tenacity._unset, 2) == 2
    assert tenacity._first_set(1, 2) == 1


def test_override_sets_marker_on_function():
    @_utils.override
    def f():
        return 1

    assert getattr(f, "__override__", False) is True


def test_to_ordinal_basic_cases():
    assert _utils.to_ordinal(1) == "1st"
    assert _utils.to_ordinal(2) == "2nd"
    assert _utils.to_ordinal(3) == "3rd"
    assert _utils.to_ordinal(4) == "4th"
    assert _utils.to_ordinal(11) == "11th"
    assert _utils.to_ordinal(22) == "22nd"


def test_get_callback_name_for_function():
    def func():
        pass

    name = _utils.get_callback_name(func)
    assert name.endswith("func")


def test_get_callback_name_for_lambda_contains_lambda():
    fn = lambda x: x  # noqa: E731
    name = _utils.get_callback_name(fn)
    assert "lambda" in name


def test_to_seconds_with_timedelta():
    assert _utils.to_seconds(timedelta(seconds=2, milliseconds=500)) == 2.5


def test_is_coroutine_callable_handles_async_function():
    async def afn():
        return 1

    assert _utils.is_coroutine_callable(afn) is True


def test_is_coroutine_callable_handles_partial_async():
    async def afn(x):
        return x

    p = functools.partial(afn, 1)
    assert _utils.is_coroutine_callable(p) is True


def test_wrap_to_async_func_wraps_sync_callable():
    def f(x):
        return x + 1

    wrapped = _utils.wrap_to_async_func(f)
    assert inspect.iscoroutinefunction(wrapped)

    async def run():
        return await wrapped(1)

    assert asyncio.run(run()) == 2


def test_future_construct_result_and_failed_property():
    fut = Future.construct(3, "x", False)
    assert fut.attempt_number == 3
    assert fut.result() == "x"
    assert fut.failed is False


def test_future_construct_exception_and_failed_property():
    err = ValueError("boom")
    fut = Future.construct(2, err, True)
    assert fut.attempt_number == 2
    assert fut.failed is True
    assert isinstance(fut.exception(), ValueError)


def test_retry_action_repr_and_str():
    a = tenacity.RetryAction(1.5)
    assert repr(a) == "RetryAction(sleep=1.5)"
    assert str(a) == repr(a)


def test_retry_call_state_repr_none_yet():
    rs = make_retry_state()
    text = repr(rs)
    assert "none yet" in text
    assert "attempt #1" in text


def test_retry_call_state_set_result_updates_outcome_and_timestamp():
    rs = make_retry_state()
    rs.set_result("ok")
    assert rs.outcome is not None
    assert rs.outcome.result() == "ok"
    assert rs.outcome_timestamp is not None
    assert rs.seconds_since_start is not None


def test_retry_call_state_set_exception_updates_outcome():
    rs = make_retry_state()
    exc = ValueError("bad")
    rs.set_exception((ValueError, exc, None))
    assert rs.outcome is not None
    assert rs.outcome.failed is True
    assert isinstance(rs.outcome.exception(), ValueError)


def test_retry_call_state_prepare_for_next_attempt_resets():
    rs = make_retry_state()
    rs.set_result("ok")
    rs.prepare_for_next_attempt()
    assert rs.outcome is None
    assert rs.next_action is None
    assert rs.attempt_number == 2


def test_retry_call_state_get_fn_name_uses_retry_object_name_when_no_fn():
    r = Retrying(name="hello")
    rs = RetryCallState(r, fn=None, args=(), kwargs={})
    assert rs.get_fn_name() == "hello"


def test_retry_call_state_get_fn_name_uses_callback_name():
    def myfunc():
        pass

    r = Retrying()
    rs = RetryCallState(r, fn=myfunc, args=(), kwargs={})
    assert rs.get_fn_name().endswith("myfunc")


def test_attempt_manager_swallow_exception_and_store_it():
    rs = make_retry_state()
    am = AttemptManager(rs)
    with am:
        raise RuntimeError("boom")
    assert rs.outcome is not None
    assert isinstance(rs.outcome.exception(), RuntimeError)


def test_attempt_manager_sets_none_result_on_normal_exit():
    rs = make_retry_state()
    am = AttemptManager(rs)
    with am:
        pass
    assert rs.outcome is not None
    assert rs.outcome.result() is None


@pytest.mark.asyncio
async def test_attempt_manager_async_exit_swallow_exception():
    rs = make_retry_state()
    am = AttemptManager(rs)
    async with am:
        raise RuntimeError("boom")
    assert rs.outcome is not None
    assert isinstance(rs.outcome.exception(), RuntimeError)


def test_base_retrying_str_and_repr():
    r = Retrying(name="n")
    assert str(r) == "n"
    assert "Retrying object" in repr(r)


def test_base_retrying_copy_preserves_and_overrides_fields():
    r = Retrying(name="a", enabled=False)
    r2 = r.copy(name="b")
    assert r2 is not r
    assert r2._name == "b"
    assert r2.enabled is False


def test_base_retrying_getstate_excludes_local():
    r = Retrying(name="x")
    state = r.__getstate__()
    assert "_local" not in state
    assert state["_name"] == "x"


def test_base_retrying_setstate_restores_local():
    r = Retrying(name="x")
    state = r.__getstate__()
    r2 = Retrying()
    r2.__setstate__(state)
    assert r2._name == "x"
    assert hasattr(r2, "_local")


def test_begin_initializes_statistics():
    r = Retrying()
    r.begin()
    stats = r.statistics
    assert stats["attempt_number"] == 1
    assert stats["idle_for"] == 0
    assert "start_time" in stats


def test_begin_clears_previous_statistics():
    r = Retrying()
    r.statistics["x"] = 1
    r.begin()
    assert "x" not in r.statistics


def test_run_wait_sets_zero_when_wait_is_falsy():
    r = Retrying(wait=None)
    rs = make_retry_state(r)
    r._run_wait(rs)
    assert rs.upcoming_sleep == 0.0


def test_run_wait_calls_wait_strategy():
    r = Retrying(wait=wait_fixed(2))
    rs = make_retry_state(r)
    rs.outcome = Future.construct(1, "ok", False)
    r._run_wait(rs)
    assert rs.upcoming_sleep == 2.0


def test_run_stop_sets_delay_since_first_attempt_and_result():
    r = Retrying(stop=stop_after_attempt(1))
    rs = make_retry_state(r)
    rs.outcome = Future.construct(1, "ok", False)
    rs.outcome_timestamp = rs.start_time + 1
    r._run_stop(rs)
    assert r.statistics["delay_since_first_attempt"] == 1.0
    assert r.iter_state.stop_run_result is True


def test_begin_iter_on_first_attempt_adds_before_and_attempt():
    r = Retrying(before=before_nothing)
    rs = make_retry_state(r)
    r._begin_iter(rs)
    assert len(r.iter_state.actions) == 2
    assert isinstance(r.iter_state.actions[-1](rs), DoAttempt)


def test_begin_iter_on_result_not_retry_returns_result():
    r = Retrying(retry=retry_never)
    rs = make_retry_state(r)
    rs.outcome = Future.construct(1, "x", False)
    r._begin_iter(rs)
    out = r.iter(rs)
    assert out == "x"


def test_begin_iter_explicit_tryagain_sets_retry_path():
    r = Retrying(stop=stop_after_attempt(1), retry=retry_never)
    rs = make_retry_state(r)
    err = TryAgain("again")
    fut = Future.construct(1, err, True)
    rs.outcome = fut
    r._begin_iter(rs)
    assert r.iter_state.is_explicit_retry is True


def test_retry_error_str():
    fut = Future.construct(1, ValueError("bad"), True)
    err = RetryError(fut)
    assert "RetryError" in str(err)


def test_retry_error_reraise_non_failed_raises_self():
    fut = Future.construct(1, "ok", False)
    err = RetryError(fut)
    with pytest.raises(RetryError):
        err.reraise()


def test_retry_error_reraise_raises_original_exception():
    fut = Future.construct(1, ValueError("bad"), True)
    err = RetryError(fut)
    with pytest.raises(ValueError):
        err.reraise()


def test_retry_error_reraise_with_tryagain_surfaces_cause():
    try:
        try:
            raise ValueError("root")
        except ValueError as e:
            raise TryAgain() from e
    except TryAgain as e:
        fut = Future.construct(1, e, True)
    err = RetryError(fut)
    with pytest.raises(ValueError):
        err.reraise()


def test_before_log_emits_expected_message():
    logger = Logger()
    fn = before_log(logger, 20)
    rs = make_retry_state()
    rs.attempt_number = 3
    rs.fn = lambda: None
    fn(rs)
    assert logger.calls[0][0] == 20
    assert "3rd" in logger.calls[0][1]


def test_after_log_emits_expected_message():
    logger = Logger()
    fn = after_log(logger, 10)
    rs = make_retry_state()
    rs.attempt_number = 2
    rs.set_result("ok")
    fn(rs)
    assert logger.calls[0][0] == 10
    assert "2nd" in logger.calls[0][1]


def test_before_sleep_log_on_result():
    logger = Logger()
    fn = before_sleep_log(logger, 30)
    rs = make_retry_state()
    rs.set_result("ok")
    rs.next_action = RetryAction(1.23)
    fn(rs)
    assert "returned ok" in logger.calls[0][1]


def test_before_sleep_log_on_exception_with_exc_info():
    logger = Logger()
    fn = before_sleep_log(logger, 30, exc_info=True)
    rs = make_retry_state()
    rs.set_exception((ValueError, ValueError("bad"), None))
    rs.next_action = RetryAction(0.1)
    fn(rs)
    assert "raised ValueError: bad" in logger.calls[0][1]


def test_before_sleep_log_requires_outcome():
    logger = Logger()
    fn = before_sleep_log(logger, 30)
    rs = make_retry_state()
    rs.next_action = RetryAction(0.1)
    with pytest.raises(RuntimeError):
        fn(rs)


def test_before_sleep_log_requires_next_action():
    logger = Logger()
    fn = before_sleep_log(logger, 30)
    rs = make_retry_state()
    rs.set_result("ok")
    with pytest.raises(RuntimeError):
        fn(rs)


def test_retry_if_exception_type_matches():
    r = retry_if_exception_type(ValueError)
    rs = make_retry_state(outcome=Future.construct(1, ValueError("x"), True))
    assert r(rs) is True


def test_retry_if_exception_type_non_match():
    r = retry_if_exception_type(ValueError)
    rs = make_retry_state(outcome=Future.construct(1, TypeError("x"), True))
    assert r(rs) is False


def test_retry_if_not_exception_type_matches_nonmatching_exception():
    r = retry_if_not_exception_type(ValueError)
    rs = make_retry_state(outcome=Future.construct(1, TypeError("x"), True))
    assert r(rs) is True


def test_retry_unless_exception_type_returns_true_for_non_exception_result():
    r = retry_unless_exception_type(ValueError)
    rs = make_retry_state(outcome=Future.construct(1, "ok", False))
    assert r(rs) is True


def test_retry_if_exception_message_exact():
    r = retry_if_exception_message(message="bad")
    rs = make_retry_state(outcome=Future.construct(1, ValueError("bad"), True))
    assert r(rs) is True


def test_retry_if_exception_message_regex():
    r = retry_if_exception_message(match=r"^b.*")
    rs = make_retry_state(outcome=Future.construct(1, ValueError("bad"), True))
    assert r(rs) is True


def test_retry_if_not_exception_message_inverts_match():
    r = retry_if_not_exception_message(message="bad")
    rs = make_retry_state(outcome=Future.construct(1, ValueError("nope"), True))
    assert r(rs) is True


def test_retry_if_result_matches_predicate():
    r = retry_if_result(lambda x: x == 3)
    rs = make_retry_state(outcome=Future.construct(1, 3, False))
    assert r(rs) is True


def test_retry_if_not_result_inverts_predicate():
    r = retry_if_not_result(lambda x: x == 3)
    rs = make_retry_state(outcome=Future.construct(1, 4, False))
    assert r(rs) is True


def test_retry_if_exception_cause_type_detects_nested_cause():
    inner = KeyError("inner")
    outer = ValueError("outer")
    outer.__cause__ = inner
    rs = make_retry_state(outcome=Future.construct(1, outer, True))
    r = retry_if_exception_cause_type(KeyError)
    assert r(rs) is True


def test_retry_if_exception_cause_type_handles_cycle():
    exc = ValueError("x")
    exc.__cause__ = exc
    rs = make_retry_state(outcome=Future.construct(1, exc, True))
    r = retry_if_exception_cause_type(KeyError)
    assert r(rs) is False


def test_stop_after_attempt_threshold():
    r = stop_after_attempt(3)
    rs = make_retry_state(attempt_number=3)
    assert r(rs) is True


def test_stop_after_delay_uses_seconds_since_start():
    r = stop_after_delay(2)
    rs = make_retry_state()
    rs.outcome = Future.construct(1, "x", False)
    rs.outcome_timestamp = rs.start_time + 2.5
    assert r(rs) is True


def test_stop_after_delay_requires_seconds_since_start():
    r = stop_after_delay(2)
    rs = make_retry_state()
    with pytest.raises(RuntimeError):
        r(rs)


def test_stop_before_delay_accounts_for_upcoming_sleep():
    r = stop_before_delay(2)
    rs = make_retry_state()
    rs.outcome = Future.construct(1, "x", False)
    rs.outcome_timestamp = rs.start_time + 1.5
    rs.upcoming_sleep = 0.6
    assert r(rs) is True


def test_stop_when_event_set():
    event = threading.Event()
    r = stop_when_event_set(event)
    rs = make_retry_state()
    assert r(rs) is False
    event.set()
    assert r(rs) is True


def test_stop_any_and_stop_all_composition():
    rs = make_retry_state(attempt_number=2)
    a = stop_after_attempt(2)
    b = stop_never
    assert stop_any(a, b)(rs) is True
    assert stop_all(a, b)(rs) is False


def test_wait_fixed_and_none():
    rs = make_retry_state()
    assert wait_fixed(3)(rs) == 3.0
    assert wait_none()(rs) == 0.0


def test_wait_random_is_within_bounds(monkeypatch):
    monkeypatch.setattr("random.random", lambda: 0.25)
    rs = make_retry_state()
    assert wait_random(4, 8)(rs) == 5.0


def test_wait_combine_sums_waits():
    rs = make_retry_state()
    w = wait_combine(wait_fixed(1), wait_fixed(2))
    assert w(rs) == 3.0


def test_wait_chain_uses_attempt_number_and_last_strategy():
    rs = make_retry_state(attempt_number=3)
    w = wait_chain(wait_fixed(1), wait_fixed(2))
    assert w(rs) == 2.0


def test_wait_chain_requires_strategy():
    with pytest.raises(ValueError):
        wait_chain()


def test_wait_incrementing_caps_and_never_goes_negative():
    rs = make_retry_state(attempt_number=3)
    assert wait_incrementing(start=1, increment=2, max=4)(rs) == 4.0


def test_wait_exponential_basic():
    rs = make_retry_state(attempt_number=3)
    assert wait_exponential(multiplier=1, exp_base=2, max=100)(rs) == 4.0


def test_wait_exponential_respects_min():
    rs = make_retry_state(attempt_number=1)
    assert wait_exponential(multiplier=0.1, min=2)(rs) == 2.0


def test_wait_random_exponential_with_monkeypatched_uniform(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda a, b: b)
    rs = make_retry_state(attempt_number=3)
    assert wait_random_exponential(multiplier=1, max=10)(rs) == 4.0


def test_wait_exponential_jitter_warns_on_initial(monkeypatch):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        we = wait_exponential_jitter(initial=2)
    assert we.multiplier == 2
    assert any("deprecated" in str(x.message) for x in w)


def test_wait_exponential_jitter_rejects_initial_and_multiplier_together():
    with pytest.raises(ValueError):
        wait_exponential_jitter(initial=2, multiplier=3)


def test_wait_exponential_jitter_uses_jitter(monkeypatch):
    monkeypatch.setattr("random.uniform", lambda a, b: b)
    rs = make_retry_state(attempt_number=2)
    assert wait_exponential_jitter(multiplier=1, jitter=0.5, max=10)(rs) == 2.5


def test_sleep_using_event_calls_wait(monkeypatch):
    event = threading.Event()
    called = {}

    def wait(timeout=None):
        called["timeout"] = timeout

    monkeypatch.setattr(event, "wait", wait)
    s = sleep_using_event(event)
    s(1.5)
    assert called["timeout"] == 1.5


def test_retry_decorator_sync_success_no_retries():
    calls = {"n": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_none())
    def f():
        calls["n"] += 1
        return 5

    assert f() == 5
    assert calls["n"] == 1
    assert f.retry is not None
    assert isinstance(f.statistics, dict)


def test_retry_decorator_sync_retries_then_succeeds():
    calls = {"n": 0}

    @retry(stop=stop_after_attempt(5), wait=wait_none())
    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("x")
        return "ok"

    assert f() == "ok"
    assert calls["n"] == 3


def test_retry_decorator_sync_reraise_false_raises_retry_error():
    calls = {"n": 0}

    @retry(stop=stop_after_attempt(2), wait=wait_none(), reraise=False)
    def f():
        calls["n"] += 1
        raise ValueError("x")

    with pytest.raises(RetryError):
        f()
    assert calls["n"] == 2


def test_retry_decorator_sync_reraise_true_raises_original():
    calls = {"n": 0}

    @retry(stop=stop_after_attempt(2), wait=wait_none(), reraise=True)
    def f():
        calls["n"] += 1
        raise ValueError("x")

    with pytest.raises(ValueError):
        f()
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_retry_decorator_async_function():
    calls = {"n": 0}

    @retry(stop=stop_after_attempt(3), wait=wait_none())
    async def f():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("x")
        return 9

    assert await f() == 9
    assert calls["n"] == 2
    assert inspect.iscoroutinefunction(f)


def test_retry_disabled_bypasses_retry_logic():
    calls = {"n": 0}
    r = Retrying(enabled=False, stop=stop_after_attempt(3))

    @r.wraps
    def f():
        calls["n"] += 1
        return 7

    assert f() == 7
    assert calls["n"] == 1


def test_retry_with_returns_modified_decorator():
    r = Retrying(stop=stop_after_attempt(2), wait=wait_none())

    @r.wraps
    def f():
        return 1

    g = f.retry_with(stop=stop_after_attempt(5))
    assert callable(g)
    assert g() == 1


def test_statistics_shared_across_wrapped_function_call():
    @retry(stop=stop_after_attempt(2), wait=wait_none())
    def f():
        return 1

    f()
    assert "attempt_number" in f.statistics


def test_wraps_preserves_name_and_docstring():
    def f(a: int) -> int:
        """doc"""
        return a

    wrapped = Retrying().wraps(f)
    assert wrapped.__name__ == "f"
    assert wrapped.__doc__ == "doc"
    assert wrapped.__wrapped__ is f


def test_retry_function_selects_sync_retrying():
    @retry
    def f():
        return 1

    assert isinstance(f.retry, Retrying)


@pytest.mark.asyncio
async def test_retry_function_selects_async_retrying_for_async_callable():
    @retry
    async def f():
        return 1

    assert isinstance(f.retry, tenacity.AsyncRetrying)
    assert await f() == 1


def test_retry_function_warns_when_passed_retry_strategy_as_callable():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wrapped = retry()(retry_never)
    assert callable(wrapped)
    assert any("probably hang indefinitely" in str(x.message) for x in w)


def test_wait_and_stop_operator_composition_with_plain_callables():
    rs = make_retry_state()

    def r1(state):
        return True

    def r2(state):
        return False

    assert (retry_any(r1, r2))(rs) is True
    assert (retry_all(r1, r2))(rs) is False


def test_wait_sum_with_builtin_sum_identity():
    rs = make_retry_state()
    waits = [wait_fixed(1), wait_fixed(2)]
    assert sum(waits)(rs) == 3.0


def test_base_retrying_iteration_context_manager_yields_attempt_manager():
    r = Retrying(stop=stop_after_attempt(1), wait=wait_none())
    it = iter(r)
    am = next(it)
    assert isinstance(am, AttemptManager)


def test_base_retrying_repr_contains_configuration():
    r = Retrying(name="abc", wait=wait_fixed(1))
    text = repr(r)
    assert "wait=" in text and "name='abc'" in text


def test_future_failed_property_with_result_false():
    fut = Future.construct(1, 3, False)
    assert fut.failed is False


def test_stop_and_retry_never_always():
    rs = make_retry_state()
    assert stop_never(rs) is False
    assert retry_never(rs) is False
    assert retry_always(rs) is True
