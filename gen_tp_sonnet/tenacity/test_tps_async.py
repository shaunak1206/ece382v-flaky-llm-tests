import importlib
import sys

import pytest

from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    _utils,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)
from tenacity.asyncio import AsyncRetrying

async_retry_mod = importlib.import_module("tenacity.asyncio.retry")


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


@pytest.mark.asyncio
async def test_async_decorator_retries_until_success_with_waits():
    sleeps = []
    calls = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(2), sleep=fake_sleep)
    async def flaky(x):
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("transient")
        return x * 2

    assert await flaky(21) == 42
    assert len(calls) == 3
    assert sleeps == [2, 2]


@pytest.mark.asyncio
async def test_async_decorator_exhaustion_raises_retry_error_or_reraises():
    async def fake_sleep(seconds):
        return None

    @retry(stop=stop_after_attempt(3), sleep=fake_sleep)
    async def always_fails():
        raise ValueError("boom")

    with pytest.raises(RetryError) as info:
        await always_fails()
    assert info.value.last_attempt.attempt_number == 3

    @retry(stop=stop_after_attempt(2), reraise=True, sleep=fake_sleep)
    async def reraises():
        raise KeyError("boom")

    with pytest.raises(KeyError):
        await reraises()


@pytest.mark.asyncio
async def test_async_retrying_call_and_async_for_context_manager():
    async def fake_sleep(seconds):
        return None

    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("transient")
        return "ok"

    controller = AsyncRetrying(stop=stop_after_attempt(5), sleep=fake_sleep)
    assert await controller(flaky) == "ok"
    assert len(calls) == 3
    assert controller.statistics["attempt_number"] == 3

    attempts = 0
    async for attempt in AsyncRetrying(stop=stop_after_attempt(4), sleep=fake_sleep):
        with attempt:
            attempts += 1
            if attempts < 3:
                raise ValueError("transient")
    assert attempts == 3

    with pytest.raises(RetryError):
        async for attempt in AsyncRetrying(stop=stop_after_attempt(2), sleep=fake_sleep):
            with attempt:
                raise ValueError("always")


@pytest.mark.asyncio
async def test_async_retry_strategies_accept_coroutine_predicates():
    async def is_none(value):
        return value is None

    async def is_value_error(exc):
        return isinstance(exc, ValueError)

    by_result = async_retry_mod.retry_if_result(is_none)
    assert await by_result(make_state(result=None))
    assert not await by_result(make_state(result=1))

    by_exception = async_retry_mod.retry_if_exception(is_value_error)
    assert await by_exception(make_state(exc=ValueError("x")))
    assert not await by_exception(make_state(exc=KeyError("x")))

    either = async_retry_mod.retry_any(by_result, retry_if_exception_type(KeyError))
    assert await either(make_state(result=None))
    assert await either(make_state(exc=KeyError("x")))
    assert not await either(make_state(result=5))

    both = async_retry_mod.retry_all(
        async_retry_mod.retry_if_result(is_none),
        async_retry_mod.retry_if_result(is_none),
    )
    assert await both(make_state(result=None))
    assert not await both(make_state(result=1))

    calls = []

    async def fake_sleep(seconds):
        return None

    @retry(
        retry=async_retry_mod.retry_if_result(is_none),
        stop=stop_after_attempt(5),
        sleep=fake_sleep,
    )
    async def eventually():
        calls.append(1)
        return None if len(calls) < 3 else "done"

    assert await eventually() == "done"
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_async_callbacks_and_wrap_to_async_func():
    events = []

    async def fake_sleep(seconds):
        return None

    async def before_cb(rs):
        events.append(("before", rs.attempt_number))

    def after_cb(rs):
        events.append(("after", rs.attempt_number))

    async def before_sleep_cb(rs):
        events.append(("sleep", rs.attempt_number))

    @retry(
        stop=stop_after_attempt(2),
        sleep=fake_sleep,
        before=before_cb,
        after=after_cb,
        before_sleep=before_sleep_cb,
    )
    async def fails():
        raise ValueError("boom")

    with pytest.raises(RetryError):
        await fails()

    assert [e for e in events if e[0] == "before"] == [("before", 1), ("before", 2)]
    assert len([e for e in events if e[0] == "after"]) == 2
    assert [e for e in events if e[0] == "sleep"] == [("sleep", 1)]

    def sync_fn(x):
        return x + 1

    wrapped = _utils.wrap_to_async_func(sync_fn)
    assert _utils.is_coroutine_callable(wrapped)
    assert await wrapped(1) == 2
