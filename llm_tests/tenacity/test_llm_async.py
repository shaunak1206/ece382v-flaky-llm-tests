"""Tests for async retrying."""
import asyncio
import time

import pytest

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry,
    retry_all,
    retry_any,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_fixed,
    wait_random,
)
from tenacity.asyncio import retry as async_retry_mod

pytestmark = pytest.mark.asyncio


async def no_sleep(seconds):
    return None


class AsyncFlaky:
    def __init__(self, failures, exc=ValueError):
        self.failures = failures
        self.exc = exc
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        await asyncio.sleep(0)
        if self.calls <= self.failures:
            raise self.exc("fail %d" % self.calls)
        return "ok"


async def test_async_retry_decorator_succeeds():
    flaky = AsyncFlaky(2)

    @retry(stop=stop_after_attempt(5))
    async def fn():
        return await flaky()

    assert await fn() == "ok"
    assert flaky.calls == 3




async def test_async_retry_gives_up_with_retry_error():
    flaky = AsyncFlaky(100)

    @retry(stop=stop_after_attempt(3))
    async def fn():
        return await flaky()

    with pytest.raises(RetryError):
        await fn()
    assert flaky.calls == 3


async def test_async_retry_reraise():
    flaky = AsyncFlaky(100, exc=KeyError)

    @retry(stop=stop_after_attempt(2), reraise=True)
    async def fn():
        return await flaky()

    with pytest.raises(KeyError):
        await fn()




async def test_async_retrying_object_call():
    flaky = AsyncFlaky(1)
    r = AsyncRetrying(stop=stop_after_attempt(3), sleep=no_sleep)
    assert await r(flaky) == "ok"




async def test_async_for_context_manager():
    flaky = AsyncFlaky(2)
    result = None
    async for attempt in AsyncRetrying(stop=stop_after_attempt(5), sleep=no_sleep):
        with attempt:
            result = await flaky()
    assert result == "ok"
    assert flaky.calls == 3






async def test_async_custom_sleep_gets_wait_values():
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    @retry(wait=wait_fixed(4), stop=stop_after_attempt(4), sleep=fake_sleep)
    async def fn():
        raise ValueError

    with pytest.raises(RetryError):
        await fn()
    assert slept == [4, 4, 4]


async def test_async_retry_if_result():
    seq = iter([0, 0, 5])

    @retry(retry=retry_if_result(lambda x: x == 0), sleep=no_sleep)
    async def fn():
        return next(seq)

    assert await fn() == 5




async def test_async_callbacks_may_be_coroutines():
    events = []

    async def before(rs):
        events.append(("before", rs.attempt_number))

    async def after(rs):
        events.append(("after", rs.attempt_number))

    async def before_sleep(rs):
        events.append(("bs", rs.attempt_number))

    @retry(
        stop=stop_after_attempt(2),
        before=before,
        after=after,
        before_sleep=before_sleep,
        sleep=no_sleep,
    )
    async def fn():
        raise ValueError

    with pytest.raises(RetryError):
        await fn()
    assert events == [
        ("before", 1), ("after", 1), ("bs", 1),
        ("before", 2), ("after", 2),
    ]




async def test_async_retry_predicate_coroutine():
    async def should_retry(exc):
        return isinstance(exc, ValueError)

    flaky = AsyncFlaky(2)

    @retry(retry=async_retry_mod.retry_if_exception(should_retry), sleep=no_sleep)
    async def fn():
        return await flaky()

    assert await fn() == "ok"








async def test_async_retry_with():
    flaky = AsyncFlaky(100)

    @retry(stop=stop_after_attempt(10), sleep=no_sleep)
    async def fn():
        return await flaky()

    with pytest.raises(RetryError):
        await fn.retry_with(stop=stop_after_attempt(2))()
    assert flaky.calls == 2




async def test_async_real_sleep_timing():
    flaky = AsyncFlaky(2)

    @retry(wait=wait_fixed(0.1), stop=stop_after_attempt(5))
    async def fn():
        return await flaky()

    start = time.time()
    assert await fn() == "ok"
    elapsed = time.time() - start
    assert 0.2 <= elapsed < 0.6




async def test_async_concurrent_calls_have_independent_state():
    async def make(n):
        calls = {"c": 0}

        @retry(stop=stop_after_attempt(10), sleep=no_sleep)
        async def fn():
            calls["c"] += 1
            await asyncio.sleep(0)
            if calls["c"] < n:
                raise ValueError
            return calls["c"]

        return await fn()

    results = await asyncio.gather(make(2), make(4), make(6))
    assert results == [2, 4, 6]
