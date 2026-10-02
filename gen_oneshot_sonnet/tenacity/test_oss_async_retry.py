"""Tests for async retries: @retry on coroutines and AsyncRetrying."""

import asyncio
import inspect

import pytest

import tenacity.asyncio.retry as aretry
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    TryAgain,
    retry,
    retry_always,
    retry_if_exception_type,
    retry_never,
    stop_after_attempt,
    wait_fixed,
)

pytestmark = pytest.mark.asyncio


class AsyncFlaky:
    """Coroutine factory that fails `fail_times` times, then succeeds."""

    def __init__(self, fail_times, exc=ValueError, result="ok"):
        self.fail_times = fail_times
        self.exc = exc
        self.result = result
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1
        await asyncio.sleep(0)
        if self.calls <= self.fail_times:
            raise self.exc("failure %d" % self.calls)
        return self.result


class AsyncSleepRecorder:
    def __init__(self):
        self.sleeps = []

    async def __call__(self, seconds):
        self.sleeps.append(seconds)


def done_state(result=None, exc=None):
    state = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    if exc is not None:
        state.set_exception((type(exc), exc, None))
    else:
        state.set_result(result)
    return state


async def test_async_retry_decorator_succeeds_and_supports_retry_with():
    flaky = AsyncFlaky(2)

    @retry(stop=stop_after_attempt(5), sleep=AsyncSleepRecorder())
    async def fn():
        """Async doc."""
        return await flaky()

    assert inspect.iscoroutinefunction(fn)
    assert await fn() == "ok"
    assert flaky.calls == 3
    assert fn.__name__ == "fn"
    assert fn.__doc__ == "Async doc."
    assert isinstance(fn.retry, AsyncRetrying)

    flaky2 = AsyncFlaky(3)

    @retry(stop=stop_after_attempt(2), sleep=AsyncSleepRecorder())
    async def short():
        return await flaky2()

    with pytest.raises(RetryError):
        await short()
    patient = short.retry_with(stop=stop_after_attempt(5))
    assert await patient() == "ok"


async def test_async_retry_gives_up_or_reraises():
    flaky = AsyncFlaky(10)

    @retry(stop=stop_after_attempt(3), sleep=AsyncSleepRecorder())
    async def fn():
        return await flaky()

    with pytest.raises(RetryError) as excinfo:
        await fn()
    assert flaky.calls == 3
    assert excinfo.value.last_attempt.attempt_number == 3
    assert isinstance(excinfo.value.last_attempt.exception(), ValueError)

    reraising = AsyncFlaky(10)

    @retry(stop=stop_after_attempt(2), reraise=True, sleep=AsyncSleepRecorder())
    async def fn_reraise():
        return await reraising()

    with pytest.raises(ValueError, match="failure 2"):
        await fn_reraise()


async def test_async_retrying_object_calls_coroutine_and_plain_functions():
    retrying = AsyncRetrying(stop=stop_after_attempt(4), sleep=AsyncSleepRecorder())
    calls = []

    async def add(a, b=0):
        calls.append((a, b))
        if len(calls) < 3:
            raise OSError("not yet")
        return a + b

    assert await retrying(add, 1, b=2) == 3
    assert calls == [(1, 2)] * 3

    sync_calls = []

    def sync_fn():
        sync_calls.append(1)
        if len(sync_calls) < 2:
            raise ValueError("again")
        return "sync ok"

    assert await retrying(sync_fn) == "sync ok"


async def test_async_sleep_receives_wait_values_and_statistics():
    recorder = AsyncSleepRecorder()
    flaky = AsyncFlaky(2)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1.5), sleep=recorder)
    async def fn():
        return await flaky()

    assert await fn() == "ok"
    assert recorder.sleeps == [1.5, 1.5]
    assert fn.statistics["attempt_number"] == 3
    assert fn.statistics["idle_for"] == 3.0


async def test_async_iteration_with_attempt_context_managers():
    calls = 0
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(5), sleep=AsyncSleepRecorder()
    ):
        with attempt:
            calls += 1
            if calls < 3:
                raise ValueError("again")
    assert calls == 3

    attempts = 0
    with pytest.raises(RetryError):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(2), sleep=AsyncSleepRecorder()
        ):
            async with attempt:
                attempts += 1
                raise ValueError("always")
    assert attempts == 2


async def test_async_callbacks_are_awaited_in_order():
    events = []

    async def before(state):
        events.append(("before", state.attempt_number))

    async def after(state):
        events.append(("after", state.attempt_number))

    async def before_sleep(state):
        events.append(("before_sleep", state.attempt_number))

    retrying = AsyncRetrying(
        stop=stop_after_attempt(3),
        before=before,
        after=after,
        before_sleep=before_sleep,
        sleep=AsyncSleepRecorder(),
    )
    flaky = AsyncFlaky(10)

    async def fn():
        return await flaky()

    with pytest.raises(RetryError):
        await retrying(fn)
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


async def test_async_retrying_accepts_plain_callbacks_and_async_error_callback():
    events = []
    flaky = AsyncFlaky(2)

    async def fn():
        return await flaky()

    retrying = AsyncRetrying(
        stop=stop_after_attempt(5),
        before=lambda state: events.append("before"),
        after=lambda state: events.append("after"),
        before_sleep=lambda state: events.append("before_sleep"),
        sleep=AsyncSleepRecorder(),
    )
    assert await retrying(fn) == "ok"
    assert events.count("before") == 3
    assert events.count("after") == 2
    assert events.count("before_sleep") == 2

    async def fallback(state):
        return "fallback"

    giving_up = AsyncRetrying(
        stop=stop_after_attempt(2),
        retry_error_callback=fallback,
        sleep=AsyncSleepRecorder(),
    )
    failing = AsyncFlaky(10)

    async def always_fails():
        return await failing()

    assert await giving_up(always_fails) == "fallback"


async def test_async_retry_conditions_and_combinators():
    async def is_none(value):
        return value is None

    async def is_value_error(exc):
        return isinstance(exc, ValueError)

    on_none = aretry.retry_if_result(is_none)
    assert await on_none(done_state(result=None)) is True
    assert await on_none(done_state(result=1)) is False
    assert await on_none(done_state(exc=ValueError())) is False

    on_value_error = aretry.retry_if_exception(is_value_error)
    assert await on_value_error(done_state(exc=ValueError())) is True
    assert await on_value_error(done_state(exc=KeyError())) is False
    assert await on_value_error(done_state(result=1)) is False

    any_of = aretry.retry_any(on_none, retry_if_exception_type(KeyError))
    assert await any_of(done_state(result=None)) is True
    assert await any_of(done_state(exc=KeyError())) is True
    assert await any_of(done_state(result=5)) is False

    all_of = aretry.retry_all(on_none, retry_always)
    assert await all_of(done_state(result=None)) is True
    assert await all_of(done_state(result=5)) is False

    assert isinstance(on_none | retry_if_exception_type(KeyError), aretry.retry_any)
    assert isinstance(on_none & retry_always, aretry.retry_all)

    values = iter([None, None, 3])

    async def produce():
        return next(values)

    retrying = AsyncRetrying(
        retry=on_none, stop=stop_after_attempt(5), sleep=AsyncSleepRecorder()
    )
    assert await retrying(produce) == 3


async def test_async_disabled_and_not_iterable():
    flaky = AsyncFlaky(3)

    async def fn():
        return await flaky()

    retrying = AsyncRetrying(
        enabled=False, stop=stop_after_attempt(5), sleep=AsyncSleepRecorder()
    )
    with pytest.raises(ValueError):
        await retrying(fn)
    assert flaky.calls == 1

    with pytest.raises(TypeError):
        iter(AsyncRetrying())


async def test_async_try_again_and_non_matching_exceptions():
    state = {"n": 0}

    @retry(retry=retry_never, stop=stop_after_attempt(5), sleep=AsyncSleepRecorder())
    async def fn():
        state["n"] += 1
        if state["n"] < 3:
            raise TryAgain
        return "done"

    assert await fn() == "done"
    assert state["n"] == 3

    @retry(
        retry=retry_if_exception_type(ValueError),
        stop=stop_after_attempt(5),
        sleep=AsyncSleepRecorder(),
    )
    async def wrong():
        raise TypeError("nope")

    with pytest.raises(TypeError):
        await wrong()


async def test_async_concurrent_calls_are_independent():
    async def run(failures):
        flaky = AsyncFlaky(failures)

        @retry(stop=stop_after_attempt(10), sleep=AsyncSleepRecorder())
        async def fn():
            return await flaky()

        return await fn(), flaky.calls

    results = await asyncio.gather(run(0), run(1), run(3))
    assert results == [("ok", 1), ("ok", 2), ("ok", 4)]


async def test_async_default_sleep_uses_asyncio():
    flaky = AsyncFlaky(2)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
    async def fn():
        return await flaky()

    assert await fn() == "ok"
    assert flaky.calls == 3
