import asyncio
import inspect

import pytest

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry,
    retry_always,
    retry_never,
    stop_after_attempt,
    wait_fixed,
)
from tenacity.asyncio import retry as aretry


def make_state(exc=None, result=None):
    rs = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    if exc is not None:
        rs.set_exception((type(exc), exc, None))
    else:
        rs.set_result(result)
    return rs


class SleepRecorder:
    def __init__(self):
        self.sleeps = []

    async def __call__(self, seconds):
        self.sleeps.append(seconds)


def test_retry_decorator_wraps_coroutine_functions_with_async_retrying():
    recorder = SleepRecorder()
    calls = []

    @retry(stop=stop_after_attempt(4), wait=wait_fixed(3), sleep=recorder)
    async def fetch():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("down")
        return "payload"

    assert inspect.iscoroutinefunction(fetch)
    assert isinstance(fetch.retry, AsyncRetrying)
    assert asyncio.run(fetch()) == "payload"
    assert len(calls) == 3
    assert recorder.sleeps == [3.0, 3.0]
    assert fetch.statistics["attempt_number"] == 3


def test_async_retrying_stop_reraise_and_error_callback():
    async def always_fails():
        raise KeyError("missing")

    async def scenario():
        r = AsyncRetrying(stop=stop_after_attempt(2), sleep=SleepRecorder())
        with pytest.raises(RetryError) as info:
            await r(always_fails)
        assert info.value.last_attempt.attempt_number == 2
        assert isinstance(info.value.__cause__, KeyError)

        reraising = AsyncRetrying(
            stop=stop_after_attempt(2), sleep=SleepRecorder(), reraise=True
        )
        with pytest.raises(KeyError):
            await reraising(always_fails)

        async def on_give_up(rs):
            return f"gave up after {rs.attempt_number}"

        with_callback = AsyncRetrying(
            stop=stop_after_attempt(3),
            sleep=SleepRecorder(),
            retry_error_callback=on_give_up,
        )
        return await with_callback(always_fails)

    assert asyncio.run(scenario()) == "gave up after 3"


def test_async_predicates_drive_retries():
    async def is_small(value):
        return value < 3

    async def is_value_error(exc):
        return isinstance(exc, ValueError)

    counter = {"n": 0}

    async def count_up():
        counter["n"] += 1
        return counter["n"]

    async def raises_twice():
        counter["n"] += 1
        if counter["n"] <= 2:
            raise ValueError("again")
        return "ok"

    async def raises_key_error():
        counter["n"] += 1
        raise KeyError("fatal")

    async def scenario():
        by_result = AsyncRetrying(
            retry=aretry.retry_if_result(is_small),
            stop=stop_after_attempt(10),
            sleep=SleepRecorder(),
        )
        first = await by_result(count_up)

        counter["n"] = 0
        by_exc = AsyncRetrying(
            retry=aretry.retry_if_exception(is_value_error),
            stop=stop_after_attempt(10),
            sleep=SleepRecorder(),
        )
        second = await by_exc(raises_twice)

        counter["n"] = 0
        with pytest.raises(KeyError):
            await by_exc(raises_key_error)
        return first, second, counter["n"]

    assert asyncio.run(scenario()) == (3, "ok", 1)


def test_async_retry_any_and_all_evaluate_sync_and_async_members():
    async def yes(_):
        return True

    async def scenario():
        rs = make_state(result=1)
        any_true = aretry.retry_any(retry_never, retry_always)
        any_false = aretry.retry_any(retry_never, retry_never)
        all_true = aretry.retry_all(retry_always, aretry.retry_if_result(yes))
        all_false = aretry.retry_all(retry_always, retry_never)
        return (
            await any_true(rs),
            await any_false(rs),
            await all_true(rs),
            await all_false(rs),
        )

    assert asyncio.run(scenario()) == (True, False, True, False)


def test_async_for_iteration_retries_the_block_and_raises_on_exhaustion():
    async def scenario():
        count = 0
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(5), sleep=SleepRecorder()
        ):
            with attempt:
                count += 1
                if count < 3:
                    raise ValueError("again")

        exhausted = 0
        with pytest.raises(RetryError):
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(2), sleep=SleepRecorder()
            ):
                async with attempt:
                    exhausted += 1
                    raise ValueError("always")
        return count, exhausted

    assert asyncio.run(scenario()) == (3, 2)


def test_async_disabled_retrying_runs_function_once():
    calls = []

    async def failing():
        calls.append(1)
        raise ValueError("direct")

    async def scenario():
        r = AsyncRetrying(enabled=False, stop=stop_after_attempt(5))
        with pytest.raises(ValueError, match="direct"):
            await r(failing)
        with pytest.raises(ValueError, match="direct"):
            async for attempt in r:
                with attempt:
                    raise ValueError("direct")
        return len(calls)

    assert asyncio.run(scenario()) == 1


def test_async_hooks_wait_and_stop_may_be_coroutines():
    events = []

    async def before(rs):
        events.append(("before", rs.attempt_number))

    async def after(rs):
        events.append(("after", rs.attempt_number))

    async def before_sleep(rs):
        events.append(("sleep", rs.attempt_number))

    async def wait(rs):
        return 0.5

    async def stop(rs):
        return rs.attempt_number >= 2

    recorder = SleepRecorder()

    async def always_fails():
        raise ValueError("x")

    async def scenario():
        r = AsyncRetrying(
            stop=stop,
            wait=wait,
            sleep=recorder,
            before=before,
            after=after,
            before_sleep=before_sleep,
        )
        with pytest.raises(RetryError):
            await r(always_fails)

    asyncio.run(scenario())
    assert events == [
        ("before", 1),
        ("after", 1),
        ("sleep", 1),
        ("before", 2),
        ("after", 2),
    ]
    assert recorder.sleeps == [0.5]
