import asyncio
import functools
import logging
import sys
from datetime import timedelta

import pytest

import tenacity
from tenacity import (
    RetryAction,
    RetryCallState,
    Retrying,
    _utils,
    after_log,
    after_nothing,
    before_log,
    before_nothing,
    before_sleep_log,
    before_sleep_nothing,
    nap,
    retry_if_result,
    sleep_using_event,
    stop_after_attempt,
    wait_fixed,
)


class FakeLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg))


def no_sleep(seconds):
    return None


def sample_task():
    return None


def test_find_ordinal_and_to_ordinal_cover_special_cases():
    expected = {
        0: "0th",
        1: "1st",
        2: "2nd",
        3: "3rd",
        4: "4th",
        11: "11th",
        12: "12th",
        13: "13th",
        14: "14th",
        20: "20th",
        21: "21st",
        22: "22nd",
        23: "23rd",
        24: "24th",
        101: "101st",
        111: "111th",
        112: "112th",
        113: "113th",
        121: "121st",
    }
    for number, text in expected.items():
        assert _utils.to_ordinal(number) == text
    assert _utils.find_ordinal(2) == "nd"


def test_get_callback_name_for_functions_and_objects():
    name = _utils.get_callback_name(sample_task)
    assert name == f"{sample_task.__module__}.sample_task"
    assert _utils.get_callback_name(len) == "builtins.len"

    class Plain:
        def __call__(self):
            return None

    plain = Plain()
    assert _utils.get_callback_name(plain) == repr(plain)
    partial = functools.partial(sample_task)
    assert _utils.get_callback_name(partial) == repr(partial)


def test_is_coroutine_callable_and_wrap_to_async_func():
    async def coro():
        return "async"

    def sync():
        return "sync"

    class AsyncCallable:
        async def __call__(self):
            return "callable"

    class SomeClass:
        pass

    assert _utils.is_coroutine_callable(coro) is True
    assert _utils.is_coroutine_callable(functools.partial(coro)) is True
    assert _utils.is_coroutine_callable(AsyncCallable()) is True
    assert _utils.is_coroutine_callable(sync) is False
    assert _utils.is_coroutine_callable(SomeClass) is False
    assert _utils.is_coroutine_callable(lambda: None) is False

    assert _utils.wrap_to_async_func(coro) is coro
    wrapped = _utils.wrap_to_async_func(sync)
    assert wrapped is not sync
    assert asyncio.run(wrapped()) == "sync"


def test_before_log_reports_attempt_ordinal():
    logger = FakeLogger()
    rs = RetryCallState(Retrying(), fn=sample_task, args=(), kwargs={})
    rs.attempt_number = 2
    before_log(logger, logging.INFO)(rs)
    assert len(logger.records) == 1
    level, msg = logger.records[0]
    assert level == logging.INFO
    assert msg == (
        f"Starting call to '{_utils.get_callback_name(sample_task)}', "
        "this is the 2nd time calling it."
    )


def test_after_log_reports_elapsed_seconds():
    logger = FakeLogger()
    rs = RetryCallState(Retrying(), fn=sample_task, args=(), kwargs={})
    rs.attempt_number = 3
    rs.set_result(None)
    rs.start_time = 100.0
    rs.outcome_timestamp = 101.5
    after_log(logger, logging.DEBUG)(rs)
    after_log(logger, logging.DEBUG, sec_format="%.2f")(rs)
    assert logger.records[0][0] == logging.DEBUG
    assert logger.records[0][1].endswith(
        "after 1.5(s), this was the 3rd time calling it."
    )
    assert "after 1.50(s)" in logger.records[1][1]

    pending = RetryCallState(Retrying(), fn=sample_task, args=(), kwargs={})
    after_log(logger, logging.DEBUG)(pending)
    assert "after ?(s)" in logger.records[2][1]


def test_before_sleep_log_describes_exceptions_and_results():
    logger = FakeLogger()
    calls = []

    def boom():
        calls.append(1)
        raise ValueError("boom")

    r = Retrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(0.5),
        sleep=no_sleep,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    with pytest.raises(tenacity.RetryError):
        r(boom)
    name = _utils.get_callback_name(boom)
    assert logger.records == [
        (
            logging.WARNING,
            f"Retrying {name} in 0.5 seconds as it raised ValueError: boom.",
        )
    ]

    logger2 = FakeLogger()
    r2 = Retrying(
        stop=stop_after_attempt(2),
        wait=wait_fixed(2),
        sleep=no_sleep,
        retry=retry_if_result(lambda v: v == "bad"),
        before_sleep=before_sleep_log(logger2, logging.INFO),
    )
    with pytest.raises(tenacity.RetryError):
        r2(lambda: "bad")
    assert logger2.records[0][1].endswith("in 2 seconds as it returned bad.")


def test_before_sleep_log_exc_info_appends_traceback_and_validates_state():
    logger = FakeLogger()

    def boom():
        raise ValueError("boom")

    r = Retrying(
        stop=stop_after_attempt(2),
        sleep=no_sleep,
        before_sleep=before_sleep_log(logger, logging.ERROR, exc_info=True),
    )
    with pytest.raises(tenacity.RetryError):
        r(boom)
    msg = logger.records[0][1]
    assert "as it raised ValueError: boom." in msg
    assert "Traceback (most recent call last)" in msg
    assert msg.endswith("ValueError: boom")

    rs = RetryCallState(Retrying(), fn=sample_task, args=(), kwargs={})
    callback = before_sleep_log(logger, logging.INFO)
    with pytest.raises(RuntimeError):
        callback(rs)  # no outcome yet
    rs.set_result(1)
    with pytest.raises(RuntimeError):
        callback(rs)  # outcome but no next_action
    rs.next_action = RetryAction(1)
    callback(rs)
    assert "as it returned 1." in logger.records[-1][1]


def test_nap_sleep_delegates_to_time_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(nap.time, "sleep", slept.append)
    nap.sleep(1.5)
    assert slept == [1.5]
    assert tenacity.sleep is nap.sleep


def test_sleep_using_event_waits_on_event_with_timeout():
    class FakeEvent:
        def __init__(self):
            self.timeouts = []

        def wait(self, timeout=None):
            self.timeouts.append(timeout)
            return False

    fake = FakeEvent()
    sleeper = sleep_using_event(fake)
    sleeper(3)
    sleeper(None)
    assert fake.timeouts == [3, None]

    import threading

    event = threading.Event()
    event.set()
    # an already-set event makes the "sleep" return immediately
    assert sleep_using_event(event)(60) is None
