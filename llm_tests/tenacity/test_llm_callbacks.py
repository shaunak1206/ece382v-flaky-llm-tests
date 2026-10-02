"""Tests for before / after / before_sleep callbacks and logging helpers."""
import logging

import pytest

from tenacity import (
    RetryError,
    Retrying,
    after_log,
    after_nothing,
    before_log,
    before_nothing,
    before_sleep_log,
    before_sleep_nothing,
    retry,
    stop_after_attempt,
    wait_fixed,
)
from tenacity import _utils


def test_before_callback_called_for_each_attempt():
    seen = []

    @retry(
        stop=stop_after_attempt(3),
        before=lambda rs: seen.append(rs.attempt_number),
        sleep=lambda s: None,
    )
    def fn():
        raise ValueError

    with pytest.raises(RetryError):
        fn()
    assert seen == [1, 2, 3]


def test_after_callback_called_after_failed_attempts():
    seen = []

    @retry(
        stop=stop_after_attempt(3),
        after=lambda rs: seen.append(rs.attempt_number),
        sleep=lambda s: None,
    )
    def fn():
        raise ValueError

    with pytest.raises(RetryError):
        fn()
    assert seen == [1, 2, 3]


def test_after_callback_not_called_on_success():
    seen = []

    @retry(after=lambda rs: seen.append(1))
    def fn():
        return "fine"

    assert fn() == "fine"
    assert seen == []


def test_before_sleep_called_between_attempts_only():
    seen = []

    @retry(
        stop=stop_after_attempt(4),
        before_sleep=lambda rs: seen.append(rs.attempt_number),
        sleep=lambda s: None,
    )
    def fn():
        raise ValueError

    with pytest.raises(RetryError):
        fn()
    assert seen == [1, 2, 3]


def test_before_sleep_sees_next_action_sleep():
    sleeps = []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(7),
        before_sleep=lambda rs: sleeps.append(rs.next_action.sleep),
        sleep=lambda s: None,
    )
    def fn():
        raise ValueError

    with pytest.raises(RetryError):
        fn()
    assert sleeps == [7, 7]


def test_callback_order_before_after_before_sleep():
    events = []

    @retry(
        stop=stop_after_attempt(2),
        before=lambda rs: events.append("before"),
        after=lambda rs: events.append("after"),
        before_sleep=lambda rs: events.append("before_sleep"),
        sleep=lambda s: events.append("sleep"),
    )
    def fn():
        raise ValueError

    with pytest.raises(RetryError):
        fn()
    assert events == [
        "before", "after", "before_sleep", "sleep",
        "before", "after",
    ]






def test_retry_state_get_fn_name_for_context_manager():
    names = []
    r = Retrying(
        stop=stop_after_attempt(2),
        name="my-block",
        before=lambda rs: names.append(rs.get_fn_name()),
        sleep=lambda s: None,
    )
    with pytest.raises(RetryError):
        for attempt in r:
            with attempt:
                raise ValueError
    assert names == ["my-block", "my-block"]




def test_before_log_writes_to_logger(caplog):
    logger = logging.getLogger("llm.before")

    @retry(
        stop=stop_after_attempt(2),
        before=before_log(logger, logging.INFO),
        sleep=lambda s: None,
    )
    def fn():
        raise ValueError

    with caplog.at_level(logging.INFO, logger="llm.before"):
        with pytest.raises(RetryError):
            fn()
    messages = [r.getMessage() for r in caplog.records]
    assert any("1st time calling it" in m for m in messages)
    assert any("2nd time calling it" in m for m in messages)




def test_before_sleep_log_message_for_exception(caplog):
    logger = logging.getLogger("llm.sleep")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        sleep=lambda s: None,
    )
    def fn():
        raise ValueError("bad thing")

    with caplog.at_level(logging.WARNING, logger="llm.sleep"):
        with pytest.raises(RetryError):
            fn()
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "in 3 seconds" in msg
    assert "raised ValueError: bad thing" in msg






