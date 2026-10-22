import functools
import logging
import threading
from datetime import timedelta

import pytest

from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    _utils,
    after_log,
    after_nothing,
    before_log,
    before_nothing,
    before_sleep_log,
    before_sleep_nothing,
    nap,
    retry,
    stop_after_attempt,
)


class FakeLogger:
    def __init__(self):
        self.records = []

    def log(self, level, msg, *args, **kwargs):
        text = msg % args if args else msg
        self.records.append((level, text))


def run_failing(**retry_kwargs):
    @retry(stop=stop_after_attempt(3), sleep=lambda s: None, **retry_kwargs)
    def f():
        raise ValueError("boom")

    with pytest.raises(RetryError):
        f()


def test_before_after_and_before_sleep_callbacks_are_invoked():
    before_numbers = []
    after_numbers = []
    sleep_numbers = []

    run_failing(
        before=lambda rs: before_numbers.append(rs.attempt_number),
        after=lambda rs: after_numbers.append(rs.attempt_number),
        before_sleep=lambda rs: sleep_numbers.append(rs.attempt_number),
    )

    assert before_numbers == [1, 2, 3]
    assert len(after_numbers) == 3
    assert len(sleep_numbers) == 2


def test_log_helpers_write_to_logger_like_object():
    before_logger = FakeLogger()
    after_logger = FakeLogger()
    sleep_logger = FakeLogger()

    run_failing(
        before=before_log(before_logger, logging.DEBUG),
        after=after_log(after_logger, logging.INFO),
        before_sleep=before_sleep_log(sleep_logger, logging.WARNING),
    )

    assert len(before_logger.records) == 3
    assert all(level == logging.DEBUG for level, _ in before_logger.records)
    assert "Starting call to" in before_logger.records[0][1]
    assert "1st" in before_logger.records[0][1]

    assert len(after_logger.records) == 3
    assert all(level == logging.INFO for level, _ in after_logger.records)
    assert "Finished call to" in after_logger.records[0][1]

    assert len(sleep_logger.records) == 2
    assert all(level == logging.WARNING for level, _ in sleep_logger.records)
    assert "Retrying" in sleep_logger.records[0][1]
    assert "boom" in sleep_logger.records[0][1]


def test_before_sleep_log_with_real_logger(caplog):
    logger = logging.getLogger("tps.tenacity.real")
    caplog.set_level(logging.DEBUG, logger="tps.tenacity.real")

    run_failing(before_sleep=before_sleep_log(logger, logging.WARNING))

    records = [r for r in caplog.records if r.name == "tps.tenacity.real"]
    assert len(records) == 2
    assert all(r.levelno == logging.WARNING for r in records)
    assert "Retrying" in records[0].getMessage()


def test_nothing_strategies_return_none():
    rs = RetryCallState(Retrying(), None, (), {})
    rs.set_result(1)
    assert before_nothing(rs) is None
    assert after_nothing(rs) is None
    assert before_sleep_nothing(rs) is None


@pytest.mark.parametrize(
    "number, expected",
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
    ],
)
def test_to_ordinal(number, expected):
    assert _utils.to_ordinal(number) == expected


def test_to_seconds_callback_name_and_coroutine_detection():
    assert _utils.to_seconds(3) == 3
    assert _utils.to_seconds(timedelta(seconds=5)) == 5
    assert _utils.to_seconds(timedelta(minutes=1)) == 60

    def my_callback(retry_state):
        return None

    async def my_async_callback(retry_state):
        return None

    assert _utils.get_callback_name(my_callback).endswith("my_callback")
    assert isinstance(_utils.get_callback_name(functools.partial(my_callback)), str)

    assert _utils.is_coroutine_callable(my_async_callback)
    assert not _utils.is_coroutine_callable(my_callback)


def test_nap_sleep_and_sleep_using_event():
    assert nap.sleep(0) is None

    event = threading.Event()
    event.set()
    assert nap.sleep_using_event(event)(5) is None

    unset = threading.Event()
    assert nap.sleep_using_event(unset)(0.01) is None
    assert not unset.is_set()
