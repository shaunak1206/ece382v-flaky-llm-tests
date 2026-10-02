"""Tests for retry_* conditions and the before/after/before_sleep logging callbacks."""

import logging
import re
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
    stop_after_attempt,
    wait_fixed,
)


class SleepRecorder:
    def __init__(self):
        self.sleeps = []

    def __call__(self, seconds):
        self.sleeps.append(seconds)


class ListLogger:
    """Minimal logger recording (level, message) pairs."""

    def __init__(self):
        self.records = []

    def log(self, level, msg, *args):
        self.records.append((level, msg))


def make_state(attempt=1, exc=None, result=None, with_outcome=True):
    state = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    state.attempt_number = attempt
    if with_outcome:
        if exc is not None:
            state.set_exception((type(exc), exc, None))
        else:
            state.set_result(result)
    return state


# ---------------------------------------------------------------------------
# retry conditions evaluated directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "types, exc, expected",
    [
        (ValueError, ValueError("x"), True),
        (ValueError, KeyError("x"), False),
        ((ValueError, KeyError), KeyError("x"), True),
        (Exception, OSError(), True),
        (OSError, FileNotFoundError(), True),
    ],
)
def test_retry_if_exception_type(types, exc, expected):
    assert retry_if_exception_type(types)(make_state(exc=exc)) is expected
    # a successful outcome never triggers an exception based retry
    assert retry_if_exception_type(types)(make_state(result=1)) is False
    assert retry_if_exception_type()(make_state(exc=RuntimeError())) is True


def test_retry_if_not_exception_type_and_unless_exception_type():
    value_error = make_state(exc=ValueError())
    key_error = make_state(exc=KeyError())
    success = make_state(result=1)

    not_type = retry_if_not_exception_type(ValueError)
    assert not_type(value_error) is False
    assert not_type(key_error) is True
    assert not_type(success) is False

    unless = retry_unless_exception_type(ValueError)
    assert unless(value_error) is False
    assert unless(key_error) is True
    assert unless(success) is True  # keeps retrying until the exception shows up


def test_retry_if_exception_predicate_always_and_never():
    predicate = retry_if_exception(lambda exc: "again" in str(exc))
    assert predicate(make_state(exc=ValueError("try again"))) is True
    assert predicate(make_state(exc=ValueError("fatal"))) is False
    assert predicate(make_state(result="again")) is False
    assert retry_always(make_state(result=1)) is True
    assert retry_never(make_state(exc=ValueError())) is False


def test_retry_if_exception_cause_type():
    outer = RuntimeError("outer")
    outer.__cause__ = KeyError("inner")
    retry = retry_if_exception_cause_type(KeyError)
    assert retry(make_state(exc=outer)) is True
    assert retry(make_state(exc=RuntimeError("no cause"))) is False
    assert retry(make_state(result=1)) is False

    deep = RuntimeError("deep")
    middle = RuntimeError("middle")
    middle.__cause__ = KeyError("root")
    deep.__cause__ = middle
    assert retry(make_state(exc=deep)) is True

    cyclic = RuntimeError("cycle")
    cyclic.__cause__ = cyclic
    assert retry_if_exception_cause_type(ValueError)(make_state(exc=cyclic)) is False


def test_retry_if_result_and_not_result():
    is_none = retry_if_result(lambda value: value is None)
    assert is_none(make_state(result=None)) is True
    assert is_none(make_state(result=1)) is False
    assert is_none(make_state(exc=ValueError())) is False

    not_positive = retry_if_not_result(lambda value: value > 0)
    assert not_positive(make_state(result=-1)) is True
    assert not_positive(make_state(result=5)) is False
    assert not_positive(make_state(exc=ValueError())) is False


def test_retry_if_exception_message():
    exact = retry_if_exception_message(message="boom")
    assert exact(make_state(exc=ValueError("boom"))) is True
    assert exact(make_state(exc=ValueError("boom!"))) is False
    assert exact(make_state(result="boom")) is False

    pattern = retry_if_exception_message(match=r"^conn.*reset$")
    assert pattern(make_state(exc=OSError("connection reset"))) is True
    assert pattern(make_state(exc=OSError("connection refused"))) is False

    compiled = retry_if_exception_message(match=re.compile("timeout"))
    assert compiled(make_state(exc=ValueError("timeout after 3s"))) is True

    with pytest.raises(TypeError):
        retry_if_exception_message(message="a", match="b")
    with pytest.raises(TypeError):
        retry_if_exception_message()


def test_retry_if_not_exception_message():
    retry = retry_if_not_exception_message(message="fatal")
    assert retry(make_state(exc=ValueError("fatal"))) is False
    assert retry(make_state(exc=ValueError("transient"))) is True
    assert retry(make_state(result=1)) is True


def test_retry_any_all_and_operators():
    state = make_state(exc=ValueError("x"))
    assert retry_any(retry_never, retry_always)(state) is True
    assert retry_any(retry_never, retry_never)(state) is False
    assert retry_all(retry_always, retry_always)(state) is True
    assert retry_all(retry_always, retry_never)(state) is False

    is_value_error = retry_if_exception_type(ValueError)
    is_none = retry_if_result(lambda value: value is None)
    either = is_value_error | is_none
    assert isinstance(either, retry_any)
    assert either(state) is True
    assert either(make_state(result=None)) is True
    assert either(make_state(result=1)) is False

    both = is_value_error & retry_if_exception_message(message="x")
    assert isinstance(both, retry_all)
    assert both(state) is True
    assert both(make_state(exc=ValueError("y"))) is False

    assert len((is_value_error | is_none | retry_always).retries) == 3
    assert len((is_value_error & is_value_error & is_value_error).retries) == 3
    assert (is_value_error | (lambda rs: True))(make_state(result=1)) is True


def test_strategies_require_an_outcome():
    state = make_state(with_outcome=False)
    strategies = [
        retry_if_exception_type(),
        retry_if_result(lambda value: True),
        retry_if_not_result(lambda value: True),
        retry_if_exception_cause_type(),
        retry_unless_exception_type(ValueError),
        retry_if_not_exception_message(message="x"),
    ]
    for strategy in strategies:
        with pytest.raises(RuntimeError):
            strategy(state)


# ---------------------------------------------------------------------------
# conditions driving a Retrying loop
# ---------------------------------------------------------------------------


def test_retry_if_result_loops_until_condition_is_met():
    counter = {"n": 0}

    def produce():
        counter["n"] += 1
        return counter["n"]

    retrying = Retrying(
        retry=retry_if_result(lambda value: value < 4),
        stop=stop_after_attempt(10),
        sleep=SleepRecorder(),
    )
    assert retrying(produce) == 4

    exhausted = Retrying(
        retry=retry_if_result(lambda value: True),
        stop=stop_after_attempt(3),
        sleep=SleepRecorder(),
    )
    with pytest.raises(RetryError) as excinfo:
        exhausted(lambda: "never good")
    assert excinfo.value.last_attempt.result() == "never good"

    values = iter([None, None, "x"])
    until_truthy = Retrying(
        retry=retry_if_not_result(bool),
        stop=stop_after_attempt(5),
        sleep=SleepRecorder(),
    )
    assert until_truthy(lambda: next(values)) == "x"


def test_retry_if_exception_message_retries_then_propagates_other_errors():
    errors = iter([ValueError("retry me"), ValueError("retry me"), ValueError("fatal")])
    calls = []

    def fn():
        calls.append(1)
        raise next(errors)

    retrying = Retrying(
        retry=retry_if_exception_message(message="retry me"),
        stop=stop_after_attempt(10),
        sleep=SleepRecorder(),
    )
    with pytest.raises(ValueError, match="fatal"):
        retrying(fn)
    assert len(calls) == 3


def test_combined_conditions_in_retrying():
    steps = iter(["raise", None, "done"])

    def fn():
        step = next(steps)
        if step == "raise":
            raise KeyError("k")
        return step

    retrying = Retrying(
        retry=retry_if_exception_type(KeyError) | retry_if_result(lambda v: v is None),
        stop=stop_after_attempt(5),
        sleep=SleepRecorder(),
    )
    assert retrying(fn) == "done"

    errors = iter(
        [ValueError("retry later"), ValueError("retry soon"), ValueError("fatal")]
    )

    def failing():
        raise next(errors)

    both = Retrying(
        retry=retry_if_exception_type(ValueError)
        & retry_if_exception_message(match="retry"),
        stop=stop_after_attempt(10),
        sleep=SleepRecorder(),
    )
    with pytest.raises(ValueError, match="fatal"):
        both(failing)


# ---------------------------------------------------------------------------
# logging callbacks
# ---------------------------------------------------------------------------


def test_before_log_and_after_log():
    logger = ListLogger()
    retrying = Retrying(
        stop=stop_after_attempt(3),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.INFO),
        sleep=SleepRecorder(),
    )

    def my_func():
        raise ValueError("boom")

    with pytest.raises(RetryError):
        retrying(my_func)

    befores = [msg for level, msg in logger.records if level == logging.DEBUG]
    afters = [msg for level, msg in logger.records if level == logging.INFO]
    assert len(befores) == 3
    assert "my_func" in befores[0]
    assert "this is the 1st time calling it." in befores[0]
    assert "2nd" in befores[1]
    assert "3rd" in befores[2]
    assert len(afters) == 3
    assert all(msg.startswith("Finished call to") for msg in afters)
    assert "this was the 1st time calling it." in afters[0]

    custom = ListLogger()
    formatted = Retrying(
        stop=stop_after_attempt(1),
        after=after_log(custom, logging.INFO, sec_format="%.1f"),
        sleep=SleepRecorder(),
    )
    with pytest.raises(RetryError):
        formatted(my_func)
    assert "after 0.0(s)" in custom.records[0][1]

    named = ListLogger()
    attempts = 0
    for attempt in Retrying(
        stop=stop_after_attempt(2),
        name="job",
        before=before_log(named, logging.INFO),
        sleep=SleepRecorder(),
    ):
        with attempt:
            attempts += 1
            if attempts < 2:
                raise ValueError("again")
    assert "'job'" in named.records[0][1]


def test_before_sleep_log_messages():
    logger = ListLogger()
    retrying = Retrying(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.5),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        sleep=SleepRecorder(),
    )

    def my_func():
        raise ValueError("boom")

    with pytest.raises(RetryError):
        retrying(my_func)
    assert len(logger.records) == 2  # only logged when about to sleep
    level, message = logger.records[0]
    assert level == logging.WARNING
    assert "Retrying" in message
    assert "my_func" in message
    assert "in 0.5 seconds as it raised ValueError: boom." in message

    result_logger = ListLogger()
    result_based = Retrying(
        retry=retry_if_result(lambda value: value is None),
        stop=stop_after_attempt(2),
        before_sleep=before_sleep_log(result_logger, logging.INFO),
        sleep=SleepRecorder(),
    )
    with pytest.raises(RetryError):
        result_based(lambda: None)
    assert "as it returned None." in result_logger.records[0][1]

    trace_logger = ListLogger()
    with_trace = Retrying(
        stop=stop_after_attempt(2),
        before_sleep=before_sleep_log(trace_logger, logging.INFO, exc_info=True),
        sleep=SleepRecorder(),
    )
    with pytest.raises(RetryError):
        with_trace(my_func)
    assert "Traceback (most recent call last)" in trace_logger.records[0][1]


def test_noop_callbacks_and_helper_functions():
    state = make_state(exc=ValueError())
    assert before_nothing(state) is None
    assert after_nothing(state) is None
    assert before_sleep_nothing(state) is None

    expected = {
        1: "1st",
        2: "2nd",
        3: "3rd",
        4: "4th",
        11: "11th",
        12: "12th",
        13: "13th",
        21: "21st",
        22: "22nd",
        23: "23rd",
        101: "101st",
        111: "111th",
    }
    for number, text in expected.items():
        assert _utils.to_ordinal(number) == text

    assert _utils.to_seconds(timedelta(minutes=1)) == 60.0
    assert _utils.to_seconds(3) == 3.0
    assert _utils.get_callback_name(len) == "builtins.len"
