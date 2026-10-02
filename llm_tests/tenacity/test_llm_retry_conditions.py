"""Tests for retry predicates (retry_if_*, retry_any, retry_all)."""
import re

import pytest

from tenacity import (
    Future,
    RetryCallState,
    Retrying,
    RetryError,
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
    stop_after_attempt,
)


def state_with_result(value):
    rs = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    rs.outcome = Future.construct(1, value, False)
    return rs


def state_with_exc(exc):
    rs = RetryCallState(Retrying(), fn=None, args=(), kwargs={})
    rs.outcome = Future.construct(1, exc, True)
    return rs


def test_retry_if_exception_type_matches_subclass():
    r = retry_if_exception_type(OSError)
    assert r(state_with_exc(FileNotFoundError("x"))) is True
    assert r(state_with_exc(ValueError("x"))) is False








def test_retry_unless_exception_type_retries_on_success():
    r = retry_unless_exception_type(ValueError)
    assert r(state_with_result("anything")) is True
    assert r(state_with_exc(ValueError())) is False
    assert r(state_with_exc(KeyError())) is True






def test_retry_if_result_predicate():
    r = retry_if_result(lambda x: x < 0)
    assert r(state_with_result(-1)) is True
    assert r(state_with_result(1)) is False
    assert r(state_with_exc(ValueError())) is False


def test_retry_if_not_result_predicate():
    r = retry_if_not_result(lambda x: x == "ready")
    assert r(state_with_result("pending")) is True
    assert r(state_with_result("ready")) is False
    assert r(state_with_exc(ValueError())) is False


def test_retry_if_result_in_decorator_returns_final_value():
    seq = iter([1, 2, 3, 10])

    @retry(retry=retry_if_result(lambda x: x < 5), sleep=lambda s: None)
    def fn():
        return next(seq)

    assert fn() == 10






def test_retry_if_exception_message_exact():
    r = retry_if_exception_message(message="try later")
    assert r(state_with_exc(ValueError("try later"))) is True
    assert r(state_with_exc(ValueError("try later please"))) is False


def test_retry_if_exception_message_regex_string():
    r = retry_if_exception_message(match=r"timeout \d+s")
    assert r(state_with_exc(ValueError("timeout 30s exceeded"))) is True
    assert r(state_with_exc(ValueError("no timeout 30s"))) is False








def test_retry_if_exception_message_in_decorator():
    calls = []

    @retry(
        retry=retry_if_exception_message(match="temporar"),
        stop=stop_after_attempt(5),
        sleep=lambda s: None,
    )
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("temporarily unavailable")
        raise RuntimeError("permanent failure")

    with pytest.raises(RuntimeError, match="permanent"):
        fn()
    assert len(calls) == 3




def test_retry_any_is_or():
    r = retry_any(retry_if_exception_type(KeyError), retry_if_result(lambda x: x == 0))
    assert r(state_with_exc(KeyError())) is True
    assert r(state_with_result(0)) is True
    assert r(state_with_result(1)) is False


def test_retry_all_is_and():
    r = retry_all(
        retry_if_exception_type(ValueError),
        retry_if_exception_message(match="retry"),
    )
    assert r(state_with_exc(ValueError("retry later"))) is True
    assert r(state_with_exc(ValueError("stop"))) is False
    assert r(state_with_exc(KeyError("retry later"))) is False


def test_retry_operators_combine():
    a = retry_if_exception_type(ValueError)
    b = retry_if_exception_type(KeyError)
    assert isinstance(a | b, retry_any)
    assert isinstance(a & b, retry_all)
    assert (a | b)(state_with_exc(KeyError())) is True
    assert (a & b)(state_with_exc(KeyError())) is False




def test_retry_never_and_always():
    s = state_with_exc(ValueError())
    assert retry_never(s) is False
    assert retry_always(s) is True






