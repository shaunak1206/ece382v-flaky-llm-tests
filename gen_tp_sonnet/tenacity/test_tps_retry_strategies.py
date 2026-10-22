import re
import sys

import pytest

from tenacity import (
    RetryCallState,
    RetryError,
    Retrying,
    TryAgain,
    retry,
    retry_all,
    retry_any,
    retry_if_exception,
    retry_if_exception_cause_type,
    retry_if_exception_message,
    retry_if_exception_type,
    retry_if_not_exception_message,
    retry_if_not_exception_type,
    retry_if_not_result,
    retry_if_result,
    retry_unless_exception_type,
    stop_after_attempt,
)


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


class ClientError(Exception):
    pass


def test_retry_if_exception_type_matches_raised_type():
    strategy = retry_if_exception_type(ValueError)
    assert strategy(make_state(exc=ValueError("x")))
    assert strategy(make_state(exc=UnicodeError("x")))  # subclass of ValueError


def test_retry_if_exception_type_ignores_other_types_and_results():
    strategy = retry_if_exception_type(ValueError)
    assert not strategy(make_state(exc=KeyError("x")))
    assert not strategy(make_state(result="fine"))


def test_retry_if_exception_type_accepts_tuple():
    strategy = retry_if_exception_type((ValueError, KeyError))
    assert strategy(make_state(exc=KeyError("x")))
    assert strategy(make_state(exc=ValueError("x")))
    assert not strategy(make_state(exc=TypeError("x")))


def test_retry_if_not_exception_type():
    strategy = retry_if_not_exception_type(ClientError)
    assert strategy(make_state(exc=ValueError("x")))
    assert not strategy(make_state(exc=ClientError("x")))


def test_retry_if_exception_uses_custom_predicate():
    strategy = retry_if_exception(lambda e: "temporary" in str(e))
    assert strategy(make_state(exc=RuntimeError("temporary glitch")))
    assert not strategy(make_state(exc=RuntimeError("permanent")))


def test_retry_if_result_predicate():
    strategy = retry_if_result(lambda r: r is None)
    assert strategy(make_state(result=None))
    assert not strategy(make_state(result=5))
    assert not strategy(make_state(exc=ValueError("x")))


def test_retry_if_not_result_predicate():
    strategy = retry_if_not_result(lambda r: r == 3)
    assert strategy(make_state(result=4))
    assert not strategy(make_state(result=3))


def test_retry_if_exception_message_exact():
    strategy = retry_if_exception_message(message="temporary")
    assert strategy(make_state(exc=ValueError("temporary")))
    assert not strategy(make_state(exc=ValueError("temporary failure")))


def test_retry_if_exception_message_regex_match_string_and_pattern():
    by_string = retry_if_exception_message(match=r"temp.*")
    by_pattern = retry_if_exception_message(match=re.compile(r"^code \d+$"))
    assert by_string(make_state(exc=ValueError("temporary failure")))
    assert not by_string(make_state(exc=ValueError("permanent")))
    assert by_pattern(make_state(exc=ValueError("code 42")))
    assert not by_pattern(make_state(exc=ValueError("code x")))


def test_retry_if_exception_message_requires_exactly_one_argument():
    with pytest.raises(TypeError):
        retry_if_exception_message(message="a", match="a")
    with pytest.raises(TypeError):
        retry_if_exception_message()


def test_retry_if_not_exception_message():
    strategy = retry_if_not_exception_message(message="fatal")
    assert strategy(make_state(exc=ValueError("other")))
    assert not strategy(make_state(exc=ValueError("fatal")))


def test_retry_if_exception_cause_type_inspects_cause_chain():
    try:
        try:
            raise KeyError("inner")
        except KeyError as inner:
            raise ValueError("outer") from inner
    except ValueError as outer:
        caused = outer

    assert retry_if_exception_cause_type(KeyError)(make_state(exc=caused))
    assert not retry_if_exception_cause_type(TypeError)(make_state(exc=caused))
    assert not retry_if_exception_cause_type(KeyError)(
        make_state(exc=ValueError("no cause"))
    )


def test_retry_any_is_true_if_one_condition_holds():
    strategy = retry_any(
        retry_if_result(lambda r: r is not None and r < 0),
        retry_if_exception_type(ValueError),
    )
    assert strategy(make_state(result=-1))
    assert strategy(make_state(exc=ValueError("x")))
    assert not strategy(make_state(result=1))


def test_retry_all_requires_every_condition():
    strategy = retry_all(
        retry_if_result(lambda r: r > 0),
        retry_if_result(lambda r: r < 10),
    )
    assert strategy(make_state(result=5))
    assert not strategy(make_state(result=20))
    assert not strategy(make_state(result=-5))


def test_retry_or_and_operators_combine_conditions():
    either = retry_if_result(lambda r: r == 1) | retry_if_result(lambda r: r == 2)
    both = retry_if_result(lambda r: r > 0) & retry_if_result(lambda r: r < 3)
    assert isinstance(either, retry_any)
    assert isinstance(both, retry_all)
    assert either(make_state(result=2))
    assert not either(make_state(result=3))
    assert both(make_state(result=2))
    assert not both(make_state(result=3))


def test_default_retry_retries_any_exception_until_success():
    calls = []

    @retry(sleep=lambda s: None)
    def flaky():
        calls.append(1)
        if len(calls) < 4:
            raise IOError("broken")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 4


def test_retry_if_exception_type_raises_other_exceptions_immediately():
    calls = []

    @retry(retry=retry_if_exception_type(IOError), stop=stop_after_attempt(5))
    def f():
        calls.append(1)
        raise KeyError("not an io error")

    with pytest.raises(KeyError):
        f()
    assert len(calls) == 1


def test_retry_if_not_exception_type_stops_on_excluded_type():
    calls = []

    @retry(
        retry=retry_if_not_exception_type(ClientError),
        stop=stop_after_attempt(5),
        sleep=lambda s: None,
    )
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("transient")
        raise ClientError("fatal")

    with pytest.raises(ClientError):
        f()
    assert len(calls) == 3


def test_retry_unless_exception_type_retries_until_that_exception():
    calls = []

    @retry(
        retry=retry_unless_exception_type(ValueError),
        stop=stop_after_attempt(10),
        sleep=lambda s: None,
    )
    def f():
        calls.append(1)
        if len(calls) < 3:
            return "keep going"
        raise ValueError("done")

    with pytest.raises(ValueError):
        f()
    assert len(calls) == 3


def test_retry_if_result_retries_until_result_acceptable():
    calls = []

    @retry(retry=retry_if_result(lambda r: r is None), sleep=lambda s: None)
    def f():
        calls.append(1)
        return None if len(calls) < 3 else "value"

    assert f() == "value"
    assert len(calls) == 3


def test_retry_if_result_exhaustion_raises_retry_error_with_last_result():
    @retry(
        retry=retry_if_result(lambda r: r is False),
        stop=stop_after_attempt(3),
        sleep=lambda s: None,
    )
    def f():
        return False

    with pytest.raises(RetryError) as info:
        f()
    assert info.value.last_attempt.result() is False
    assert info.value.last_attempt.attempt_number == 3


def test_retry_if_not_result_flow():
    counter = {"n": 0}

    @retry(retry=retry_if_not_result(lambda r: r == 3), sleep=lambda s: None)
    def f():
        counter["n"] += 1
        return counter["n"]

    assert f() == 3


def test_combined_result_or_exception_retry_from_readme():
    calls = []

    @retry(
        retry=retry_if_result(lambda v: v is None) | retry_if_exception_type(),
        sleep=lambda s: None,
    )
    def f():
        calls.append(1)
        if len(calls) == 1:
            return None
        if len(calls) == 2:
            raise RuntimeError("fail")
        return "good"

    assert f() == "good"
    assert len(calls) == 3


def test_try_again_forces_retry_regardless_of_retry_strategy():
    calls = []

    @retry(
        retry=retry_if_exception_type(KeyError),
        stop=stop_after_attempt(10),
        sleep=lambda s: None,
    )
    def f():
        calls.append(1)
        if len(calls) < 3:
            raise TryAgain
        return "done"

    assert f() == "done"
    assert len(calls) == 3
