import sys

import pytest

from tenacity import (
    Future,
    RetryAction,
    RetryCallState,
    RetryError,
    Retrying,
    retry,
    retry_if_result,
    stop_after_attempt,
    wait_fixed,
)


class MyException(Exception):
    pass


def helper_function():
    return None


def _value(obj):
    return obj() if callable(obj) else obj


def test_decorated_function_forwards_arguments_and_keeps_metadata():
    @retry(stop=stop_after_attempt(3))
    def add(a, b=0):
        """adds"""
        return a + b

    assert add(1, b=2) == 3
    assert add.__name__ == "add"
    assert add.__doc__ == "adds"


def test_retry_error_exposes_last_attempt():
    calls = []

    @retry(stop=stop_after_attempt(3), sleep=lambda s: None)
    def f():
        calls.append(1)
        raise MyException("Fail")

    with pytest.raises(RetryError) as info:
        f()
    last = info.value.last_attempt
    assert len(calls) == 3
    assert last.attempt_number == 3
    assert isinstance(last.exception(), MyException)
    assert _value(last.failed) is True


def test_reraise_true_raises_original_exception():
    calls = []

    @retry(reraise=True, stop=stop_after_attempt(3), sleep=lambda s: None)
    def f():
        calls.append(1)
        raise MyException("Fail")

    with pytest.raises(MyException):
        f()
    assert len(calls) == 3


def test_retry_error_reraise_method():
    failed = Future.construct(1, MyException("bad"), True)
    with pytest.raises(MyException):
        RetryError(failed).reraise()

    succeeded = Future.construct(1, "ok", False)
    with pytest.raises(RetryError):
        RetryError(succeeded).reraise()


def test_custom_retry_error_cls_is_raised():
    class MyRetryError(RetryError):
        pass

    @retry(
        stop=stop_after_attempt(2),
        retry_error_cls=MyRetryError,
        sleep=lambda s: None,
    )
    def f():
        raise MyException("Fail")

    with pytest.raises(MyRetryError):
        f()


def test_retry_error_callback_supplies_return_value():
    seen = []

    def last_value(retry_state):
        seen.append(retry_state.attempt_number)
        return retry_state.outcome.result()

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_result(lambda v: v is False),
        retry_error_callback=last_value,
        sleep=lambda s: None,
    )
    def eventually_false():
        return False

    assert eventually_false() is False
    assert seen == [3]


def test_statistics_empty_before_run_and_populated_after():
    controller = Retrying(stop=stop_after_attempt(3), sleep=lambda s: None)
    assert _value(controller.statistics) == {}

    def failing():
        raise MyException("Fail")

    with pytest.raises(RetryError):
        controller(failing)
    assert _value(controller.statistics)["attempt_number"] == 3

    @retry(stop=stop_after_attempt(2), sleep=lambda s: None)
    def decorated():
        raise MyException("Fail")

    with pytest.raises(RetryError):
        decorated()
    assert decorated.statistics["attempt_number"] == 2


def test_copy_overrides_only_given_parameters_and_retry_with():
    original = Retrying(stop=stop_after_attempt(3), wait=wait_fixed(1))
    clone = original.copy(stop=stop_after_attempt(5), reraise=True)
    assert clone.stop is not original.stop
    assert clone.wait is original.wait
    assert clone.reraise is True
    assert original.reraise is False

    calls = []

    @retry(stop=stop_after_attempt(2), sleep=lambda s: None)
    def f():
        calls.append(1)
        raise MyException("Fail")

    with pytest.raises(RetryError):
        f()
    assert len(calls) == 2
    calls.clear()
    with pytest.raises(RetryError):
        f.retry_with(stop=stop_after_attempt(4))()
    assert len(calls) == 4


def test_retrying_object_call_and_wraps():
    calls = []

    def flaky(x, y=1):
        calls.append(1)
        if len(calls) < 3:
            raise MyException("Fail")
        return x * y

    controller = Retrying(stop=stop_after_attempt(5), sleep=lambda s: None)
    assert controller(flaky, 3, y=4) == 12
    assert len(calls) == 3

    calls.clear()
    wrapped = Retrying(stop=stop_after_attempt(5), sleep=lambda s: None).wraps(flaky)
    assert wrapped(2, y=5) == 10
    assert len(calls) == 3


def test_context_manager_style_retries_and_exhausts():
    attempts = 0
    for attempt in Retrying(stop=stop_after_attempt(3), sleep=lambda s: None):
        with attempt:
            attempts += 1
            if attempts < 3:
                raise MyException("Fail")
    assert attempts == 3

    tries = []
    with pytest.raises(RetryError):
        for attempt in Retrying(stop=stop_after_attempt(2), sleep=lambda s: None):
            with attempt:
                tries.append(1)
                raise MyException("Fail")
    assert len(tries) == 2


def test_enabled_false_disables_retrying():
    calls = []

    @retry(enabled=False, stop=stop_after_attempt(5), sleep=lambda s: None)
    def f():
        calls.append(1)
        raise MyException("Fail")

    with pytest.raises((MyException, RetryError)):
        f()
    assert len(calls) == 1


def test_retry_call_state_lifecycle_and_fn_name():
    controller = Retrying()
    rs = RetryCallState(controller, helper_function, (1,), {"k": 2})
    assert rs.attempt_number == 1
    assert rs.outcome is None
    assert rs.args == (1,)
    assert rs.kwargs == {"k": 2}
    assert rs.retry_object is controller
    assert _value(rs.seconds_since_start) is None
    assert rs.get_fn_name().endswith("helper_function")

    rs.set_result(5)
    assert rs.outcome.result() == 5
    assert _value(rs.outcome.failed) is False
    assert _value(rs.seconds_since_start) >= 0

    rs.prepare_for_next_attempt()
    assert rs.attempt_number == 2
    assert rs.outcome is None

    try:
        raise MyException("bad")
    except MyException:
        rs.set_exception(sys.exc_info())
    assert _value(rs.outcome.failed) is True
    assert isinstance(rs.outcome.exception(), MyException)

    assert RetryCallState(Retrying(name="ctx"), None, (), {}).get_fn_name() == "ctx"
    assert RetryCallState(Retrying(), None, (), {}).get_fn_name() == "<unknown>"


def test_future_construct_and_retry_action():
    ok = Future.construct(2, "value", False)
    assert ok.attempt_number == 2
    assert ok.result() == "value"
    assert _value(ok.failed) is False

    error = MyException("bad")
    bad = Future.construct(3, error, True)
    assert _value(bad.failed) is True
    assert bad.exception() is error
    with pytest.raises(MyException):
        bad.result()

    assert RetryAction(3).sleep == 3.0
