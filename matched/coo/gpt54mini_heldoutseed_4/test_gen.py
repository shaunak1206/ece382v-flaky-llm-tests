import inspect

import pytest

from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError


def test_package_exports_coo_class():
    import coo as coo_pkg

    assert hasattr(coo_pkg, "Coo")
    assert coo_pkg.Coo is Coo


def test_coo_is_a_class():
    assert inspect.isclass(Coo)


def test_coo_class_comes_from_coo_module():
    assert Coo.__module__ == "coo.coo"


def test_coo_is_callable():
    assert callable(Coo)


def test_coo_has_expected_class_name():
    assert Coo.__name__ == "Coo"


def test_tweet_type_error_is_typeerror_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_has_expected_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_string_representation_without_args_is_empty():
    err = TweetTypeError()
    assert str(err) == ""


def test_tweet_type_error_can_carry_custom_message():
    err = TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(err) == TweetTypeError.wrongListMsg


def test_schedule_error_is_typeerror_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_has_expected_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_schedule_error_string_representation_without_args_is_empty():
    err = ScheduleError()
    assert str(err) == ""


def test_schedule_error_can_carry_custom_message():
    err = ScheduleError(ScheduleError.pastDateError)
    assert str(err) == ScheduleError.pastDateError


def test_exceptions_are_distinct_types():
    assert TweetTypeError is not ScheduleError
    assert TweetTypeError != ScheduleError


def test_exception_instances_are_instances_of_typeerror():
    assert isinstance(TweetTypeError(), TypeError)
    assert isinstance(ScheduleError(), TypeError)


def test_exception_constants_are_plain_strings():
    for value in (
        TweetTypeError.wrongListMsg,
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    ):
        assert isinstance(value, str)
        assert value.strip()
