import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


def test_coo_is_exported_class():
    assert inspect.isclass(Coo)
    assert coo.Coo is Coo


def test_coo_defined_in_coo_module():
    from coo.coo import Coo as InnerCoo

    assert InnerCoo is Coo


def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__url__.startswith("https://github.com/")


def test_tweet_type_error_is_type_error():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(TweetTypeError, Exception)


def test_schedule_error_is_type_error():
    assert issubclass(ScheduleError, TypeError)


def test_exceptions_are_distinct_classes():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)
    assert not issubclass(ScheduleError, TemplateError)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_template_error_message_constants():
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_raise_tweet_type_error_with_message():
    with pytest.raises(TweetTypeError) as exc_info:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(exc_info.value) == "A List of Strings is required."


def test_raise_schedule_error_with_each_message():
    for msg in (
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    ):
        with pytest.raises(ScheduleError, match=msg.rstrip(".")):
            raise ScheduleError(msg)


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("bad")


def test_schedule_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise ScheduleError("bad")


def test_schedule_error_not_caught_as_tweet_type_error():
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError("bad")
        except TweetTypeError:  # pragma: no cover
            pytest.fail("ScheduleError must not be a TweetTypeError")


def test_exceptions_without_message_have_empty_str():
    assert str(TweetTypeError()) == ""
    assert str(ScheduleError()) == ""


def test_exception_instances_keep_args():
    err = ScheduleError("a", 3)
    assert err.args == ("a", 3)
    assert isinstance(err, ScheduleError)
