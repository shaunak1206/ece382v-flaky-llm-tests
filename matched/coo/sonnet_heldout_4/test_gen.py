import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


# ---------------------------------------------------------------------------
# TweetTypeError
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_is_exception_subclass():
    assert issubclass(TweetTypeError, Exception)


def test_tweet_type_error_wrong_list_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_with_message():
    with pytest.raises(TweetTypeError) as exc_info:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(exc_info.value) == "A List of Strings is required."


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("bad msg")


def test_tweet_type_error_without_args_has_empty_str():
    assert str(TweetTypeError()) == ""


def test_tweet_type_error_message_available_on_instance():
    err = TweetTypeError("x")
    assert err.wrongListMsg == TweetTypeError.wrongListMsg
    assert err.args == ("x",)


# ---------------------------------------------------------------------------
# ScheduleError
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_messages():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_schedule_error_raise_with_each_message():
    for msg in (
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    ):
        with pytest.raises(ScheduleError, match=msg.rstrip(".")):
            raise ScheduleError(msg)


def test_schedule_error_not_caught_as_tweet_type_error():
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError("x")
        except TweetTypeError:  # pragma: no cover
            pytest.fail("ScheduleError must not be a TweetTypeError")


def test_schedule_error_and_tweet_type_error_are_distinct():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)


# ---------------------------------------------------------------------------
# TemplateError (related exception in the same module)
# ---------------------------------------------------------------------------

def test_template_error_messages_and_base():
    assert issubclass(TemplateError, TypeError)
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


# ---------------------------------------------------------------------------
# Package-level / Coo
# ---------------------------------------------------------------------------

def test_coo_is_exported_class():
    assert inspect.isclass(Coo)
    assert coo.Coo is Coo


def test_coo_defined_in_coo_module():
    assert Coo.__module__ == "coo.coo"


def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"
    assert "@" in coo.__contact__
