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
    assert issubclass(TweetTypeError, Exception)


def test_tweet_type_error_wrong_list_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_and_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_str_contains_message():
    err = TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(err) == "A List of Strings is required."


def test_tweet_type_error_match_with_pytest_raises():
    with pytest.raises(TweetTypeError, match="List of Strings"):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_message_available_on_instance():
    err = TweetTypeError()
    assert err.wrongListMsg == TweetTypeError.wrongListMsg


# ---------------------------------------------------------------------------
# ScheduleError
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_messages():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


@pytest.mark.parametrize(
    "attr",
    ["wrongListMsg", "tupleLenError", "pastDateError"],
)
def test_schedule_error_raise_with_each_message(attr):
    message = getattr(ScheduleError, attr)
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(message)
    assert str(excinfo.value) == message


def test_schedule_error_not_caught_as_tweet_type_error():
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError(ScheduleError.pastDateError)
        except TweetTypeError:  # pragma: no cover
            pytest.fail("ScheduleError must not be a TweetTypeError")


def test_exception_classes_are_distinct():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)
    assert not issubclass(TemplateError, ScheduleError)


def test_schedule_and_tweet_messages_differ():
    assert ScheduleError.wrongListMsg != TweetTypeError.wrongListMsg


def test_template_error_messages():
    assert issubclass(TemplateError, TypeError)
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


# ---------------------------------------------------------------------------
# Coo / package
# ---------------------------------------------------------------------------

def test_coo_is_a_class():
    assert inspect.isclass(Coo)
    assert Coo.__name__ == "Coo"


def test_coo_exposed_from_package_and_submodule():
    import coo.coo as coo_module

    assert coo.Coo is coo_module.Coo


def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"
