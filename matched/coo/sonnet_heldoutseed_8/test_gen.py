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


def test_tweet_type_error_wrong_list_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_and_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_str_contains_message():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(excinfo.value) == "A List of Strings is required."


def test_tweet_type_error_without_args_has_empty_str():
    assert str(TweetTypeError()) == ""


def test_tweet_type_error_message_accessible_from_instance():
    err = TweetTypeError()
    assert err.wrongListMsg == TweetTypeError.wrongListMsg


# ---------------------------------------------------------------------------
# ScheduleError
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


@pytest.mark.parametrize(
    "attr, expected",
    [
        ("wrongListMsg", "A list of tuples is required."),
        ("tupleLenError", "Every tuple need a length of 3 or 4."),
        ("pastDateError", "A future date is needed."),
    ],
)
def test_schedule_error_messages(attr, expected):
    assert getattr(ScheduleError, attr) == expected


@pytest.mark.parametrize(
    "attr", ["wrongListMsg", "tupleLenError", "pastDateError"]
)
def test_schedule_error_raise_with_each_message(attr):
    message = getattr(ScheduleError, attr)
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(message)
    assert str(excinfo.value) == message


def test_schedule_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise ScheduleError(ScheduleError.pastDateError)


def test_schedule_and_tweet_errors_are_distinct():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError("x")
        except TweetTypeError:  # pragma: no cover
            pytest.fail("ScheduleError must not be caught as TweetTypeError")


def test_schedule_error_wrong_list_message_differs_from_tweet_error():
    assert ScheduleError.wrongListMsg != TweetTypeError.wrongListMsg


# ---------------------------------------------------------------------------
# TemplateError (sibling exception in the same module)
# ---------------------------------------------------------------------------

def test_template_error_is_type_error_with_messages():
    assert issubclass(TemplateError, TypeError)
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


# ---------------------------------------------------------------------------
# Coo / package exports
# ---------------------------------------------------------------------------

def test_coo_is_exported_class():
    assert inspect.isclass(Coo)
    assert coo.Coo is Coo


def test_coo_is_same_class_as_in_submodule():
    from coo.coo import Coo as SubmoduleCoo

    assert SubmoduleCoo is Coo


def test_coo_is_callable_with_inspectable_signature():
    assert callable(Coo)
    sig = inspect.signature(Coo)
    assert isinstance(sig, inspect.Signature)


def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"
