import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


# ---------------------------------------------------------------------------
# Package-level metadata / exports
# ---------------------------------------------------------------------------

def test_coo_is_exported_class():
    assert inspect.isclass(Coo)
    assert Coo.__name__ == "Coo"


def test_coo_exported_from_package_is_same_as_submodule_class():
    from coo.coo import Coo as SubmoduleCoo

    assert coo.Coo is SubmoduleCoo


def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__url__.startswith("https://github.com/")


# ---------------------------------------------------------------------------
# TweetTypeError
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_with_message():
    with pytest.raises(TweetTypeError, match="A List of Strings is required."):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("bad")


def test_tweet_type_error_str_matches_argument():
    err = TweetTypeError("custom message")
    assert str(err) == "custom message"


# ---------------------------------------------------------------------------
# ScheduleError
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


@pytest.mark.parametrize(
    "message",
    [
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    ],
)
def test_schedule_error_raised_with_each_message(message):
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(message)
    assert str(excinfo.value) == message


def test_schedule_error_not_caught_as_tweet_type_error():
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError("x")
        except TweetTypeError:  # pragma: no cover - must not trigger
            pytest.fail("ScheduleError must not be a TweetTypeError")


def test_schedule_error_and_tweet_type_error_are_distinct_classes():
    assert ScheduleError is not TweetTypeError
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)


# ---------------------------------------------------------------------------
# TemplateError (companion exception in the same module)
# ---------------------------------------------------------------------------

def test_template_error_constants_and_base():
    assert issubclass(TemplateError, TypeError)
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."
