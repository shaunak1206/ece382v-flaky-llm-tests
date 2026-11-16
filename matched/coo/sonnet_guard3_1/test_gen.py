import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_template_error_is_type_error_subclass():
    assert issubclass(TemplateError, TypeError)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_template_error_message_constants():
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_raise_tweet_type_error_with_class_message():
    with pytest.raises(TweetTypeError, match="List of Strings"):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("bad")


def test_schedule_error_str_matches_argument():
    err = ScheduleError(ScheduleError.pastDateError)
    assert str(err) == "A future date is needed."
    assert err.args == ("A future date is needed.",)


def test_schedule_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise ScheduleError(ScheduleError.tupleLenError)


def test_schedule_error_not_caught_as_tweet_type_error():
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError("x")
        except TweetTypeError:  # pragma: no cover
            pytest.fail("ScheduleError must not be a TweetTypeError")


def test_exception_classes_are_distinct():
    assert TweetTypeError is not ScheduleError
    assert not issubclass(TweetTypeError, ScheduleError)
    assert not issubclass(ScheduleError, TweetTypeError)


def test_exception_without_args_has_empty_message():
    assert str(TweetTypeError()) == ""
    assert str(ScheduleError()) == ""


def test_coo_is_a_class_exported_from_package():
    assert isinstance(Coo, type)
    assert coo.Coo is Coo
    assert Coo.__name__ == "Coo"


def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"
