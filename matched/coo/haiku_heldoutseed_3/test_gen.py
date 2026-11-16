import inspect
import re

import pytest

import coo
from coo import Coo
from coo import coo as coo_module
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_template_error_is_type_error_subclass():
    assert issubclass(TemplateError, TypeError)


def test_tweet_type_error_and_schedule_error_are_distinct():
    assert not issubclass(TweetTypeError, ScheduleError)
    assert not issubclass(ScheduleError, TweetTypeError)


def test_tweet_type_error_wrong_list_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_wrong_list_message():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tuple_len_message():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_past_date_message():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_template_error_messages():
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_tweet_type_error_can_be_raised_and_matched_by_message():
    with pytest.raises(TweetTypeError, match="A List of Strings is required."):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_schedule_error_can_be_raised_and_caught_as_type_error():
    with pytest.raises(TypeError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)
    assert isinstance(excinfo.value, ScheduleError)
    assert str(excinfo.value) == "A future date is needed."


def test_exception_str_preserves_passed_message():
    err = TweetTypeError("custom message")
    assert str(err) == "custom message"


def test_exception_without_args_has_empty_str():
    assert str(ScheduleError()) == ""


def test_schedule_error_not_caught_by_unrelated_except():
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError(ScheduleError.tupleLenError)
        except ValueError:
            pytest.fail("ScheduleError should not be caught as ValueError")


def test_package_metadata_values():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


def test_package_version_is_dotted_numeric():
    assert re.match(r"^\d+\.\d+\.\d+$", coo.__version__)


def test_coo_is_a_class():
    assert inspect.isclass(Coo)


def test_coo_exported_from_package_is_same_as_submodule_class():
    assert coo_module.Coo is Coo
    assert Coo.__module__.startswith("coo")
