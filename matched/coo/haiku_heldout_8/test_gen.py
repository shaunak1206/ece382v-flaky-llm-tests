import inspect
import re

import pytest

import coo
from coo import Coo
from coo import coo as coo_module
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


# --- Package metadata -------------------------------------------------------

def test_package_title():
    assert coo.__title__ == "coo"


def test_package_license():
    assert coo.__license__ == "Apache 2.0"


def test_package_version_is_semver_like():
    assert re.fullmatch(r"\d+\.\d+\.\d+", coo.__version__)


def test_package_author_and_contact():
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"


def test_package_url():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


# --- Coo class export -------------------------------------------------------

def test_coo_is_exported_as_class():
    assert inspect.isclass(Coo)


def test_coo_is_same_object_as_submodule_class():
    assert Coo is coo_module.Coo


def test_coo_name_is_coo():
    assert Coo.__name__ == "Coo"


# --- TweetTypeError ---------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_is_exception_instance():
    assert isinstance(TweetTypeError("x"), Exception)


def test_tweet_type_error_wrong_list_message_value():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_message_is_preserved_when_raised():
    with pytest.raises(TweetTypeError, match="List of Strings"):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("bad input")


def test_tweet_type_error_does_not_subclass_schedule_error():
    assert not issubclass(TweetTypeError, ScheduleError)


# --- ScheduleError ----------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_class_message_values():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_schedule_error_messages_are_strings():
    for attr in ("wrongListMsg", "tupleLenError", "pastDateError"):
        assert isinstance(getattr(ScheduleError, attr), str)


def test_schedule_error_messages_are_distinct():
    messages = {
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    }
    assert len(messages) == 3


def test_schedule_error_raised_with_tuple_length_message():
    with pytest.raises(ScheduleError, match="length of 3 or 4"):
        raise ScheduleError(ScheduleError.tupleLenError)


def test_schedule_error_raised_with_past_date_message():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)
    assert str(excinfo.value) == "A future date is needed."


def test_schedule_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise ScheduleError("bad schedule")


def test_schedule_error_does_not_subclass_tweet_type_error():
    assert not issubclass(ScheduleError, TweetTypeError)


def test_template_error_is_separate_type_error_subclass():
    assert issubclass(TemplateError, TypeError)
    assert not issubclass(TemplateError, ScheduleError)
