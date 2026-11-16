import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError
from coo import coo as coo_module


# ---------------------------------------------------------------------------
# Package metadata (coo/__init__.py)
# ---------------------------------------------------------------------------

def test_title_is_coo():
    assert coo.__title__ == "coo"


def test_license_is_apache_2():
    assert coo.__license__ == "Apache 2.0"


def test_version_is_three_part_numeric_string():
    parts = coo.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_version_value():
    assert coo.__version__ == "0.1.3"


def test_author_is_set():
    assert coo.__author__ == "Carlos Montecinos Geisse"


def test_contact_is_set():
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"


def test_url_points_to_github_repo():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


# ---------------------------------------------------------------------------
# Coo export (coo/__init__.py)
# ---------------------------------------------------------------------------

def test_coo_is_exported_as_class():
    assert inspect.isclass(Coo)


def test_coo_top_level_is_same_object_as_submodule_class():
    assert Coo is coo_module.Coo


def test_coo_is_defined_in_submodule():
    assert Coo.__module__ == "coo.coo"


# ---------------------------------------------------------------------------
# TweetTypeError (coo/_exceptions.py)
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_wrong_list_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_and_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_message_is_preserved():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(excinfo.value) == "A List of Strings is required."


# ---------------------------------------------------------------------------
# ScheduleError (coo/_exceptions.py)
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_wrong_list_message():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tuple_length_message():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_past_date_message():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_schedule_error_can_be_raised_and_caught():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)
    assert str(excinfo.value) == "A future date is needed."


# ---------------------------------------------------------------------------
# Relationship between the exception classes
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_not_schedule_error():
    assert not issubclass(TweetTypeError, ScheduleError)


def test_schedule_error_is_not_tweet_type_error():
    assert not issubclass(ScheduleError, TweetTypeError)


def test_schedule_error_messages_are_distinct():
    messages = {
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    }
    assert len(messages) == 3
