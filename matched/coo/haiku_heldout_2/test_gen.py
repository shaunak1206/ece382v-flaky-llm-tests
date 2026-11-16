import re

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError
from coo import coo as coo_module


# ---------------------------------------------------------------------------
# Package metadata (coo/__init__.py)
# ---------------------------------------------------------------------------

def test_package_title():
    assert coo.__title__ == "coo"


def test_package_license():
    assert coo.__license__ == "Apache 2.0"


def test_package_version_is_string():
    assert isinstance(coo.__version__, str)
    assert coo.__version__ == "0.1.3"


def test_package_author_and_contact():
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"


def test_package_url():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


# ---------------------------------------------------------------------------
# Coo class export (coo/__init__.py)
# ---------------------------------------------------------------------------

def test_coo_is_exported_at_package_level():
    assert hasattr(coo, "Coo")


def test_coo_exported_object_is_a_class():
    assert isinstance(Coo, type)


def test_coo_is_same_object_as_submodule_class():
    assert Coo is coo_module.Coo


def test_coo_class_defined_in_coo_submodule():
    assert Coo.__module__ == "coo.coo"


# ---------------------------------------------------------------------------
# TweetTypeError (coo/_exceptions.py)
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_is_exception_instance_when_created():
    err = TweetTypeError("msg")
    assert isinstance(err, Exception)
    assert isinstance(err, TypeError)


def test_tweet_type_error_wrong_list_message_value():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_with_class_message():
    with pytest.raises(TweetTypeError, match=re.escape(TweetTypeError.wrongListMsg)):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_is_not_schedule_error():
    assert not issubclass(TweetTypeError, ScheduleError)


# ---------------------------------------------------------------------------
# ScheduleError (coo/_exceptions.py)
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_wrong_list_message_value():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tuple_len_message_value():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_past_date_message_value():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_schedule_error_messages_are_all_strings():
    for name in ("wrongListMsg", "tupleLenError", "pastDateError"):
        assert isinstance(getattr(ScheduleError, name), str)


def test_schedule_error_can_be_raised_with_past_date_message():
    with pytest.raises(ScheduleError, match=re.escape(ScheduleError.pastDateError)):
        raise ScheduleError(ScheduleError.pastDateError)


def test_schedule_error_wrong_list_and_tweet_type_messages_differ():
    assert ScheduleError.wrongListMsg != TweetTypeError.wrongListMsg
