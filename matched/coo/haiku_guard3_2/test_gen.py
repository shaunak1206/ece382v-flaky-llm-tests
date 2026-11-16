"""Unit tests for the coo package: the Coo export, ScheduleError and TweetTypeError.

Only coo/__init__.py and coo/_exceptions.py were provided, so these tests cover
the public surface visible from those files: the exception classes, their
messages and hierarchy, the package metadata, and the Coo export.
They use no randomness, no clock, no sleeps, no global state changes and
no set or dict ordering.
"""

import importlib
import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError


# --- TweetTypeError ---------------------------------------------------------

def test_tweet_type_error_is_subclass_of_type_error():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_wrong_list_msg_value():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_with_its_message():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert excinfo.value.args[0] == "A List of Strings is required."


def test_tweet_type_error_is_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("boom")


def test_tweet_type_error_str_reflects_custom_message():
    assert str(TweetTypeError("custom")) == "custom"


# --- ScheduleError ----------------------------------------------------------

def test_schedule_error_is_subclass_of_type_error():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_wrong_list_msg_value():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tuple_len_msg_value():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_past_date_msg_value():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_schedule_error_messages_are_all_distinct():
    messages = {
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    }
    assert len(messages) == 3


def test_schedule_error_can_be_raised_and_caught_as_type_error():
    with pytest.raises(TypeError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)
    assert isinstance(excinfo.value, ScheduleError)
    assert excinfo.value.args[0] == "A future date is needed."


def test_schedule_error_str_reflects_custom_message():
    assert str(ScheduleError("custom")) == "custom"


# --- Relationship between the two exceptions --------------------------------

def test_schedule_error_and_tweet_type_error_are_unrelated():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)


def test_tweet_type_error_instance_is_not_schedule_error():
    exc = TweetTypeError(TweetTypeError.wrongListMsg)
    assert not isinstance(exc, ScheduleError)


# --- Coo --------------------------------------------------------------------

def test_coo_is_a_class():
    assert inspect.isclass(Coo)


def test_coo_is_not_an_exception():
    assert not issubclass(Coo, BaseException)


def test_coo_export_is_same_object_as_submodule_class():
    submodule = importlib.import_module("coo.coo")
    assert Coo is submodule.Coo


def test_package_exposes_coo_attribute():
    assert hasattr(coo, "Coo")
    assert coo.Coo is Coo


# --- Package metadata -------------------------------------------------------

def test_package_version():
    assert coo.__version__ == "0.1.3"


def test_package_title_and_license():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"


def test_package_author_and_url():
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"
