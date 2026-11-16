import inspect
import re

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError


# ---------------------------------------------------------------------------
# TweetTypeError
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_wrong_list_message_value():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_wrong_list_message_is_str():
    assert isinstance(TweetTypeError.wrongListMsg, str)


def test_tweet_type_error_can_be_raised_and_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_message_is_preserved():
    with pytest.raises(TweetTypeError, match=re.escape(TweetTypeError.wrongListMsg)):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_without_args_has_empty_str():
    assert str(TweetTypeError()) == ""


# ---------------------------------------------------------------------------
# ScheduleError
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


def test_schedule_error_can_be_raised_with_each_message():
    for message in (
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    ):
        with pytest.raises(ScheduleError, match=re.escape(message)):
            raise ScheduleError(message)


def test_schedule_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise ScheduleError(ScheduleError.pastDateError)


# ---------------------------------------------------------------------------
# Relationship between the exception classes
# ---------------------------------------------------------------------------

def test_schedule_error_and_tweet_type_error_are_distinct():
    assert ScheduleError is not TweetTypeError
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)


def test_exceptions_are_exported_from_private_module():
    assert ScheduleError.__module__ == "coo._exceptions"
    assert TweetTypeError.__module__ == "coo._exceptions"


# ---------------------------------------------------------------------------
# Package-level exports and metadata
# ---------------------------------------------------------------------------

def test_package_exposes_coo_class():
    assert inspect.isclass(Coo)
    assert coo.Coo is Coo
    assert Coo.__name__ == "Coo"


def test_package_metadata_title_and_license():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"


def test_package_version_is_semver_string():
    assert isinstance(coo.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", coo.__version__)


def test_package_metadata_author_and_url():
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"
    assert "@" in coo.__contact__
