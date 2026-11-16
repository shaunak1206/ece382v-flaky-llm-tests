import inspect
import re

import pytest

import coo
from coo import Coo
from coo import coo as coo_module
from coo._exceptions import ScheduleError, TweetTypeError


# ---------------------------------------------------------------------------
# Coo class and package exports
# ---------------------------------------------------------------------------

def test_coo_is_exported_at_package_level():
    assert hasattr(coo, "Coo")
    assert coo.Coo is Coo


def test_coo_is_a_class():
    assert inspect.isclass(Coo)


def test_coo_is_same_object_as_defined_in_submodule():
    assert Coo is coo_module.Coo


def test_coo_defined_in_coo_module_path():
    assert Coo.__module__ == "coo.coo"


def test_coo_class_has_a_constructor_signature():
    # Inspecting the signature must not raise; this confirms a callable class.
    sig = inspect.signature(Coo)
    assert isinstance(sig.parameters, dict) or sig.parameters is not None


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

def test_metadata_title():
    assert coo.__title__ == "coo"


def test_metadata_version():
    assert coo.__version__ == "0.1.3"


def test_metadata_license():
    assert coo.__license__ == "Apache 2.0"


def test_metadata_author():
    assert coo.__author__ == "Carlos Montecinos Geisse"


def test_metadata_url():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


# ---------------------------------------------------------------------------
# TweetTypeError
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_wrong_list_message_value():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_raised_and_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_message_matches_pattern():
    with pytest.raises(TweetTypeError, match=re.escape(TweetTypeError.wrongListMsg)):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_instance_args_hold_message():
    err = TweetTypeError(TweetTypeError.wrongListMsg)
    assert err.args == (TweetTypeError.wrongListMsg,)


# ---------------------------------------------------------------------------
# ScheduleError
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_wrong_list_message_value():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tuple_length_message_value():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_past_date_message_value():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_schedule_error_messages_are_distinct():
    messages = {
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    }
    assert len(messages) == 3


def test_schedule_error_raised_and_caught_as_type_error():
    with pytest.raises(TypeError, match=re.escape(ScheduleError.pastDateError)):
        raise ScheduleError(ScheduleError.pastDateError)


def test_schedule_error_and_tweet_type_error_are_independent():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)
