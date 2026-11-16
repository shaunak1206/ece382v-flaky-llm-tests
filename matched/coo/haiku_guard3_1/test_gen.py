import inspect
import re

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


# ---------------------------------------------------------------------------
# Package-level Coo export
# ---------------------------------------------------------------------------

def test_package_exports_coo_class():
    assert hasattr(coo, "Coo")
    assert coo.Coo is Coo


def test_coo_is_a_class():
    assert inspect.isclass(Coo)


def test_coo_is_defined_in_coo_submodule():
    from coo.coo import Coo as SubmoduleCoo

    assert SubmoduleCoo is Coo
    assert Coo.__module__ == "coo.coo"


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

def test_package_title():
    assert coo.__title__ == "coo"


def test_package_license():
    assert coo.__license__ == "Apache 2.0"


def test_package_version():
    assert coo.__version__ == "0.1.3"


def test_package_author():
    assert coo.__author__ == "Carlos Montecinos Geisse"


def test_package_contact():
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"


def test_package_url():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


# ---------------------------------------------------------------------------
# TweetTypeError
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_wrong_list_message_value():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_and_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_str_matches_message():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(excinfo.value) == "A List of Strings is required."


def test_tweet_type_error_has_docstring():
    assert TweetTypeError.__doc__ is not None
    assert "list of strings" in TweetTypeError.__doc__


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


def test_schedule_error_messages_are_distinct():
    messages = {
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    }
    assert len(messages) == 3


def test_schedule_error_can_be_raised_with_message_and_matched():
    message = ScheduleError.pastDateError
    with pytest.raises(ScheduleError, match=re.escape(message)):
        raise ScheduleError(message)


# ---------------------------------------------------------------------------
# Relationships between exception classes
# ---------------------------------------------------------------------------

def test_tweet_type_error_and_schedule_error_are_unrelated():
    assert not issubclass(TweetTypeError, ScheduleError)
    assert not issubclass(ScheduleError, TweetTypeError)


def test_schedule_error_not_caught_by_tweet_type_error_handler():
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError(ScheduleError.wrongListMsg)
        except TweetTypeError:
            pytest.fail("ScheduleError should not be caught as TweetTypeError")


def test_template_error_is_type_error_subclass_for_context():
    assert issubclass(TemplateError, TypeError)
