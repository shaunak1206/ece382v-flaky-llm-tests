import inspect
import re

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


# ---------------------------------------------------------------------------
# Package-level exports and metadata
# ---------------------------------------------------------------------------

def test_coo_is_exported_as_class():
    assert inspect.isclass(Coo)


def test_coo_export_is_same_object_as_defined_in_submodule():
    from coo.coo import Coo as SubmoduleCoo

    assert Coo is SubmoduleCoo


def test_coo_defined_in_coo_module():
    assert Coo.__module__ == "coo.coo"


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


# ---------------------------------------------------------------------------
# TweetTypeError
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_and_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_carries_message():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(excinfo.value) == "A List of Strings is required."


def test_tweet_type_error_is_not_template_error():
    assert not issubclass(TweetTypeError, TemplateError)


# ---------------------------------------------------------------------------
# ScheduleError
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_wrong_list_message():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tuple_length_message():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_past_date_message():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_schedule_error_messages_are_distinct():
    messages = {
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    }
    assert len(messages) == 3


def test_schedule_error_can_be_raised_with_each_message():
    for message in (
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    ):
        with pytest.raises(ScheduleError) as excinfo:
            raise ScheduleError(message)
        assert str(excinfo.value) == message


def test_schedule_error_is_not_tweet_type_error():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)
