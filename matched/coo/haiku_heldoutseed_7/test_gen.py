import inspect

import pytest

import coo
from coo import Coo
from coo import coo as coo_module
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


# --- Package metadata (coo/__init__.py) ---

def test_package_title():
    assert coo.__title__ == "coo"


def test_package_license():
    assert coo.__license__ == "Apache 2.0"


def test_package_version_value():
    assert coo.__version__ == "0.1.3"


def test_package_version_format():
    parts = coo.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_package_author_and_contact():
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"


def test_package_url():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


# --- Coo class export (coo/__init__.py) ---

def test_coo_is_a_class():
    assert inspect.isclass(Coo)


def test_coo_exported_from_package_is_same_object():
    assert coo.Coo is coo_module.Coo


def test_coo_defined_in_coo_module():
    assert Coo.__module__ == "coo.coo"


def test_coo_has_callable_init():
    assert callable(Coo.__init__)


def test_coo_is_not_an_exception():
    assert not issubclass(Coo, BaseException)


# --- TweetTypeError (coo/_exceptions.py) ---

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_message_attribute():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_with_message():
    with pytest.raises(TweetTypeError, match="A List of Strings is required."):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_can_be_instantiated_without_args():
    err = TweetTypeError()
    assert isinstance(err, TweetTypeError)
    assert err.args == ()


# --- ScheduleError (coo/_exceptions.py) ---

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


@pytest.mark.parametrize(
    "attr, expected",
    [
        ("wrongListMsg", "A list of tuples is required."),
        ("tupleLenError", "Every tuple need a length of 3 or 4."),
        ("pastDateError", "A future date is needed."),
    ],
)
def test_schedule_error_message_attributes(attr, expected):
    assert getattr(ScheduleError, attr) == expected


def test_schedule_error_messages_are_distinct():
    messages = {
        ScheduleError.wrongListMsg,
        ScheduleError.tupleLenError,
        ScheduleError.pastDateError,
    }
    assert len(messages) == 3


def test_schedule_error_raised_with_tuple_message():
    with pytest.raises(ScheduleError, match="tuples"):
        raise ScheduleError(ScheduleError.wrongListMsg)


def test_schedule_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise ScheduleError(ScheduleError.pastDateError)


# --- Relationships between exception classes ---

def test_tweet_and_schedule_errors_are_unrelated():
    assert not issubclass(TweetTypeError, ScheduleError)
    assert not issubclass(ScheduleError, TweetTypeError)


def test_template_error_is_type_error_subclass():
    assert issubclass(TemplateError, TypeError)


@pytest.mark.parametrize("exc_cls", [TweetTypeError, ScheduleError])
def test_custom_errors_not_caught_by_unrelated_handler(exc_cls):
    with pytest.raises(exc_cls):
        try:
            raise exc_cls("boom")
        except ValueError:
            pytest.fail("ValueError handler should not catch TypeError subclasses")
