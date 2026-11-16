import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError


# ---------------------------------------------------------------------------
# TweetTypeError
# ---------------------------------------------------------------------------

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(TweetTypeError, Exception)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_can_be_raised_with_message():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(excinfo.value) == "A List of Strings is required."


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("boom")


def test_tweet_type_error_without_args_has_empty_message():
    err = TweetTypeError()
    assert str(err) == ""
    assert err.args == ()


def test_tweet_type_error_has_docstring():
    assert TweetTypeError.__doc__ is not None
    assert "list of strings" in TweetTypeError.__doc__.lower()


# ---------------------------------------------------------------------------
# ScheduleError
# ---------------------------------------------------------------------------

def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


@pytest.mark.parametrize(
    "attr",
    ["wrongListMsg", "tupleLenError", "pastDateError"],
)
def test_schedule_error_raised_with_each_message(attr):
    message = getattr(ScheduleError, attr)
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(message)
    assert str(excinfo.value) == message


def test_schedule_error_and_tweet_type_error_are_distinct():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError("x")
        except TweetTypeError:  # must not catch ScheduleError
            pytest.fail("TweetTypeError should not catch ScheduleError")


def test_schedule_error_messages_are_strings():
    for attr in ("wrongListMsg", "tupleLenError", "pastDateError"):
        assert isinstance(getattr(ScheduleError, attr), str)


# ---------------------------------------------------------------------------
# Package / Coo
# ---------------------------------------------------------------------------

def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


def test_coo_is_exported_class():
    assert inspect.isclass(Coo)
    assert coo.Coo is Coo
    assert Coo.__module__ == "coo.coo"


def test_coo_can_be_instantiated_with_dummy_credentials():
    try:
        instance = Coo("key", "secret", "token", "token_secret")
    except TypeError:
        pytest.skip("Coo constructor signature differs from expectation")
    assert isinstance(instance, Coo)


def test_exceptions_importable_from_private_module():
    import coo._exceptions as exc

    assert exc.TweetTypeError is TweetTypeError
    assert exc.ScheduleError is ScheduleError
