import pytest
from coo import Coo
from coo._exceptions import TweetTypeError, ScheduleError


def test_Coo_instance_creation():
    c = Coo()
    assert isinstance(c, Coo)


def test_Coo_invalid_msg_type_raises_TweetTypeError():
    c = Coo()
    # Assuming Coo expects a list of strings as msg and raises TweetTypeError otherwise
    with pytest.raises(TweetTypeError):
        c.tweet(msg="not a list")  # passing string instead of list
    with pytest.raises(TweetTypeError):
        c.tweet(msg=[1, 2, 3])  # list but not all strings
    with pytest.raises(TweetTypeError):
        c.tweet(msg=[None])  # list with None


def test_TweetTypeError_message():
    err = TweetTypeError()
    assert hasattr(err, "wrongListMsg")
    assert err.wrongListMsg == "A List of Strings is required."
    # Also check that raising it works normally
    with pytest.raises(TweetTypeError):
        raise TweetTypeError(err.wrongListMsg)


def test_ScheduleError_class_attributes():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_ScheduleError_raises_and_has_correct_message():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.wrongListMsg)
    assert str(excinfo.value) == ScheduleError.wrongListMsg


def test_coo_tweet_accepts_valid_msg_list():
    c = Coo()
    valid_msg = ["Hello", "World"]
    # If Coo has a tweet method, it should accept list of strings without error
    try:
        c.tweet(msg=valid_msg)
    except TweetTypeError:
        pytest.fail("tweet() raised TweetTypeError unexpectedly with valid input.")


def test_ScheduleError_inheritance():
    # ScheduleError inherits from TypeError
    assert issubclass(ScheduleError, TypeError)
    assert isinstance(ScheduleError(), ScheduleError)
    assert isinstance(ScheduleError(), TypeError)


def test_TweetTypeError_inheritance():
    # TweetTypeError inherits from TypeError
    assert issubclass(TweetTypeError, TypeError)
    assert isinstance(TweetTypeError(), TweetTypeError)
    assert isinstance(TweetTypeError(), TypeError)


def test_ScheduleError_tupleLenError_message():
    err = ScheduleError(ScheduleError.tupleLenError)
    assert str(err) == "Every tuple need a length of 3 or 4."


def test_raise_ScheduleError_pastDateError_message():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)
    assert str(excinfo.value) == "A future date is needed."


def test_raise_TweetTypeError_with_wrongListMsg():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(excinfo.value) == "A List of Strings is required."


def test_Coo_version_and_metadata():
    # Check if coo module metadata exist and are strings
    import coo
    assert hasattr(coo, "__version__")
    assert isinstance(coo.__version__, str)
    assert hasattr(coo, "__title__")
    assert isinstance(coo.__title__, str)
    assert hasattr(coo, "__license__")
    assert isinstance(coo.__license__, str)
    assert hasattr(coo, "__author__")
    assert isinstance(coo.__author__, str)
    assert hasattr(coo, "__contact__")
    assert isinstance(coo.__contact__, str)
    assert hasattr(coo, "__url__")
    assert isinstance(coo.__url__, str)
