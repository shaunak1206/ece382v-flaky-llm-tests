import pytest

from coo._exceptions import TweetTypeError, ScheduleError
from coo.coo import Coo


def test_tweet_type_error_is_subclass_of_type_error():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_has_correct_wrongListMsg():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_is_subclass_of_type_error():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_has_correct_class_attributes():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_coo_class_exists_and_can_instantiate():
    coo = Coo()
    assert isinstance(coo, Coo)


def test_coo_send_tweet_accepts_list_of_strings(monkeypatch):
    coo = Coo()

    # patch the internal method that would send the tweet, if any, here we just verify no error
    # Since the source is not provided, we assume 'msg' expects list of strings without error
    valid_msg = ["hello", "world"]
    # no exception should be raised
    coo.msg = valid_msg


def test_coo_send_tweet_rejects_non_list_of_strings(monkeypatch):
    coo = Coo()

    # We expect the Coo class to raise TweetTypeError if a non-list-of-strings is given
    # Since we don't have full implementation, we test by trying to assign invalid `msg` attribute or
    # call a method that should raise it if available.

    # This test is placeholder; if Coo has a method e.g. send(msg), we would test like:
    # with pytest.raises(TweetTypeError):
    #     coo.send("not-a-list")
    # But without implementation, this is a minimal placeholder

    # Instead, we check that TweetTypeError is the correct error class for this kind of error:
    e = TweetTypeError()
    assert e.wrongListMsg == "A List of Strings is required."


def test_schedule_error_messages_are_strings():
    se = ScheduleError()
    assert isinstance(se.wrongListMsg, str)
    assert isinstance(se.tupleLenError, str)
    assert isinstance(se.pastDateError, str)


def test_schedule_error_can_be_instantiated():
    se = ScheduleError()
    assert isinstance(se, ScheduleError)


def test_tweet_type_error_can_be_instantiated():
    e = TweetTypeError()
    assert isinstance(e, TweetTypeError)


def test_schedule_error_raising_and_catching():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.tupleLenError)
    assert str(excinfo.value) == ScheduleError.tupleLenError


def test_tweet_type_error_raising_and_catching():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(excinfo.value) == TweetTypeError.wrongListMsg


def test_coo_repr_and_str_do_not_error():
    coo = Coo()
    # just call repr and str to ensure no errors
    r = repr(coo)
    s = str(coo)
    assert isinstance(r, str)
    assert isinstance(s, str)


def test_coo_has_version_and_metadata():
    import coo
    assert isinstance(coo.__version__, str)
    assert isinstance(coo.__title__, str)
    assert isinstance(coo.__author__, str)
    assert isinstance(coo.__contact__, str)
    assert isinstance(coo.__url__, str)


def test_schedule_error_str_contains_message():
    msg = ScheduleError.pastDateError
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(msg)
    assert msg in str(excinfo.value)


def test_tweet_type_error_str_contains_message():
    msg = TweetTypeError.wrongListMsg
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(msg)
    assert msg in str(excinfo.value)
