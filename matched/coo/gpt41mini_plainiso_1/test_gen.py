import pytest
from coo._exceptions import TweetTypeError, ScheduleError


def test_tweet_type_error_inheritance():
    err = TweetTypeError()
    assert isinstance(err, TypeError)
    assert isinstance(err, TweetTypeError)


def test_tweet_type_error_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_inheritance():
    err = ScheduleError()
    assert isinstance(err, TypeError)
    assert isinstance(err, ScheduleError)


def test_schedule_error_messages():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweet_type_error_str_message():
    # When raised with a message, the message should be str(err)
    msg = "Invalid tweet list"
    err = TweetTypeError(msg)
    assert str(err) == msg


def test_schedule_error_str_message():
    msg = "Invalid schedule"
    err = ScheduleError(msg)
    assert str(err) == msg


def test_tweet_type_error_default_message_empty():
    # default error has no message if not set explicitly
    err = TweetTypeError()
    assert str(err) == ""


def test_schedule_error_default_message_empty():
    err = ScheduleError()
    assert str(err) == ""


def test_tweet_type_error_usage_example():
    # simulate a function raising this error when input not list of strings
    def func(msg):
        if not (isinstance(msg, list) and all(isinstance(i, str) for i in msg)):
            raise TweetTypeError(TweetTypeError.wrongListMsg)
        return True

    assert func(["hello", "world"])
    with pytest.raises(TweetTypeError) as e:
        func("not a list")
    assert str(e.value) == TweetTypeError.wrongListMsg


def test_schedule_error_usage_example():
    # simulate a function raising ScheduleError on invalid schedule
    def func(schedule):
        if not (isinstance(schedule, list) and all(isinstance(tpl, tuple) for tpl in schedule)):
            raise ScheduleError(ScheduleError.wrongListMsg)
        for tpl in schedule:
            if len(tpl) not in (3, 4):
                raise ScheduleError(ScheduleError.tupleLenError)
        return True

    assert func([(1, 2, 3), (4, 5, 6, 7)])
    with pytest.raises(ScheduleError) as e:
        func("not a list")
    assert str(e.value) == ScheduleError.wrongListMsg

    with pytest.raises(ScheduleError) as e:
        func([(1, 2)])
    assert str(e.value) == ScheduleError.tupleLenError


def test_schedule_error_past_date_error_message():
    # test the presence and content of the pastDateError message constant
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweet_type_error_repr_and_str():
    err = TweetTypeError(TweetTypeError.wrongListMsg)
    assert repr(err) and str(err) == TweetTypeError.wrongListMsg


def test_schedule_error_repr_and_str():
    err = ScheduleError(ScheduleError.tupleLenError)
    assert repr(err) and str(err) == ScheduleError.tupleLenError
