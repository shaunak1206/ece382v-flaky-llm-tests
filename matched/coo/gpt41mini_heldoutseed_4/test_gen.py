import pytest
from coo._exceptions import TweetTypeError, ScheduleError
from coo.coo import Coo


def test_tweet_type_error_inheritance_and_message():
    err = TweetTypeError()
    assert isinstance(err, TypeError)
    assert hasattr(err, "wrongListMsg")
    assert err.wrongListMsg == "A List of Strings is required."


def test_schedule_error_inheritance_and_messages():
    err = ScheduleError()
    assert isinstance(err, TypeError)
    assert hasattr(err, "wrongListMsg")
    assert hasattr(err, "tupleLenError")
    assert hasattr(err, "pastDateError")

    assert err.wrongListMsg == "A list of tuples is required."
    assert err.tupleLenError == "Every tuple need a length of 3 or 4."
    assert err.pastDateError == "A future date is needed."


def test_coo_init_without_params():
    c = Coo()
    # Default attributes should be set (checked for example attributes)
    # as Coo is the core class (basic attributes presence)
    assert hasattr(c, "tweets")
    assert hasattr(c, "schedule")
    # tweets should be a list (default)
    assert isinstance(c.tweets, list)
    # schedule should be a list (default)
    assert isinstance(c.schedule, list)


def test_coo_accepts_valid_tweets_list():
    # a normal list of string tweets
    tweets = ["Hello", "World!"]
    c = Coo(tweets=tweets)
    assert c.tweets == tweets


def test_coo_rejects_tweets_wrong_type():
    # tweets argument should be a list of strings, else raise TweetTypeError
    with pytest.raises(TweetTypeError):
        Coo(tweets="Hello World")  # str not list

    with pytest.raises(TweetTypeError):
        Coo(tweets=[1, 2, 3])  # list of ints

    with pytest.raises(TweetTypeError):
        Coo(tweets=[["nested"], "string"])  # list with nested list


def test_coo_accepts_valid_schedule_list():
    # schedule should be list of tuples with len 3 or 4, no past dates
    schedule = [
        ("msg1", "2025-01-01", 1),
        ("msg2", "2025-12-31", 4, "extra"),
    ]
    c = Coo(schedule=schedule)
    assert c.schedule == schedule


def test_coo_rejects_schedule_wrong_type():
    with pytest.raises(ScheduleError):
        Coo(schedule="not a list")

    with pytest.raises(ScheduleError):
        Coo(schedule=[("msg", "date")])  # tuple length too short

    with pytest.raises(ScheduleError):
        Coo(schedule=[("msg", "date", 1, 2, 3)])  # tuple length too long


def test_coo_rejects_schedule_past_date():
    # We test with a date in the past format "YYYY-MM-DD"
    from datetime import datetime, timedelta

    past_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    schedule = [("msg", past_date, 1)]

    with pytest.raises(ScheduleError):
        Coo(schedule=schedule)


def test_coo_additional_tweet_types_raise_error():
    # tweets must be list of strings only, no other iterable
    with pytest.raises(TweetTypeError):
        Coo(tweets=[b"bytes", "valid"])

    with pytest.raises(TweetTypeError):
        Coo(tweets=[None, "valid"])


def test_coo_str_and_repr_methods():
    c = Coo(tweets=["test"])
    s = str(c)
    r = repr(c)
    assert isinstance(s, str)
    assert isinstance(r, str)
    assert "Coo" in r


def test_coo_schedule_tuple_lengths_valid_and_invalid():
    # valid lengths: 3 or 4
    valid_3 = ("msg", "2099-12-31", 2)
    valid_4 = ("msg", "2099-12-31", 2, "optional")

    c = Coo(schedule=[valid_3, valid_4])
    assert c.schedule == [valid_3, valid_4]

    # invalid lengths:
    with pytest.raises(ScheduleError):
        Coo(schedule=[("msg", "2099-12-31")])  # length 2

    with pytest.raises(ScheduleError):
        Coo(schedule=[("msg", "2099-12-31", 2, "extra", "too much")])  # length 5


def test_tweet_type_error_message_is_class_variable():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_messages_are_class_variables():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."
