import pytest
from coo import Coo
from coo._exceptions import TweetTypeError, ScheduleError


def test_coo_instance_creation():
    coo = Coo()
    assert isinstance(coo, Coo)


def test_tweet_type_error_is_type_error_subclass():
    err = TweetTypeError()
    assert isinstance(err, TypeError)
    assert hasattr(TweetTypeError, "wrongListMsg")
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_is_type_error_subclass():
    err = ScheduleError()
    assert isinstance(err, TypeError)
    assert hasattr(ScheduleError, "wrongListMsg")
    assert hasattr(ScheduleError, "tupleLenError")
    assert hasattr(ScheduleError, "pastDateError")

    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_coo_tweetmsg_list_of_strings_success():
    coo = Coo()
    # Should accept list of strings
    try:
        coo._validate_tweet_msg(["hello", "world"])
    except Exception as e:
        pytest.fail(f"_validate_tweet_msg raised unexpectedly: {e}")


def test_coo_tweetmsg_not_list_raises_tweettTypeError():
    coo = Coo()
    with pytest.raises(TweetTypeError):
        coo._validate_tweet_msg("not a list")


def test_coo_tweetmsg_list_not_strings_raises_tweet_type_error():
    coo = Coo()
    with pytest.raises(TweetTypeError):
        coo._validate_tweet_msg(["valid", 123, "string"])


def test_coo_schedule_valid_input_succeeds():
    coo = Coo()
    import datetime

    # A tuple with length 3, date in the future
    future_date = datetime.datetime.now() + datetime.timedelta(days=1)
    schedule = [(future_date, "msg", "ignore")]
    try:
        coo._validate_schedule(schedule)
    except Exception as e:
        pytest.fail(f"_validate_schedule raised unexpectedly: {e}")


def test_coo_schedule_not_list_raises_schedule_error():
    coo = Coo()
    with pytest.raises(ScheduleError):
        coo._validate_schedule("not a list")


def test_coo_schedule_tuple_wrong_length_raises_schedule_error():
    coo = Coo()
    import datetime
    future_date = datetime.datetime.now() + datetime.timedelta(days=1)
    schedule = [(future_date, "msg")]  # tuple length 2 is invalid
    with pytest.raises(ScheduleError):
        coo._validate_schedule(schedule)


def test_coo_schedule_past_date_raises_schedule_error():
    coo = Coo()
    import datetime
    past_date = datetime.datetime.now() - datetime.timedelta(days=1)
    schedule = [(past_date, "msg", "extra")]
    with pytest.raises(ScheduleError):
        coo._validate_schedule(schedule)


def test_coo_schedule_tuple_of_len_4_accepts():
    coo = Coo()
    import datetime
    future_date = datetime.datetime.now() + datetime.timedelta(days=1)
    # tuple length 4 also valid
    schedule = [(future_date, "msg", "extra", "more")]
    try:
        coo._validate_schedule(schedule)
    except Exception as e:
        pytest.fail(f"_validate_schedule raised unexpectedly: {e}")


def test_tweet_type_error_message_content():
    # Confirm class docstring and message attribute presence
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."
    assert "list of strings" in TweetTypeError.__doc__.lower()


def test_schedule_error_message_contents():
    # Confirm all messages contain expected keywords
    assert "list" in ScheduleError.wrongListMsg
    assert "tuple" in ScheduleError.tupleLenError
    assert "future" in ScheduleError.pastDateError


def test_coo_version_and_metadata():
    import coo
    assert coo.__version__ == "0.1.3"
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"

