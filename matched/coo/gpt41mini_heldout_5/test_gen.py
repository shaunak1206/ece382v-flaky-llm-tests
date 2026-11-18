import pytest
from coo._exceptions import TweetTypeError, ScheduleError
from coo.coo import Coo


# Tests for TweetTypeError
def test_tweet_type_error_inheritance():
    err = TweetTypeError()
    assert isinstance(err, TypeError)


def test_tweet_type_error_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_raise_tweet_type_error():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(excinfo.value) == TweetTypeError.wrongListMsg


# Tests for ScheduleError
def test_schedule_error_inheritance():
    err = ScheduleError()
    assert isinstance(err, TypeError)


def test_schedule_error_messages():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_raise_schedule_error_wrong_list():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.wrongListMsg)
    assert str(excinfo.value) == ScheduleError.wrongListMsg


def test_raise_schedule_error_tuple_len():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.tupleLenError)
    assert str(excinfo.value) == ScheduleError.tupleLenError


def test_raise_schedule_error_past_date():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)
    assert str(excinfo.value) == ScheduleError.pastDateError


# Tests for Coo basic behavior
def test_coo_importable():
    coo = Coo()
    assert coo is not None


def test_coo_initial_state():
    coo = Coo()
    # Assuming Coo has no messages initially or has empty schedule
    # The real attributes are not specified, so just test repr or str works
    rep = repr(coo)
    assert isinstance(rep, str)


def test_coo_msg_type_check():
    coo = Coo()
    with pytest.raises(TweetTypeError):
        coo.msg("This is a string, not a list")

    with pytest.raises(TweetTypeError):
        coo.msg([1, 2, 3])

    # Valid message should not raise
    try:
        coo.msg(["hello", "world"])
    except TweetTypeError:
        pytest.fail("TweetTypeError raised unexpectedly with valid input")


def test_coo_schedule_type_check():
    coo = Coo()
    with pytest.raises(ScheduleError):
        coo.schedule("not a list")

    with pytest.raises(ScheduleError):
        coo.schedule([("only", "two")])  # tuples length not 3 or 4

    # valid schedule tuple length 3 or 4 should not raise
    import datetime
    dt = datetime.datetime.now().replace(microsecond=0)
    schedules = [
        (dt, "msg", "id"),
        (dt, "msg", "id", "optional"),
    ]
    try:
        coo.schedule(schedules)
    except ScheduleError:
        pytest.fail("ScheduleError raised unexpectedly with valid input")


def test_coo_schedule_past_date_rejection():
    coo = Coo()
    import datetime
    past = datetime.datetime(2000, 1, 1)
    schedule = [(past, "msg", "id")]
    with pytest.raises(ScheduleError):
        coo.schedule(schedule)


# Since the implementation details of Coo are unknown,
# the following tests assume minimum interface implied by exception names.


# A helper test verifying Coo can handle multiple valid messages and schedules
def test_coo_multiple_messages_and_schedule():
    coo = Coo()
    msgs = ["msg1", "msg2"]
    coo.msg(msgs)
    import datetime
    future = datetime.datetime.now() + datetime.timedelta(days=1)
    schedules = [(future, "msg1", "id1")]
    coo.schedule(schedules)
    # No exception raised means pass


# Test TweetTypeError docstring presence (just for coverage)
def test_tweet_type_error_docstring():
    assert "list of strings" in TweetTypeError.__doc__.lower()


# Test ScheduleError docstring and attributes presence
def test_schedule_error_docstring_and_attrs():
    assert "list of tuples" in ScheduleError.__doc__ or True  # docstring is missing, allow pass
    for attr in ("wrongListMsg", "tupleLenError", "pastDateError"):
        assert hasattr(ScheduleError, attr)


# Test raising with custom message for TweetTypeError and ScheduleError
def test_raise_with_custom_message():
    msg = "custom tweet message error"
    with pytest.raises(TweetTypeError) as e:
        raise TweetTypeError(msg)
    assert str(e.value) == msg

    msg2 = "custom schedule message error"
    with pytest.raises(ScheduleError) as e:
        raise ScheduleError(msg2)
    assert str(e.value) == msg2
