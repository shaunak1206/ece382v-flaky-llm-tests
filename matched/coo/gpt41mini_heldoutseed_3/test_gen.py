import pytest
from coo._exceptions import TweetTypeError, ScheduleError


def test_tweet_type_error_is_type_error():
    err = TweetTypeError("error message")
    assert isinstance(err, TypeError)
    assert str(err) == "error message"


def test_tweet_type_error_wrongListMsg():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_template_error_attributes():
    from coo._exceptions import TemplateError

    assert issubclass(TemplateError, TypeError)
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_schedule_error_is_type_error():
    err = ScheduleError("schedule error")
    assert isinstance(err, TypeError)
    assert str(err) == "schedule error"


def test_schedule_error_wrongListMsg():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tupleLenError():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_pastDateError():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweet_type_error_raises_with_wrong_msg_type():
    # simulate raising TweetTypeError when msg is invalid (list of strings required)
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert TweetTypeError.wrongListMsg in str(excinfo.value)


def test_schedule_error_raises_with_wrong_list():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.wrongListMsg)
    assert ScheduleError.wrongListMsg in str(excinfo.value)


def test_schedule_error_raises_with_tuple_len_error():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.tupleLenError)
    assert ScheduleError.tupleLenError in str(excinfo.value)


def test_schedule_error_raises_with_past_date_error():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)
    assert ScheduleError.pastDateError in str(excinfo.value)


def test_all_exceptions_inherit_from_type_error():
    from coo._exceptions import TemplateError
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(TemplateError, TypeError)
    assert issubclass(ScheduleError, TypeError)
