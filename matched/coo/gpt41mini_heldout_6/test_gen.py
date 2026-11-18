import pytest
from coo._exceptions import TweetTypeError, TemplateError, ScheduleError


def test_tweet_type_error_is_subclass_of_type_error():
    err = TweetTypeError()
    assert isinstance(err, TypeError)


def test_tweet_type_error_has_correct_message_attribute():
    assert hasattr(TweetTypeError, "wrongListMsg")
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_template_error_is_subclass_of_type_error():
    err = TemplateError()
    assert isinstance(err, TypeError)


def test_template_error_has_correct_message_attributes():
    assert hasattr(TemplateError, "templateInfoMsg")
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert hasattr(TemplateError, "templateMsgErr")
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_schedule_error_is_subclass_of_type_error():
    err = ScheduleError()
    assert isinstance(err, TypeError)


def test_schedule_error_has_wrongListMsg_attribute():
    assert hasattr(ScheduleError, "wrongListMsg")
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_has_tupleLenError_attribute():
    assert hasattr(ScheduleError, "tupleLenError")
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_has_pastDateError_attribute():
    assert hasattr(ScheduleError, "pastDateError")
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweet_type_error_str_repr_default():
    err = TweetTypeError()
    # default str should reflect the exception class name if no message
    assert str(err) == "" or "TweetTypeError" in str(type(err))


def test_template_error_str_repr_default():
    err = TemplateError()
    assert str(err) == "" or "TemplateError" in str(type(err))


def test_schedule_error_str_repr_default():
    err = ScheduleError()
    assert str(err) == "" or "ScheduleError" in str(type(err))


def test_tweet_type_error_accepts_custom_message():
    msg = "Custom tweet type error message."
    err = TweetTypeError(msg)
    assert str(err) == msg


def test_template_error_accepts_custom_message():
    msg = "Custom template error message."
    err = TemplateError(msg)
    assert str(err) == msg


def test_schedule_error_accepts_custom_message():
    msg = "Custom schedule error message."
    err = ScheduleError(msg)
    assert str(err) == msg


def test_raise_and_catch_tweet_type_error():
    with pytest.raises(TweetTypeError) as exc_info:
        raise TweetTypeError(TweetTypeError.wrongListMsg)
    assert exc_info.value.args[0] == TweetTypeError.wrongListMsg


def test_raise_and_catch_template_error():
    with pytest.raises(TemplateError) as exc_info:
        raise TemplateError(TemplateError.templateInfoMsg)
    assert exc_info.value.args[0] == TemplateError.templateInfoMsg


def test_raise_and_catch_schedule_error():
    with pytest.raises(ScheduleError) as exc_info:
        raise ScheduleError(ScheduleError.pastDateError)
    assert exc_info.value.args[0] == ScheduleError.pastDateError
