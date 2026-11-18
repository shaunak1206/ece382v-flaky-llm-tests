import pytest
from coo._exceptions import TweetTypeError, TemplateError, ScheduleError


def test_tweet_type_error_is_type_error():
    err = TweetTypeError("error")
    assert isinstance(err, TypeError)
    assert str(err) == "error"
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_template_error_is_type_error():
    err = TemplateError("template error")
    assert isinstance(err, TypeError)
    assert str(err) == "template error"
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_schedule_error_is_type_error():
    err = ScheduleError("schedule error")
    assert isinstance(err, TypeError)
    assert str(err) == "schedule error"
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweet_type_error_str_message():
    err = TweetTypeError()
    # default __str__ should be empty string if no message
    assert str(err) == ""


def test_template_error_default_message_empty():
    err = TemplateError()
    assert str(err) == ""


def test_schedule_error_default_message_empty():
    err = ScheduleError()
    assert str(err) == ""


def test_raising_tweet_type_error():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError("Must be a list of strings")
    assert "Must be a list of strings" in str(excinfo.value)


def test_raising_template_error():
    with pytest.raises(TemplateError) as excinfo:
        raise TemplateError("Template error occurred")
    assert "Template error occurred" in str(excinfo.value)


def test_raising_schedule_error():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError("Schedule invalid")
    assert "Schedule invalid" in str(excinfo.value)


def test_exception_inheritance():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(TemplateError, TypeError)
    assert issubclass(ScheduleError, TypeError)


def test_exception_attributes_are_class_level():
    # Changing instance attribute should not affect class attribute
    ex = TweetTypeError("msg")
    ex.wrongListMsg = "Changed"
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."
    assert ex.wrongListMsg == "Changed"


def test_multiple_instances_have_separate_messages():
    ex1 = TemplateError("First")
    ex2 = TemplateError("Second")
    assert str(ex1) == "First"
    assert str(ex2) == "Second"


def test_schedule_error_class_attrs_are_correct():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."
