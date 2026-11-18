import random
import pytest
import numpy as np

from coo import Coo
from coo._exceptions import TweetTypeError, ScheduleError


def test_Coo_basic_instantiation():
    random.seed(123)
    np.random.seed(123)
    c = Coo()
    # Just test it creates an instance without error
    assert isinstance(c, Coo)


def test_TweetTypeError_attributes():
    err = TweetTypeError()
    assert issubclass(type(err), TypeError)
    # The message attribute should be exact string
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_ScheduleError_attributes():
    err = ScheduleError()
    assert issubclass(type(err), TypeError)
    # These strings should be exact
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_TweetTypeError_is_subclass_of_TypeError():
    assert issubclass(TweetTypeError, TypeError)


def test_ScheduleError_is_subclass_of_TypeError():
    assert issubclass(ScheduleError, TypeError)


def test_exception_instances_are_exceptions():
    # Instances must be instances of BaseException
    tweet_err = TweetTypeError()
    schedule_err = ScheduleError()
    assert isinstance(tweet_err, BaseException)
    assert isinstance(schedule_err, BaseException)


def test_exception_message_defaults():
    # Instantiating the errors without args produces default TypeError message, not the class attribute
    tweet_err = TweetTypeError()
    schedule_err = ScheduleError()
    # The string representation should fallback to default since no argument provided
    assert str(tweet_err) == ""
    assert str(schedule_err) == ""


def test_instantiating_exceptions_with_message():
    msg = "some error message"
    e1 = TweetTypeError(msg)
    e2 = ScheduleError(msg)
    assert str(e1) == msg
    assert str(e2) == msg


@pytest.mark.parametrize("exc_class,expected_msg", [
    (TweetTypeError, "A List of Strings is required."),
    (ScheduleError, "A list of tuples is required."),
])
def test_class_attributes_do_not_change(exc_class, expected_msg):
    # Test that the class attribute string does not change, and is correct type
    assert isinstance(exc_class.wrongListMsg, str)
    assert exc_class.wrongListMsg == expected_msg


def test_Coo_module_metadata():
    import coo
    assert hasattr(coo, "__title__")
    assert hasattr(coo, "__license__")
    assert hasattr(coo, "__version__")
    assert hasattr(coo, "__author__")
    assert hasattr(coo, "__contact__")
    assert hasattr(coo, "__url__")

    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


def test_TweetTypeError_wrongListMsg_not_modified(monkeypatch):
    # monkeypatch the wrongListMsg and check it is restored after patch
    original_msg = TweetTypeError.wrongListMsg
    new_msg = "changed message"
    monkeypatch.setattr(TweetTypeError, "wrongListMsg", new_msg)
    assert TweetTypeError.wrongListMsg == new_msg
    # After test completes, monkeypatch will restore it


def test_ScheduleError_tupleLenError_immutable(monkeypatch):
    original = ScheduleError.tupleLenError
    monkeypatch.setattr(ScheduleError, "tupleLenError", "modified")
    assert ScheduleError.tupleLenError == "modified"
    # monkeypatch restores after test


def test_TemplateError_class_attributes():
    from coo._exceptions import TemplateError
    assert isinstance(TemplateError.templateInfoMsg, str)
    assert isinstance(TemplateError.templateMsgErr, str)
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."
    assert issubclass(TemplateError, TypeError)


def test_exception_str_repr_consistency():
    # For exceptions with no message, str() and repr() are consistent and sensible
    te = TweetTypeError()
    se = ScheduleError()
    assert str(te) == ""
    assert str(se) == ""
    # repr is of type: ClassName()
    assert repr(te).startswith("TweetTypeError")
    assert repr(se).startswith("ScheduleError")


def test_exception_with_non_string_message():
    # Instantiating exceptions with non-string messages must keep the message as is (usually converted to str)
    e = TweetTypeError(123)
    assert str(e) == "123"
    e2 = ScheduleError(None)
    assert str(e2) == "None"


def test_exceptions_are_picklable(monkeypatch):
    import pickle
    # Patch attributes to restore after test
    tweet_error = TweetTypeError("msg")
    schedule_error = ScheduleError("msg")

    b1 = pickle.dumps(tweet_error)
    b2 = pickle.dumps(schedule_error)

    e1 = pickle.loads(b1)
    e2 = pickle.loads(b2)

    assert isinstance(e1, TweetTypeError)
    assert isinstance(e2, ScheduleError)
    assert str(e1) == "msg"
    assert str(e2) == "msg"
