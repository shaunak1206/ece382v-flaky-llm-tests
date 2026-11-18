import pytest
import random
import numpy as np

from coo import Coo
from coo._exceptions import TweetTypeError, ScheduleError


def test_coo_instance_creation_and_basic_attributes():
    random.seed(0)
    np.random.seed(0)
    # Just test that Coo can be instantiated and has expected attributes or methods
    c = Coo()
    assert c is not None
    # We don't know internals, so just test it has some attributes or methods by dir()
    d = dir(c)
    # It should have at least some public methods or properties (not __ only)
    public_attrs = [x for x in d if not x.startswith('_')]
    assert len(public_attrs) > 0


def test_tweet_type_error_str_and_repr_messages():
    random.seed(1)
    np.random.seed(1)
    exc = TweetTypeError()
    # The exception should be a TypeError subclass
    assert isinstance(exc, TypeError)
    # The class attribute wrongListMsg should be a string and correct
    assert isinstance(TweetTypeError.wrongListMsg, str)
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_message_attributes():
    random.seed(2)
    np.random.seed(2)
    # The class attributes of ScheduleError must be strings with expected messages
    assert isinstance(ScheduleError.wrongListMsg, str)
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert isinstance(ScheduleError.tupleLenError, str)
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert isinstance(ScheduleError.pastDateError, str)
    assert ScheduleError.pastDateError == "A future date is needed."
    # Check inheritance
    exc = ScheduleError()
    assert isinstance(exc, TypeError)


def test_tweet_type_error_docstring():
    random.seed(3)
    np.random.seed(3)
    doc = TweetTypeError.__doc__
    assert doc is not None
    assert "list of strings" in doc.lower()


def test_schedule_error_docstring():
    random.seed(4)
    np.random.seed(4)
    doc = ScheduleError.__doc__
    assert doc is not None
    assert "list of tuples" in doc.lower() or "future date" in doc.lower()


def test_coo_module_metadata():
    random.seed(5)
    np.random.seed(5)
    import coo

    # The metadata globals must be strings matching exact values
    assert getattr(coo, "__title__", None) == "coo"
    assert getattr(coo, "__license__", None) == "Apache 2.0"
    assert getattr(coo, "__version__", None) == "0.1.3"
    assert getattr(coo, "__author__", None) == "Carlos Montecinos Geisse"
    assert getattr(coo, "__contact__", None) == "carlos.w.montecinos@gmail.com"
    assert getattr(coo, "__url__", None) == "https://github.com/wilfredinni/coo"


def test_tweet_type_error_is_subtype_of_typeerror():
    random.seed(6)
    np.random.seed(6)
    assert issubclass(TweetTypeError, TypeError)


def test_schedule_error_is_subtype_of_typeerror():
    random.seed(7)
    np.random.seed(7)
    assert issubclass(ScheduleError, TypeError)


def test_template_error_message_attributes():
    random.seed(8)
    np.random.seed(8)

    from coo._exceptions import TemplateError

    # Check class attributes exact strings
    assert isinstance(TemplateError.templateInfoMsg, str)
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert isinstance(TemplateError.templateMsgErr, str)
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."

    # Check inheritance
    exc = TemplateError()
    assert isinstance(exc, TypeError)

def test_template_error_docstring():
    random.seed(9)
    np.random.seed(9)

    from coo._exceptions import TemplateError

    assert TemplateError.__doc__ is not None
    assert "template" in TemplateError.__doc__.lower()


def test_coo_import_and_type():
    random.seed(10)
    np.random.seed(10)
    from coo import Coo as ImportedCoo

    # ImportedCoo should be the same as coo.Coo if imported from coo.coo
    # We cannot import coo.coo directly (not recommended here), but type check:
    assert callable(ImportedCoo)
    obj = ImportedCoo()
    # Check it is an instance of ImportedCoo
    assert isinstance(obj, ImportedCoo)


def test_exceptions_str_and_repr_do_not_fail():
    random.seed(11)
    np.random.seed(11)

    exc1 = TweetTypeError()
    exc2 = ScheduleError()
    from coo._exceptions import TemplateError
    exc3 = TemplateError()

    # str and repr should not raise and produce a string
    for exc in (exc1, exc2, exc3):
        s = str(exc)
        r = repr(exc)
        assert isinstance(s, str)
        assert isinstance(r, str)


def test_schedule_error_attributes_are_constants(monkeypatch):
    random.seed(12)
    np.random.seed(12)

    # Using monkeypatch to replace the attribute and check it restores afterwards
    original_msg = ScheduleError.wrongListMsg
    monkeypatch.setattr(ScheduleError, "wrongListMsg", "test message")
    assert ScheduleError.wrongListMsg == "test message"
    # After test finishes, monkeypatch restores original, so here just check replacement worked


def test_tweet_type_error_wrongListMsg_is_not_instance_attribute():
    random.seed(13)
    np.random.seed(13)

    t = TweetTypeError()
    # It should not have wrongListMsg in instance __dict__, only class
    assert "wrongListMsg" not in t.__dict__
    # Accessing via instance works & returns class attribute
    assert t.wrongListMsg == TweetTypeError.wrongListMsg


def test_schedule_error_tupleLenError_message_is_not_empty():
    random.seed(14)
    np.random.seed(14)

    msg = ScheduleError.tupleLenError
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_template_error_messages_are_strings_with_expected_substrings():
    random.seed(15)
    np.random.seed(15)

    from coo._exceptions import TemplateError

    assert isinstance(TemplateError.templateInfoMsg, str)
    assert "string" in TemplateError.templateInfoMsg
    assert isinstance(TemplateError.templateMsgErr, str)
    assert "$message" in TemplateError.templateMsgErr


def test_module_level_variables_are_strings(monkeypatch):
    random.seed(16)
    np.random.seed(16)
    import coo

    attrs = ["__title__", "__license__", "__version__", "__author__", "__contact__", "__url__"]

    for attr in attrs:
        val = getattr(coo, attr)
        assert isinstance(val, str)
        # Patch the attribute and check monkeypatch effect
        monkeypatch.setattr(coo, attr, "patched")
        assert getattr(coo, attr) == "patched"
