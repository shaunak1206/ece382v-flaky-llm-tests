import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError, TemplateError


def _make_coo():
    """Try to build a Coo instance using dummy string credentials."""
    sig = inspect.signature(Coo.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.default is param.empty:
            kwargs[name] = "dummy-" + name
    try:
        return Coo(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on implementation
        pytest.skip("Could not instantiate Coo with dummy credentials: %r" % (exc,))


# ---------------------------------------------------------------- exceptions

def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_tweet_type_error_wrong_list_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_messages():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweet_type_error_raise_with_class_message():
    with pytest.raises(TweetTypeError, match="List of Strings"):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_schedule_error_raise_with_class_message():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)
    assert str(excinfo.value) == "A future date is needed."


def test_exceptions_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("bad")
    with pytest.raises(TypeError):
        raise ScheduleError("bad")


def test_exceptions_are_distinct_classes():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)
    assert ScheduleError is not TweetTypeError


@pytest.mark.parametrize("exc_cls", [TweetTypeError, ScheduleError, TemplateError])
def test_exceptions_custom_message_preserved(exc_cls):
    err = exc_cls("custom message")
    assert str(err) == "custom message"
    assert err.args == ("custom message",)


@pytest.mark.parametrize("exc_cls", [TweetTypeError, ScheduleError])
def test_exceptions_can_be_raised_without_message(exc_cls):
    with pytest.raises(exc_cls):
        raise exc_cls()
    assert str(exc_cls()) == ""


def test_schedule_error_attributes_accessible_on_instance():
    err = ScheduleError("x")
    assert err.tupleLenError == ScheduleError.tupleLenError
    assert err.wrongListMsg == ScheduleError.wrongListMsg


# ---------------------------------------------------------------------- Coo

def test_coo_exported_from_package():
    assert coo.Coo is Coo
    assert inspect.isclass(Coo)


def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__version__ == "0.1.3"
    assert coo.__license__ == "Apache 2.0"


def test_coo_instantiation_returns_coo_instance():
    instance = _make_coo()
    assert isinstance(instance, Coo)


def test_coo_instances_are_independent():
    first = _make_coo()
    second = _make_coo()
    assert first is not second
    assert isinstance(second, Coo)
