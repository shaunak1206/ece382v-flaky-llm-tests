import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


def test_package_exports_coo_class():
    import coo.coo as coo_module

    assert inspect.isclass(Coo)
    assert Coo is coo_module.Coo


def test_package_metadata():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__version__ == "0.1.3"
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


def test_coo_init_signature_is_inspectable():
    sig = inspect.signature(Coo.__init__)
    assert "self" in sig.parameters


def test_coo_instantiation_with_dummy_credentials():
    sig = inspect.signature(Coo.__init__)
    kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            kwargs[name] = "dummy"
    try:
        instance = Coo(**kwargs)
    except Exception as exc:  # constructor may validate credentials
        pytest.skip("Coo could not be built with dummy credentials: %r" % (exc,))
    assert isinstance(instance, Coo)


def test_tweet_type_error_is_type_error():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(TweetTypeError, Exception)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_tweet_type_error_raise_and_match():
    with pytest.raises(TweetTypeError, match="List of Strings"):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_tweet_type_error_caught_as_type_error():
    with pytest.raises(TypeError):
        raise TweetTypeError("bad")


def test_tweet_type_error_str_contains_message():
    err = TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(err) == "A List of Strings is required."
    assert err.args == ("A List of Strings is required.",)


def test_schedule_error_is_type_error():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


@pytest.mark.parametrize(
    "attr", ["wrongListMsg", "tupleLenError", "pastDateError"]
)
def test_schedule_error_raise_with_each_message(attr):
    message = getattr(ScheduleError, attr)
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(message)
    assert str(excinfo.value) == message


def test_schedule_error_and_tweet_error_are_distinct():
    assert not issubclass(ScheduleError, TweetTypeError)
    assert not issubclass(TweetTypeError, ScheduleError)
    with pytest.raises(ScheduleError):
        try:
            raise ScheduleError("x")
        except TweetTypeError:  # must not catch
            pytest.fail("TweetTypeError handler caught ScheduleError")


def test_template_error_constants_and_base():
    assert issubclass(TemplateError, TypeError)
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_exceptions_can_be_raised_without_arguments():
    for exc in (TweetTypeError, ScheduleError, TemplateError):
        with pytest.raises(exc):
            raise exc()
