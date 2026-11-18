import importlib

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError


def test_package_exports_coo_class():
    assert coo.Coo is Coo


def test_imported_coo_is_class():
    assert isinstance(Coo, type)
    assert Coo.__name__ == "Coo"


def test_package_version_constant():
    assert coo.__version__ == "0.1.3"


def test_package_metadata_constants():
    assert coo.__title__ == "coo"
    assert coo.__license__ == "Apache 2.0"
    assert coo.__author__ == "Carlos Montecinos Geisse"
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


def test_package_allows_reimport():
    reloaded = importlib.reload(coo)
    assert reloaded.Coo is Coo
    assert reloaded.__version__ == coo.__version__


def test_tweet_type_error_is_typeerror_subclass():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(TweetTypeError, Exception)


def test_schedule_error_is_typeerror_subclass():
    assert issubclass(ScheduleError, TypeError)
    assert issubclass(ScheduleError, Exception)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_wrong_list_message_constant():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tuple_length_message_constant():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_past_date_message_constant():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_exception_instances_are_message_free_by_default():
    tweet_err = TweetTypeError()
    schedule_err = ScheduleError()

    assert isinstance(tweet_err, TypeError)
    assert isinstance(schedule_err, TypeError)
    assert str(tweet_err) == ""
    assert str(schedule_err) == ""


def test_exception_classes_expose_expected_class_attributes():
    for cls, attrs in [
        (TweetTypeError, ["wrongListMsg"]),
        (ScheduleError, ["wrongListMsg", "tupleLenError", "pastDateError"]),
    ]:
        for attr in attrs:
            assert hasattr(cls, attr)
            assert isinstance(getattr(cls, attr), str)


def test_cannot_raise_schedule_error_without_message_but_can_raise_class():
    with pytest.raises(ScheduleError):
        raise ScheduleError


def test_cannot_raise_tweet_type_error_without_message_but_can_raise_class():
    with pytest.raises(TweetTypeError):
        raise TweetTypeError


def test_module_imports_are_consistent():
    import coo._exceptions as exc_mod

    assert exc_mod.TweetTypeError is TweetTypeError
    assert exc_mod.ScheduleError is ScheduleError
