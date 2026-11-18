import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError


def test_package_exports_coo_class():
    assert hasattr(coo, "Coo")
    assert coo.Coo is Coo
    assert inspect.isclass(Coo)


def test_package_metadata_title():
    assert coo.__title__ == "coo"


def test_package_metadata_license():
    assert coo.__license__ == "Apache 2.0"


def test_package_metadata_version():
    assert coo.__version__ == "0.1.3"


def test_package_metadata_author():
    assert coo.__author__ == "Carlos Montecinos Geisse"


def test_package_metadata_contact():
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"


def test_package_metadata_url():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


def test_tweettypeerror_is_typeerror_subclass():
    assert issubclass(TweetTypeError, TypeError)
    assert issubclass(TweetTypeError, Exception)


def test_scheduleerror_is_typeerror_subclass():
    assert issubclass(ScheduleError, TypeError)
    assert issubclass(ScheduleError, Exception)


def test_tweettypeerror_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_scheduleerror_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_tweettypeerror_can_be_raised_and_caught():
    with pytest.raises(TweetTypeError) as excinfo:
        raise TweetTypeError(TweetTypeError.wrongListMsg)

    assert isinstance(excinfo.value, TweetTypeError)
    assert str(excinfo.value) == TweetTypeError.wrongListMsg
    assert excinfo.value.args == (TweetTypeError.wrongListMsg,)


def test_scheduleerror_can_be_raised_and_caught():
    with pytest.raises(ScheduleError) as excinfo:
        raise ScheduleError(ScheduleError.pastDateError)

    assert isinstance(excinfo.value, ScheduleError)
    assert str(excinfo.value) == ScheduleError.pastDateError
    assert excinfo.value.args == (ScheduleError.pastDateError,)
