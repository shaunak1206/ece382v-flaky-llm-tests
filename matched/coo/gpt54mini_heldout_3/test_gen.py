import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


def test_package_exports_coo_class():
    assert coo.Coo is Coo


def test_package_title_metadata():
    assert coo.__title__ == "coo"


def test_package_license_metadata():
    assert coo.__license__ == "Apache 2.0"


def test_package_version_metadata():
    assert coo.__version__ == "0.1.3"


def test_package_author_metadata():
    assert coo.__author__ == "Carlos Montecinos Geisse"


def test_package_contact_metadata():
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"


def test_package_url_metadata():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_exceptions_can_be_raised_and_caught():
    with pytest.raises(TweetTypeError, match="A List of Strings is required\\."):
        raise TweetTypeError(TweetTypeError.wrongListMsg)


def test_schedule_error_can_be_raised_and_caught():
    with pytest.raises(ScheduleError, match="A list of tuples is required\\."):
        raise ScheduleError(ScheduleError.wrongListMsg)


def test_template_error_is_type_error_subclass():
    assert issubclass(TemplateError, TypeError)


def test_template_error_message_constants():
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."


def test_exception_instances_have_expected_str_for_message():
    err = TweetTypeError(TweetTypeError.wrongListMsg)
    assert str(err) == TweetTypeError.wrongListMsg


def test_schedule_error_instances_have_expected_str_for_message():
    err = ScheduleError(ScheduleError.pastDateError)
    assert str(err) == ScheduleError.pastDateError
