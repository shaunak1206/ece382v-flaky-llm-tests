import inspect

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TweetTypeError, TemplateError


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


def test_coo_is_reexported_from_package_root():
    assert hasattr(coo, "Coo")
    assert Coo is coo.Coo


def test_coo_is_a_class():
    assert inspect.isclass(Coo)


def test_tweet_type_error_is_subclass_of_type_error():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_message_constant():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_is_subclass_of_type_error():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_message_constants():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."
    assert ScheduleError.pastDateError == "A future date is needed."


def test_template_error_is_subclass_of_type_error():
    assert issubclass(TemplateError, TypeError)


def test_template_error_message_constants():
    assert TemplateError.templateInfoMsg == "template must be a string."
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."
