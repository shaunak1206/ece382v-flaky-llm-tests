import inspect

import pytest

import coo
from coo import Coo
from coo._exceptions import ScheduleError, TemplateError, TweetTypeError


def test_coo_reexport_matches_module_attribute():
    assert coo.Coo is Coo


def test_coo_class_is_importable_and_is_a_class():
    assert inspect.isclass(Coo)


def test_coo_module_metadata_title():
    assert coo.__title__ == "coo"


def test_coo_module_metadata_license():
    assert coo.__license__ == "Apache 2.0"


def test_coo_module_metadata_version():
    assert coo.__version__ == "0.1.3"


def test_coo_module_metadata_author():
    assert coo.__author__ == "Carlos Montecinos Geisse"


def test_coo_module_metadata_contact():
    assert coo.__contact__ == "carlos.w.montecinos@gmail.com"


def test_coo_module_metadata_url():
    assert coo.__url__ == "https://github.com/wilfredinni/coo"


def test_tweet_type_error_is_type_error_subclass():
    assert issubclass(TweetTypeError, TypeError)


def test_tweet_type_error_wrong_list_message():
    assert TweetTypeError.wrongListMsg == "A List of Strings is required."


def test_schedule_error_is_type_error_subclass():
    assert issubclass(ScheduleError, TypeError)


def test_schedule_error_wrong_list_message():
    assert ScheduleError.wrongListMsg == "A list of tuples is required."


def test_schedule_error_tuple_length_message():
    assert ScheduleError.tupleLenError == "Every tuple need a length of 3 or 4."


def test_schedule_error_past_date_message():
    assert ScheduleError.pastDateError == "A future date is needed."


def test_template_error_is_type_error_subclass():
    assert issubclass(TemplateError, TypeError)


def test_template_error_template_info_message():
    assert TemplateError.templateInfoMsg == "template must be a string."


def test_template_error_template_message_error():
    assert TemplateError.templateMsgErr == "The template must contain a '$message'."
