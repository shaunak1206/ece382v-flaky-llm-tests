import gettext

import pytest

import humanize
from humanize import i18n


# natural_list

def test_natural_list_three_items():
    assert humanize.natural_list(["one", "two", "three"]) == "one, two and three"


def test_natural_list_two_items():
    assert humanize.natural_list(["one", "two"]) == "one and two"


def test_natural_list_single_item():
    assert humanize.natural_list(["one"]) == "one"


# i18n

def test_default_separators():
    humanize.i18n.deactivate()
    assert i18n.thousands_separator() == ","
    assert i18n.decimal_separator() == "."


def test_activate_none_and_deactivate():
    translations = i18n.activate(None)
    assert isinstance(translations, gettext.NullTranslations)
    assert i18n.thousands_separator() == ","
    assert i18n.deactivate() is None
    assert humanize.intcomma(1000) == "1,000"


def test_activate_missing_locale_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        i18n.activate("xx_NOWHERE", path=str(tmp_path))
