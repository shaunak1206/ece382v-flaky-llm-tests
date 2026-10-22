"""Unit tests for humanize.i18n and locale-dependent number formatting."""

import gettext
import threading

import pytest

import humanize
from humanize import i18n


@pytest.fixture(autouse=True)
def _reset_locale():
    humanize.deactivate()
    yield
    humanize.deactivate()


def _set_locale(monkeypatch, name):
    monkeypatch.setattr(i18n._CURRENT, "locale", name, raising=False)


def test_default_separators_and_english_activation():
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."
    translation = humanize.activate("en_GB")
    assert isinstance(translation, gettext.NullTranslations)
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."
    assert humanize.activate(None) is translation
    humanize.deactivate()
    assert humanize.thousands_separator() == ","


def test_german_locale_swaps_separators_in_intcomma_and_intword(monkeypatch):
    _set_locale(monkeypatch, "de_DE")
    assert humanize.thousands_separator() == "."
    assert humanize.decimal_separator() == ","
    assert humanize.intcomma(1234567.5) == "1.234.567,5"
    assert humanize.intcomma(1000) == "1.000"
    assert humanize.intcomma("1.234.567,5") == "1.234.567,5"
    assert humanize.intword(1_200_000) == "1,2 million"


@pytest.mark.parametrize(
    "locale, thousands, decimal, expected",
    [
        ("fr_FR", " ", ".", "1 234 567.5"),
        ("it_IT", ".", ",", "1.234.567,5"),
        ("pt_BR", ".", ",", "1.234.567,5"),
        ("hu_HU", " ", ",", "1 234 567,5"),
        ("lv", " ", ",", "1 234 567,5"),
    ],
)
def test_locale_separator_tables(monkeypatch, locale, thousands, decimal, expected):
    _set_locale(monkeypatch, locale)
    assert humanize.thousands_separator() == thousands
    assert humanize.decimal_separator() == decimal
    assert humanize.intcomma(1234567.5) == expected


def test_activate_unknown_locale_raises_and_keeps_current_locale(tmp_path):
    with pytest.raises(FileNotFoundError):
        humanize.activate("xx_YY", path=tmp_path)
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."


def test_locale_is_thread_local(monkeypatch):
    _set_locale(monkeypatch, "de_DE")
    results = []

    def worker():
        results.append((humanize.thousands_separator(), humanize.intcomma(1234.5)))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert results == [(",", "1,234.5")]
    assert humanize.thousands_separator() == "."
