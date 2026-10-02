"""Tests for humanize.i18n and localized output of the other modules."""

import datetime as dt
import gettext
import threading

import pytest

import humanize

WHEN = dt.datetime(2024, 6, 15, 12, 0, 0)


@pytest.fixture(autouse=True)
def _reset_locale():
    humanize.deactivate()
    yield
    humanize.deactivate()


def activate_or_skip(name):
    """Activate a bundled locale, skipping the test if its catalog is unavailable."""
    try:
        return humanize.activate(name)
    except OSError as exc:  # compiled .mo file missing
        pytest.skip(f"locale {name} unavailable: {exc}")


@pytest.fixture
def french():
    activate_or_skip("fr_FR")


@pytest.fixture
def german():
    activate_or_skip("de_DE")


# ---------------------------------------------------------------- activate / deactivate


def test_default_separators_are_english():
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."


def test_activate_english_and_none_return_null_translations():
    for name in ("en_US", "en", None):
        translation = humanize.activate(name)
        assert isinstance(translation, gettext.NullTranslations)
        assert humanize.thousands_separator() == ","
        assert humanize.decimal_separator() == "."
        assert humanize.naturalday(dt.date.today()) == "today"


def test_activate_unknown_locale_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        humanize.activate("xx_XX", path=tmp_path)


def test_activate_french_then_deactivate(french):
    assert humanize.naturaltime(30) == "il y a 30 secondes"
    humanize.deactivate()
    assert humanize.naturaltime(30) == "30 seconds ago"
    assert humanize.ordinal(1) == "1st"


def test_deactivate_restores_english_separators(german):
    assert humanize.thousands_separator() == "."
    humanize.deactivate()
    assert humanize.thousands_separator() == ","
    assert humanize.decimal_separator() == "."
    assert humanize.intcomma(1234567) == "1,234,567"


def test_switching_between_locales():
    activate_or_skip("fr_FR")
    assert humanize.naturalday(dt.date.today()) == "aujourd'hui"
    activate_or_skip("de_DE")
    assert humanize.naturalday(dt.date.today()) == "heute"
    humanize.activate("en")
    assert humanize.naturalday(dt.date.today()) == "today"


def test_activation_is_thread_local(french):
    seen = {}

    def worker():
        seen["thousands"] = humanize.thousands_separator()
        seen["today"] = humanize.naturalday(dt.date.today())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=10)
    assert seen == {"thousands": ",", "today": "today"}
    assert humanize.thousands_separator() == " "


@pytest.mark.parametrize(
    "locale, thousands, decimal",
    [
        ("de_DE", ".", ","),
        ("fr_FR", " ", "."),
        ("it_IT", ".", ","),
        ("pt_BR", ".", ","),
        ("hu_HU", " ", ","),
        ("lv", " ", ","),
    ],
)
def test_separators_per_locale(locale, thousands, decimal):
    activate_or_skip(locale)
    assert humanize.thousands_separator() == thousands
    assert humanize.decimal_separator() == decimal


# ---------------------------------------------------------------- French output


def test_french_naturalday(french):
    today = dt.date.today()
    assert humanize.naturalday(today) == "aujourd'hui"
    assert humanize.naturalday(today + dt.timedelta(days=1)) == "demain"
    assert humanize.naturalday(today - dt.timedelta(days=1)) == "hier"


def test_french_naturaltime_with_when(french):
    assert humanize.naturaltime(WHEN + dt.timedelta(hours=3), when=WHEN) == "dans 3 heures"
    assert humanize.naturaltime(WHEN - dt.timedelta(minutes=1), when=WHEN) == (
        "il y a une minute"
    )
    assert humanize.naturaltime(WHEN, when=WHEN) == "maintenant"


def test_french_naturaldelta(french):
    assert humanize.naturaldelta(dt.timedelta(0)) == "un instant"
    assert humanize.naturaldelta(dt.timedelta(minutes=1)) == "une minute"
    assert humanize.naturaldelta(dt.timedelta(hours=1)) == "une heure"
    assert humanize.naturaldelta(dt.timedelta(days=1)) == "un jour"
    assert humanize.naturaldelta(dt.timedelta(days=2)) == "2 jours"


def test_french_apnumber_and_precisedelta(french):
    assert humanize.apnumber(0) == "zéro"
    assert humanize.apnumber(3) == "trois"
    assert humanize.apnumber(5) == "cinq"
    assert humanize.apnumber(12) == "12"
    assert humanize.precisedelta(dt.timedelta(seconds=90)) == "1 minute et 30 secondes"


def test_french_intcomma_and_intword(french):
    assert humanize.intcomma(1234567) == "1 234 567"
    assert humanize.intcomma(1234567.25) == "1 234 567.25"
    assert humanize.intword(1_000_000) == "1.0 million"
    assert humanize.intword(1_500_000) == "1.5 millions"


def test_french_naturalsize(french):
    assert humanize.naturalsize(1) == "1 octet"
    assert humanize.naturalsize(300) == "300 octets"
    assert humanize.naturalsize(1500) == "1.5 Ko"
    assert humanize.naturalsize(3_000_000) == "3.0 Mo"


# ---------------------------------------------------------------- German output


def test_german_naturalday(german):
    today = dt.date.today()
    assert humanize.naturalday(today) == "heute"
    assert humanize.naturalday(today + dt.timedelta(days=1)) == "morgen"
    assert humanize.naturalday(today - dt.timedelta(days=1)) == "gestern"


def test_german_naturaltime_and_delta(german):
    assert humanize.naturaltime(WHEN - dt.timedelta(seconds=30), when=WHEN) == (
        "vor 30 Sekunden"
    )
    assert humanize.naturaltime(WHEN + dt.timedelta(hours=3), when=WHEN) == (
        "3 Stunden ab jetzt"
    )
    assert humanize.naturaltime(WHEN, when=WHEN) == "jetzt"
    assert humanize.naturaldelta(dt.timedelta(0)) == "ein Moment"
    assert humanize.naturaldelta(dt.timedelta(minutes=1)) == "eine Minute"
    assert humanize.naturaldelta(dt.timedelta(minutes=5)) == "5 Minuten"
    assert humanize.naturaldelta(dt.timedelta(hours=1)) == "eine Stunde"


def test_german_apnumber(german):
    assert humanize.apnumber(1) == "eins"
    assert humanize.apnumber(2) == "zwei"


def test_german_intcomma_uses_dot_thousands_and_comma_decimal(german):
    assert humanize.intcomma(1234567) == "1.234.567"
    assert humanize.intcomma(1234567.25) == "1.234.567,25"
    assert humanize.intcomma(1234.5678, 2) == "1.234,57"
    assert humanize.intcomma("1.234.567,25") == "1.234.567,25"
    assert humanize.intcomma("1.000") == "1.000"


def test_german_intword_uses_decimal_comma(german):
    assert humanize.intword(1_200_000) == "1,2 Millionen"
    assert humanize.intword(1_000_000) == "1,0 Million"
    assert humanize.intword(12_400) == "12,4 Tausend"


# ---------------------------------------------------------------- not translated


def test_natural_list_is_not_localized(french):
    assert humanize.natural_list(["a", "b"]) == "a and b"
