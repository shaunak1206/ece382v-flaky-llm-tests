import datetime

import arrow
from arrow import Arrow

BASE = Arrow(2013, 5, 11, 21, 23, 58)  # a Saturday, ISO week 19


def test_format_explicit_pattern():
    assert BASE.format("YYYY-MM-DD HH:mm:ss") == "2013-05-11 21:23:58"


def test_format_offset_tokens():
    pacific = BASE.to("US/Pacific")
    assert pacific.format("ZZ") == "-07:00"
    assert pacific.format("Z") == "-0700"


def test_format_month_and_day_names():
    assert BASE.format("MMMM") == "May"
    assert BASE.format("MMM") == "May"
    assert BASE.format("dddd") == "Saturday"
    assert BASE.format("ddd") == "Sat"
    assert BASE.format("YY") == "13"
    assert BASE.format("Do") == "11th"


def test_format_twelve_hour_clock():
    assert BASE.format("hh:mm A") == "09:23 PM"
    assert BASE.format("a") == "pm"


def test_format_bracket_escape():
    assert BASE.format("YYYY [at] HH") == "2013 at 21"


def test_format_iso_week_token():
    assert BASE.format("W") == "2013-W19-6"


def test_format_timestamp_token():
    assert float(Arrow(2013, 5, 11, 20, 23, 58).format("X")) == 1368303838


def test_format_french_locale():
    a = Arrow(2013, 5, 11)
    assert a.format("MMMM", locale="fr") == "mai"
    assert a.format("dddd", locale="fr") == "samedi"


def test_humanize_hour_ago_and_in():
    earlier = BASE.shift(hours=-1)
    assert earlier.humanize(BASE) == "an hour ago"
    assert BASE.humanize(earlier) == "in an hour"


def test_humanize_same_time_is_just_now():
    assert BASE.humanize(BASE) == "just now"


def test_humanize_days():
    assert BASE.shift(days=-2).humanize(BASE) == "2 days ago"
    assert BASE.shift(days=2).humanize(BASE) == "in 2 days"


def test_humanize_only_distance():
    earlier = BASE.shift(hours=-1)
    assert earlier.humanize(BASE, only_distance=True) == "an hour"


def test_humanize_granularity_single_and_list():
    later = BASE.shift(hours=2, minutes=30)
    assert BASE.shift(hours=2).humanize(BASE, granularity="hour") == "in 2 hours"
    assert (
        later.humanize(BASE, granularity=["hour", "minute"])
        == "in 2 hours and 30 minutes"
    )


def test_dehumanize_future_and_past():
    assert BASE.dehumanize("in 2 hours") == BASE.shift(hours=2)
    assert BASE.dehumanize("an hour ago") == BASE.shift(hours=-1)
    assert BASE.dehumanize("2 days ago") == BASE.shift(days=-2)
    assert isinstance(BASE.dehumanize("just now"), arrow.Arrow)
