"""Tests for humanize, dehumanize, comparisons, arithmetic and is_between."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from dateutil.relativedelta import relativedelta

import arrow
from arrow import Arrow

BASE = Arrow(2013, 1, 1)


# ---------------------------------------------------------------- humanize relative to now


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"seconds": -30}, "30 seconds ago"),
        ({"minutes": -5}, "5 minutes ago"),
        ({"hours": -1}, "an hour ago"),
        ({"hours": -2}, "2 hours ago"),
        ({"days": -3}, "3 days ago"),
        ({"weeks": -2}, "2 weeks ago"),
        ({"months": -3}, "3 months ago"),
        ({"years": -2}, "2 years ago"),
    ],
)
def test_humanize_past_relative_to_now(kwargs, expected):
    assert arrow.utcnow().shift(**kwargs).humanize() == expected


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"seconds": 30}, "in 30 seconds"),
        ({"minutes": 5}, "in 5 minutes"),
        ({"hours": 3}, "in 3 hours"),
        ({"days": 3}, "in 3 days"),
        ({"weeks": 2}, "in 2 weeks"),
        ({"months": 3}, "in 3 months"),
        ({"years": 2}, "in 2 years"),
    ],
)
def test_humanize_future_relative_to_now(kwargs, expected):
    assert arrow.utcnow().shift(**kwargs).humanize() == expected


def test_humanize_just_now_and_its_boundaries():
    assert arrow.utcnow().humanize() == "just now"
    assert arrow.utcnow().shift(seconds=-5).humanize() == "just now"
    assert arrow.utcnow().shift(seconds=5).humanize() == "just now"
    assert BASE.shift(seconds=9).humanize(BASE) == "just now"
    assert BASE.shift(seconds=-9).humanize(BASE) == "just now"
    assert BASE.shift(seconds=10).humanize(BASE) == "in 10 seconds"
    assert BASE.shift(seconds=-10).humanize(BASE) == "10 seconds ago"


def test_humanize_in_other_timezones_relative_to_now():
    assert arrow.now("US/Pacific").shift(minutes=-10).humanize() == "10 minutes ago"
    assert arrow.now("Asia/Tokyo").shift(hours=2).humanize() == "in 2 hours"
    assert arrow.get(datetime.now(timezone.utc) - timedelta(days=1, hours=1)).humanize() == (
        "a day ago"
    )
    assert arrow.now().shift(hours=-1).humanize() == "an hour ago"


# ---------------------------------------------------------------- humanize with explicit other


@pytest.mark.parametrize(
    "kwargs, text",
    [
        ({"seconds": 44}, "44 seconds"),
        ({"seconds": 60}, "a minute"),
        ({"seconds": 119}, "a minute"),
        ({"minutes": 2}, "2 minutes"),
        ({"minutes": 45}, "45 minutes"),
        ({"hours": 1}, "an hour"),
        ({"hours": 2}, "2 hours"),
        ({"hours": 23}, "23 hours"),
        ({"days": 1}, "a day"),
        ({"days": 2}, "2 days"),
        ({"days": 6}, "6 days"),
        ({"weeks": 1}, "a week"),
        ({"weeks": 2}, "2 weeks"),
        ({"days": 21}, "a month"),
        ({"months": 1}, "a month"),
        ({"months": 2}, "2 months"),
        ({"months": 6}, "6 months"),
        ({"years": 1}, "a year"),
        ({"years": 2}, "2 years"),
        ({"years": 5}, "5 years"),
    ],
)
def test_humanize_with_explicit_other(kwargs, text):
    later = BASE.shift(**kwargs)
    earlier = BASE.shift(**{unit: -amount for unit, amount in kwargs.items()})
    assert later.humanize(BASE) == f"in {text}"
    assert earlier.humanize(BASE) == f"{text} ago"


def test_humanize_only_distance():
    earlier = Arrow(2013, 5, 5, 12)
    later = earlier.shift(hours=4)
    assert later.humanize(earlier) == "in 4 hours"
    assert earlier.humanize(later) == "4 hours ago"
    assert later.humanize(earlier, only_distance=True) == "4 hours"
    assert earlier.humanize(later, only_distance=True) == "4 hours"
    assert earlier.humanize(earlier, only_distance=True) == "instantly"


def test_humanize_granularity():
    earlier = Arrow(2013, 5, 5, 12)
    later = earlier.shift(hours=4)
    assert later.humanize(earlier, granularity="hour") == "in 4 hours"
    assert later.humanize(earlier, granularity="minute") == "in 240 minutes"
    assert later.humanize(earlier, granularity="second") == "in 14400 seconds"
    assert later.humanize(earlier, granularity="day") == "in 0 days"
    assert earlier.shift(seconds=1).humanize(earlier, granularity="second") == "just now"
    assert earlier.shift(seconds=2).humanize(earlier, granularity="second") == "in 2 seconds"
    assert earlier.shift(weeks=1).humanize(earlier, granularity="week") == "in a week"
    assert earlier.shift(weeks=2).humanize(earlier, granularity="week") == "in 2 weeks"
    assert earlier.shift(days=365).humanize(earlier, granularity="year") == "in a year"

    detailed = earlier.shift(hours=2, minutes=30)
    assert detailed.humanize(earlier, granularity=["hour", "minute"]) == (
        "in 2 hours and 30 minutes"
    )
    assert earlier.humanize(detailed, granularity=["hour", "minute"]) == (
        "2 hours and 30 minutes ago"
    )
    assert detailed.humanize(earlier, granularity=["hour"]) == "in 2 hours"
    long_gap = earlier.shift(days=1, hours=2, minutes=3)
    assert long_gap.humanize(earlier, granularity=["day", "hour", "minute"]) == (
        "in a day 2 hours and 3 minutes"
    )

    with pytest.raises(ValueError):
        later.humanize(earlier, granularity="fortnight")
    with pytest.raises(ValueError):
        later.humanize(earlier, granularity=[])
    with pytest.raises(ValueError):
        later.humanize(earlier, granularity=["hour", "fortnight"])


def test_humanize_accepts_arrow_and_datetime_others_and_rejects_the_rest():
    noon = Arrow(2013, 5, 5, 12)
    assert noon.humanize(datetime(2013, 5, 5, 16)) == "4 hours ago"  # naive: same tz
    aware = datetime(2013, 5, 5, 9, tzinfo=ZoneInfo("US/Pacific"))  # 16:00 UTC
    assert noon.humanize(aware) == "4 hours ago"
    assert noon.humanize(Arrow(2013, 5, 5, 9, tzinfo="US/Pacific")) == "4 hours ago"
    assert Arrow(2013, 5, 5, 5, tzinfo="US/Pacific").humanize(noon) == "just now"
    with pytest.raises(TypeError):
        noon.humanize("tomorrow")
    with pytest.raises(TypeError):
        noon.humanize(5)


# ---------------------------------------------------------------- humanize locales


@pytest.mark.parametrize(
    "locale, future, past",
    [
        ("en-us", "in 4 hours", "4 hours ago"),
        ("en_GB", "in 4 hours", "4 hours ago"),
        ("fr", "dans 4 heures", "il y a 4 heures"),
        ("fr-fr", "dans 4 heures", "il y a 4 heures"),
        ("de", "in 4 Stunden", "vor 4 Stunden"),
        ("es", "en 4 horas", "hace 4 horas"),
        ("it", "tra 4 ore", "4 ore fa"),
        ("ja", "4時間後", "4時間前"),
    ],
)
def test_humanize_hours_in_several_locales(locale, future, past):
    earlier = Arrow(2013, 5, 5, 12)
    later = earlier.shift(hours=4)
    assert later.humanize(earlier, locale=locale) == future
    assert earlier.humanize(later, locale=locale) == past


def test_humanize_locale_details():
    noon = Arrow(2013, 5, 5, 12)
    for locale, text in [
        ("en", "just now"),
        ("fr", "maintenant"),
        ("de", "gerade eben"),
        ("es", "ahora"),
        ("it", "adesso"),
        ("ja", "現在"),
    ]:
        assert noon.humanize(noon, locale=locale) == text

    tomorrow = noon.shift(days=1)
    assert tomorrow.humanize(noon, locale="fr") == "dans un jour"
    assert noon.humanize(tomorrow, locale="fr") == "il y a un jour"
    assert tomorrow.humanize(noon, locale="de") == "in einem Tag"
    assert noon.humanize(tomorrow, locale="de") == "vor einem Tag"
    assert noon.shift(weeks=2).humanize(noon, locale="fr") == "dans 2 semaines"

    # German drops the "in"/"vor" and changes case for bare distances
    assert noon.shift(days=3).humanize(noon, locale="de", only_distance=True) == "3 Tage"
    assert tomorrow.humanize(noon, locale="de", only_distance=True) == "ein Tag"
    assert noon.shift(hours=4).humanize(noon, locale="de", only_distance=True) == "4 Stunden"
    assert noon.shift(months=2).humanize(noon, locale="de", only_distance=True) == "2 Monate"

    assert arrow.utcnow().shift(hours=-3).humanize(locale="fr") == "il y a 3 heures"
    assert arrow.utcnow().shift(hours=3).humanize(locale="de") == "in 3 Stunden"

    with pytest.raises(ValueError):
        noon.humanize(tomorrow, locale="xx-yy")
    with pytest.raises(ValueError):  # German has no "quarter" translation
        noon.shift(hours=4).humanize(noon, locale="de", granularity="quarter")


# ---------------------------------------------------------------- dehumanize


@pytest.mark.parametrize(
    "text, kwargs",
    [
        ("in 2 hours", {"hours": 2}),
        ("2 hours ago", {"hours": -2}),
        ("in an hour", {"hours": 1}),
        ("an hour ago", {"hours": -1}),
        ("in 44 seconds", {"seconds": 44}),
        ("44 seconds ago", {"seconds": -44}),
        ("in a minute", {"minutes": 1}),
        ("5 minutes ago", {"minutes": -5}),
        ("in a day", {"days": 1}),
        ("2 days ago", {"days": -2}),
        ("in a week", {"weeks": 1}),
        ("3 weeks ago", {"weeks": -3}),
        ("in a month", {"months": 1}),
        ("3 months ago", {"months": -3}),
        ("in a year", {"years": 1}),
        ("2 years ago", {"years": -2}),
        ("in 2 days and 5 hours", {"days": 2, "hours": 5}),
        ("just now", {}),
    ],
)
def test_dehumanize_english(text, kwargs):
    start = Arrow(2013, 5, 5, 12, 0, 0)
    assert start.dehumanize(text) == start.shift(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seconds": 30},
        {"minutes": -5},
        {"hours": -3},
        {"hours": 3},
        {"days": -2},
        {"weeks": -2},
        {"months": -2},
        {"years": -1},
    ],
)
def test_dehumanize_round_trips_humanize(kwargs):
    start = Arrow(2013, 5, 5, 12, 0, 0)
    target = start.shift(**kwargs)
    assert start.dehumanize(target.humanize(start)) == target


def test_dehumanize_other_locales_and_errors():
    start = Arrow(2013, 5, 5, 12, 0, 0)
    assert start.dehumanize("dans 2 heures", locale="fr") == start.shift(hours=2)
    assert start.dehumanize("il y a 3 jours", locale="fr") == start.shift(days=-3)
    assert start.dehumanize("il y a une heure", locale="fr") == start.shift(hours=-1)
    assert start.dehumanize("vor 2 Tagen", locale="de") == start.shift(days=-2)
    assert start.dehumanize("in 3 Stunden", locale="de") == start.shift(hours=3)

    with pytest.raises(ValueError):
        start.dehumanize("gibberish")
    with pytest.raises(ValueError):
        start.dehumanize("")
    with pytest.raises(ValueError):
        start.dehumanize("2 hours")  # no direction
    with pytest.raises(ValueError):
        start.dehumanize("in 2 hours", locale="xx-yy")


# ---------------------------------------------------------------- comparisons


def test_equality_and_hashing():
    noon = Arrow(2013, 5, 5, 12)
    assert noon == Arrow(2013, 5, 5, 12)
    assert noon == datetime(2013, 5, 5, 12, tzinfo=timezone.utc)
    assert datetime(2013, 5, 5, 12, tzinfo=timezone.utc) == noon
    assert noon != Arrow(2013, 5, 5, 13)
    assert noon == noon.to("US/Pacific") == noon.to("Asia/Tokyo")
    assert Arrow(2013, 5, 5, 5, tzinfo="US/Pacific") == noon

    assert not (noon == datetime(2013, 5, 5, 12))  # naive datetimes never equal
    assert noon != datetime(2013, 5, 5, 12)
    assert noon != 5
    assert not (noon == None)  # noqa: E711
    assert noon != "2013-05-05T12:00:00+00:00"

    assert hash(noon) == hash(noon.to("US/Pacific")) == hash(noon.clone())
    assert len({noon, noon.to("Asia/Tokyo"), noon.clone()}) == 1
    assert {noon: "x"}[noon.to("US/Pacific")] == "x"


def test_ordering_comparisons():
    earlier, later = Arrow(2013, 5, 5, 12), Arrow(2013, 5, 5, 13)
    assert earlier < later
    assert earlier <= later
    assert later > earlier
    assert later >= earlier
    assert earlier <= Arrow(2013, 5, 5, 12)
    assert earlier >= Arrow(2013, 5, 5, 12)
    assert not earlier > later
    assert not later < earlier

    assert earlier < datetime(2013, 5, 5, 13, tzinfo=timezone.utc)
    assert datetime(2013, 5, 5, 11, tzinfo=timezone.utc) < earlier
    assert Arrow(2013, 5, 5, 5, tzinfo="US/Pacific") < later  # 12:00 UTC < 13:00 UTC

    with pytest.raises(TypeError):
        earlier < datetime(2013, 5, 5, 13)  # naive
    with pytest.raises(TypeError):
        earlier < 5
    with pytest.raises(TypeError):
        earlier >= "2013-05-05"
    with pytest.raises(TypeError):
        earlier > None


def test_sorting_min_and_max_across_timezones():
    a = Arrow(2013, 5, 5, 12)
    b = Arrow(2013, 5, 5, 6, tzinfo="US/Pacific")  # 13:00 UTC
    c = Arrow(2013, 5, 5, 23, tzinfo="Asia/Tokyo")  # 14:00 UTC
    d = Arrow(2013, 5, 5, 10)
    shuffled = [c, a, d, b]
    assert sorted(shuffled) == [d, a, b, c]
    assert sorted(shuffled, reverse=True) == [c, b, a, d]
    assert min(shuffled) is d
    assert max(shuffled) is c


# ---------------------------------------------------------------- arithmetic


def test_adding_and_subtracting_durations():
    moment = Arrow(2013, 5, 5, 12, 30, 45, 123456)
    assert (moment + timedelta(days=1)).isoformat() == "2013-05-06T12:30:45.123456+00:00"
    assert (moment - timedelta(hours=1)).isoformat() == "2013-05-05T11:30:45.123456+00:00"
    assert (timedelta(days=1) + moment) == moment + timedelta(days=1)
    assert (moment + relativedelta(months=1)).isoformat() == "2013-06-05T12:30:45.123456+00:00"
    assert (moment - relativedelta(years=1)).isoformat() == "2012-05-05T12:30:45.123456+00:00"
    assert Arrow(2013, 1, 31) + relativedelta(months=1) == Arrow(2013, 2, 28)
    assert moment + timedelta(0) == moment
    assert isinstance(moment + timedelta(days=1), Arrow)
    assert moment.isoformat() == "2013-05-05T12:30:45.123456+00:00"

    pacific = Arrow(2013, 3, 9, 12, tzinfo="US/Pacific")
    next_day = pacific + timedelta(days=1)
    assert next_day.isoformat() == "2013-03-10T12:00:00-07:00"
    assert next_day.tzinfo == pacific.tzinfo


def test_subtracting_arrows_and_datetimes_gives_timedeltas():
    earlier, later = Arrow(2013, 5, 5, 12), Arrow(2013, 5, 5, 13, 30)
    assert later - earlier == timedelta(hours=1, minutes=30)
    assert earlier - later == -timedelta(hours=1, minutes=30)
    assert (later - earlier).total_seconds() == 5400.0
    assert later - datetime(2013, 5, 5, 12, tzinfo=timezone.utc) == timedelta(hours=1, minutes=30)
    assert datetime(2013, 5, 5, 14, tzinfo=timezone.utc) - later == timedelta(minutes=30)
    assert Arrow(2013, 5, 5, 5, tzinfo="US/Pacific") - earlier == timedelta(0)
    assert isinstance(later - earlier, timedelta)
    with pytest.raises(TypeError):
        later - datetime(2013, 5, 5, 12)  # naive


def test_unsupported_arithmetic_raises_type_error():
    moment = Arrow(2013, 5, 5, 12)
    with pytest.raises(TypeError):
        moment + 1
    with pytest.raises(TypeError):
        moment + "1 day"
    with pytest.raises(TypeError):
        moment + moment
    with pytest.raises(TypeError):
        moment - 5
    with pytest.raises(TypeError):
        moment - "x"
    with pytest.raises(TypeError):
        5 - moment
    with pytest.raises(TypeError):
        timedelta(days=1) - moment


# ---------------------------------------------------------------- is_between


def test_is_between_bounds():
    start = Arrow(2013, 5, 5, 12, 30, 10)
    end = Arrow(2013, 5, 5, 12, 30, 36)
    inside = Arrow(2013, 5, 5, 12, 30, 27)
    assert inside.is_between(start, end)
    assert not Arrow(2013, 5, 5, 12, 30, 9).is_between(start, end)
    assert not Arrow(2013, 5, 5, 12, 30, 37).is_between(start, end)

    start, end = Arrow(2013, 5, 5), Arrow(2013, 5, 8)
    assert end.is_between(start, end, "[]")
    assert not end.is_between(start, end, "[)")
    assert not end.is_between(start, end)
    assert end.is_between(start, end, "(]")
    assert start.is_between(start, end, "[)")
    assert start.is_between(start, end, "[]")
    assert not start.is_between(start, end, "()")
    assert not start.is_between(start, end, "(]")

    assert Arrow(2013, 5, 6, 12, tzinfo="US/Pacific").is_between(start, end)
    assert Arrow(2013, 5, 7, 20, tzinfo="US/Pacific").is_between(start, end) is False

    with pytest.raises(ValueError):
        inside.is_between(start, end, "<>")
    with pytest.raises(TypeError):
        inside.is_between(datetime(2013, 5, 5), end)
    with pytest.raises(TypeError):
        inside.is_between(start, datetime(2013, 5, 9))
