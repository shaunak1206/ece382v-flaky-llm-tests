"""Unit tests for timemarker.TimeMarker.

Time comes from a fake clock that replaces timemarker.timer.default_timer, so
every measured duration is exact and no test sleeps.
"""

import pytest

import timemarker.timer as timer_module
from timemarker import TimeMarker


class FakeClock(object):
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(timer_module, "default_timer", fake)
    return fake


def _collect(tm, **kwargs):
    lines = []
    tm.stats(print_function=lines.append, **kwargs)
    return lines


def _marker_raw(clock):
    """Total 8s: start 1s, a 4s, b 3s."""
    tm = TimeMarker()
    clock.now = 0.0
    with tm:
        clock.now = 1.0
        tm.tag("a")
        clock.now = 5.0
        tm.tag("b")
        clock.now = 8.0
    return tm


def _marker_pctg(clock):
    """Total 8s: start 4s (0.5), a 2s (0.25), b 1s (0.125), c 1s (0.125)."""
    tm = TimeMarker()
    clock.now = 0.0
    with tm:
        clock.now = 4.0
        tm.tag("a")
        clock.now = 6.0
        tm.tag("b")
        clock.now = 7.0
        tm.tag("c")
        clock.now = 8.0
    return tm


def test_init_defaults():
    tm = TimeMarker()
    assert tm._start is None
    assert tm._end is None
    assert tm._cumulative_time is None
    assert tm._last_tag is None
    assert tm._last_ts is None
    assert dict(tm._timing) == {}


def test_repr_before_enter():
    assert repr(TimeMarker()) == "TimeMarker(start=None, last=(None, None))"


def test_repr_after_enter(clock):
    clock.now = 2.5
    tm = TimeMarker()
    with tm:
        assert repr(tm) == "TimeMarker(start=2.5, last=(start, 2.5))"


def test_enter_returns_self_and_records_start(clock):
    clock.now = 7.0
    tm = TimeMarker()
    with tm as entered:
        assert entered is tm
        assert tm._start == 7.0
        assert tm._last_ts == 7.0
        assert tm._last_tag == "start"


def test_reserved_tag_names_are_rejected(clock):
    tm = TimeMarker()
    with tm:
        for name in ("start", "stop"):
            with pytest.raises(ValueError, match="reserved"):
                tm.tag(name)


def test_reserved_tag_allowed_with_override(clock):
    tm = TimeMarker()
    with tm:
        clock.now = 2.0
        tm.tag("start", _override=True)
        assert tm._last_tag == "start"
        assert tm._last_ts == 2.0


def test_tag_charges_elapsed_time_to_previous_tag(clock):
    clock.now = 0.0
    tm = TimeMarker()
    with tm:
        clock.now = 3.0
        tm.tag("a")
        clock.now = 5.0
        tm.tag("b")
        clock.now = 9.0
    assert tm._timing == {"start": 3.0, "a": 2.0, "b": 4.0}


def test_repeated_tag_accumulates_time(clock):
    clock.now = 0.0
    tm = TimeMarker()
    with tm:
        clock.now = 2.0
        tm.tag("a")
        clock.now = 3.0
        tm.tag("b")
        clock.now = 7.0
        tm.tag("a")
        clock.now = 10.0
    # a: 1s (2 to 3) then 3s (7 to 10); b: 4s (3 to 7); start: 2s
    assert tm._timing == {"start": 2.0, "a": 4.0, "b": 4.0}


def test_tag_updates_cumulative_time_during_block(clock):
    clock.now = 0.0
    tm = TimeMarker()
    with tm:
        clock.now = 4.0
        tm.tag("a")
        assert tm._cumulative_time == 4.0


def test_exit_records_stop_and_cumulative_time(clock):
    clock.now = 0.0
    tm = TimeMarker()
    with tm:
        clock.now = 9.0
    assert tm._last_tag == "stop"
    assert tm._end == 9.0
    assert tm._cumulative_time == 9.0


def test_exception_propagates_and_stop_is_recorded(clock):
    clock.now = 0.0
    tm = TimeMarker()
    with pytest.raises(RuntimeError, match="boom"):
        with tm:
            clock.now = 2.0
            tm.tag("work")
            clock.now = 5.0
            raise RuntimeError("boom")
    assert tm._cumulative_time == 5.0
    assert tm._timing == {"start": 2.0, "work": 3.0}
    assert tm._last_tag == "stop"


def test_instances_reject_arbitrary_attributes():
    tm = TimeMarker()
    with pytest.raises(AttributeError):
        tm.unexpected = 1


def test_stats_invalid_fmt_raises(clock):
    tm = _marker_raw(clock)
    with pytest.raises(ValueError, match="Invalid format: bogus"):
        tm.stats(fmt="bogus")


def test_stats_invalid_pctg_cap_raises(clock):
    tm = _marker_pctg(clock)
    for cap in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="pctg must be"):
            tm.stats(pctg_cap=cap)


def test_stats_pctg_cap_needs_percentage_format(clock):
    tm = _marker_raw(clock)
    with pytest.raises(ValueError, match="requires fmt"):
        tm.stats(fmt="raw", pctg_cap=0.5)


def test_stats_pctg_cap_needs_sort(clock):
    tm = _marker_pctg(clock)
    with pytest.raises(ValueError, match="requires sort"):
        tm.stats(sort=False, pctg_cap=0.5)


def test_stats_raw_sorted_by_duration(clock):
    tm = _marker_raw(clock)
    assert _collect(tm, fmt="raw") == [
        "TIME:8.000000s a:4.000s b:3.000s start:1.000s"
    ]


def test_stats_raw_unsorted_keeps_chronological_order(clock):
    tm = _marker_raw(clock)
    assert _collect(tm, fmt="raw", sort=False) == [
        "TIME:8.000000s start:1.000s a:4.000s b:3.000s"
    ]


def test_stats_oneline_false_uses_bullets(clock):
    tm = _marker_raw(clock)
    assert _collect(tm, fmt="raw", oneline=False) == [
        "TIME:8.000000s\n  - a:4.000s\n  - b:3.000s\n  - start:1.000s"
    ]


def test_stats_percentage_default_output(clock):
    tm = _marker_pctg(clock)
    assert _collect(tm) == ["TIME:8.000000s start:0.500 a:0.250 b:0.125 c:0.125"]


def test_stats_percentage_cap_truncates(clock):
    tm = _marker_pctg(clock)
    # running totals 0.5, 0.75, ...; 0.75 exceeds the 0.6 cap so it stops
    assert _collect(tm, pctg_cap=0.6) == ["TIME:8.000000s start:0.500"]


def test_stats_percentage_cap_includes_tag_reaching_cap(clock):
    tm = _marker_pctg(clock)
    # running total hits exactly 0.75, which is not above the 0.75 cap
    assert _collect(tm, pctg_cap=0.75) == ["TIME:8.000000s start:0.500 a:0.250"]


def test_stats_default_print_goes_to_stdout(clock, capsys):
    tm = _marker_pctg(clock)
    tm.stats()
    captured = capsys.readouterr()
    assert captured.out == "TIME:8.000000s start:0.500 a:0.250 b:0.125 c:0.125\n"


def test_calculate_tag_percentage_static_helper():
    result = TimeMarker._calculate_tag_percentage(8.0, 0.6, [("a", 4.0), ("b", 2.0)])
    assert result == [("a", 0.5)]
