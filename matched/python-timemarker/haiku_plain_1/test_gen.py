import logging

import pytest

import timemarker.timer as timer_module
from timemarker import TimeMarker


def _install_clock(monkeypatch, start=0.0):
    """Replace the timer's default_timer with a manually advanced fake clock."""
    clock = {"now": start}
    monkeypatch.setattr(timer_module, "default_timer", lambda: clock["now"])
    return clock


def _run_scenario(monkeypatch):
    """Enter at t=0, tag 'a' at t=1, tag 'b' at t=4, exit at t=4.

    Resulting timing: start=1.0, a=3.0, b=0.0; cumulative time 4.0.
    """
    clock = _install_clock(monkeypatch, 0.0)
    with TimeMarker() as tm:
        clock["now"] = 1.0
        tm.tag("a")
        clock["now"] = 4.0
        tm.tag("b")
    return tm


def test_init_defaults():
    tm = TimeMarker()
    assert tm._start is None
    assert tm._end is None
    assert tm._cumulative_time is None
    assert tm._last_tag is None
    assert tm._last_ts is None
    assert len(tm._timing) == 0


def test_enter_records_start_and_returns_self(monkeypatch):
    _install_clock(monkeypatch, 5.0)
    tm = TimeMarker()
    with tm as entered:
        assert entered is tm
        assert tm._start == 5.0
        assert tm._last_ts == 5.0
        assert tm._last_tag == "start"


def test_exit_records_end_and_cumulative_time(monkeypatch):
    clock = _install_clock(monkeypatch, 5.0)
    tm = TimeMarker()
    with tm:
        clock["now"] = 9.0
    assert tm._end == 9.0
    assert tm._cumulative_time == 4.0
    assert tm._last_tag == "stop"


def test_with_block_accumulates_per_tag_durations(monkeypatch):
    clock = _install_clock(monkeypatch, 10.0)
    with TimeMarker() as tm:
        clock["now"] = 12.0
        tm.tag("load")
        clock["now"] = 15.0
        tm.tag("compute")
    assert dict(tm._timing) == {"start": 2.0, "load": 3.0, "compute": 0.0}
    assert tm._cumulative_time == 5.0


def test_repeated_tag_name_accumulates(monkeypatch):
    clock = _install_clock(monkeypatch, 0.0)
    with TimeMarker() as tm:
        clock["now"] = 1.0
        tm.tag("a")
        clock["now"] = 2.0
        tm.tag("a")
        clock["now"] = 3.0
        tm.tag("b")
    assert tm._timing["start"] == 1.0
    assert tm._timing["a"] == 2.0
    assert tm._timing["b"] == 0.0


@pytest.mark.parametrize("name", ["start", "stop"])
def test_tag_rejects_reserved_names(monkeypatch, name):
    _install_clock(monkeypatch, 0.0)
    with TimeMarker() as tm:
        with pytest.raises(ValueError, match="reserved"):
            tm.tag(name)


def test_tag_override_allows_reserved_name(monkeypatch):
    _install_clock(monkeypatch, 0.0)
    with TimeMarker() as tm:
        tm.tag("start", _override=True)
        assert tm._last_tag == "start"


def test_tag_updates_cumulative_time(monkeypatch):
    clock = _install_clock(monkeypatch, 0.0)
    with TimeMarker() as tm:
        clock["now"] = 2.0
        tm.tag("a")
        assert tm._cumulative_time == 2.0
        clock["now"] = 5.0
        tm.tag("b")
        assert tm._cumulative_time == 5.0


def test_repr_before_and_after_enter(monkeypatch):
    assert repr(TimeMarker()) == "TimeMarker(start=None, last=(None, None))"
    clock = _install_clock(monkeypatch, 7.0)
    with TimeMarker() as tm:
        clock["now"] = 8.0
        tm.tag("x")
        assert repr(tm) == "TimeMarker(start=7.0, last=(x, 8.0))"


def test_stats_default_is_sorted_percentage_one_line(monkeypatch):
    tm = _run_scenario(monkeypatch)
    out = []
    tm.stats(print_function=out.append)
    assert out == ["TIME:4.000000s a:0.750 start:0.250 b:0.000"]


def test_stats_raw_format_appends_seconds_suffix(monkeypatch):
    tm = _run_scenario(monkeypatch)
    out = []
    tm.stats(fmt="raw", print_function=out.append)
    assert out == ["TIME:4.000000s a:3.000s start:1.000s b:0.000s"]


def test_stats_multiline_uses_bullet_separator(monkeypatch):
    tm = _run_scenario(monkeypatch)
    out = []
    tm.stats(oneline=False, print_function=out.append)
    assert out == ["TIME:4.000000s\n  - a:0.750\n  - start:0.250\n  - b:0.000"]


def test_stats_unsorted_keeps_insertion_order(monkeypatch):
    tm = _run_scenario(monkeypatch)
    out = []
    tm.stats(sort=False, print_function=out.append)
    assert out == ["TIME:4.000000s start:0.250 a:0.750 b:0.000"]


def test_stats_pctg_cap_truncates_lower_ranked_tags(monkeypatch):
    tm = _run_scenario(monkeypatch)
    out = []
    tm.stats(pctg_cap=0.8, print_function=out.append)
    assert out == ["TIME:4.000000s a:0.750"]


def test_stats_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid format"):
        TimeMarker().stats(fmt="json")


@pytest.mark.parametrize("cap", [0.0, -0.5, 1.5])
def test_stats_out_of_range_pctg_cap_raises(cap):
    with pytest.raises(ValueError, match="pctg must be"):
        TimeMarker().stats(pctg_cap=cap)


def test_stats_cap_below_one_requires_percentage_format():
    with pytest.raises(ValueError, match="requires fmt"):
        TimeMarker().stats(pctg_cap=0.5, fmt="raw")


def test_stats_cap_below_one_requires_sort():
    with pytest.raises(ValueError, match="requires sort"):
        TimeMarker().stats(pctg_cap=0.5, sort=False)


def test_calculate_tag_percentage_basic():
    result = TimeMarker._calculate_tag_percentage(
        10.0, 1.0, [("a", 5.0), ("b", 3.0)])
    assert result == [("a", 0.5), ("b", 0.3)]


def test_calculate_tag_percentage_stops_at_first_overflow():
    # running total after "b" is 1.1, which exceeds the cap, so "c" is dropped
    # even though it would fit on its own.
    result = TimeMarker._calculate_tag_percentage(
        10.0, 1.0, [("a", 6.0), ("b", 5.0), ("c", 1.0)])
    assert result == [("a", 0.6)]


def test_calculate_tag_percentage_empty_input():
    assert TimeMarker._calculate_tag_percentage(10.0, 1.0, []) == []


def test_instances_reject_unknown_attributes():
    tm = TimeMarker()
    with pytest.raises(AttributeError):
        tm.not_a_field = 1


def test_tag_emits_debug_log(monkeypatch, caplog):
    _install_clock(monkeypatch, 0.0)
    with caplog.at_level(logging.DEBUG, logger="timemarker.timer"):
        with TimeMarker() as tm:
            tm.tag("a")
    assert any("tag took" in rec.getMessage() for rec in caplog.records)
