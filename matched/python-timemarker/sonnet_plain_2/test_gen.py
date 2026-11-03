import importlib

import pytest

from timemarker import TimeMarker

timer_module = importlib.import_module("timemarker.timer")


class FakeClock(object):
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock(0.0)
    monkeypatch.setattr(timer_module, "default_timer", c)
    return c


def _run_sample(clock):
    """start=1s, a=2s, b=4s, total 7s."""
    tm = TimeMarker()
    clock.now = 0.0
    with tm:
        clock.now = 1.0
        tm.tag("a")
        clock.now = 3.0
        tm.tag("b")
        clock.now = 7.0
    return tm


def test_init_defaults():
    tm = TimeMarker()
    assert tm._start is None
    assert tm._end is None
    assert tm._cumulative_time is None
    assert tm._last_tag is None
    assert tm._last_ts is None
    assert len(tm._timing) == 0


def test_enter_returns_self_and_sets_start(clock):
    clock.now = 5.0
    tm = TimeMarker()
    with tm as entered:
        assert entered is tm
        assert tm._start == 5.0
        assert tm._last_ts == 5.0
        assert tm._last_tag == "start"


def test_exit_sets_end_and_cumulative_time(clock):
    tm = TimeMarker()
    clock.now = 10.0
    with tm:
        clock.now = 12.5
    assert tm._end == 12.5
    assert tm._cumulative_time == pytest.approx(2.5)
    assert tm._last_tag == "stop"


def test_tag_records_elapsed_time_per_tag(clock):
    tm = _run_sample(clock)
    assert tm._timing["start"] == pytest.approx(1.0)
    assert tm._timing["a"] == pytest.approx(2.0)
    assert tm._timing["b"] == pytest.approx(4.0)
    assert tm._cumulative_time == pytest.approx(7.0)


def test_stop_is_not_recorded_in_timing(clock):
    tm = _run_sample(clock)
    assert "stop" not in tm._timing


def test_repeated_tag_accumulates(clock):
    tm = TimeMarker()
    with tm:
        clock.now = 1.0
        tm.tag("loop")
        clock.now = 3.0
        tm.tag("loop")
        clock.now = 6.0
        tm.tag("loop")
        clock.now = 6.0
    # time spent after the first "loop" tag, attributed to "loop"
    assert tm._timing["loop"] == pytest.approx(5.0)
    assert tm._timing["start"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["start", "stop"])
def test_reserved_tags_raise_value_error(clock, name):
    tm = TimeMarker()
    with tm:
        with pytest.raises(ValueError) as excinfo:
            tm.tag(name)
    assert name in str(excinfo.value)
    assert "reserved" in str(excinfo.value)


def test_exception_in_block_still_finalizes_and_propagates(clock):
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        with tm:
            clock.now = 2.0
            raise RuntimeError("boom")
    assert tm._cumulative_time == pytest.approx(2.0)
    assert tm._end == 2.0


def test_repr_contains_state(clock):
    clock.now = 3.0
    tm = TimeMarker()
    with tm:
        assert repr(tm) == "TimeMarker(start=3.0, last=(start, 3.0))"
        clock.now = 4.0
        tm.tag("x")
        assert repr(tm) == "TimeMarker(start=3.0, last=(x, 4.0))"


def test_stats_raw_sorted_by_time(clock):
    tm = _run_sample(clock)
    out = []
    tm.stats(fmt="raw", print_function=out.append)
    assert out == ["TIME:7.000000s b:4.000s a:2.000s start:1.000s"]


def test_stats_raw_unsorted_keeps_tag_order(clock):
    tm = _run_sample(clock)
    out = []
    tm.stats(sort=False, fmt="raw", print_function=out.append)
    assert out == ["TIME:7.000000s start:1.000s a:2.000s b:4.000s"]


def test_stats_percentage_default(clock):
    tm = _run_sample(clock)
    out = []
    tm.stats(print_function=out.append)
    assert out == ["TIME:7.000000s b:0.571 a:0.286 start:0.143"]


def test_stats_multiline(clock):
    tm = _run_sample(clock)
    out = []
    tm.stats(fmt="raw", oneline=False, print_function=out.append)
    assert out == ["TIME:7.000000s\n  - b:4.000s\n  - a:2.000s\n  - start:1.000s"]


def test_stats_pctg_cap_truncates_results(clock):
    tm = _run_sample(clock)
    out = []
    tm.stats(pctg_cap=0.9, print_function=out.append)
    assert out == ["TIME:7.000000s b:0.571 a:0.286"]


def test_stats_pctg_cap_small_leaves_only_total(clock):
    tm = _run_sample(clock)
    out = []
    tm.stats(pctg_cap=0.5, print_function=out.append)
    assert out == ["TIME:7.000000s"]


def test_stats_prints_by_default(clock, capsys):
    tm = _run_sample(clock)
    tm.stats(fmt="raw")
    captured = capsys.readouterr()
    assert captured.out.strip() == "TIME:7.000000s b:4.000s a:2.000s start:1.000s"


def test_stats_invalid_format_raises(clock):
    tm = _run_sample(clock)
    with pytest.raises(ValueError, match="Invalid format"):
        tm.stats(fmt="bogus", print_function=lambda s: None)


@pytest.mark.parametrize("cap", [0.0, -0.5, 1.5])
def test_stats_invalid_pctg_cap_raises(clock, cap):
    tm = _run_sample(clock)
    with pytest.raises(ValueError):
        tm.stats(pctg_cap=cap, print_function=lambda s: None)


def test_stats_cap_with_raw_format_raises(clock):
    tm = _run_sample(clock)
    with pytest.raises(ValueError, match="requires fmt"):
        tm.stats(fmt="raw", pctg_cap=0.5, print_function=lambda s: None)


def test_stats_cap_without_sort_raises(clock):
    tm = _run_sample(clock)
    with pytest.raises(ValueError, match="requires sort"):
        tm.stats(sort=False, pctg_cap=0.5, print_function=lambda s: None)


def test_calculate_tag_percentage_static():
    result = TimeMarker._calculate_tag_percentage(
        10.0, 1.0, [("a", 5.0), ("b", 2.5)])
    assert result == [("a", 0.5), ("b", 0.25)]
    capped = TimeMarker._calculate_tag_percentage(
        10.0, 0.6, [("a", 5.0), ("b", 2.5)])
    assert capped == [("a", 0.5)]
