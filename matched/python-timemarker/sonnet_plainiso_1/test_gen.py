import pytest

import timemarker.timer as timer_module
from timemarker import TimeMarker


class FakeClock(object):
    """Deterministic replacement for timeit.default_timer."""

    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock(0.0)
    monkeypatch.setattr(timer_module, "default_timer", fake)
    return fake


def run_scenario(clock):
    """start->a: 1s, a->b: 3s, b->stop: 4s (total 8s). All values exact in binary."""
    with TimeMarker() as tm:
        clock.advance(1)
        tm.tag("a")
        clock.advance(3)
        tm.tag("b")
        clock.advance(4)
    return tm


def collect(tm, **kwargs):
    out = []
    tm.stats(print_function=out.append, **kwargs)
    return out[0]


def test_initial_state_repr(clock):
    tm = TimeMarker()
    assert repr(tm) == "TimeMarker(start=None, last=(None, None))"


def test_enter_returns_self_and_sets_start(clock):
    clock.now = 5.0
    tm = TimeMarker()
    with tm as entered:
        assert entered is tm
        assert repr(tm) == "TimeMarker(start=5.0, last=(start, 5.0))"


def test_tag_updates_repr(clock):
    with TimeMarker() as tm:
        clock.advance(2)
        tm.tag("load")
        assert repr(tm) == "TimeMarker(start=0.0, last=(load, 2.0))"


def test_exit_records_stop_tag(clock):
    tm = run_scenario(clock)
    assert repr(tm) == "TimeMarker(start=0.0, last=(stop, 8.0))"


def test_tag_returns_none(clock):
    with TimeMarker() as tm:
        assert tm.tag("x") is None


@pytest.mark.parametrize("name", ["start", "stop"])
def test_reserved_tag_names_rejected(clock, name):
    with TimeMarker() as tm:
        with pytest.raises(ValueError, match="reserved"):
            tm.tag(name)


def test_tag_before_enter_fails(clock):
    tm = TimeMarker()
    with pytest.raises(TypeError):
        tm.tag("early")


def test_exception_in_block_propagates_and_finalizes(clock):
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        with tm:
            clock.advance(2)
            raise RuntimeError("boom")
    assert repr(tm) == "TimeMarker(start=0.0, last=(stop, 2.0))"
    assert collect(tm, fmt="raw") == "TIME:2.000000s start:2.000s"


def test_stats_raw_sorted_by_duration(clock):
    tm = run_scenario(clock)
    assert collect(tm, fmt="raw") == "TIME:8.000000s b:4.000s a:3.000s start:1.000s"


def test_stats_raw_unsorted_keeps_tag_order(clock):
    tm = run_scenario(clock)
    assert (
        collect(tm, fmt="raw", sort=False)
        == "TIME:8.000000s start:1.000s a:3.000s b:4.000s"
    )


def test_stats_percentage_default(clock):
    tm = run_scenario(clock)
    assert collect(tm) == "TIME:8.000000s b:0.500 a:0.375 start:0.125"


def test_stats_multiline(clock):
    tm = run_scenario(clock)
    result = collect(tm, fmt="raw", oneline=False)
    assert result == "TIME:8.000000s\n  - b:4.000s\n  - a:3.000s\n  - start:1.000s"


def test_stats_percentage_cap_truncates(clock):
    tm = run_scenario(clock)
    assert collect(tm, pctg_cap=0.6) == "TIME:8.000000s b:0.500"
    assert collect(tm, pctg_cap=0.9) == "TIME:8.000000s b:0.500 a:0.375"


def test_repeated_tag_accumulates_time(clock):
    with TimeMarker() as tm:
        clock.advance(1)
        tm.tag("x")
        clock.advance(2)
        tm.tag("y")
        clock.advance(1)
        tm.tag("x")
        clock.advance(3)
    # x was active for 2s and 3s, y for 1s, start for 1s; total 7s
    assert collect(tm, fmt="raw") == "TIME:7.000000s x:5.000s y:1.000s start:1.000s"


def test_stats_uses_print_function_once(clock):
    tm = run_scenario(clock)
    calls = []
    assert tm.stats(print_function=calls.append) is None
    assert len(calls) == 1
    assert isinstance(calls[0], str)


def test_stats_default_prints_to_stdout(clock, capsys):
    tm = run_scenario(clock)
    tm.stats(fmt="raw")
    captured = capsys.readouterr()
    assert captured.out == "TIME:8.000000s b:4.000s a:3.000s start:1.000s\n"


def test_stats_invalid_format_raises():
    tm = TimeMarker()
    with pytest.raises(ValueError, match="Invalid format"):
        tm.stats(fmt="bogus")


@pytest.mark.parametrize("cap", [0.0, -0.5, 1.5])
def test_stats_pctg_cap_out_of_range_raises(cap):
    tm = TimeMarker()
    with pytest.raises(ValueError):
        tm.stats(pctg_cap=cap)


def test_stats_cap_requires_percentage_format():
    tm = TimeMarker()
    with pytest.raises(ValueError, match="requires fmt"):
        tm.stats(fmt="raw", pctg_cap=0.5)


def test_stats_cap_requires_sort():
    tm = TimeMarker()
    with pytest.raises(ValueError, match="requires sort"):
        tm.stats(sort=False, pctg_cap=0.5)


def test_calculate_tag_percentage_helper():
    result = TimeMarker._calculate_tag_percentage(
        10.0, 1.0, [("a", 5.0), ("b", 2.5), ("c", 2.5)]
    )
    assert result == [("a", 0.5), ("b", 0.25), ("c", 0.25)]
    capped = TimeMarker._calculate_tag_percentage(
        10.0, 0.6, [("a", 5.0), ("b", 2.5)]
    )
    assert capped == [("a", 0.5)]
