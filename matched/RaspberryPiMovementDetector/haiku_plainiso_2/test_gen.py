import os
import sys
import threading
from unittest import mock

import pytest

# Make the project root importable (tests are run from the project root).
PROJECT_ROOT = os.path.abspath(os.getcwd())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import MovementDetector.Watch as watch_module  # noqa: E402
from MovementDetector.Watch import Watch  # noqa: E402


class FakeTime(object):
    """Replaces the `time` module inside Watch.py."""

    def __init__(self, times=None):
        self.sleeps = []
        self._times = list(times or [])

    def sleep(self, seconds):
        self.sleeps.append(seconds)

    def time(self):
        return self._times.pop(0)


class FakeGPIO(object):
    """Minimal stand-in for an RPi.GPIO-like object."""

    OUT = 0
    IN = 1

    def __init__(self, echo_values=()):
        self.setup_calls = []
        self.output_calls = []
        self.cleanup_calls = 0
        self._echo_values = list(echo_values)

    def setup(self, pin, mode):
        self.setup_calls.append((pin, mode))

    def output(self, pin, value):
        self.output_calls.append((pin, value))

    def input(self, pin):
        return self._echo_values.pop(0)

    def cleanup(self):
        self.cleanup_calls += 1


@pytest.fixture
def fake_time(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(watch_module, "time", fake)
    return fake


def make_watch(**kwargs):
    kwargs.setdefault("gpio", FakeGPIO())
    return Watch(**kwargs)


def ping_with_distances(watch, monkeypatch, distances):
    """Run watch.ping() once per distance, feeding fixed sensor readings."""
    readings = iter(distances)
    monkeypatch.setattr(watch, "get_distance", lambda: next(readings))
    for _ in distances:
        watch.ping()


def test_default_trigger_and_echo_pins():
    watch = make_watch()
    assert watch.trigger_pin() == 23
    assert watch.echo_pin() == 24


def test_custom_trigger_and_echo_pins():
    watch = make_watch(trig=5, echo=6)
    assert watch.trigger_pin() == 5
    assert watch.echo_pin() == 6


def test_default_offset_is_200_cm(monkeypatch):
    watch = make_watch(func_in=mock.Mock(), func_out=mock.Mock())
    in_handler = mock.Mock()
    watch = make_watch(func_in=in_handler)
    # 199 is inside the zone (< 200), so ObjectIn fires.
    ping_with_distances(watch, monkeypatch, [199])
    in_handler.assert_called_once_with(199)

    in_handler2 = mock.Mock()
    watch2 = make_watch(func_in=in_handler2)
    # 200 is not less than the offset, so ObjectIn must not fire.
    ping_with_distances(watch2, monkeypatch, [200])
    in_handler2.assert_not_called()


def test_ping_emits_object_in_when_entering_zone(monkeypatch):
    in_handler = mock.Mock()
    watch = make_watch(func_in=in_handler, offset=100)
    ping_with_distances(watch, monkeypatch, [50])
    in_handler.assert_called_once_with(50)


def test_ping_does_not_repeat_object_in_while_inside(monkeypatch):
    in_handler = mock.Mock()
    watch = make_watch(func_in=in_handler, offset=100)
    ping_with_distances(watch, monkeypatch, [50, 40, 30])
    assert in_handler.call_count == 1


def test_ping_emits_object_out_when_leaving_zone(monkeypatch):
    out_handler = mock.Mock()
    watch = make_watch(func_out=out_handler, offset=100)
    ping_with_distances(watch, monkeypatch, [50, 150])
    out_handler.assert_called_once_with(150)


def test_ping_does_not_emit_object_out_without_prior_entry(monkeypatch):
    out_handler = mock.Mock()
    watch = make_watch(func_out=out_handler, offset=100)
    ping_with_distances(watch, monkeypatch, [150, 300])
    out_handler.assert_not_called()


def test_ping_reentry_emits_second_object_in(monkeypatch):
    in_handler = mock.Mock()
    out_handler = mock.Mock()
    watch = make_watch(func_in=in_handler, func_out=out_handler, offset=100)
    ping_with_distances(watch, monkeypatch, [50, 150, 60])
    assert in_handler.call_count == 2
    assert out_handler.call_count == 1


def test_object_in_handler_receives_distance(monkeypatch):
    in_handler = mock.Mock()
    watch = make_watch(func_in=in_handler, offset=100)
    ping_with_distances(watch, monkeypatch, [42.5])
    in_handler.assert_called_once_with(42.5)


def test_object_out_handler_receives_distance(monkeypatch):
    out_handler = mock.Mock()
    watch = make_watch(func_out=out_handler, offset=100)
    ping_with_distances(watch, monkeypatch, [10, 123.45])
    out_handler.assert_called_once_with(123.45)


def test_default_handlers_print_when_not_provided(monkeypatch, capsys):
    watch = make_watch(offset=100)
    ping_with_distances(watch, monkeypatch, [10, 500])
    captured = capsys.readouterr().out
    assert "Dummy in function" in captured
    assert "Dummy out function" in captured


def test_get_distance_configures_pins(fake_time):
    gpio = FakeGPIO(echo_values=[0, 1, 1, 0])
    fake_time._times = [0.0, 0.0, 1.0, 1.001]
    watch = make_watch(gpio=gpio, trig=23, echo=24)
    watch.get_distance()
    assert gpio.setup_calls == [(23, FakeGPIO.OUT), (24, FakeGPIO.IN)]


def test_get_distance_sends_trigger_pulse_and_settles(fake_time):
    gpio = FakeGPIO(echo_values=[0, 1, 1, 0])
    fake_time._times = [0.0, 0.0, 1.0, 1.001]
    watch = make_watch(gpio=gpio, trig=23, echo=24)
    watch.get_distance()
    assert gpio.output_calls == [(23, False), (23, True), (23, False)]
    assert fake_time.sleeps == [2, 0.00001]


def test_get_distance_computes_distance_from_echo_duration(fake_time):
    gpio = FakeGPIO(echo_values=[0, 1, 1, 0])
    # Echo pulse lasts 0.001 s -> 0.001 * 17150 = 17.15 cm
    fake_time._times = [0.0, 0.0, 1.0, 1.001]
    watch = make_watch(gpio=gpio)
    distance = watch.get_distance()
    assert distance == pytest.approx(17.15, abs=0.01)


def test_get_distance_rounds_to_two_decimals(fake_time):
    gpio = FakeGPIO(echo_values=[0, 1, 1, 0])
    fake_time._times = [0.0, 0.0, 1.0, 1.0012345]
    watch = make_watch(gpio=gpio)
    distance = watch.get_distance()
    assert distance == round(distance, 2)
    assert distance == pytest.approx(0.0012345 * 17150, abs=0.01)


def test_get_distance_cleans_up_gpio(fake_time):
    gpio = FakeGPIO(echo_values=[0, 1, 1, 0])
    fake_time._times = [0.0, 0.0, 1.0, 1.001]
    watch = make_watch(gpio=gpio)
    watch.get_distance()
    assert gpio.cleanup_calls == 1


def test_poll_returns_immediately_when_stopped(monkeypatch):
    watch = make_watch()
    ping = mock.Mock()
    monkeypatch.setattr(watch, "ping", ping)
    watch._observer_on = False
    assert watch.poll() is None
    ping.assert_not_called()


def test_observe_pings_until_stopped(monkeypatch, fake_time):
    watch = make_watch()
    first_ping = threading.Event()

    def fake_ping():
        first_ping.set()

    monkeypatch.setattr(watch, "ping", fake_ping)
    watch.observe()
    assert first_ping.wait(timeout=5)
    watch.stop()
    assert watch._future.result(timeout=5) is None


def test_stop_disables_observer_and_sleeps_three_seconds(monkeypatch, fake_time):
    watch = make_watch()
    monkeypatch.setattr(watch, "ping", lambda: None)
    watch.observe()
    watch.stop()
    assert watch._observer_on is False
    assert fake_time.sleeps == [3]
    assert watch._future.result(timeout=5) is None
