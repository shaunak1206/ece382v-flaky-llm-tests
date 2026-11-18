import types
import pytest

import MovementDetector.Watch as watch_module
from MovementDetector.Watch import Watch


class FakeGPIO:
    OUT = "OUT"
    IN = "IN"

    def __init__(self, input_values=None):
        self.input_values = list(input_values or [])
        self.setup_calls = []
        self.output_calls = []
        self.cleanup_called = False

    def setup(self, pin, mode):
        self.setup_calls.append((pin, mode))

    def output(self, pin, value):
        self.output_calls.append((pin, value))

    def input(self, pin):
        if not self.input_values:
            raise AssertionError("FakeGPIO.input called more times than expected")
        return self.input_values.pop(0)

    def cleanup(self):
        self.cleanup_called = True


class FakeFuture:
    def __init__(self, shutdown_side_effect=None, running=False):
        self.shutdown_calls = []
        self.shutdown_side_effect = shutdown_side_effect
        self._running = running

    def shutdown(self, wait=False):
        self.shutdown_calls.append(wait)
        if self.shutdown_side_effect:
            raise self.shutdown_side_effect

    def running(self):
        return self._running


class FakeExecutor:
    def __init__(self, future=None):
        self.submit_calls = []
        self.shutdown_calls = []
        self.future = future or FakeFuture()

    def submit(self, fn, *args, **kwargs):
        self.submit_calls.append((fn, args, kwargs))
        return self.future

    def shutdown(self, wait=False):
        self.shutdown_calls.append(wait)


def make_watch(monkeypatch, **kwargs):
    monkeypatch.setattr(watch_module.time, "sleep", lambda *_args, **_kwargs: None)
    return Watch(**kwargs)


def test_init_uses_defaults_and_registers_handlers(monkeypatch):
    events = []

    def on_in(distance):
        events.append(("in", distance))

    def on_out(distance):
        events.append(("out", distance))

    w = make_watch(monkeypatch, func_in=on_in, func_out=on_out)

    assert w.trigger_pin() == 23
    assert w.echo_pin() == 24
    assert w._offset == 200
    assert w._gpio is None
    assert w._wasIn is False

    w._ee.emit("ObjectIn", 12)
    w._ee.emit("ObjectOut", 34)
    assert events == [("in", 12), ("out", 34)]


def test_init_accepts_custom_pins_and_offset(monkeypatch):
    gpio = FakeGPIO()
    w = make_watch(monkeypatch, gpio=gpio, trig=11, echo=12, offset=99)

    assert w._gpio is gpio
    assert w.trigger_pin() == 11
    assert w.echo_pin() == 12
    assert w._offset == 99


def test_ping_emits_object_in_when_distance_below_offset_and_not_previously_in(monkeypatch):
    gpio = FakeGPIO()
    events = []
    w = make_watch(monkeypatch, gpio=gpio, offset=100, func_in=lambda d: events.append(("in", d)))
    w.get_distance = lambda: 42

    w.ping()

    assert events == [("in", 42)]
    assert w._wasIn is True


def test_ping_does_not_emit_object_in_again_while_still_in(monkeypatch):
    gpio = FakeGPIO()
    events = []
    w = make_watch(monkeypatch, gpio=gpio, offset=100, func_in=lambda d: events.append(("in", d)))
    w.get_distance = lambda: 42

    w.ping()
    w.ping()

    assert events == [("in", 42)]
    assert w._wasIn is True


def test_ping_emits_object_out_when_distance_at_or_above_offset_and_was_in(monkeypatch):
    gpio = FakeGPIO()
    events = []
    w = make_watch(monkeypatch, gpio=gpio, offset=100, func_out=lambda d: events.append(("out", d)))
    w._wasIn = True
    w.get_distance = lambda: 123

    w.ping()

    assert events == [("out", 123)]
    assert w._wasIn is False


def test_ping_does_not_emit_object_out_when_already_out(monkeypatch):
    gpio = FakeGPIO()
    events = []
    w = make_watch(monkeypatch, gpio=gpio, offset=100, func_out=lambda d: events.append(("out", d)))
    w._wasIn = False
    w.get_distance = lambda: 123

    w.ping()

    assert events == []
    assert w._wasIn is False


def test_poll_calls_ping_until_observer_is_disabled(monkeypatch):
    w = make_watch(monkeypatch)
    calls = {"count": 0}

    def fake_ping():
        calls["count"] += 1
        if calls["count"] >= 3:
            w._observer_on = False

    w.ping = fake_ping

    w.poll()

    assert calls["count"] == 3


def test_poll_returns_immediately_when_observer_is_off(monkeypatch):
    w = make_watch(monkeypatch)
    called = {"ping": False}
    w._observer_on = False

    def fake_ping():
        called["ping"] = True

    w.ping = fake_ping
    assert w.poll() is None
    assert called["ping"] is False


def test_observe_submits_poll_to_executor(monkeypatch):
    w = make_watch(monkeypatch)
    fake_future = FakeFuture()
    fake_executor = FakeExecutor(future=fake_future)
    w._observer = fake_executor

    w.observe()

    assert fake_executor.submit_calls
    fn, args, kwargs = fake_executor.submit_calls[0]
    assert fn == w.poll
    assert args == ()
    assert kwargs == {}
    assert w._future is fake_future


def test_stop_calls_future_shutdown_when_available(monkeypatch):
    w = make_watch(monkeypatch)
    fake_future = FakeFuture()
    w._future = fake_future
    w._observer = FakeExecutor()

    w.stop()

    assert w._observer_on is False
    assert fake_future.shutdown_calls == [False]
    assert w._observer.shutdown_calls == []


def test_stop_falls_back_to_observer_shutdown_when_future_shutdown_raises_and_future_running(monkeypatch):
    w = make_watch(monkeypatch)
    fake_future = FakeFuture(shutdown_side_effect=AttributeError("no shutdown"), running=True)
    fake_observer = FakeExecutor()
    w._future = fake_future
    w._observer = fake_observer

    w.stop()

    assert w._observer_on is False
    assert fake_future.shutdown_calls == [False]
    assert fake_observer.shutdown_calls == [False]


def test_get_distance_measures_distance_and_cleans_up(monkeypatch):
    gpio = FakeGPIO(input_values=[0, 0, 1, 1, 0])
    w = make_watch(monkeypatch, gpio=gpio, trig=5, echo=6)

    times = iter([100.0, 100.0, 100.0, 100.1, 100.1, 100.1])
    monkeypatch.setattr(watch_module.time, "time", lambda: next(times))

    distance = w.get_distance()

    assert distance == round((100.1 - 100.0) * 17150, 2)
    assert gpio.cleanup_called is True
    assert gpio.setup_calls == [(5, gpio.OUT), (6, gpio.IN)]
    assert gpio.output_calls[:3] == [(5, False), (5, True), (5, False)]


def test_get_distance_uses_rounding_to_two_decimals(monkeypatch):
    gpio = FakeGPIO(input_values=[0, 1, 0])
    w = make_watch(monkeypatch, gpio=gpio)

    times = iter([10.0, 10.0, 10.0, 10.123456])
    monkeypatch.setattr(watch_module.time, "time", lambda: next(times))

    distance = w.get_distance()

    assert distance == round((10.123456 - 10.0) * 17150, 2)
    assert isinstance(distance, float)
