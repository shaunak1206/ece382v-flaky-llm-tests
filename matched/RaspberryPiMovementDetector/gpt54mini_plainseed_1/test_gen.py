import types
import pytest

import MovementDetector.Watch as watch_module
from MovementDetector.Watch import Watch


class FakeEmitter:
    def __init__(self, *args, **kwargs):
        self.handlers = {}
        self.emitted = []

    def on(self, name, handler):
        self.handlers[name] = handler

    def emit(self, name, *args, **kwargs):
        self.emitted.append((name, args, kwargs))
        if name in self.handlers:
            return self.handlers[name](*args, **kwargs)


class FakeGPIO:
    OUT = "OUT"
    IN = "IN"

    def __init__(self, input_sequence=None):
        self.calls = []
        self.input_sequence = list(input_sequence or [])

    def setup(self, pin, mode):
        self.calls.append(("setup", pin, mode))

    def output(self, pin, value):
        self.calls.append(("output", pin, value))

    def input(self, pin):
        self.calls.append(("input", pin))
        if self.input_sequence:
            return self.input_sequence.pop(0)
        return 0

    def cleanup(self):
        self.calls.append(("cleanup",))


class FakeFuture:
    def __init__(self, shutdown_raises=False, running_value=False):
        self.shutdown_calls = []
        self.shutdown_raises = shutdown_raises
        self.running_value = running_value

    def shutdown(self, wait=False):
        self.shutdown_calls.append(wait)
        if self.shutdown_raises:
            raise RuntimeError("boom")

    def running(self):
        return self.running_value


class FakeObserver:
    def __init__(self, future=None):
        self.future = future if future is not None else FakeFuture()
        self.submitted = []

    def submit(self, fn):
        self.submitted.append(fn)
        return self.future


@pytest.fixture(autouse=True)
def patch_emitter(monkeypatch):
    monkeypatch.setattr(watch_module, "EventEmitter", FakeEmitter)
    yield


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(watch_module.time, "sleep", lambda *args, **kwargs: None)


def test_init_defaults_registers_handlers():
    w = Watch()
    assert w.trigger_pin() == 23
    assert w.echo_pin() == 24
    assert w._offset == 200
    assert w._wasIn is False
    assert "ObjectIn" in w._ee.handlers
    assert "ObjectOut" in w._ee.handlers


def test_init_custom_kwargs_are_used():
    gpio = object()

    def func_in(distance):
        return distance + 1

    def func_out(distance):
        return distance - 1

    w = Watch(gpio=gpio, trig=11, echo=12, func_in=func_in, func_out=func_out, offset=99)
    assert w._gpio is gpio
    assert w.trigger_pin() == 11
    assert w.echo_pin() == 12
    assert w._offset == 99
    assert w._ee.handlers["ObjectIn"] is func_in
    assert w._ee.handlers["ObjectOut"] is func_out


def test_trigger_and_echo_pin_accessors():
    w = Watch(trig=7, echo=8)
    assert w.trigger_pin() == 7
    assert w.echo_pin() == 8


def test_ping_emits_object_in_when_distance_below_offset_and_not_was_in(monkeypatch):
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    w.get_distance = lambda: 42

    w.ping()

    assert w._wasIn is True
    assert w._ee.emitted == [("ObjectIn", (42,), {})]


def test_ping_does_not_emit_object_in_again_when_already_in(monkeypatch):
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    w._wasIn = True
    w.get_distance = lambda: 42

    w.ping()

    assert w._wasIn is True
    assert w._ee.emitted == []


def test_ping_emits_object_out_when_distance_above_offset_and_was_in(monkeypatch):
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    w._wasIn = True
    w.get_distance = lambda: 101

    w.ping()

    assert w._wasIn is False
    assert w._ee.emitted == [("ObjectOut", (101,), {})]


def test_ping_does_not_emit_object_out_when_not_was_in(monkeypatch):
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    w._wasIn = False
    w.get_distance = lambda: 101

    w.ping()

    assert w._wasIn is False
    assert w._ee.emitted == []


def test_ping_uses_strict_less_than_offset_for_in_zone():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    w.get_distance = lambda: 100

    w.ping()

    assert w._wasIn is False
    assert w._ee.emitted == []


def test_poll_calls_ping_until_observer_off_then_returns():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)

    calls = {"n": 0}

    def fake_ping():
        calls["n"] += 1
        if calls["n"] >= 3:
            w._observer_on = False

    w.ping = fake_ping

    w.poll()

    assert calls["n"] == 3


def test_observe_submits_poll_to_observer():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)
    observer = FakeObserver()
    w._observer = observer

    w.observe()

    assert observer.submitted == [w.poll]
    assert w._future is observer.future


def test_stop_sets_observer_off_and_shuts_down_future(no_sleep):
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)
    future = FakeFuture()
    w._future = future

    w.stop()

    assert w._observer_on is False
    assert future.shutdown_calls == [False]


def test_stop_handles_shutdown_exception_when_future_not_running(no_sleep):
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)
    future = FakeFuture(shutdown_raises=True, running_value=False)
    w._future = future

    w.stop()

    assert w._observer_on is False
    assert future.shutdown_calls == [False]


def test_stop_handles_shutdown_exception_when_future_running(no_sleep):
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)
    future = FakeFuture(shutdown_raises=True, running_value=True)

    class RecordingObserver:
        def __init__(self):
            self.shutdown_calls = []

        def shutdown(self, wait=False):
            self.shutdown_calls.append(wait)

    observer = RecordingObserver()
    w._future = future
    w._observer = observer

    w.stop()

    assert w._observer_on is False
    assert future.shutdown_calls == [False]
    assert observer.shutdown_calls == [False]


def test_get_distance_performs_gpio_sequence_and_returns_expected_distance(monkeypatch):
    gpio = FakeGPIO(input_sequence=[0, 0, 1, 1, 1, 0])
    w = Watch(gpio=gpio, trig=17, echo=27)

    times = iter([10.0, 10.5, 11.0, 11.0, 11.2, 11.2, 11.2])
    monkeypatch.setattr(watch_module.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(watch_module.time, "time", lambda: next(times))

    distance = w.get_distance()

    assert distance == round((11.2 - 11.0) * 17150, 2)
    assert ("setup", 17, gpio.OUT) in gpio.calls
    assert ("setup", 27, gpio.IN) in gpio.calls
    assert ("output", 17, False) in gpio.calls
    assert ("output", 17, True) in gpio.calls
    assert gpio.calls[-1] == ("cleanup",)


def test_get_distance_calls_cleanup_even_with_simple_sequence(monkeypatch):
    gpio = FakeGPIO(input_sequence=[0, 1, 0])
    w = Watch(gpio=gpio)

    times = iter([1.0, 1.1, 1.1, 1.3, 1.3])
    monkeypatch.setattr(watch_module.time, "sleep", lambda *args, **kwargs: None)
    monkeypatch.setattr(watch_module.time, "time", lambda: next(times))

    _ = w.get_distance()

    assert ("cleanup",) in gpio.calls
