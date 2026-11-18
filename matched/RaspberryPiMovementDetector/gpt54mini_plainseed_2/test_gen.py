import types

import pytest

import MovementDetector.Watch as watch_module
from MovementDetector.Watch import Watch


class DummyEventEmitter:
    def __init__(self, *args, **kwargs):
        self.handlers = {}
        self.emitted = []

    def on(self, event_name, handler):
        self.handlers.setdefault(event_name, []).append(handler)

    def emit(self, event_name, *args, **kwargs):
        self.emitted.append((event_name, args, kwargs))
        for handler in self.handlers.get(event_name, []):
            handler(*args, **kwargs)


class DummyFuture:
    def __init__(self):
        self.shutdown_called = False
        self.shutdown_wait = None
        self.running_called = False

    def shutdown(self, wait=False):
        self.shutdown_called = True
        self.shutdown_wait = wait

    def running(self):
        self.running_called = True
        return True


class DummyExecutor:
    def __init__(self, *args, **kwargs):
        self.max_workers = kwargs.get("max_workers")
        self.submitted = []
        self.shutdown_called = False
        self.shutdown_wait = None

    def submit(self, fn, *args, **kwargs):
        self.submitted.append((fn, args, kwargs))
        return DummyFuture()

    def shutdown(self, wait=False):
        self.shutdown_called = True
        self.shutdown_wait = wait


class FakeGPIO:
    OUT = "OUT"
    IN = "IN"

    def __init__(self, input_values=None):
        self.calls = []
        self.input_values = list(input_values or [])
        self.cleanup_called = False
        self._input_index = 0

    def setup(self, pin, mode):
        self.calls.append(("setup", pin, mode))

    def output(self, pin, value):
        self.calls.append(("output", pin, value))

    def input(self, pin):
        self.calls.append(("input", pin))
        if self._input_index < len(self.input_values):
            value = self.input_values[self._input_index]
            self._input_index += 1
            return value
        return self.input_values[-1] if self.input_values else 0

    def cleanup(self):
        self.calls.append(("cleanup",))
        self.cleanup_called = True


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    monkeypatch.setattr(watch_module, "EventEmitter", DummyEventEmitter)
    monkeypatch.setattr(watch_module, "ThreadPoolExecutor", DummyExecutor)
    monkeypatch.setattr(watch_module.time, "sleep", lambda *args, **kwargs: None)


def test_init_uses_default_pins_and_offset():
    w = Watch(gpio=FakeGPIO())
    assert w.trigger_pin() == 23
    assert w.echo_pin() == 24
    assert w._offset == 200
    assert w._wasIn is False
    assert w._observer_on is True


def test_init_accepts_custom_pins_and_offset():
    w = Watch(gpio=FakeGPIO(), trig=5, echo=6, offset=123)
    assert w.trigger_pin() == 5
    assert w.echo_pin() == 6
    assert w._offset == 123


def test_init_registers_default_handlers_when_none_given(capsys):
    w = Watch(gpio=FakeGPIO(), func_in=None, func_out=None)
    w._ee.emit("ObjectIn", 10)
    w._ee.emit("ObjectOut", 20)
    out = capsys.readouterr().out
    assert "Dummy in function" in out
    assert "Dummy out function" in out


def test_init_registers_custom_handlers():
    events = []

    def on_in(distance):
        events.append(("in", distance))

    def on_out(distance):
        events.append(("out", distance))

    w = Watch(gpio=FakeGPIO(), func_in=on_in, func_out=on_out)
    w._ee.emit("ObjectIn", 11)
    w._ee.emit("ObjectOut", 22)
    assert events == [("in", 11), ("out", 22)]


def test_ping_emits_object_in_once_when_distance_below_offset():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    w.get_distance = lambda: 42

    w.ping()
    w.ping()

    assert w._wasIn is True
    emitted = [event for event, _, _ in w._ee.emitted]
    assert emitted == ["ObjectIn"]


def test_ping_emits_object_out_once_when_distance_above_offset_after_in():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    distances = iter([42, 150])
    w.get_distance = lambda: next(distances)

    w.ping()
    w.ping()

    emitted = [event for event, _, _ in w._ee.emitted]
    assert emitted == ["ObjectIn", "ObjectOut"]
    assert w._wasIn is False


def test_ping_does_not_emit_object_out_when_not_previous_in():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    w.get_distance = lambda: 150

    w.ping()

    assert w._ee.emitted == []
    assert w._wasIn is False


def test_ping_does_not_emit_object_in_when_already_in():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio, offset=100)
    distances = iter([50, 40, 30])
    w.get_distance = lambda: next(distances)

    w.ping()
    w.ping()
    w.ping()

    emitted = [event for event, _, _ in w._ee.emitted]
    assert emitted == ["ObjectIn"]
    assert w._wasIn is True


def test_poll_calls_ping_until_observer_flag_is_cleared():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)
    calls = {"count": 0}

    def fake_ping():
        calls["count"] += 1
        w._observer_on = False

    w.ping = fake_ping
    w.poll()

    assert calls["count"] == 1


def test_observe_submits_poll_to_executor():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)

    w.observe()

    assert isinstance(w._future, DummyFuture)
    assert len(w._observer.submitted) == 1
    submitted_fn, args, kwargs = w._observer.submitted[0]
    assert submitted_fn == w.poll
    assert args == ()
    assert kwargs == {}


def test_stop_disables_observer_and_shuts_future_down():
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)
    w._future = DummyFuture()

    w.stop()

    assert w._observer_on is False
    assert w._future.shutdown_called is True
    assert w._future.shutdown_wait is False


def test_stop_uses_observer_shutdown_when_future_shutdown_raises(monkeypatch):
    gpio = FakeGPIO()
    w = Watch(gpio=gpio)

    class BadFuture:
        def __init__(self):
            self.running_called = False

        def shutdown(self, wait=False):
            raise RuntimeError("boom")

        def running(self):
            self.running_called = True
            return True

    w._future = BadFuture()

    w.stop()

    assert w._observer_on is False
    assert w._future.running_called is True
    assert w._observer.shutdown_called is True
    assert w._observer.shutdown_wait is False


def test_get_distance_returns_expected_distance_and_cleans_up(monkeypatch):
    gpio = FakeGPIO(input_values=[0, 0, 1, 1, 0])
    w = Watch(gpio=gpio, trig=17, echo=27)

    times = iter([100.0, 102.0, 102.1, 102.2])
    monkeypatch.setattr(watch_module.time, "time", lambda: next(times))

    distance = w.get_distance()

    assert distance == round((102.1 - 102.0) * 17150, 2)
    assert gpio.cleanup_called is True


def test_get_distance_uses_gpio_setup_output_and_input_sequence(monkeypatch):
    gpio = FakeGPIO(input_values=[0, 1, 0])
    w = Watch(gpio=gpio, trig=12, echo=16)

    times = iter([200.0, 201.0, 201.2, 201.3])
    monkeypatch.setattr(watch_module.time, "time", lambda: next(times))

    w.get_distance()

    assert ("setup", 12, gpio.OUT) in gpio.calls
    assert ("setup", 16, gpio.IN) in gpio.calls
    assert ("output", 12, False) in gpio.calls
    assert ("output", 12, True) in gpio.calls
    assert gpio.cleanup_called is True
