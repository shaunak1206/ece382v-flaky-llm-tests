import pytest

from MovementDetector.Watch import Watch


class DummyGPIO:
    OUT = "OUT"
    IN = "IN"

    def __init__(self, input_values=None):
        self.input_values = list(input_values or [])
        self.setup_calls = []
        self.output_calls = []
        self.cleanup_called = False
        self.input_calls = 0

    def setup(self, pin, mode):
        self.setup_calls.append((pin, mode))

    def output(self, pin, value):
        self.output_calls.append((pin, value))

    def input(self, pin):
        self.input_calls += 1
        if self.input_values:
            return self.input_values.pop(0)
        return 0

    def cleanup(self):
        self.cleanup_called = True


class DummyFuture:
    def __init__(self, running=False, shutdown_raises=False):
        self.running_flag = running
        self.shutdown_raises = shutdown_raises
        self.shutdown_called = False
        self.shutdown_wait = None

    def shutdown(self, wait=False):
        self.shutdown_called = True
        self.shutdown_wait = wait
        if self.shutdown_raises:
            raise RuntimeError("boom")

    def running(self):
        return self.running_flag


class DummyExecutor:
    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args, **kwargs):
        self.submitted.append((fn, args, kwargs))
        return DummyFuture()


def test_init_defaults_sets_default_pins_and_offset():
    watch = Watch(gpio=DummyGPIO())
    assert watch.trigger_pin() == 23
    assert watch.echo_pin() == 24
    assert watch._offset == 200
    assert watch._wasIn is False


def test_init_custom_values_are_used():
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio, trig=5, echo=6, offset=123)
    assert watch.trigger_pin() == 5
    assert watch.echo_pin() == 6
    assert watch._offset == 123


def test_ping_emits_object_in_when_distance_below_offset(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio, offset=50)
    events = []

    watch._ee.on("ObjectIn", lambda d: events.append(("in", d)))
    watch._ee.on("ObjectOut", lambda d: events.append(("out", d)))

    monkeypatch.setattr(watch, "get_distance", lambda: 12.5)
    watch.ping()

    assert events == [("in", 12.5)]
    assert watch._wasIn is True


def test_ping_does_not_emit_object_in_twice_while_still_in(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio, offset=50)
    events = []

    watch._ee.on("ObjectIn", lambda d: events.append(d))
    monkeypatch.setattr(watch, "get_distance", lambda: 10)

    watch.ping()
    watch.ping()

    assert events == [10]
    assert watch._wasIn is True


def test_ping_emits_object_out_when_distance_reaches_or_exceeds_offset(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio, offset=50)
    events = []

    watch._ee.on("ObjectOut", lambda d: events.append(("out", d)))
    watch._wasIn = True
    monkeypatch.setattr(watch, "get_distance", lambda: 55)

    watch.ping()

    assert events == [("out", 55)]
    assert watch._wasIn is False


def test_ping_does_not_emit_object_out_if_not_previous_in(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio, offset=50)
    events = []

    watch._ee.on("ObjectOut", lambda d: events.append(d))
    monkeypatch.setattr(watch, "get_distance", lambda: 75)

    watch.ping()

    assert events == []
    assert watch._wasIn is False


def test_poll_calls_ping_once_and_returns_when_observer_disabled(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio)
    calls = []

    def fake_ping():
        calls.append("ping")
        watch._observer_on = False

    monkeypatch.setattr(watch, "ping", fake_ping)

    watch.poll()

    assert calls == ["ping"]


def test_observe_submits_poll_to_executor(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio)
    dummy_executor = DummyExecutor()
    watch._observer = dummy_executor

    watch.observe()

    assert len(dummy_executor.submitted) == 1
    fn, args, kwargs = dummy_executor.submitted[0]
    assert fn == watch.poll
    assert args == ()
    assert kwargs == {}
    assert watch._future is not None


def test_stop_sets_observer_off_and_shutdowns_future(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio)
    future = DummyFuture()
    watch._future = future

    monkeypatch.setattr("MovementDetector.Watch.time.sleep", lambda s: None)

    watch.stop()

    assert watch._observer_on is False
    assert future.shutdown_called is True
    assert future.shutdown_wait is False


def test_stop_handles_shutdown_exception_and_calls_observer_shutdown_if_running(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio)
    future = DummyFuture(running=True, shutdown_raises=True)
    watch._future = future

    class Observer:
        def __init__(self):
            self.shutdown_called = False
            self.shutdown_wait = None

        def shutdown(self, wait=False):
            self.shutdown_called = True
            self.shutdown_wait = wait

    observer = Observer()
    watch._observer = observer

    monkeypatch.setattr("MovementDetector.Watch.time.sleep", lambda s: None)

    watch.stop()

    assert future.shutdown_called is True
    assert observer.shutdown_called is True
    assert observer.shutdown_wait is False


def test_get_distance_measures_and_cleans_up(monkeypatch):
    gpio = DummyGPIO(input_values=[0, 0, 1, 1, 0])
    watch = Watch(gpio=gpio, trig=17, echo=27)

    monkeypatch.setattr("MovementDetector.Watch.time.sleep", lambda s: None)
    times = iter([100.0, 100.0, 100.0, 100.002, 100.002, 100.002])
    monkeypatch.setattr("MovementDetector.Watch.time.time", lambda: next(times))

    distance = watch.get_distance()

    assert distance == round((100.002 - 100.0) * 17150, 2)
    assert gpio.cleanup_called is True
    assert (17, gpio.OUT) in gpio.setup_calls
    assert (27, gpio.IN) in gpio.setup_calls
    assert gpio.output_calls[:3] == [(17, False), (17, True), (17, False)]


def test_get_distance_uses_correct_pins_in_output_sequence(monkeypatch):
    gpio = DummyGPIO(input_values=[0, 1, 0])
    watch = Watch(gpio=gpio, trig=9, echo=10)

    monkeypatch.setattr("MovementDetector.Watch.time.sleep", lambda s: None)
    monkeypatch.setattr("MovementDetector.Watch.time.time", lambda: 1.0)

    watch.get_distance()

    assert gpio.setup_calls == [(9, gpio.OUT), (10, gpio.IN)]
    assert gpio.output_calls[0] == (9, False)
    assert gpio.cleanup_called is True


def test_get_distance_returns_float_with_two_decimals(monkeypatch):
    gpio = DummyGPIO(input_values=[0, 1, 0])
    watch = Watch(gpio=gpio)

    monkeypatch.setattr("MovementDetector.Watch.time.sleep", lambda s: None)
    monkeypatch.setattr("MovementDetector.Watch.time.time", lambda: 1.0)

    distance = watch.get_distance()

    assert isinstance(distance, float)
    assert distance == 0.0


def test_multiple_ping_transitions_in_and_out(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio, offset=50)
    events = []

    watch._ee.on("ObjectIn", lambda d: events.append(("in", d)))
    watch._ee.on("ObjectOut", lambda d: events.append(("out", d)))

    readings = iter([10, 60, 20])
    monkeypatch.setattr(watch, "get_distance", lambda: next(readings))

    watch.ping()
    watch.ping()
    watch.ping()

    assert events == [("in", 10), ("out", 60), ("in", 20)]


def test_poll_returns_immediately_when_observer_flag_false(monkeypatch):
    gpio = DummyGPIO()
    watch = Watch(gpio=gpio)
    watch._observer_on = False

    called = []
    monkeypatch.setattr(watch, "ping", lambda: called.append("ping"))

    watch.poll()

    assert called == []


def test_custom_handlers_receive_emitted_distances():
    gpio = DummyGPIO()
    received_in = []
    received_out = []

    watch = Watch(
        gpio=gpio,
        func_in=lambda d: received_in.append(d),
        func_out=lambda d: received_out.append(d),
        offset=100,
    )

    watch._ee.emit("ObjectIn", 33)
    watch._ee.emit("ObjectOut", 101)

    assert received_in == [33]
    assert received_out == [101]
