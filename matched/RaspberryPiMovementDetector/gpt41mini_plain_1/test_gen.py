import time
import pytest
from concurrent.futures import Future
from MovementDetector.Watch import Watch

class DummyGPIO:
    """A dummy GPIO class to simulate GPIO behavior for testing."""
    OUT = 0
    IN = 1

    def __init__(self):
        self._setup = {}
        self._outputs = {}
        self._inputs = {}
        self.cleanup_called = False
        self._input_queue_0 = []
        self._input_queue_1 = []

    def setup(self, pin, mode):
        self._setup[pin] = mode

    def output(self, pin, state):
        self._outputs[pin] = state

    def input(self, pin):
        # Return 0 or 1 depending on queues, or static if queues empty
        if self._input_queue_0:
            return self._input_queue_0.pop(0)
        if self._input_queue_1:
            return self._input_queue_1.pop(0)
        # Default simulate immediate transition from 0 to 1
        # Let's return 0 once, then 1
        if not hasattr(self, "_input_called"):
            self._input_called = True
            return 0
        return 1

    def add_input_sequence_0(self, seq):
        self._input_queue_0.extend(seq)

    def add_input_sequence_1(self, seq):
        self._input_queue_1.extend(seq)

    def cleanup(self):
        self.cleanup_called = True


@pytest.fixture
def dummy_gpio():
    return DummyGPIO()

@pytest.fixture
def watch_instance(dummy_gpio):
    return Watch(gpio=dummy_gpio, trig=17, echo=27, offset=150)

def test_trigger_pin_returns_correct_pin(watch_instance):
    assert watch_instance.trigger_pin() == 17

def test_echo_pin_returns_correct_pin(watch_instance):
    assert watch_instance.echo_pin() == 27

def test_default_constructor_functions_exist():
    # When no func_in or func_out given, dummy functions print something
    w = Watch()
    # The event emitter should have listeners for ObjectIn and ObjectOut
    assert len(w._ee.listeners("ObjectIn")) >= 1
    assert len(w._ee.listeners("ObjectOut")) >= 1

def test_custom_func_in_and_func_out():
    called = {"in": False, "out": False}
    def func_in(distance):
        called["in"] = True
        called["dist_in"] = distance
    def func_out(distance):
        called["out"] = True
        called["dist_out"] = distance
    w = Watch(func_in=func_in, func_out=func_out, offset=100)
    w._wasIn = False
    w._ee.emit("ObjectIn", 50)
    assert called["in"] is True
    w._ee.emit("ObjectOut", 120)
    assert called["out"] is True

def test_ping_emits_ObjectIn_when_distance_less_than_offset(monkeypatch):
    triggered = {"in": False, "out": False}
    def func_in(dist):
        triggered["in"] = True
        triggered["dist"] = dist
    def func_out(dist):
        triggered["out"] = True
    w = Watch(func_in=func_in, func_out=func_out, offset=50)
    w._wasIn = False

    # monkeypatch get_distance to a value less than offset
    monkeypatch.setattr(w, "get_distance", lambda: 40)
    w.ping()
    assert triggered["in"] is True
    assert triggered["dist"] == 40
    assert w._wasIn is True
    # No ObjectOut emitted
    assert not triggered["out"]

def test_ping_emits_ObjectOut_when_distance_greater_than_offset(monkeypatch):
    triggered = {"in": False, "out": False}
    def func_in(dist):
        triggered["in"] = True
    def func_out(dist):
        triggered["out"] = True
        triggered["dist"] = dist
    w = Watch(func_in=func_in, func_out=func_out, offset=50)
    w._wasIn = True

    monkeypatch.setattr(w, "get_distance", lambda: 100)
    w.ping()
    assert triggered["out"] is True
    assert triggered["dist"] == 100
    assert w._wasIn is False
    # No ObjectIn emitted
    assert not triggered["in"]

def test_ping_no_emit_when_distance_above_offset_and_wasIn_false(monkeypatch):
    triggered = {"in": False, "out": False}
    def func_in(dist):
        triggered["in"] = True
    def func_out(dist):
        triggered["out"] = True
    w = Watch(func_in=func_in, func_out=func_out, offset=50)
    w._wasIn = False
    monkeypatch.setattr(w, "get_distance", lambda: 100)
    w.ping()
    assert not triggered["in"]
    assert not triggered["out"]
    assert w._wasIn is False

def test_ping_no_emit_when_distance_below_offset_and_wasIn_true(monkeypatch):
    triggered = {"in": False, "out": False}
    def func_in(dist):
        triggered["in"] = True
    def func_out(dist):
        triggered["out"] = True
    w = Watch(func_in=func_in, func_out=func_out, offset=50)
    w._wasIn = True
    monkeypatch.setattr(w, "get_distance", lambda: 40)
    w.ping()
    assert not triggered["in"]
    assert not triggered["out"]
    assert w._wasIn is True

def test_poll_runs_ping_until_observer_off(monkeypatch):
    call_count = {"ping": 0}
    def fake_ping():
        call_count["ping"] += 1
        if call_count["ping"] >= 3:
            w._observer_on = False
    w = Watch()
    monkeypatch.setattr(w, "ping", fake_ping)
    w._observer_on = True
    w.poll()
    assert call_count["ping"] >= 3
    assert w._observer_on is False

def test_observe_submits_poll(monkeypatch):
    w = Watch()
    called = {"submitted": False}
    def fake_submit(fn):
        called["submitted"] = True
        fut = Future()
        fut.set_result(None)
        return fut
    monkeypatch.setattr(w._observer, "submit", fake_submit)
    w.observe()
    assert called["submitted"] is True
    assert isinstance(w._future, Future)

def test_stop_calls_shutdown(monkeypatch):
    w = Watch()
    # Setup fake future with shutdown
    class DummyFuture:
        def __init__(self):
            self.shutdown_called = False
        def shutdown(self, wait):
            self.shutdown_called = True
        def running(self):
            return False
    dummy_future = DummyFuture()
    w._future = dummy_future
    w._observer_on = True
    # Patch time.sleep to avoid delay
    monkeypatch.setattr(time, "sleep", lambda x: None)
    w.stop()
    assert w._observer_on is False
    assert dummy_future.shutdown_called is True

def test_stop_shutdown_running_shutdown_observer(monkeypatch):
    w = Watch()
    class DummyFuture:
        def __init__(self):
            self.shutdown_called = False
            self._running = True
        def shutdown(self, wait):
            self.shutdown_called = True
            raise Exception("fail to shutdown",)
        def running(self):
            return self._running
    class DummyObserver:
        def __init__(self):
            self.shutdown_called = False
        def shutdown(self, wait):
            self.shutdown_called = True

    dummy_future = DummyFuture()
    dummy_observer = DummyObserver()
    w._future = dummy_future
    w._observer = dummy_observer
    # Patch time.sleep to avoid delay
    monkeypatch.setattr(time, "sleep", lambda x: None)
    w._observer_on = True

    w.stop()
    assert w._observer_on is False
    assert dummy_future.shutdown_called is True
    assert dummy_observer.shutdown_called is True

def test_get_distance_basic_flow(dummy_gpio, monkeypatch):
    # Prepare dummy gpio input to simulate sensor pulse
    # input returns 0 (waiting), then 1 (pulse high), then ends
    # Put 3 zeros for loop wait, then 2 ones for loop
    dummy_gpio.add_input_sequence_0([0,0,0])
    dummy_gpio.add_input_sequence_1([1,1])
    w = Watch(gpio=dummy_gpio, trig=5, echo=6)
    # Patch time.sleep to very short to speed up test
    monkeypatch.setattr(time, "sleep", lambda x: None)
    # Patch time.time to simulate times with controlled increments
    times = [100.0, 100.0, 100.1, 100.25]
    def fake_time():
        return times.pop(0)
    monkeypatch.setattr(time, "time", fake_time)
    dist = w.get_distance()

    # Should call setup for trig (OUT) and echo (IN)
    assert dummy_gpio._setup[5] == dummy_gpio.OUT
    assert dummy_gpio._setup[6] == dummy_gpio.IN
    # Should have output True then False on trig pin during pulse
    # The last output should be False because of sequence
    assert dummy_gpio._outputs[5] is False
    # Distance should be computed as (pulse_end - pulse_start) * 17150
    # pulse_duration = 100.25 - 100.1 = 0.15
    # distance = 0.15 * 17150 = 2572.5
    assert abs(dist - 2572.5) < 0.01
    # Cleanup called
    assert dummy_gpio.cleanup_called is True

def test_get_distance_loops_input_until_non_monotonic(monkeypatch):
    dummy_gpio = DummyGPIO()
    # simulate inputs so the loops run once each
    dummy_gpio.add_input_sequence_0([0])
    dummy_gpio.add_input_sequence_1([1])
    w = Watch(gpio=dummy_gpio, trig=5, echo=6)
    # Provide sleep and time
    monkeypatch.setattr(time, "sleep", lambda x: None)
    start_time = 100.0
    call_count = {"time": 0}
    def fake_time():
        call_count["time"] += 1
        # On first call pulse_start = 100.0
        # On last pulse_end = 100.01
        return start_time + (call_count["time"] -1)*0.01
    monkeypatch.setattr(time, "time", fake_time)
    d = w.get_distance()
    assert isinstance(d, float)
    assert dummy_gpio.cleanup_called

def test_get_distance_prints(capsys, dummy_gpio, monkeypatch):
    dummy_gpio.add_input_sequence_0([0])
    dummy_gpio.add_input_sequence_1([1])
    w = Watch(gpio=dummy_gpio)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    monkeypatch.setattr(time, "time", lambda: 123.0 if not hasattr(w, "_called") else 123.01)
    w._called = False
    def fake_time():
        if not hasattr(w, "_called"):
            w._called = True
            return 123.0
        return 123.01
    monkeypatch.setattr(time, "time", fake_time)
    dist = w.get_distance()
    out = capsys.readouterr().out
    assert "Distance Measurement In Progress" in out
    assert "Distance:" in out

def test_wasIn_flag_behavior(monkeypatch):
    called_in = False
    called_out = False
    def func_in(d):
        nonlocal called_in
        called_in = True
    def func_out(d):
        nonlocal called_out
        called_out = True
    w = Watch(func_in=func_in, func_out=func_out, offset=100)
    w._wasIn = False
    monkeypatch.setattr(w, "get_distance", lambda: 50)
    w.ping()
    assert called_in
    called_in = False
    monkeypatch.setattr(w, "get_distance", lambda: 150)
    w.ping()
    assert called_out
    assert w._wasIn is False

def test_watch_multiple_object_in_and_out(monkeypatch):
    events = []
    def func_in(d): events.append(("in", d))
    def func_out(d): events.append(("out", d))
    w = Watch(func_in=func_in, func_out=func_out, offset=50)
    w._wasIn = False
    monkeypatch.setattr(w, "get_distance", lambda: 40)
    w.ping()  # should emit ObjectIn
    monkeypatch.setattr(w, "get_distance", lambda: 40)
    w.ping()  # no event, already in
    monkeypatch.setattr(w, "get_distance", lambda: 60)
    w.ping()  # should emit ObjectOut
    monkeypatch.setattr(w, "get_distance", lambda: 60)
    w.ping()  # no event, already out
    assert events == [("in", 40), ("out", 60)]

def test_constructor_default_values():
    w = Watch()
    assert w._trig == 23
    assert w._echo == 24
    assert w._offset == 200
    assert w._gpio is None
    assert w._wasIn is False
    assert w._observer_on is True

def test_future_is_none_before_observe():
    w = Watch()
    assert w._future is None

def test_stop_waits_and_shutdown(monkeypatch):
    w = Watch()
    class DummyFuture:
        def __init__(self):
            self.shutdown_called = False
        def shutdown(self, wait):
            self.shutdown_called = True
        def running(self):
            return False
    f = DummyFuture()
    w._future = f
    monkeypatch.setattr(time, "sleep", lambda x: None)
    w.stop()
    assert f.shutdown_called is True
    assert w._observer_on is False
