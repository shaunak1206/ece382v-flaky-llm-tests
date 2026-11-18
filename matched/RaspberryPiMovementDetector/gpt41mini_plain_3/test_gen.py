import pytest
import time
from unittest.mock import MagicMock, patch

from MovementDetector.Watch import Watch


class DummyGPIO:
    OUT = "out"
    IN = "in"

    def __init__(self):
        self.calls = []
        self.input_values = []
        self.output_values = {}

    def setup(self, pin, mode):
        self.calls.append(("setup", pin, mode))

    def output(self, pin, value):
        self.output_values[pin] = value
        self.calls.append(("output", pin, value))

    def input(self, pin):
        # Pop first value from input_values if available, else 0
        if self.input_values:
            return self.input_values.pop(0)
        return 0

    def cleanup(self):
        self.calls.append(("cleanup",))


def test_trigger_pin_default():
    w = Watch()
    assert w.trigger_pin() == 23

def test_echo_pin_default():
    w = Watch()
    assert w.echo_pin() == 24

def test_trigger_echo_pin_custom():
    w = Watch(trig=5, echo=6)
    assert w.trigger_pin() == 5
    assert w.echo_pin() == 6

def test_offset_default():
    w = Watch()
    assert w._offset == 200

def test_wasIn_initial_false():
    w = Watch()
    assert w._wasIn is False

def test_ping_emits_ObjectIn_and_ObjectOut(tmp_path):
    # Setup dummy gpio with controlled input to simulate distance < offset and > offset
    gpio = DummyGPIO()
    # To test ping, we need to patch get_distance to simulate distance values

    emitted = {"in": [], "out": []}
    def on_in(distance):
        emitted["in"].append(distance)
    def on_out(distance):
        emitted["out"].append(distance)

    w = Watch(gpio=gpio, func_in=on_in, func_out=on_out, offset=100)
    
    # First ping, distance < offset --> triggers ObjectIn
    w.get_distance = MagicMock(return_value=50)
    w._wasIn = False
    w.ping()
    assert emitted["in"] == [50]
    assert emitted["out"] == []
    assert w._wasIn is True

    # Second ping, still < offset, no new event
    emitted["in"].clear()
    w.ping()
    assert emitted["in"] == []
    assert w._wasIn is True

    # Third ping, distance > offset --> triggers ObjectOut
    w.get_distance = MagicMock(return_value=150)
    w.ping()
    assert emitted["out"] == [150]
    assert w._wasIn is False

    # Fourth ping, distance > offset again, no new event
    emitted["out"].clear()
    w.ping()
    assert emitted["out"] == []
    assert w._wasIn is False

def test_poll_stops_when_observer_off():
    w = Watch()
    w._observer_on = False

    # Since poll loops infinitely while _observer_on is True,
    # but here it's False so poll returns immediately.
    # We patch ping to track calls.
    w.ping = MagicMock()

    w.poll()
    w.ping.assert_not_called()

def test_poll_runs_ping_once_and_stops(monkeypatch):
    # Patch ping to stop after first call by toggling _observer_on
    w = Watch()
    count = {"calls":0}
    original_ping = w.ping
    def ping_side_effect():
        count["calls"] += 1
        if count["calls"] >= 3:
            w._observer_on = False
        original_ping()

    w.ping = MagicMock(side_effect=ping_side_effect)
    # Run poll with limited loop by toggling _observer_on off after some calls
    w.poll()
    assert count["calls"] == 3

def test_observe_submits_poll(monkeypatch):
    w = Watch()
    called = {"submitted": False}

    def fake_submit(func):
        called["submitted"] = True
        # return a dummy future with shutdown method
        class DummyFuture:
            def shutdown(self, wait):
                pass
            def running(self):
                return False
        return DummyFuture()

    w._observer.submit = fake_submit
    w.observe()
    assert called["submitted"] is True

def test_stop_calls_shutdown_and_sleep(monkeypatch):
    w = Watch()
    dummy_future = MagicMock()
    dummy_future.shutdown = MagicMock(side_effect=Exception("fail"))
    dummy_future.running = MagicMock(return_value=True)
    w._future = dummy_future
    w._observer.shutdown = MagicMock()

    # Patch time.sleep to avoid real sleep
    monkeypatch.setattr(time, "sleep", lambda s: None)

    w.stop()
    dummy_future.shutdown.assert_called_once_with(wait=False)
    dummy_future.running.assert_called_once()
    w._observer.shutdown.assert_called_once_with(wait=False)

def test_stop_shutdown_no_exception(monkeypatch):
    w = Watch()
    dummy_future = MagicMock()
    dummy_future.shutdown = MagicMock()
    w._future = dummy_future
    w._observer.shutdown = MagicMock()

    monkeypatch.setattr(time, "sleep", lambda s: None)

    w.stop()
    dummy_future.shutdown.assert_called_once_with(wait=False)
    w._observer.shutdown.assert_not_called()

def test_get_distance_sequence(monkeypatch):
    gpio = DummyGPIO()
    # We'll set inputs to simulate the sensor echo pin changes

    # The input sequence: [0,0,...,1,1,...]
    # so first few inputs 0, then some 1's.
    gpio.input_values = [0,0,0,1,1,1,1]

    w = Watch(gpio=gpio, trig=17, echo=18)

    # Patch time.sleep to speed up test
    monkeypatch.setattr(time, "sleep", lambda s: None)

    dist = w.get_distance()

    # Check that distance is float rounded to 2 decimals
    assert isinstance(dist, float)
    # Check gpio calls include setup for trig and echo
    setups = [call for call in gpio.calls if call[0] == "setup"]
    assert ("setup", 17, gpio.OUT) in setups
    assert ("setup", 18, gpio.IN) in setups
    # Check gpio.output to trig was called with False, True, False in order
    outputs = [call for call in gpio.calls if call[0] == "output"]
    output_calls = [(c[1], c[2]) for c in outputs]
    expected_sequence = [(17, False), (17, True), (17, False)]
    assert all(seq in output_calls for seq in expected_sequence)
    # Check cleanup was called
    assert ("cleanup",) in gpio.calls
    # distance should be >= 0 and reasonable (pulse_duration * 17150)
    assert dist >= 0

def test_dummy_func_in_and_out_print(capsys):
    # Create Watch without func_in and func_out to trigger dummy prints
    w = Watch()
    w._ee.emit("ObjectIn", 0)
    w._ee.emit("ObjectOut", 0)
    captured = capsys.readouterr()

    assert "Dummy in function" in captured.out
    assert "Dummy out function" in captured.out

def test_get_distance_times(monkeypatch):
    gpio = DummyGPIO()
    w = Watch(gpio=gpio)

    # patch time.time to simulate controlled pulse durations
    times = [1.0, 1.0, 1.0, 1.005, 1.01]
    time_iter = iter(times)
    monkeypatch.setattr(time, "time", lambda: next(time_iter))

    # patch time.sleep to no-op
    monkeypatch.setattr(time, "sleep", lambda s: None)

    # input returns 0 while pulse_start; then 1 for pulse_end
    gpio.input_values = [0, 1]

    dist = w.get_distance()
    expected_duration = 1.01 - 1.0
    expected_distance = round(expected_duration * 17150, 2)
    assert dist == expected_distance

def test_wasIn_flag_changes_on_ping_events():
    w = Watch()
    w.get_distance = MagicMock(return_value=50)
    w._wasIn = False

    w.ping()
    assert w._wasIn is True

    w.get_distance = MagicMock(return_value=250)
    w.ping()
    assert w._wasIn is False

def test_stop_wait_and_shutdown_order(monkeypatch):
    w = Watch()
    called = []

    class DummyFuture:
        def shutdown(self, wait):
            called.append("shutdown")
        def running(self):
            return False

    w._future = DummyFuture()
    w._observer.shutdown = lambda wait: called.append("observer_shutdown")
    monkeypatch.setattr(time, "sleep", lambda s: called.append(f"sleep_{s}"))

    w.stop()
    assert called[0].startswith("sleep_")
    assert called[1] == "shutdown"

def test_stop_shutdown_runs_observer_shutdown_on_running(monkeypatch):
    w = Watch()
    called = []

    class DummyFuture:
        def shutdown(self, wait):
            raise Exception("fail shutdown")
        def running(self):
            called.append("running_called")
            return True

    w._future = DummyFuture()
    w._observer.shutdown = lambda wait: called.append("observer_shutdown")
    monkeypatch.setattr(time, "sleep", lambda s: None)

    w.stop()
    assert "running_called" in called
    assert "observer_shutdown" in called

def test_ping_does_not_emit_if_in_same_state():
    emitted = {"in": 0, "out": 0}
    def on_in(d): emitted["in"] += 1
    def on_out(d): emitted["out"] += 1
    w = Watch(func_in=on_in, func_out=on_out, offset=100)
    w.get_distance = MagicMock(return_value=50)

    # First call emits ObjectIn
    w._wasIn = False
    w.ping()
    assert emitted["in"] == 1
    assert emitted["out"] == 0

    # Subsequent calls with distance < offset do not emit again
    w.ping()
    assert emitted["in"] == 1
    assert emitted["out"] == 0

    # Change distance > offset, emit ObjectOut
    w.get_distance = MagicMock(return_value=150)
    w.ping()
    assert emitted["out"] == 1

    # Next call with distance > offset no emit
    w.ping()
    assert emitted["out"] == 1

def test_observe_returns_none():
    w = Watch()
    w._observer.submit = MagicMock(return_value="future")
    ret = w.observe()
    assert ret is None
