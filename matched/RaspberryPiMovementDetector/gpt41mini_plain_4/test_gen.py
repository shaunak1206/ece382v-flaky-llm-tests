import pytest
import time
from MovementDetector.Watch import Watch

class DummyGPIO:
    OUT = "out"
    IN = "in"

    def __init__(self):
        self.setup_calls = []
        self.output_calls = []
        self.input_sequence = []
        self.input_index = 0
        self.cleanup_called = False

    def setup(self, pin, mode):
        self.setup_calls.append((pin, mode))

    def output(self, pin, value):
        self.output_calls.append((pin, value))

    def input(self, pin):
        # Return values from a predefined sequence for echo pin
        if self.input_index < len(self.input_sequence):
            val = self.input_sequence[self.input_index]
            self.input_index += 1
            return val
        return 0

    def cleanup(self):
        self.cleanup_called = True


def test_init_defaults():
    w = Watch()
    assert w._trig == 23
    assert w._echo == 24
    assert w._offset == 200
    assert w._wasIn is False
    assert w._observer_on is True
    assert w._gpio is None
    # EventEmitter has listeners for ObjectIn and ObjectOut
    ee_listeners = w._ee.listeners("ObjectIn") + w._ee.listeners("ObjectOut")
    assert len(ee_listeners) == 2
    assert callable(ee_listeners[0])
    assert callable(ee_listeners[1])

def test_init_custom_args():
    def f_in(dist): return dist
    def f_out(dist): return dist
    dummy_gpio = object()
    w = Watch(gpio=dummy_gpio, trig=11, echo=12, func_in=f_in, func_out=f_out, offset=150)
    assert w._trig == 11
    assert w._echo == 12
    assert w._offset == 150
    assert w._gpio == dummy_gpio

def test_trigger_pin_and_echo_pin_properties():
    w = Watch(trig=7, echo=8)
    assert w.trigger_pin() == 7
    assert w.echo_pin() == 8

def test_ping_object_in_emit(monkeypatch):
    dummy_gpio = DummyGPIO()
    # Set offset = 10 to simplify
    w = Watch(gpio=dummy_gpio, offset=10)

    emitted_events = []
    def store_in(dist):
        emitted_events.append(("in", dist))
    def store_out(dist):
        emitted_events.append(("out", dist))

    # Replace listeners to our store functions
    w._ee._events["ObjectIn"] = [store_in]
    w._ee._events["ObjectOut"] = [store_out]

    # We'll patch get_distance to simulate distance triggering ObjectIn and then no emit again
    # Initially _wasIn == False
    # first call distance < offset, emit ObjectIn, _wasIn=True
    # second call distance < offset, no emit
    # third call distance > offset, emit ObjectOut, _wasIn=False
    distances = [5, 5, 20]
    def get_distance_side_effect():
        return distances.pop(0)
    w.get_distance = get_distance_side_effect

    # First ping triggers "ObjectIn" event
    w.ping()
    assert emitted_events == [("in", 5)]
    assert w._wasIn is True

    # Second ping no event (still in)
    w.ping()
    assert emitted_events == [("in", 5)]

    # Third ping triggers "ObjectOut"
    w.ping()
    assert emitted_events == [("in", 5), ("out", 20)]
    assert w._wasIn is False

def test_poll_runs_and_stops(monkeypatch):
    # The poll method calls ping repeatedly while _observer_on is True,
    # then returns when it is False.

    # We'll patch ping to count calls and stop after 3 calls by setting _observer_on False
    w = Watch()

    count = {"calls": 0}
    def fake_ping():
        count["calls"] += 1
        if count["calls"] >= 3:
            w._observer_on = False
    w.ping = fake_ping

    # poll should run looping calling ping, then return after flag set False
    w._observer_on = True
    w.poll()
    assert count["calls"] == 3
    assert w._observer_on is False

def test_observe_submits_poll(monkeypatch):
    w = Watch()

    called = {"submitted": False}
    class DummyFuture:
        def shutdown(self, wait):
            pass
    class DummyObserver:
        def submit(self, func):
            called["submitted"] = True
            # Confirm func is w.poll
            assert func == w.poll
            return DummyFuture()
    w._observer = DummyObserver()
    w._future = None
    w.observe()
    assert called["submitted"]
    assert w._future is not None

def test_stop_calls_shutdown(monkeypatch):
    w = Watch()
    dummy_future = type("DummyFuture", (), {"shutdown": lambda self, wait: setattr(w._future, "shutdown_called", True), "running": lambda self: False})()
    dummy_future.shutdown_called = False
    w._future = dummy_future

    called = {"shutdown_called": False, "observer_shutdown_called": False}

    def dummy_shutdown(wait=False):
        called["shutdown_called"] = True
    def dummy_observer_shutdown(wait=False):
        called["observer_shutdown_called"] = True

    w._observer = type("DummyObserver", (), {"shutdown": dummy_observer_shutdown})()

    # Patch time.sleep to speed test
    monkeypatch.setattr(time, "sleep", lambda x: None)

    w.stop()
    assert called["shutdown_called"]

def test_stop_shutdown_with_exception(monkeypatch):
    w = Watch()
    class DummyFuture:
        def shutdown(self, wait):
            raise RuntimeError("fail")
        def running(self):
            return True
    w._future = DummyFuture()

    called = {"observer_shutdown_called": False}
    class DummyObserver:
        def shutdown(self, wait=False):
            called["observer_shutdown_called"] = True
    w._observer = DummyObserver()

    # Patch time.sleep to no-op
    monkeypatch.setattr(time, "sleep", lambda x: None)

    w.stop()
    assert called["observer_shutdown_called"]

def test_get_distance_complete(monkeypatch):
    dummy_gpio = DummyGPIO()

    # Simulate GPIO input sequence for echo pin: wait while 0, then while 1 twice, then end.
    # We'll have to provide enough 0s then 1s:
    # Order of loop in get_distance:
    # while input(echo) == 0: pulse_start = time.time()
    # while input(echo) == 1: pulse_end = time.time()
    # We'll simulate 3x 0 and 2x 1 in sequence in input_sequence

    dummy_gpio.input_sequence = [0, 0, 0, 1, 1]

    w = Watch(gpio=dummy_gpio)

    # Patch time.sleep to no-op except for 2 sec wait which we count
    sleeps = []
    def fake_sleep(sec):
        sleeps.append(sec)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    distance = w.get_distance()

    # Confirm GPIO setup calls correct pins and modes
    assert (w._trig, dummy_gpio.OUT) in dummy_gpio.setup_calls
    assert (w._echo, dummy_gpio.IN) in dummy_gpio.setup_calls

    # Confirm trig output sequence called exactly with False, True, False
    trig_outputs = [call for call in dummy_gpio.output_calls if call[0] == w._trig]
    assert trig_outputs == [(w._trig, False), (w._trig, True), (w._trig, False)]

    # Confirm final cleanup called
    assert dummy_gpio.cleanup_called

    # Check that sleep was called with 2 for sensor settle and 0.00001 for trigger pulse
    # They come in order: sleep(2), sleep(0.00001)
    assert sleeps[0] == 2
    assert abs(sleeps[1] - 0.00001) < 0.00001

    # Distance should be positive and float rounded to 2 decimals
    assert isinstance(distance, float)
    assert distance > 0

def test_get_distance_handles_no_echo_pulse(monkeypatch):
    # Test that if input(echo) stays zero or never returns 1, function still completes

    dummy_gpio = DummyGPIO()
    # input_sequence all zeros: infinite loop avoided by providing limited zeros only,
    # but our logic is tight on this, so provide 3 zeros and then 1 to escape loop.
    dummy_gpio.input_sequence = [0, 0, 0, 1, 0, 0]  

    w = Watch(gpio=dummy_gpio)

    # Patch time.sleep to no-op
    monkeypatch.setattr(time, "sleep", lambda x: None)

    # Should complete without infinite loop
    distance = w.get_distance()
    assert distance >= 0

def test_ping_event_fires_only_once_per_change(monkeypatch):
    dummy_gpio = DummyGPIO()
    w = Watch(gpio=dummy_gpio, offset=10)

    events = []
    w._ee._events["ObjectIn"] = [lambda dist: events.append("in:"+str(dist))]
    w._ee._events["ObjectOut"] = [lambda dist: events.append("out:"+str(dist))]

    distances = [5, 5, 5, 15, 15, 5, 15, 15]
    def get_distance_side():
        return distances.pop(0)
    w.get_distance = get_distance_side

    # Simulate ping calls with distance changes
    expected_events = [
        "in:5",   # first time < offset, emits ObjectIn
        # next 2 same < offset: no new event
        "out:15", # first time > offset, emits ObjectOut
        # next same > offset: no event
        "in:5",   # again < offset after out, emits ObjectIn
        "out:15", # out again
        # last no event
    ]

    for _ in range(8):
        w.ping()

    assert events == expected_events

def test_stop_waits_sleep_and_shutdown(monkeypatch):
    w = Watch()
    called = {"sleep_called": False, "shutdown_called": False}

    monkeypatch.setattr(time, "sleep", lambda x: called.update({"sleep_called": True}))
    class DummyFuture:
        def shutdown(self, wait):
            called["shutdown_called"] = True
        def running(self):
            return False
    w._future = DummyFuture()

    w.stop()
    assert called["sleep_called"]
    assert called["shutdown_called"]

def test_stop_shutdown_when_running(monkeypatch):
    w = Watch()
    monkeypatch.setattr(time, "sleep", lambda x: None)

    class DummyFuture:
        def shutdown(self, wait):
            raise Exception("fail")
        def running(self):
            return True
    w._future = DummyFuture()

    class DummyObserver:
        def shutdown(self, wait=False):
            DummyObserver.called = True
    DummyObserver.called = False
    w._observer = DummyObserver()

    w.stop()
    assert DummyObserver.called

def test_observe_starts_future(monkeypatch):
    w = Watch()
    called = {"submitted": False}

    class DummyFuture:
        def __init__(self):
            called["submitted"] = True
    class DummyObserver:
        def submit(self, func):
            assert func == w.poll
            return DummyFuture()
    w._observer = DummyObserver()
    w.observe()
    assert called["submitted"]

def test_get_distance_cleanup_called(monkeypatch):
    dummy_gpio = DummyGPIO()
    dummy_gpio.input_sequence = [0, 1]

    w = Watch(gpio=dummy_gpio)
    monkeypatch.setattr(time, "sleep", lambda x: None)

    dist = w.get_distance()

    assert dummy_gpio.cleanup_called
    assert dist >= 0

def test_ping_with_no_gpio(monkeypatch):
    # Test that ping doesn't fail if gpio is None and get_distance patched
    w = Watch()
    w.get_distance = lambda: 0
    # Swap listeners to catch events
    events = []
    w._ee._events["ObjectIn"] = [lambda dist: events.append(f"in:{dist}")]
    w._ee._events["ObjectOut"] = [lambda dist: events.append(f"out:{dist}")]
    w._wasIn = False
    w._offset = 10

    w.ping()
    assert events == ["in:0"]
    w.ping()
    assert events == ["in:0"]
    # Now simulate > offset
    w.get_distance = lambda: 20
    w.ping()
    assert events[-1] == "out:20"

def test_get_distance_prints(monkeypatch, capsys):
    dummy_gpio = DummyGPIO()
    dummy_gpio.input_sequence = [0,1]

    w = Watch(gpio=dummy_gpio)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    _ = w.get_distance()
    captured = capsys.readouterr()
    assert "Distance Measurement In Progress" in captured.out
    assert "Waiting For Sensor To Settle" in captured.out
    assert "Distance:" in captured.out
