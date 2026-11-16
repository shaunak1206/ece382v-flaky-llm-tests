import importlib

import pytest

watch_module = importlib.import_module("MovementDetector.Watch")
Watch = watch_module.Watch


class FakeClock:
    """Replacement for the time module used in Watch.py."""

    def __init__(self, step=0.01):
        self.now = 0.0
        self.step = step
        self.sleeps = []

    def time(self):
        self.now += self.step
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)


class FakeGPIO:
    OUT = "OUT"
    IN = "IN"

    def __init__(self, echo_values):
        self.echo_values = list(echo_values)
        self.calls = []

    def setup(self, pin, mode):
        self.calls.append(("setup", pin, mode))

    def output(self, pin, value):
        self.calls.append(("output", pin, value))

    def input(self, pin):
        self.calls.append(("input", pin))
        return self.echo_values.pop(0)

    def cleanup(self):
        self.calls.append(("cleanup",))


def make_watch(distances=None, **kwargs):
    """Create a Watch whose get_distance returns queued values."""
    ins, outs = [], []
    kwargs.setdefault("func_in", lambda d: ins.append(d))
    kwargs.setdefault("func_out", lambda d: outs.append(d))
    w = Watch(**kwargs)
    if distances is not None:
        queue = list(distances)
        w.get_distance = lambda: queue.pop(0)
    return w, ins, outs


def test_default_constructor_values():
    w = Watch()
    assert w.trigger_pin() == 23
    assert w.echo_pin() == 24
    assert w._offset == 200
    assert w._gpio is None
    assert w._wasIn is False
    assert w._observer_on is True
    assert w._future is None


def test_constructor_custom_values():
    gpio = object()
    w = Watch(gpio=gpio, trig=5, echo=6, offset=50)
    assert w._gpio is gpio
    assert w.trigger_pin() == 5
    assert w.echo_pin() == 6
    assert w._offset == 50


def test_ping_object_in_emits_with_distance():
    w, ins, outs = make_watch([30.5], offset=100)
    w.ping()
    assert ins == [30.5]
    assert outs == []
    assert w._wasIn is True


def test_ping_far_object_without_previous_in_does_not_emit():
    w, ins, outs = make_watch([500], offset=100)
    w.ping()
    assert ins == []
    assert outs == []
    assert w._wasIn is False


def test_ping_in_only_emitted_once_while_object_stays():
    w, ins, outs = make_watch([10, 20, 30], offset=100)
    w.ping()
    w.ping()
    w.ping()
    assert ins == [10]
    assert outs == []


def test_ping_in_then_out_emits_both():
    w, ins, outs = make_watch([10, 150], offset=100)
    w.ping()
    w.ping()
    assert ins == [10]
    assert outs == [150]
    assert w._wasIn is False


def test_ping_out_only_emitted_once():
    w, ins, outs = make_watch([10, 150, 200, 300], offset=100)
    for _ in range(4):
        w.ping()
    assert ins == [10]
    assert outs == [150]


def test_ping_distance_equal_to_offset_counts_as_out():
    w, ins, outs = make_watch([100, 100], offset=100)
    w.ping()
    assert ins == []
    # Enter, then exactly at offset should leave.
    w.get_distance = lambda: 99.99
    w.ping()
    w.get_distance = lambda: 100
    w.ping()
    assert ins == [99.99]
    assert outs == [100]


def test_ping_can_reenter_after_leaving():
    w, ins, outs = make_watch([10, 300, 20], offset=100)
    w.ping()
    w.ping()
    w.ping()
    assert ins == [10, 20]
    assert outs == [300]


def test_get_distance_computes_distance_and_uses_gpio(monkeypatch):
    clock = FakeClock(step=0.01)
    monkeypatch.setattr(watch_module, "time", clock)
    # Input reads: 0,0 then 1 (exit first loop), 1,1 then 0 (exit second loop)
    gpio = FakeGPIO([0, 0, 1, 1, 1, 0])
    w = Watch(gpio=gpio, trig=7, echo=8)
    distance = w.get_distance()
    # pulse_start is 4th time() call, pulse_end is 6th -> 0.02 s
    assert distance == pytest.approx(0.02 * 17150, abs=0.01)
    assert ("setup", 7, FakeGPIO.OUT) in gpio.calls
    assert ("setup", 8, FakeGPIO.IN) in gpio.calls
    outputs = [c for c in gpio.calls if c[0] == "output"]
    assert outputs == [("output", 7, False), ("output", 7, True), ("output", 7, False)]
    assert gpio.calls[-1] == ("cleanup",)
    assert clock.sleeps == [2, 0.00001]


def test_get_distance_result_is_rounded_to_two_decimals(monkeypatch):
    clock = FakeClock(step=0.0003337)
    monkeypatch.setattr(watch_module, "time", clock)
    gpio = FakeGPIO([0, 1, 1, 0])
    w = Watch(gpio=gpio)
    distance = w.get_distance()
    assert distance == round(distance, 2)


def test_poll_returns_immediately_when_observer_off():
    w, _, _ = make_watch()
    calls = []
    w.ping = lambda: calls.append(1)
    w._observer_on = False
    assert w.poll() is None
    assert calls == []


def test_poll_pings_until_observer_turned_off():
    w, _, _ = make_watch()
    calls = []

    def fake_ping():
        calls.append(1)
        if len(calls) == 3:
            w._observer_on = False

    w.ping = fake_ping
    w.poll()
    assert len(calls) == 3


def test_observe_submits_poll_to_executor():
    w, _, _ = make_watch()
    w.poll = lambda: "polled"
    w.observe()
    assert w._future is not None
    assert w._future.result(timeout=5) == "polled"
    w._observer.shutdown(wait=True)


def test_observe_runs_ping_loop_in_background_and_stops():
    w, ins, outs = make_watch([10, 10, 300], offset=100)
    done = []
    original_get = w.get_distance

    def get_distance():
        try:
            return original_get()
        except IndexError:
            w._observer_on = False
            done.append(True)
            return 300

    w.get_distance = get_distance
    w.observe()
    w._future.result(timeout=5)
    assert ins == [10]
    assert outs == [300]
    assert done
    w._observer.shutdown(wait=True)


def test_stop_turns_off_observer_and_sleeps(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(watch_module, "time", clock)
    w, _, _ = make_watch()

    class FakeFuture:
        def running(self):
            return False

    w._future = FakeFuture()
    assert w.stop() is None
    assert w._observer_on is False
    assert clock.sleeps == [3]


def test_stop_shuts_down_executor_when_future_still_running(monkeypatch):
    monkeypatch.setattr(watch_module, "time", FakeClock())
    w, _, _ = make_watch()
    shutdown_calls = []

    class FakeExecutor:
        def shutdown(self, wait=True):
            shutdown_calls.append(wait)

    class FakeFuture:
        def running(self):
            return True

    w._observer = FakeExecutor()
    w._future = FakeFuture()
    w.stop()
    assert shutdown_calls == [False]


def test_stop_does_not_shutdown_executor_when_future_finished(monkeypatch):
    monkeypatch.setattr(watch_module, "time", FakeClock())
    w, _, _ = make_watch()
    shutdown_calls = []

    class FakeExecutor:
        def shutdown(self, wait=True):
            shutdown_calls.append(wait)

    class FakeFuture:
        def running(self):
            return False

    w._observer = FakeExecutor()
    w._future = FakeFuture()
    w.stop()
    assert shutdown_calls == []
