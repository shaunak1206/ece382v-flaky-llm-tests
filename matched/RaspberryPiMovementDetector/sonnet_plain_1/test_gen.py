import os
import sys
from unittest.mock import MagicMock

import pytest

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.getcwd(), os.path.dirname(_here), _here):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import MovementDetector.Watch as watch_module
from MovementDetector.Watch import Watch


@pytest.fixture
def handlers():
    return MagicMock(name="func_in"), MagicMock(name="func_out")


@pytest.fixture
def watch(handlers):
    func_in, func_out = handlers
    w = Watch(gpio=MagicMock(), func_in=func_in, func_out=func_out, offset=100)
    yield w
    w._observer.shutdown(wait=False)


def set_distance(w, value):
    w.get_distance = MagicMock(return_value=value)


def test_default_pins_and_offset():
    w = Watch(func_in=MagicMock(), func_out=MagicMock())
    assert w.trigger_pin() == 23
    assert w.echo_pin() == 24
    assert w._offset == 200
    assert w._gpio is None
    assert w._wasIn is False
    assert w._observer_on is True
    assert w._future is None
    w._observer.shutdown(wait=False)


def test_custom_pins_and_offset_and_gpio():
    gpio = MagicMock()
    w = Watch(gpio=gpio, trig=5, echo=6, offset=50)
    assert w.trigger_pin() == 5
    assert w.echo_pin() == 6
    assert w._offset == 50
    assert w._gpio is gpio
    w._observer.shutdown(wait=False)


def test_ping_object_in_emits_event(watch, handlers):
    func_in, func_out = handlers
    set_distance(watch, 50)
    watch.ping()
    func_in.assert_called_once_with(50)
    func_out.assert_not_called()
    assert watch._wasIn is True


def test_ping_in_twice_emits_once(watch, handlers):
    func_in, _ = handlers
    set_distance(watch, 50)
    watch.ping()
    watch.ping()
    assert func_in.call_count == 1
    assert watch._wasIn is True


def test_ping_out_without_prior_in_does_not_emit(watch, handlers):
    func_in, func_out = handlers
    set_distance(watch, 150)
    watch.ping()
    func_in.assert_not_called()
    func_out.assert_not_called()
    assert watch._wasIn is False


def test_ping_in_then_out_emits_out(watch, handlers):
    func_in, func_out = handlers
    set_distance(watch, 10)
    watch.ping()
    set_distance(watch, 300)
    watch.ping()
    func_in.assert_called_once_with(10)
    func_out.assert_called_once_with(300)
    assert watch._wasIn is False


def test_ping_distance_equal_to_offset_is_out(watch, handlers):
    func_in, func_out = handlers
    watch._wasIn = True
    set_distance(watch, 100)
    watch.ping()
    func_in.assert_not_called()
    func_out.assert_called_once_with(100)
    assert watch._wasIn is False


def test_ping_distance_just_below_offset_is_in(watch, handlers):
    func_in, _ = handlers
    set_distance(watch, 99.99)
    watch.ping()
    func_in.assert_called_once_with(99.99)


def test_ping_in_out_in_cycle(watch, handlers):
    func_in, func_out = handlers
    for d in (10, 500, 20):
        set_distance(watch, d)
        watch.ping()
    assert func_in.call_count == 2
    assert func_out.call_count == 1
    assert watch._wasIn is True


def test_ping_returns_none(watch):
    set_distance(watch, 10)
    assert watch.ping() is None


def test_poll_returns_immediately_when_observer_off(watch):
    watch.ping = MagicMock()
    watch._observer_on = False
    assert watch.poll() is None
    watch.ping.assert_not_called()


def test_poll_pings_until_observer_off(watch):
    calls = []

    def fake_ping():
        calls.append(1)
        if len(calls) == 3:
            watch._observer_on = False

    watch.ping = fake_ping
    watch.poll()
    assert len(calls) == 3


def test_observe_submits_poll_to_executor(watch):
    watch._observer = MagicMock()
    watch.observe()
    watch._observer.submit.assert_called_once_with(watch.poll)
    assert watch._future is watch._observer.submit.return_value


def test_stop_turns_observer_off_and_shuts_down_future(watch, monkeypatch):
    fake_time = MagicMock()
    monkeypatch.setattr(watch_module, "time", fake_time)
    watch._future = MagicMock()
    watch._observer = MagicMock()
    watch.stop()
    assert watch._observer_on is False
    fake_time.sleep.assert_called_once_with(3)
    watch._future.shutdown.assert_called_once_with(wait=False)
    watch._observer.shutdown.assert_not_called()


def test_stop_falls_back_to_executor_shutdown_when_running(watch, monkeypatch):
    monkeypatch.setattr(watch_module, "time", MagicMock())
    watch._future = MagicMock()
    watch._future.shutdown.side_effect = AttributeError("no shutdown")
    watch._future.running.return_value = True
    watch._observer = MagicMock()
    watch.stop()
    watch._observer.shutdown.assert_called_once_with(wait=False)


def test_stop_does_not_shutdown_executor_when_future_not_running(watch, monkeypatch):
    monkeypatch.setattr(watch_module, "time", MagicMock())
    watch._future = MagicMock()
    watch._future.shutdown.side_effect = AttributeError("no shutdown")
    watch._future.running.return_value = False
    watch._observer = MagicMock()
    watch.stop()
    watch._observer.shutdown.assert_not_called()


def _install_fake_time(monkeypatch, times):
    it = iter(times)
    fake_time = MagicMock()
    fake_time.time.side_effect = lambda: next(it)
    monkeypatch.setattr(watch_module, "time", fake_time)
    return fake_time


def test_get_distance_computes_distance_from_pulse(watch, monkeypatch):
    # time() calls: init start, init end, start x2 (echo 0), end x1 (echo 1)
    fake_time = _install_fake_time(monkeypatch, [0, 0, 1, 2, 3])
    watch._gpio.input.side_effect = [0, 0, 1, 1, 0]
    distance = watch.get_distance()
    assert distance == 17150.0
    assert fake_time.sleep.call_count == 2


def test_get_distance_rounds_to_two_decimals(watch, monkeypatch):
    _install_fake_time(monkeypatch, [0, 0, 1.0, 1.001234])
    watch._gpio.input.side_effect = [0, 1, 1, 0]
    # start=1.0 (one echo-0 iteration), end=1.001234
    distance = watch.get_distance()
    assert distance == round(0.001234 * 17150, 2)


def test_get_distance_configures_pins_and_cleans_up(watch, monkeypatch):
    _install_fake_time(monkeypatch, [0, 0, 1, 2])
    gpio = watch._gpio
    gpio.input.side_effect = [0, 1, 1, 0]
    watch.get_distance()

    trig, echo = watch.trigger_pin(), watch.echo_pin()
    gpio.setup.assert_any_call(trig, gpio.OUT)
    gpio.setup.assert_any_call(echo, gpio.IN)
    output_calls = [c.args for c in gpio.output.call_args_list]
    assert output_calls == [(trig, False), (trig, True), (trig, False)]
    gpio.cleanup.assert_called_once_with()
