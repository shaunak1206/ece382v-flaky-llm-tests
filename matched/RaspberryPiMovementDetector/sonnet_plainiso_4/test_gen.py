import types
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

import MovementDetector.Watch as watch_module
from MovementDetector.Watch import Watch


def make_watch(**kwargs):
    kwargs.setdefault("func_in", MagicMock())
    kwargs.setdefault("func_out", MagicMock())
    return Watch(**kwargs)


def test_default_pins_and_offset():
    w = Watch()
    assert w.trigger_pin() == 23
    assert w.echo_pin() == 24
    assert w._offset == 200
    assert w._gpio is None


def test_custom_kwargs_are_stored():
    gpio = object()
    w = Watch(gpio=gpio, trig=5, echo=6, offset=50)
    assert w._gpio is gpio
    assert w.trigger_pin() == 5
    assert w.echo_pin() == 6
    assert w._offset == 50


def test_initial_state():
    w = Watch()
    assert w._wasIn is False
    assert w._observer_on is True
    assert w._future is None


def test_ping_emits_object_in_when_below_offset():
    f_in, f_out = MagicMock(), MagicMock()
    w = Watch(func_in=f_in, func_out=f_out, offset=100)
    w.get_distance = MagicMock(return_value=50)
    w.ping()
    f_in.assert_called_once_with(50)
    f_out.assert_not_called()
    assert w._wasIn is True


def test_ping_does_not_emit_in_twice():
    f_in, f_out = MagicMock(), MagicMock()
    w = Watch(func_in=f_in, func_out=f_out, offset=100)
    w.get_distance = MagicMock(return_value=50)
    w.ping()
    w.ping()
    assert f_in.call_count == 1
    f_out.assert_not_called()


def test_ping_no_emit_when_far_and_was_out():
    f_in, f_out = MagicMock(), MagicMock()
    w = Watch(func_in=f_in, func_out=f_out, offset=100)
    w.get_distance = MagicMock(return_value=300)
    w.ping()
    f_in.assert_not_called()
    f_out.assert_not_called()
    assert w._wasIn is False


def test_ping_emits_object_out_after_in():
    f_in, f_out = MagicMock(), MagicMock()
    w = Watch(func_in=f_in, func_out=f_out, offset=100)
    w.get_distance = MagicMock(side_effect=[20, 150])
    w.ping()
    w.ping()
    f_in.assert_called_once_with(20)
    f_out.assert_called_once_with(150)
    assert w._wasIn is False


def test_ping_distance_equal_to_offset_counts_as_out():
    f_in, f_out = MagicMock(), MagicMock()
    w = Watch(func_in=f_in, func_out=f_out, offset=100)
    w.get_distance = MagicMock(side_effect=[100, 100])
    w.ping()
    f_in.assert_not_called()
    assert w._wasIn is False
    # Enter then measure exactly at offset: should leave
    w._wasIn = True
    w.ping()
    f_out.assert_called_once_with(100)


def test_ping_out_only_emitted_once():
    f_in, f_out = MagicMock(), MagicMock()
    w = Watch(func_in=f_in, func_out=f_out, offset=100)
    w.get_distance = MagicMock(side_effect=[10, 500, 600])
    w.ping()
    w.ping()
    w.ping()
    assert f_in.call_count == 1
    assert f_out.call_count == 1


def test_ping_returns_none():
    w = make_watch(offset=100)
    w.get_distance = MagicMock(return_value=10)
    assert w.ping() is None


def test_poll_returns_immediately_when_observer_off():
    w = make_watch()
    w.ping = MagicMock()
    w._observer_on = False
    assert w.poll() is None
    w.ping.assert_not_called()


def test_poll_pings_until_observer_turned_off():
    w = make_watch()
    calls = []

    def fake_ping():
        calls.append(1)
        if len(calls) == 3:
            w._observer_on = False

    w.ping = fake_ping
    w.poll()
    assert len(calls) == 3


def test_observe_submits_poll_to_executor():
    w = make_watch()
    w.poll = MagicMock(return_value=None)
    w.observe()
    assert isinstance(w._future, Future)
    w._future.result(timeout=5)
    w.poll.assert_called_once_with()
    w._observer.shutdown(wait=True)


def test_stop_sets_flag_and_sleeps(monkeypatch):
    sleeps = []
    fake_time = types.SimpleNamespace(sleep=sleeps.append, time=lambda: 0.0)
    monkeypatch.setattr(watch_module, "time", fake_time)
    w = make_watch()
    w._future = MagicMock()
    w.stop()
    assert w._observer_on is False
    assert sleeps == [3]
    w._future.shutdown.assert_called_once_with(wait=False)


def test_stop_shuts_down_observer_when_future_running(monkeypatch):
    fake_time = types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0.0)
    monkeypatch.setattr(watch_module, "time", fake_time)
    w = make_watch()
    future = MagicMock()
    future.shutdown.side_effect = AttributeError("no shutdown")
    future.running.return_value = True
    w._future = future
    w._observer = MagicMock()
    w.stop()
    w._observer.shutdown.assert_called_once_with(wait=False)


def test_stop_with_real_future_not_running_does_not_shutdown(monkeypatch):
    fake_time = types.SimpleNamespace(sleep=lambda s: None, time=lambda: 0.0)
    monkeypatch.setattr(watch_module, "time", fake_time)
    w = make_watch()
    w._future = Future()  # pending, not running; has no shutdown attribute
    w._observer = MagicMock()
    w.stop()
    assert w._observer_on is False
    w._observer.shutdown.assert_not_called()


def _install_fake_time(monkeypatch, times):
    times_iter = iter(times)
    sleeps = []
    fake_time = types.SimpleNamespace(
        sleep=sleeps.append, time=lambda: next(times_iter)
    )
    monkeypatch.setattr(watch_module, "time", fake_time)
    return sleeps


def test_get_distance_computes_distance(monkeypatch):
    sleeps = _install_fake_time(monkeypatch, [0.0, 0.0, 1.0, 2.0, 3.0])
    gpio = MagicMock()
    gpio.input.side_effect = [0, 0, 1, 1, 0]
    w = Watch(gpio=gpio, trig=7, echo=8)
    distance = w.get_distance()
    # pulse_start=2.0, pulse_end=3.0 -> 1.0 * 17150
    assert distance == 17150.0
    assert sleeps == [2, 0.00001]


def test_get_distance_configures_pins_and_cleans_up(monkeypatch):
    _install_fake_time(monkeypatch, [0.0, 0.0, 1.0, 1.5])
    gpio = MagicMock()
    gpio.input.side_effect = [0, 1, 0]
    w = Watch(gpio=gpio, trig=7, echo=8)
    w.get_distance()
    gpio.setup.assert_any_call(7, gpio.OUT)
    gpio.setup.assert_any_call(8, gpio.IN)
    output_calls = [c.args for c in gpio.output.call_args_list]
    assert output_calls == [(7, False), (7, True), (7, False)]
    gpio.cleanup.assert_called_once_with()


def test_get_distance_rounds_to_two_decimals(monkeypatch):
    # input: 0 -> loop1 sets start, 1 -> exit; 1 -> loop2 sets end, 0 -> exit
    _install_fake_time(monkeypatch, [0.0, 0.0, 1.0, 1.001])
    gpio = MagicMock()
    gpio.input.side_effect = [0, 1, 1, 0]
    w = Watch(gpio=gpio)
    # start=1.0? loop1 runs once (start=1.0), loop2 runs once (end=1.001)
    distance = w.get_distance()
    assert distance == round(0.001 * 17150, 2)
