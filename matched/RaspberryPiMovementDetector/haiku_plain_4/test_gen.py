import os
import sys
from unittest import mock

import pytest

try:
    from MovementDetector.Watch import Watch
except ImportError:
    sys.path.insert(0, os.path.join(os.getcwd(), "MovementDetector"))
    sys.path.insert(0, os.getcwd())
    from Watch import Watch


def make_gpio():
    gpio = mock.MagicMock()
    gpio.OUT = 0
    gpio.IN = 1
    gpio.input.return_value = 0
    return gpio


def make_watch(**kwargs):
    kwargs.setdefault("gpio", make_gpio())
    return Watch(**kwargs)


# ---------- constructor and accessors ----------

def test_defaults_are_applied_when_no_kwargs_given():
    w = Watch()
    assert w.trigger_pin() == 23
    assert w.echo_pin() == 24
    assert w._offset == 200
    assert w._gpio is None
    assert w._wasIn is False
    assert w._observer_on is True
    assert w._future is None


def test_custom_kwargs_override_defaults():
    gpio = make_gpio()
    w = Watch(gpio=gpio, trig=5, echo=6, offset=50)
    assert w._gpio is gpio
    assert w.trigger_pin() == 5
    assert w.echo_pin() == 6
    assert w._offset == 50


def test_default_handlers_print_when_events_emitted(capsys):
    w = make_watch()
    w._ee.emit("ObjectIn", 10)
    w._ee.emit("ObjectOut", 300)
    out = capsys.readouterr().out
    assert "Dummy in function" in out
    assert "Dummy out function" in out


# ---------- ping behaviour ----------

def test_ping_emits_object_in_on_first_detection():
    func_in = mock.MagicMock()
    w = make_watch(func_in=func_in)
    w.get_distance = mock.MagicMock(return_value=50.0)
    w.ping()
    func_in.assert_called_once_with(50.0)
    assert w._wasIn is True


def test_ping_does_not_repeat_object_in_while_still_inside():
    func_in = mock.MagicMock()
    w = make_watch(func_in=func_in)
    w.get_distance = mock.MagicMock(return_value=50.0)
    w.ping()
    w.ping()
    w.ping()
    assert func_in.call_count == 1


def test_ping_emits_object_out_when_leaving_zone():
    func_out = mock.MagicMock()
    w = make_watch(func_out=func_out)
    w._wasIn = True
    w.get_distance = mock.MagicMock(return_value=250.0)
    w.ping()
    func_out.assert_called_once_with(250.0)
    assert w._wasIn is False


def test_ping_emits_nothing_when_outside_and_never_entered():
    func_in = mock.MagicMock()
    func_out = mock.MagicMock()
    w = make_watch(func_in=func_in, func_out=func_out)
    w.get_distance = mock.MagicMock(return_value=500.0)
    w.ping()
    func_in.assert_not_called()
    func_out.assert_not_called()
    assert w._wasIn is False


def test_ping_distance_equal_to_offset_counts_as_outside():
    func_in = mock.MagicMock()
    w = make_watch(func_in=func_in, offset=200)
    w.get_distance = mock.MagicMock(return_value=200)
    w.ping()
    func_in.assert_not_called()
    assert w._wasIn is False


def test_ping_full_in_out_in_cycle_emits_expected_events():
    func_in = mock.MagicMock()
    func_out = mock.MagicMock()
    w = make_watch(func_in=func_in, func_out=func_out, offset=100)
    distances = [50, 60, 150, 40]
    w.get_distance = mock.MagicMock(side_effect=distances)
    for _ in distances:
        w.ping()
    assert func_in.call_args_list == [mock.call(50), mock.call(40)]
    assert func_out.call_args_list == [mock.call(150)]


# ---------- get_distance behaviour ----------

def test_get_distance_computes_centimetres_from_echo_pulse():
    gpio = make_gpio()
    # echo: low (0) -> high (1) -> low (0)
    gpio.input.side_effect = [0, 1, 1, 0]
    w = Watch(gpio=gpio)
    with mock.patch("time.sleep"), \
            mock.patch("time.time", side_effect=[0.0, 0.0, 5.0, 5.001]):
        distance = w.get_distance()
    assert distance == pytest.approx(17.15)


def test_get_distance_rounds_to_two_decimals():
    gpio = make_gpio()
    gpio.input.side_effect = [0, 1, 1, 0]
    w = Watch(gpio=gpio)
    # duration 0.00012345 s * 17150 = 2.1171675 -> 2.12
    with mock.patch("time.sleep"), \
            mock.patch("time.time", side_effect=[0.0, 0.0, 0.0, 0.00012345]):
        distance = w.get_distance()
    assert distance == 2.12


def test_get_distance_configures_pins_and_triggers_sensor():
    gpio = make_gpio()
    gpio.input.side_effect = [0, 1, 1, 0]
    w = Watch(gpio=gpio, trig=17, echo=27)
    with mock.patch("time.sleep") as sleep_mock, \
            mock.patch("time.time", side_effect=[0.0, 0.0, 5.0, 5.001]):
        w.get_distance()
    gpio.setup.assert_any_call(17, gpio.OUT)
    gpio.setup.assert_any_call(27, gpio.IN)
    gpio.output.assert_any_call(17, False)
    gpio.output.assert_any_call(17, True)
    sleep_mock.assert_any_call(2)


def test_get_distance_cleans_up_gpio_after_measurement():
    gpio = make_gpio()
    gpio.input.side_effect = [0, 1, 1, 0]
    w = Watch(gpio=gpio)
    with mock.patch("time.sleep"), \
            mock.patch("time.time", side_effect=[0.0, 0.0, 5.0, 5.001]):
        w.get_distance()
    gpio.cleanup.assert_called_once_with()


# ---------- observer lifecycle ----------

def test_observe_submits_poll_to_executor_and_stores_future():
    w = make_watch()
    w._observer = mock.MagicMock()
    result = w.observe()
    w._observer.submit.assert_called_once_with(w.poll)
    assert w._future is w._observer.submit.return_value
    assert result is None


def test_poll_returns_immediately_when_observer_disabled():
    w = make_watch()
    w._observer_on = False
    w.ping = mock.MagicMock()
    w.poll()
    w.ping.assert_not_called()


def test_poll_pings_repeatedly_until_disabled():
    w = make_watch()
    state = {"calls": 0}

    def fake_ping():
        state["calls"] += 1
        if state["calls"] == 3:
            w._observer_on = False

    w.ping = mock.MagicMock(side_effect=fake_ping)
    w.poll()
    assert w.ping.call_count == 3


def test_stop_disables_observer_and_shuts_down_future():
    w = make_watch()
    w._future = mock.MagicMock()
    with mock.patch("time.sleep"):
        w.stop()
    assert w._observer_on is False
    w._future.shutdown.assert_called_once_with(wait=False)


def test_stop_waits_three_seconds_before_shutdown():
    w = make_watch()
    w._future = mock.MagicMock()
    with mock.patch("time.sleep") as sleep_mock:
        w.stop()
    sleep_mock.assert_called_once_with(3)
