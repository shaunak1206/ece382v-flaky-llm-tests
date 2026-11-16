import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (os.getcwd(), _here, os.path.dirname(_here)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from MovementDetector.Watch import Watch  # noqa: E402


@pytest.fixture
def gpio():
    g = MagicMock()
    g.OUT = "OUT"
    g.IN = "IN"
    return g


@pytest.fixture
def handlers():
    return MagicMock(name="func_in"), MagicMock(name="func_out")


@pytest.fixture
def watch(gpio, handlers):
    f_in, f_out = handlers
    w = Watch(gpio=gpio, trig=5, echo=6, func_in=f_in, func_out=f_out, offset=100)
    yield w
    w._observer.shutdown(wait=False)


def test_default_pins_and_offset():
    w = Watch()
    try:
        assert w.trigger_pin() == 23
        assert w.echo_pin() == 24
        assert w._offset == 200
        assert w._gpio is None
        assert w._wasIn is False
        assert w._observer_on is True
        assert w._future is None
    finally:
        w._observer.shutdown(wait=False)


def test_custom_pins_and_offset(watch, gpio):
    assert watch.trigger_pin() == 5
    assert watch.echo_pin() == 6
    assert watch._offset == 100
    assert watch._gpio is gpio


def test_ping_object_comes_in_emits_in(watch, handlers):
    f_in, f_out = handlers
    with patch.object(watch, "get_distance", return_value=50):
        watch.ping()
    f_in.assert_called_once_with(50)
    f_out.assert_not_called()
    assert watch._wasIn is True


def test_ping_in_only_emits_once(watch, handlers):
    f_in, f_out = handlers
    with patch.object(watch, "get_distance", side_effect=[50, 60, 70]):
        watch.ping()
        watch.ping()
        watch.ping()
    assert f_in.call_count == 1
    f_out.assert_not_called()


def test_ping_far_object_when_not_in_emits_nothing(watch, handlers):
    f_in, f_out = handlers
    with patch.object(watch, "get_distance", return_value=300):
        watch.ping()
    f_in.assert_not_called()
    f_out.assert_not_called()
    assert watch._wasIn is False


def test_ping_object_goes_out_emits_out(watch, handlers):
    f_in, f_out = handlers
    with patch.object(watch, "get_distance", side_effect=[50, 250]):
        watch.ping()
        watch.ping()
    f_in.assert_called_once_with(50)
    f_out.assert_called_once_with(250)
    assert watch._wasIn is False


def test_ping_distance_equal_to_offset_counts_as_out(watch, handlers):
    f_in, f_out = handlers
    with patch.object(watch, "get_distance", return_value=100):
        watch.ping()
    f_in.assert_not_called()
    f_out.assert_not_called()
    assert watch._wasIn is False


def test_ping_distance_equal_to_offset_triggers_out_when_was_in(watch, handlers):
    f_in, f_out = handlers
    watch._wasIn = True
    with patch.object(watch, "get_distance", return_value=100):
        watch.ping()
    f_out.assert_called_once_with(100)
    assert watch._wasIn is False


def test_ping_in_out_in_cycle(watch, handlers):
    f_in, f_out = handlers
    with patch.object(watch, "get_distance", side_effect=[10, 500, 20]):
        watch.ping()
        watch.ping()
        watch.ping()
    assert f_in.call_count == 2
    assert f_out.call_count == 1


def test_poll_returns_immediately_when_observer_off(watch):
    watch._observer_on = False
    with patch.object(watch, "ping") as ping:
        assert watch.poll() is None
    ping.assert_not_called()


def test_poll_pings_until_observer_turned_off(watch):
    calls = []

    def fake_ping():
        calls.append(1)
        if len(calls) >= 3:
            watch._observer_on = False

    with patch.object(watch, "ping", side_effect=fake_ping):
        watch.poll()
    assert len(calls) == 3


def test_observe_submits_poll_to_executor(watch):
    with patch.object(watch, "poll", return_value=None) as poll:
        watch.observe()
        assert watch._future is not None
        watch._future.result(timeout=5)
    poll.assert_called_once_with()


def test_stop_sets_flag_and_calls_future_shutdown(watch):
    future = MagicMock()
    watch._future = future
    watch._observer = MagicMock()
    with patch("MovementDetector.Watch.time") as t:
        watch.stop()
        t.sleep.assert_called_once_with(3)
    assert watch._observer_on is False
    future.shutdown.assert_called_once_with(wait=False)
    watch._observer.shutdown.assert_not_called()


def test_stop_shuts_down_observer_if_future_running(watch):
    future = MagicMock()
    future.shutdown.side_effect = AttributeError("no shutdown")
    future.running.return_value = True
    watch._future = future
    watch._observer = MagicMock()
    with patch("MovementDetector.Watch.time"):
        watch.stop()
    watch._observer.shutdown.assert_called_once_with(wait=False)


def test_stop_does_not_shutdown_observer_if_future_not_running(watch):
    future = MagicMock()
    future.shutdown.side_effect = AttributeError("no shutdown")
    future.running.return_value = False
    watch._future = future
    watch._observer = MagicMock()
    with patch("MovementDetector.Watch.time"):
        watch.stop()
    watch._observer.shutdown.assert_not_called()


def test_get_distance_computes_distance_and_cleans_up(watch, gpio):
    gpio.input.side_effect = [0, 1, 1, 0]
    with patch("MovementDetector.Watch.time") as t:
        t.time.side_effect = [10.0, 10.0, 11.0, 11.5]
        distance = watch.get_distance()
    assert distance == pytest.approx(0.5 * 17150)
    gpio.cleanup.assert_called_once_with()


def test_get_distance_configures_pins_and_pulses_trigger(watch, gpio):
    gpio.input.side_effect = [0, 1, 1, 0]
    with patch("MovementDetector.Watch.time") as t:
        t.time.side_effect = [1.0, 1.0, 2.0, 2.25]
        watch.get_distance()
    gpio.setup.assert_any_call(5, "OUT")
    gpio.setup.assert_any_call(6, "IN")
    trig_outputs = [c.args for c in gpio.output.call_args_list]
    assert trig_outputs == [(5, False), (5, True), (5, False)]


def test_get_distance_is_rounded_to_two_decimals(watch, gpio):
    gpio.input.side_effect = [0, 1, 1, 0]
    with patch("MovementDetector.Watch.time") as t:
        t.time.side_effect = [0.0, 0.0, 1.0, 1.0001234]
        distance = watch.get_distance()
    assert distance == round(distance, 2)
    assert distance == pytest.approx(0.0001234 * 17150, abs=0.01)


def test_ping_uses_get_distance_from_gpio_reading(watch, gpio, handlers):
    f_in, _ = handlers
    gpio.input.side_effect = [0, 1, 1, 0]
    with patch("MovementDetector.Watch.time") as t:
        t.time.side_effect = [1.0, 1.0, 2.0, 2.0009765625]  # 2**-10 s -> ~16.75 cm
        watch.ping()
    assert f_in.call_count == 1
    assert f_in.call_args.args[0] < 100
    assert watch._wasIn is True
