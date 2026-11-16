import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Make the MovementDetector package importable when pytest runs from the project root.
PROJECT_ROOT = os.getcwd()
MOVEMENT_DIR = os.path.join(PROJECT_ROOT, "MovementDetector")
for path in (PROJECT_ROOT, MOVEMENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import Watch as watch_module  # noqa: E402
from Watch import Watch  # noqa: E402


@pytest.fixture
def gpio():
    mock_gpio = MagicMock(name="gpio")
    mock_gpio.OUT = "OUT"
    mock_gpio.IN = "IN"
    return mock_gpio


@pytest.fixture
def func_in():
    return MagicMock(name="func_in")


@pytest.fixture
def func_out():
    return MagicMock(name="func_out")


@pytest.fixture
def watch(gpio, func_in, func_out):
    return Watch(gpio=gpio, func_in=func_in, func_out=func_out)


def test_init_uses_default_values():
    w = Watch()
    assert w._gpio is None
    assert w._trig == 23
    assert w._echo == 24
    assert w._offset == 200
    assert w._wasIn is False
    assert w._observer_on is True
    assert w._future is None


def test_init_uses_custom_values(gpio):
    w = Watch(gpio=gpio, trig=5, echo=6, offset=50)
    assert w._gpio is gpio
    assert w._trig == 5
    assert w._echo == 6
    assert w._offset == 50


def test_trigger_pin_returns_trig_pin(gpio):
    w = Watch(gpio=gpio, trig=17)
    assert w.trigger_pin() == 17


def test_echo_pin_returns_echo_pin(gpio):
    w = Watch(gpio=gpio, echo=18)
    assert w.echo_pin() == 18


def test_ping_emits_object_in_when_distance_below_offset(watch, func_in):
    watch.get_distance = MagicMock(return_value=100)
    watch.ping()
    func_in.assert_called_once_with(100)
    assert watch._wasIn is True


def test_ping_does_not_emit_object_in_twice(watch, func_in):
    watch.get_distance = MagicMock(side_effect=[100, 90, 80])
    watch.ping()
    watch.ping()
    watch.ping()
    func_in.assert_called_once_with(100)


def test_ping_emits_object_out_after_object_was_in(watch, func_in, func_out):
    watch.get_distance = MagicMock(side_effect=[100, 300])
    watch.ping()
    watch.ping()
    func_in.assert_called_once_with(100)
    func_out.assert_called_once_with(300)
    assert watch._wasIn is False


def test_ping_emits_nothing_when_object_never_entered(watch, func_in, func_out):
    watch.get_distance = MagicMock(return_value=500)
    watch.ping()
    watch.ping()
    func_in.assert_not_called()
    func_out.assert_not_called()
    assert watch._wasIn is False


def test_ping_treats_distance_equal_to_offset_as_outside(watch, func_in, func_out):
    watch.get_distance = MagicMock(return_value=200)
    watch.ping()
    func_in.assert_not_called()
    func_out.assert_not_called()


def test_ping_allows_reentry_after_object_leaves(watch, func_in, func_out):
    watch.get_distance = MagicMock(side_effect=[50, 400, 50])
    watch.ping()
    watch.ping()
    watch.ping()
    assert func_in.call_count == 2
    func_out.assert_called_once_with(400)


def test_get_distance_computes_centimetres_and_cleans_up(gpio):
    w = Watch(gpio=gpio, trig=23, echo=24)
    gpio.input.side_effect = [1, 1, 0]
    with patch("Watch.time") as mock_time:
        mock_time.time.side_effect = [10.0, 10.0, 10.001]
        distance = w.get_distance()
    # 0.001 s * 17150 cm/s = 17.15 cm
    assert distance == pytest.approx(17.15)
    gpio.cleanup.assert_called_once()


def test_get_distance_configures_pins_and_triggers_sensor(gpio):
    w = Watch(gpio=gpio, trig=23, echo=24)
    gpio.input.side_effect = [1, 1, 0]
    with patch("Watch.time") as mock_time:
        mock_time.time.side_effect = [0.0, 0.0, 0.001]
        w.get_distance()
    gpio.setup.assert_any_call(23, "OUT")
    gpio.setup.assert_any_call(24, "IN")
    output_calls = [c.args for c in gpio.output.call_args_list]
    assert output_calls == [(23, False), (23, True), (23, False)]


def test_get_distance_waits_for_settle_and_trigger_pulse(gpio):
    w = Watch(gpio=gpio)
    gpio.input.side_effect = [1, 1, 0]
    with patch("Watch.time") as mock_time:
        mock_time.time.side_effect = [0.0, 0.0, 0.001]
        w.get_distance()
    sleep_args = [c.args[0] for c in mock_time.sleep.call_args_list]
    assert sleep_args == [2, 0.00001]


def test_get_distance_rounds_to_two_decimal_places(gpio):
    w = Watch(gpio=gpio)
    gpio.input.side_effect = [1, 1, 0]
    with patch("Watch.time") as mock_time:
        mock_time.time.side_effect = [0.0, 0.0, 0.0012345]
        distance = w.get_distance()
    assert distance == 21.17


def test_poll_returns_immediately_when_observer_off(watch):
    watch._observer_on = False
    watch.ping = MagicMock()
    watch.poll()
    watch.ping.assert_not_called()


def test_poll_pings_until_observer_stopped(watch):
    def stop_after_three_pings():
        if watch.ping.call_count == 3:
            watch._observer_on = False

    watch.ping = MagicMock(side_effect=stop_after_three_pings)
    watch.poll()
    assert watch.ping.call_count == 3


def test_observe_submits_poll_to_executor(watch):
    watch._observer = MagicMock(name="executor")
    sentinel_future = MagicMock(name="future")
    watch._observer.submit.return_value = sentinel_future
    watch.observe()
    watch._observer.submit.assert_called_once_with(watch.poll)
    assert watch._future is sentinel_future


def test_stop_disables_observer_and_shuts_down_future(watch):
    watch._future = MagicMock(name="future")
    with patch("Watch.time.sleep") as mock_sleep:
        watch.stop()
    assert watch._observer_on is False
    mock_sleep.assert_called_once_with(3)
    watch._future.shutdown.assert_called_once_with(wait=False)


def test_stop_shuts_down_executor_when_future_shutdown_fails_and_running(watch):
    watch._future = MagicMock(name="future")
    watch._future.shutdown.side_effect = Exception("boom")
    watch._future.running.return_value = True
    watch._observer = MagicMock(name="executor")
    with patch("Watch.time.sleep"):
        watch.stop()
    watch._observer.shutdown.assert_called_once_with(wait=False)


def test_stop_does_not_shut_down_executor_when_future_not_running(watch):
    watch._future = MagicMock(name="future")
    watch._future.shutdown.side_effect = Exception("boom")
    watch._future.running.return_value = False
    watch._observer = MagicMock(name="executor")
    with patch("Watch.time.sleep"):
        watch.stop()
    watch._observer.shutdown.assert_not_called()


def test_stop_raises_when_observe_was_never_called(watch):
    with patch("Watch.time.sleep"):
        with pytest.raises(AttributeError):
            watch.stop()
