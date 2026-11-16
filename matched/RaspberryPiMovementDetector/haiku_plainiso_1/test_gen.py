import os
import sys
from unittest import mock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (
    os.path.join(_HERE, "..", "MovementDetector"),
    os.path.join(os.getcwd(), "MovementDetector"),
):
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

import Watch as watch_module  # noqa: E402
from Watch import Watch  # noqa: E402


@pytest.fixture
def gpio():
    gpio_mock = mock.Mock(name="gpio")
    gpio_mock.OUT = "OUT"
    gpio_mock.IN = "IN"
    return gpio_mock


@pytest.fixture
def watch(gpio):
    return Watch(gpio=gpio)


def test_default_pins_and_offset():
    w = Watch()
    assert w.trigger_pin() == 23
    assert w.echo_pin() == 24
    assert w._offset == 200


def test_custom_pins_are_used():
    w = Watch(trig=5, echo=6)
    assert w.trigger_pin() == 5
    assert w.echo_pin() == 6


def test_initial_state(watch):
    assert watch._wasIn is False
    assert watch._observer_on is True
    assert watch._future is None


def test_custom_offset_is_stored():
    w = Watch(offset=350)
    assert w._offset == 350


def test_ping_emits_object_in_when_entering_zone(watch):
    handler_in = mock.Mock()
    watch._ee.on("ObjectIn", handler_in)
    watch.get_distance = mock.Mock(return_value=50)

    watch.ping()

    handler_in.assert_called_once_with(50)
    assert watch._wasIn is True


def test_ping_does_not_repeat_object_in_while_still_inside(watch):
    handler_in = mock.Mock()
    watch._ee.on("ObjectIn", handler_in)
    watch.get_distance = mock.Mock(side_effect=[50, 60, 70])

    watch.ping()
    watch.ping()
    watch.ping()

    assert handler_in.call_count == 1


def test_ping_emits_object_out_when_leaving_zone(watch):
    handler_out = mock.Mock()
    watch._ee.on("ObjectOut", handler_out)
    watch.get_distance = mock.Mock(side_effect=[50, 300])

    watch.ping()
    watch.ping()

    handler_out.assert_called_once_with(300)
    assert watch._wasIn is False


def test_ping_no_events_when_outside_and_never_entered(watch):
    handler_in = mock.Mock()
    handler_out = mock.Mock()
    watch._ee.on("ObjectIn", handler_in)
    watch._ee.on("ObjectOut", handler_out)
    watch.get_distance = mock.Mock(return_value=500)

    watch.ping()

    handler_in.assert_not_called()
    handler_out.assert_not_called()
    assert watch._wasIn is False


def test_ping_distance_equal_to_offset_is_out_of_zone(watch):
    handler_in = mock.Mock()
    watch._ee.on("ObjectIn", handler_in)
    watch.get_distance = mock.Mock(return_value=200)

    watch.ping()

    handler_in.assert_not_called()
    assert watch._wasIn is False


def test_ping_returns_none(watch):
    watch.get_distance = mock.Mock(return_value=10)
    assert watch.ping() is None


def test_custom_func_in_and_func_out_are_registered(gpio):
    func_in = mock.Mock()
    func_out = mock.Mock()
    w = Watch(gpio=gpio, func_in=func_in, func_out=func_out)
    w.get_distance = mock.Mock(side_effect=[10, 900])

    w.ping()
    w.ping()

    func_in.assert_called_once_with(10)
    func_out.assert_called_once_with(900)


def test_default_handlers_print_when_events_fire(gpio, capsys):
    w = Watch(gpio=gpio)
    w.get_distance = mock.Mock(side_effect=[10, 900])

    w.ping()
    w.ping()

    captured = capsys.readouterr().out
    assert "Dummy in function" in captured
    assert "Dummy out function" in captured


def test_poll_loops_until_observer_turned_off(watch):
    call_count = {"n": 0}

    def fake_ping():
        call_count["n"] += 1
        if call_count["n"] == 3:
            watch._observer_on = False

    watch.ping = mock.Mock(side_effect=fake_ping)

    watch.poll()

    assert watch.ping.call_count == 3


def test_poll_returns_immediately_when_observer_off(watch):
    watch._observer_on = False
    watch.ping = mock.Mock()

    result = watch.poll()

    assert result is None
    watch.ping.assert_not_called()


def test_observe_submits_poll_to_executor(watch):
    fake_executor = mock.Mock()
    fake_future = mock.Mock()
    fake_executor.submit.return_value = fake_future
    watch._observer = fake_executor

    result = watch.observe()

    fake_executor.submit.assert_called_once_with(watch.poll)
    assert watch._future is fake_future
    assert result is None


def test_stop_disables_observer_and_shuts_down_future(watch):
    future = mock.Mock()
    watch._future = future

    with mock.patch.object(watch_module.time, "sleep") as sleep_mock:
        watch.stop()

    assert watch._observer_on is False
    sleep_mock.assert_called_once_with(3)
    future.shutdown.assert_called_once_with(wait=False)


def test_stop_shuts_down_executor_when_future_shutdown_fails_and_running(watch):
    future = mock.Mock()
    future.shutdown.side_effect = RuntimeError("cannot shutdown")
    future.running.return_value = True
    executor = mock.Mock()
    watch._future = future
    watch._observer = executor

    with mock.patch.object(watch_module.time, "sleep"):
        watch.stop()

    executor.shutdown.assert_called_once_with(wait=False)


def test_stop_does_not_shut_down_executor_when_future_not_running(watch):
    future = mock.Mock()
    future.shutdown.side_effect = RuntimeError("cannot shutdown")
    future.running.return_value = False
    executor = mock.Mock()
    watch._future = future
    watch._observer = executor

    with mock.patch.object(watch_module.time, "sleep"):
        watch.stop()

    executor.shutdown.assert_not_called()


def test_get_distance_computes_centimeters_from_echo_pulse(watch, gpio):
    # Echo reads: low (start), high (end of wait), high (pulse), low (pulse finished)
    gpio.input.side_effect = [0, 1, 1, 0]
    # time.time calls: initial start, initial end, loop-1 body, loop-2 body
    times = [10.0, 10.0, 10.0, 10.001]

    with mock.patch.object(watch_module.time, "sleep"), \
            mock.patch.object(watch_module.time, "time", side_effect=times):
        distance = watch.get_distance()

    assert distance == pytest.approx(17.15, abs=0.01)


def test_get_distance_configures_pins_and_triggers_sensor(watch, gpio):
    gpio.input.side_effect = [0, 1, 1, 0]
    times = [0.0, 0.0, 0.0, 0.001]

    with mock.patch.object(watch_module.time, "sleep"), \
            mock.patch.object(watch_module.time, "time", side_effect=times):
        watch.get_distance()

    gpio.setup.assert_any_call(23, gpio.OUT)
    gpio.setup.assert_any_call(24, gpio.IN)
    gpio.output.assert_any_call(23, False)
    gpio.output.assert_any_call(23, True)
    assert gpio.output.call_args_list[-1] == mock.call(23, False)


def test_get_distance_cleans_up_gpio(watch, gpio):
    gpio.input.side_effect = [0, 1, 1, 0]
    times = [0.0, 0.0, 0.0, 0.001]

    with mock.patch.object(watch_module.time, "sleep"), \
            mock.patch.object(watch_module.time, "time", side_effect=times):
        watch.get_distance()

    gpio.cleanup.assert_called_once()


def test_get_distance_rounds_to_two_decimals(watch, gpio):
    gpio.input.side_effect = [0, 1, 1, 0]
    # duration 0.0000123456 -> 0.0000123456 * 17150 = 0.21172... -> 0.21
    times = [1.0, 1.0, 1.0, 1.0 + 0.0000123456]

    with mock.patch.object(watch_module.time, "sleep"), \
            mock.patch.object(watch_module.time, "time", side_effect=times):
        distance = watch.get_distance()

    assert distance == 0.21
