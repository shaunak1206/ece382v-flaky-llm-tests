import time
import pytest
from unittest.mock import MagicMock, patch, call
from MovementDetector.Watch import Watch

@pytest.fixture
def gpio_mock():
    gpio = MagicMock()
    gpio.OUT = 1
    gpio.IN = 0
    return gpio

def test_init_defaults(gpio_mock):
    w = Watch(gpio=gpio_mock)
    assert w._gpio is gpio_mock
    # default trig and echo pins
    assert w._trig == 23
    assert w._echo == 24
    assert w._offset == 200
    # Check the initial wasIn flag
    assert w._wasIn is False
    # Check that _observer is ThreadPoolExecutor
    assert w._observer_on is True
    assert w._future is None

def test_init_custom_args(gpio_mock):
    func_in = MagicMock()
    func_out = MagicMock()
    w = Watch(gpio=gpio_mock, trig=5, echo=6, func_in=func_in, func_out=func_out, offset=100)
    assert w._trig == 5
    assert w._echo == 6
    assert w._offset == 100
    # EventEmitter should have these listeners attached, but no direct attribute to check,
    # so we test behavior later with emitting.

def test_trigger_pin_and_echo_pin(gpio_mock):
    w = Watch(gpio=gpio_mock, trig=12, echo=18)
    assert w.trigger_pin() == 12
    assert w.echo_pin() == 18

def test_get_distance_returns_correct_value_and_calls_gpio_methods(gpio_mock):
    # Arrange: simulate gpio.input calls to produce a pulse duration
    # We'll patch time.time to control pulse timing precisely
    w = Watch(gpio=gpio_mock)
    # Define times for pulse start and end
    times = iter([
        1.0, # pulse_start time 1st call
        1.0, # pulse_end time 1st call
        1.1, # pulse_start update inside first while
        1.1, # pulse_end update before second while
        1.15, # pulse_end during second while
    ])

    # Side effect for time.time(), consume next from times
    def fake_time():
        return next(times)
    
    # We need to simulate gpio.input that returns 0 then 1 to exit loops
    # first while: while gpio.input(echo) == 0: -> returns 0 once, then 1
    # second while: while gpio.input(echo) == 1: -> returns 1 twice, then 0

    inputs = iter([0,1,1,0])
    gpio_mock.input.side_effect = lambda pin: next(inputs)

    with patch("time.time", side_effect=fake_time):
        with patch("time.sleep", return_value=None) as sleep_mock:
            distance = w.get_distance()

    # pulse_duration = pulse_end - pulse_start = 1.15 - 1.1 = 0.05
    # expected distance = 0.05 * 17150 = 857.5 rounded to 857.5
    assert isinstance(distance, float)
    assert distance == round(0.05 * 17150, 2)

    # Confirm that correct gpio methods were called (setup, output, cleanup)
    calls = [
        call.setup(w._trig, gpio_mock.OUT),
        call.setup(w._echo, gpio_mock.IN),
        call.output(w._trig, False),
        call.output(w._trig, True),
        call.output(w._trig, False),
        call.input(w._echo),
        call.input(w._echo),
        call.cleanup()
    ]
    gpio_mock.assert_has_calls(calls, any_order=False)
    # Confirm sleep called with expected delays: 2 and 0.00001 seconds
    sleep_mock.assert_any_call(2)
    sleep_mock.assert_any_call(0.00001)

def test_ping_emits_object_in_event(gpio_mock):
    func_in = MagicMock()
    func_out = MagicMock()
    w = Watch(gpio=gpio_mock, func_in=func_in, func_out=func_out, offset=100)

    # Patch get_distance to return less than offset to trigger ObjectIn
    w.get_distance = MagicMock(return_value=50)
    w._wasIn = False

    w.ping()
    func_in.assert_called_once_with(50)
    func_out.assert_not_called()
    assert w._wasIn is True

def test_ping_emits_object_out_event(gpio_mock):
    func_in = MagicMock()
    func_out = MagicMock()
    w = Watch(gpio=gpio_mock, func_in=func_in, func_out=func_out, offset=100)

    # Patch get_distance to return greater than offset to trigger ObjectOut
    w.get_distance = MagicMock(return_value=150)
    w._wasIn = True

    w.ping()
    func_out.assert_called_once_with(150)
    func_in.assert_not_called()
    assert w._wasIn is False

def test_ping_no_event_when_in_zone_and_wasIn_true(gpio_mock):
    func_in = MagicMock()
    func_out = MagicMock()
    w = Watch(gpio=gpio_mock, func_in=func_in, func_out=func_out, offset=100)
    w.get_distance = MagicMock(return_value=50)
    w._wasIn = True
    w.ping()
    func_in.assert_not_called()
    func_out.assert_not_called()
    assert w._wasIn is True

def test_ping_no_event_when_out_zone_and_wasIn_false(gpio_mock):
    func_in = MagicMock()
    func_out = MagicMock()
    w = Watch(gpio=gpio_mock, func_in=func_in, func_out=func_out, offset=100)
    w.get_distance = MagicMock(return_value=150)
    w._wasIn = False
    w.ping()
    func_in.assert_not_called()
    func_out.assert_not_called()
    assert w._wasIn is False

def test_poll_runs_and_calls_ping(mocker):
    w = Watch()
    w._observer_on = True
    ping_mock = mocker.patch.object(w, "ping")
    
    # Use side effect to break infinite loop after 3 calls
    def side_effect():
        side_effect.counter += 1
        if side_effect.counter >= 3:
            w._observer_on = False
    side_effect.counter = 0

    ping_mock.side_effect = side_effect

    w.poll()
    assert ping_mock.call_count == 3

def test_poll_returns_immediately_when_observer_off(mocker):
    w = Watch()
    w._observer_on = False
    ping_mock = mocker.patch.object(w, "ping")

    w.poll()
    ping_mock.assert_not_called()

def test_observe_submits_poll(mocker):
    w = Watch()
    submit_mock = mocker.patch.object(w._observer, "submit", wraps=w._observer.submit)
    w.observe()
    submit_mock.assert_called_once_with(w.poll)
    assert w._future is not None

def test_stop_shuts_down_future_and_observer(mocker):
    w = Watch()
    w._future = MagicMock()
    w._observer = MagicMock()

    # Simulate future.shutdown runs without error
    w._future.shutdown = MagicMock(side_effect=None)
    w._future.running = MagicMock(return_value=False)
    w._observer.shutdown = MagicMock()

    with patch("time.sleep", return_value=None):
        w.stop()

    w._future.shutdown.assert_called_once_with(wait=False)
    w._observer.shutdown.assert_not_called()

def test_stop_fallback_shutdown_if_shutdown_raises(mocker):
    w = Watch()
    w._future = MagicMock()
    w._observer = MagicMock()

    # Simulate future.shutdown raises Exception, running() returns True, so observer.shutdown called
    def raise_exc(*a, **kw):
        raise Exception("shutdown error")
    w._future.shutdown = MagicMock(side_effect=raise_exc)
    w._future.running = MagicMock(return_value=True)
    w._observer.shutdown = MagicMock()

    with patch("time.sleep", return_value=None):
        w.stop()

    w._future.shutdown.assert_called_once_with(wait=False)
    w._observer.shutdown.assert_called_once_with(wait=False)

def test_dummy_func_in_and_out_prints(capsys):
    w = Watch()
    # Access internal EventEmitter listeners by emitting events
    w._ee.emit("ObjectIn", 123)
    out = capsys.readouterr().out
    assert "Dummy in function" in out

    w._ee.emit("ObjectOut", 456)
    out = capsys.readouterr().out
    assert "Dummy out function" in out

def test_get_distance_prints_expected_output(gpio_mock, capsys):
    w = Watch(gpio=gpio_mock)

    # Setup for input side effect: one 0 then one 1 to exit loops quickly
    gpio_mock.input.side_effect = [0,1,0]

    # Patch time.time to simulate pulse_duration
    times = iter([1.0,1.0,1.1,1.2])
    def fake_time():
        return next(times)

    with patch("time.time", side_effect=fake_time):
        with patch("time.sleep", return_value=None):
            distance = w.get_distance()

    output = capsys.readouterr().out
    # Should print measurement progress and distance messages
    assert "Distance Measurement In Progress" in output
    assert "Waiting For Sensor To Settle" in output
    assert "Distance:" in output

def test_multiple_ping_events_change_wasIn_flag_correctly(gpio_mock):
    func_in = MagicMock()
    func_out = MagicMock()
    w = Watch(gpio=gpio_mock, func_in=func_in, func_out=func_out, offset=100)

    # Start with wasIn = False and distance < offset: should trigger ObjectIn
    w._wasIn = False
    w.get_distance = MagicMock(return_value=50)
    w.ping()
    func_in.assert_called_once()
    assert w._wasIn is True
    func_in.reset_mock()

    # Call ping again with distance < offset - no event expected
    w.get_distance = MagicMock(return_value=50)
    w.ping()
    func_in.assert_not_called()
    func_out.assert_not_called()
    assert w._wasIn is True

    # Change distance to > offset: should trigger ObjectOut
    w.get_distance = MagicMock(return_value=150)
    w.ping()
    func_out.assert_called_once()
    assert w._wasIn is False

    # Call ping again with distance > offset - no event
    func_out.reset_mock()
    w.get_distance = MagicMock(return_value=150)
    w.ping()
    func_in.assert_not_called()
    func_out.assert_not_called()
    assert w._wasIn is False

def test_observe_and_stop_integration(mocker):
    w = Watch()
    mocked_poll = mocker.patch.object(w, "poll", autospec=True)
    w.observe()
    assert w._future is not None
    assert mocked_poll.call_count == 0  # poll hasn't run yet in this test

    w._future.running = MagicMock(return_value=False)
    w._future.shutdown = MagicMock()
    w._observer.shutdown = MagicMock()
    with patch("time.sleep", return_value=None):
        w.stop()

    w._future.shutdown.assert_called_once_with(wait=False)
    w._observer.shutdown.assert_not_called()

def test_stop_handles_running_future(mocker):
    w = Watch()
    w._future = MagicMock()
    w._observer = MagicMock()
    w._future.shutdown = MagicMock(side_effect=Exception("fail"))
    w._future.running = MagicMock(return_value=True)
    w._observer.shutdown = MagicMock()

    with patch("time.sleep", return_value=None):
        w.stop()

    w._future.shutdown.assert_called_once_with(wait=False)
    w._observer.shutdown.assert_called_once_with(wait=False)
