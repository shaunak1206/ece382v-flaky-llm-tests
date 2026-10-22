from datetime import timedelta
from unittest import mock

import pybreaker
import pytest
from freezegun import freeze_time


class Boom(Exception):
    pass


def boom():
    raise Boom("boom")


def ok():
    return "ok"


def trip(cb):
    for _ in range(cb.fail_max):
        try:
            cb.call(boom)
        except (Boom, pybreaker.CircuitBreakerError):
            pass


def test_failures_below_threshold_keep_circuit_closed():
    cb = pybreaker.CircuitBreaker(fail_max=3)
    for _ in range(2):
        with pytest.raises(Boom):
            cb.call(boom)
    assert cb.current_state == "closed"
    assert cb.fail_counter == 2


def test_threshold_failure_opens_circuit_and_raises_circuit_error():
    cb = pybreaker.CircuitBreaker(fail_max=3)
    for _ in range(2):
        with pytest.raises(Boom):
            cb.call(boom)
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(boom)
    assert cb.current_state == "open"


def test_default_breaker_opens_on_fifth_failure():
    cb = pybreaker.CircuitBreaker()
    for _ in range(4):
        with pytest.raises(Boom):
            cb.call(boom)
    assert cb.current_state == "closed"
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(boom)
    assert cb.current_state == "open"


def test_success_resets_fail_counter_in_closed_state():
    cb = pybreaker.CircuitBreaker(fail_max=5)
    for _ in range(3):
        with pytest.raises(Boom):
            cb.call(boom)
    assert cb.call(ok) == "ok"
    assert cb.fail_counter == 0


def test_open_circuit_does_not_invoke_function():
    cb = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=60)
    trip(cb)
    func = mock.Mock(return_value=1)
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(func)
    func.assert_not_called()


def test_open_circuit_blocks_before_timeout_and_recovers_after():
    with freeze_time("2024-01-01 12:00:00") as frozen:
        cb = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=10)
        trip(cb)
        frozen.tick(timedelta(seconds=4))
        with pytest.raises(pybreaker.CircuitBreakerError):
            cb.call(ok)
        assert cb.current_state == "open"
        frozen.tick(timedelta(seconds=10))
        assert cb.call(ok) == "ok"
        assert cb.current_state == "closed"


def test_half_open_failure_reopens_circuit():
    with freeze_time("2024-01-01 12:00:00") as frozen:
        cb = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=10)
        trip(cb)
        frozen.tick(timedelta(seconds=11))
        with pytest.raises((Boom, pybreaker.CircuitBreakerError)):
            cb.call(boom)
        assert cb.current_state == "open"


def test_reopened_circuit_needs_a_fresh_timeout():
    with freeze_time("2024-01-01 12:00:00") as frozen:
        cb = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=10)
        trip(cb)
        frozen.tick(timedelta(seconds=11))
        with pytest.raises((Boom, pybreaker.CircuitBreakerError)):
            cb.call(boom)
        frozen.tick(timedelta(seconds=5))
        with pytest.raises(pybreaker.CircuitBreakerError):
            cb.call(ok)
        frozen.tick(timedelta(seconds=10))
        assert cb.call(ok) == "ok"
        assert cb.current_state == "closed"


def test_manual_open_blocks_calls():
    cb = pybreaker.CircuitBreaker()
    cb.open()
    assert cb.current_state == "open"
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(ok)


def test_manual_close_allows_calls_again():
    cb = pybreaker.CircuitBreaker(fail_max=1)
    trip(cb)
    cb.close()
    assert cb.current_state == "closed"
    assert cb.fail_counter == 0
    assert cb.call(ok) == "ok"


def test_manual_half_open_then_success_closes():
    cb = pybreaker.CircuitBreaker()
    cb.half_open()
    assert cb.current_state == "half-open"
    assert cb.call(ok) == "ok"
    assert cb.current_state == "closed"


def test_manual_half_open_then_failure_opens():
    cb = pybreaker.CircuitBreaker()
    cb.half_open()
    with pytest.raises((Boom, pybreaker.CircuitBreakerError)):
        cb.call(boom)
    assert cb.current_state == "open"


def test_success_threshold_needs_multiple_successes_to_close():
    cb = pybreaker.CircuitBreaker(success_threshold=3)
    cb.half_open()
    cb.call(ok)
    assert cb.current_state == "half-open"
    cb.call(ok)
    assert cb.current_state == "half-open"
    cb.call(ok)
    assert cb.current_state == "closed"


def test_success_counter_counts_in_half_open_state():
    cb = pybreaker.CircuitBreaker(success_threshold=3)
    cb.half_open()
    cb.call(ok)
    assert cb.success_counter == 1
    cb.call(ok)
    assert cb.success_counter == 2


def test_throw_new_error_on_trip_false_raises_original_error():
    cb = pybreaker.CircuitBreaker(fail_max=2, throw_new_error_on_trip=False)
    with pytest.raises(Boom):
        cb.call(boom)
    with pytest.raises(Boom):
        cb.call(boom)
    assert cb.current_state == "open"
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(ok)


def test_manual_open_recovers_after_reset_timeout():
    with freeze_time("2024-01-01 12:00:00") as frozen:
        cb = pybreaker.CircuitBreaker(reset_timeout=30)
        cb.open()
        with pytest.raises(pybreaker.CircuitBreakerError):
            cb.call(ok)
        frozen.tick(timedelta(seconds=31))
        assert cb.call(ok) == "ok"
        assert cb.current_state == "closed"


def test_reset_timeout_setter_takes_effect_on_open_circuit():
    with freeze_time("2024-01-01 12:00:00") as frozen:
        cb = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=1000)
        trip(cb)
        cb.reset_timeout = 5
        frozen.tick(timedelta(seconds=6))
        assert cb.call(ok) == "ok"
        assert cb.current_state == "closed"


def test_failures_after_close_start_counting_from_zero():
    cb = pybreaker.CircuitBreaker(fail_max=3)
    trip(cb)
    cb.close()
    for _ in range(2):
        with pytest.raises(Boom):
            cb.call(boom)
    assert cb.current_state == "closed"
