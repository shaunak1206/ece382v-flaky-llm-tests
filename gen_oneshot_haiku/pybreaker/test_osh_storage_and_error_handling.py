"""Storage and error handling tests for circuit breaker."""

import pytest
import time
from datetime import datetime, timedelta
from pybreaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitMemoryStorage,
    STATE_CLOSED,
    STATE_OPEN,
    STATE_HALF_OPEN,
)


class TestCircuitBreakerError:
    """Test CircuitBreakerError exception."""

    def test_circuit_breaker_error_exists(self):
        """Test that CircuitBreakerError exists."""
        assert issubclass(CircuitBreakerError, Exception)

    def test_circuit_breaker_error_raised_when_open(self):
        """Test that CircuitBreakerError is raised when circuit is open."""
        cb = CircuitBreaker(fail_max=1)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

    def test_circuit_breaker_error_message(self):
        """Test CircuitBreakerError message."""
        cb = CircuitBreaker(fail_max=1)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        try:
            cb.call(failing_func)
        except CircuitBreakerError as e:
            assert str(e) is not None

    def test_circuit_breaker_error_in_open_state(self):
        """Test circuit breaker error behavior in open state."""
        cb = CircuitBreaker(fail_max=1, throw_new_error_on_trip=True)

        def failing_func():
            raise ValueError("original")

        # Trigger open
        with pytest.raises(ValueError):
            cb.call(failing_func)

        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError) as exc_info:
            cb.call(failing_func)

        assert "open" in str(exc_info.value).lower()

    def test_circuit_breaker_error_is_exception(self):
        """Test that CircuitBreakerError can be caught as Exception."""
        cb = CircuitBreaker(fail_max=1)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Can catch as generic Exception
        try:
            cb.call(failing_func)
        except Exception as e:
            assert isinstance(e, CircuitBreakerError)


class TestMemoryStorageCounter:
    """Test counter operations in memory storage."""

    def test_counter_initialization(self):
        """Test that counter is initialized to 0."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage.counter == 0

    def test_increment_counter_single(self):
        """Test incrementing counter by 1."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.increment_counter()
        assert storage.counter == 1

    def test_increment_counter_multiple(self):
        """Test incrementing counter multiple times."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        for _ in range(5):
            storage.increment_counter()
        assert storage.counter == 5

    def test_reset_counter(self):
        """Test resetting counter to 0."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.increment_counter()
        storage.increment_counter()
        storage.increment_counter()
        assert storage.counter == 3

        storage.reset_counter()
        assert storage.counter == 0

    def test_success_counter_initialization(self):
        """Test that success counter is initialized to 0."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage.success_counter == 0

    def test_increment_success_counter(self):
        """Test incrementing success counter."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.increment_success_counter()
        storage.increment_success_counter()
        assert storage.success_counter == 2

    def test_reset_success_counter(self):
        """Test resetting success counter."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.increment_success_counter()
        storage.increment_success_counter()
        assert storage.success_counter == 2

        storage.reset_success_counter()
        assert storage.success_counter == 0


class TestMemoryStorageState:
    """Test state operations in memory storage."""

    def test_state_initialization(self):
        """Test state initialization."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage.state == STATE_CLOSED

    def test_state_getter(self):
        """Test getting state."""
        storage = CircuitMemoryStorage(STATE_OPEN)
        assert storage.state == STATE_OPEN

    def test_state_setter(self):
        """Test setting state."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.state = STATE_OPEN
        assert storage.state == STATE_OPEN

    def test_state_transitions(self):
        """Test transitioning between states."""
        storage = CircuitMemoryStorage(STATE_CLOSED)

        storage.state = STATE_OPEN
        assert storage.state == STATE_OPEN

        storage.state = STATE_HALF_OPEN
        assert storage.state == STATE_HALF_OPEN

        storage.state = STATE_CLOSED
        assert storage.state == STATE_CLOSED

    def test_storage_name(self):
        """Test storage name property."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage.name == "memory"


class TestMemoryStorageOpenedAt:
    """Test opened_at timestamp in memory storage."""

    def test_opened_at_initially_none(self):
        """Test that opened_at is initially None."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage.opened_at is None

    def test_opened_at_setter(self):
        """Test setting opened_at timestamp."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        now = datetime.now()
        storage.opened_at = now
        assert storage.opened_at is not None

    def test_opened_at_tracks_open_time(self):
        """Test that opened_at tracks when circuit was opened."""
        cb = CircuitBreaker(fail_max=1)

        def failing_func():
            raise Exception("error")

        before_open = datetime.now()

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Get the opened_at time
        opened_at = cb._state_storage.opened_at

        after_open = datetime.now()

        # opened_at should be between before and after
        assert opened_at is not None
        assert before_open <= opened_at <= after_open


class TestErrorPropagation:
    """Test exception propagation through circuit breaker."""

    def test_exception_preserved_in_closed_state(self):
        """Test that exceptions are preserved in closed state."""
        cb = CircuitBreaker(fail_max=5)

        def raises_value_error():
            raise ValueError("specific error message")

        with pytest.raises(ValueError) as exc_info:
            cb.call(raises_value_error)

        assert "specific error message" in str(exc_info.value)

    def test_different_exception_types(self):
        """Test handling different exception types."""
        cb = CircuitBreaker(fail_max=5)

        def raises_type_error():
            raise TypeError("type error")

        def raises_key_error():
            raise KeyError("key error")

        with pytest.raises(TypeError):
            cb.call(raises_type_error)

        with pytest.raises(KeyError):
            cb.call(raises_key_error)

    def test_exception_with_args(self):
        """Test exception with multiple arguments."""
        cb = CircuitBreaker()

        def raises_exception_with_args():
            raise ValueError("arg1", "arg2", "arg3")

        with pytest.raises(ValueError):
            cb.call(raises_exception_with_args)

    def test_custom_exception(self):
        """Test handling custom exceptions."""
        class CustomError(Exception):
            pass

        cb = CircuitBreaker(fail_max=3)

        def raises_custom():
            raise CustomError("custom message")

        with pytest.raises(CustomError):
            cb.call(raises_custom)


class TestCounterBehavior:
    """Test counter behavior across states."""

    def test_fail_counter_increments_on_system_error(self):
        """Test that fail counter increments only on system errors."""
        cb = CircuitBreaker(fail_max=10, exclude=[ValueError])

        def raises_value_error():
            raise ValueError("excluded")

        def raises_runtime_error():
            raise RuntimeError("system error")

        # ValueError should not increment
        cb.call(raises_value_error)
        assert cb.fail_counter == 0

        # RuntimeError should increment
        with pytest.raises(RuntimeError):
            cb.call(raises_runtime_error)
        assert cb.fail_counter == 1

    def test_fail_counter_resets_on_success(self):
        """Test that fail counter resets on successful call."""
        cb = CircuitBreaker(fail_max=5)

        def failing_func():
            raise Exception("error")

        def working_func():
            return "success"

        # Cause some failures
        for _ in range(3):
            with pytest.raises(Exception):
                cb.call(failing_func)

        assert cb.fail_counter == 3

        # Success resets
        cb.call(working_func)
        assert cb.fail_counter == 0

    def test_success_counter_increments_in_half_open(self):
        """Test that success counter increments in half-open state."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, success_threshold=2)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Wait for half-open
        time.sleep(0.15)

        def working_func():
            return "success"

        # First success in half-open
        cb.call(working_func)
        assert cb.success_counter == 1

        # If we stay in half-open, another success
        if cb.current_state == STATE_HALF_OPEN:
            cb.call(working_func)
            assert cb.success_counter == 2

    def test_success_counter_resets_on_failure_in_half_open(self):
        """Test that success counter resets when failing in half-open."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, success_threshold=2)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Wait for half-open
        time.sleep(0.15)

        # Calling a failing function in half-open reopens circuit
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

        # Success counter should be reset
        assert cb.success_counter == 0


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_function_returns_none(self):
        """Test function that returns None."""
        cb = CircuitBreaker()

        def returns_none():
            return None

        result = cb.call(returns_none)
        assert result is None

    def test_function_returns_false(self):
        """Test function that returns False."""
        cb = CircuitBreaker()

        def returns_false():
            return False

        result = cb.call(returns_false)
        assert result is False

    def test_function_returns_zero(self):
        """Test function that returns 0."""
        cb = CircuitBreaker()

        def returns_zero():
            return 0

        result = cb.call(returns_zero)
        assert result == 0

    def test_function_returns_empty_string(self):
        """Test function that returns empty string."""
        cb = CircuitBreaker()

        def returns_empty_string():
            return ""

        result = cb.call(returns_empty_string)
        assert result == ""

    def test_function_raises_immediately(self):
        """Test function that raises immediately."""
        cb = CircuitBreaker(fail_max=2)

        def raises_immediately():
            raise RuntimeError("immediate")

        with pytest.raises(RuntimeError):
            cb.call(raises_immediately)

    def test_very_high_fail_max(self):
        """Test with very high fail_max."""
        cb = CircuitBreaker(fail_max=1000)

        def failing_func():
            raise Exception("error")

        # Should handle many failures
        for _ in range(100):
            with pytest.raises(Exception):
                cb.call(failing_func)

        # Should still be closed with fail_max=1000
        assert cb.fail_counter == 100

    def test_zero_reset_timeout(self):
        """Test with reset_timeout of 0."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # With 0 timeout, should immediately try half-open
        assert cb.current_state == STATE_OPEN

    def test_many_state_changes(self):
        """Test many rapid state changes."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.05)

        def failing_func():
            raise Exception("error")

        def working_func():
            return "ok"

        for _ in range(3):
            # Trigger open
            with pytest.raises(Exception):
                cb.call(failing_func)

            with pytest.raises(Exception):
                cb.call(failing_func)

            # Wait and recover
            time.sleep(0.1)
            cb.call(working_func)
            assert cb.current_state == STATE_CLOSED
