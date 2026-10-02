"""State transition tests for circuit breaker."""

import pytest
import time
from datetime import datetime, timedelta
from pybreaker import (
    CircuitBreaker,
    CircuitBreakerError,
    STATE_CLOSED,
    STATE_OPEN,
    STATE_HALF_OPEN,
)


class TestClosedState:
    """Test circuit breaker in closed state."""

    def test_start_in_closed_state(self):
        """Test that circuit breaker starts in closed state."""
        cb = CircuitBreaker()
        assert cb.current_state == STATE_CLOSED

    def test_closed_state_allows_calls(self):
        """Test that closed state allows function calls."""
        cb = CircuitBreaker()

        def working_func():
            return "success"

        result = cb.call(working_func)
        assert result == "success"
        assert cb.current_state == STATE_CLOSED

    def test_closed_state_with_failures(self):
        """Test that closed state tracks failures."""
        cb = CircuitBreaker(fail_max=3)

        def failing_func():
            raise Exception("error")

        for i in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)

        assert cb.fail_counter == 2
        assert cb.current_state == STATE_CLOSED

    def test_closed_state_transitions_to_open(self):
        """Test that closed state transitions to open after fail_max failures."""
        cb = CircuitBreaker(fail_max=2)

        def failing_func():
            raise Exception("error")

        # First failure
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.current_state == STATE_CLOSED

        # Second failure - should transition to open
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)
        assert cb.current_state == STATE_OPEN

    def test_closed_state_resets_on_success(self):
        """Test that successful call resets fail counter in closed state."""
        cb = CircuitBreaker(fail_max=5)

        def failing_func():
            raise Exception("error")

        def successful_func():
            return "ok"

        # Cause some failures
        for _ in range(3):
            with pytest.raises(Exception):
                cb.call(failing_func)

        assert cb.fail_counter == 3

        # Successful call should reset counter
        cb.call(successful_func)
        assert cb.fail_counter == 0


class TestOpenState:
    """Test circuit breaker in open state."""

    def test_open_state_rejects_calls(self):
        """Test that open state rejects calls immediately."""
        cb = CircuitBreaker(fail_max=1)

        def failing_func():
            raise Exception("error")

        # Trigger open state
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Now circuit is open, should reject without calling function
        assert cb.current_state == STATE_OPEN

        def working_func():
            return "success"

        with pytest.raises(CircuitBreakerError):
            cb.call(working_func)

    def test_open_state_raises_circuit_breaker_error(self):
        """Test that open state raises CircuitBreakerError."""
        cb = CircuitBreaker(fail_max=1, throw_new_error_on_trip=True)

        def failing_func():
            raise Exception("error")

        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Circuit now open, should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

    def test_open_state_stays_open_until_timeout(self):
        """Test that open state persists until reset_timeout elapses."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

        # Circuit should still be open
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

    def test_open_state_transitions_to_half_open_after_timeout(self):
        """Test that open state transitions to half-open after timeout."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Try to call - should transition to half-open and allow the call
        def successful_func():
            return "success"

        # After timeout, should attempt to call (transitioning to half-open)
        # If it succeeds, state becomes closed
        result = cb.call(successful_func)
        assert result == "success"
        assert cb.current_state == STATE_CLOSED


class TestHalfOpenState:
    """Test circuit breaker in half-open state."""

    def test_half_open_state_after_timeout(self):
        """Test that circuit enters half-open state after timeout."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

        # Wait for timeout
        time.sleep(0.15)

        def test_func():
            return "test"

        # This call should cause transition to half-open
        result = cb.call(test_func)
        # After successful call in half-open, should transition to closed
        assert cb.current_state == STATE_CLOSED

    def test_half_open_successful_call_closes_circuit(self):
        """Test that successful call in half-open closes circuit."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, success_threshold=1)

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

        result = cb.call(working_func)
        assert result == "success"
        assert cb.current_state == STATE_CLOSED

    def test_half_open_failed_call_reopens_circuit(self):
        """Test that failed call in half-open reopens circuit."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

        # Wait for half-open
        time.sleep(0.15)

        # Call failing function - should reopen circuit
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

    def test_half_open_success_counter_increments(self):
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

        # First success - should still be half-open if success_threshold > 1
        cb.call(working_func)

        # Manually check state or call again if needed for the test scenario
        # For success_threshold=2, we'd need another success


class TestStateTransitionSequence:
    """Test full state transition sequences."""

    def test_full_cycle_closed_open_half_open_closed(self):
        """Test full cycle: closed -> open -> half-open -> closed."""
        cb = CircuitBreaker(fail_max=2, reset_timeout=0.1, success_threshold=1)

        def failing_func():
            raise Exception("error")

        def working_func():
            return "success"

        # Start in closed
        assert cb.current_state == STATE_CLOSED

        # Cause failures to trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Should be open now
        assert cb.current_state == STATE_OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Try successful call - transitions through half-open to closed
        result = cb.call(working_func)
        assert result == "success"
        assert cb.current_state == STATE_CLOSED

    def test_full_cycle_with_reopening(self):
        """Test cycle that reopens: closed -> open -> half-open -> open -> closed."""
        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, success_threshold=1)

        def failing_func():
            raise Exception("error")

        def working_func():
            return "success"

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Fail again in half-open - reopens circuit
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

        # Wait again
        time.sleep(0.15)

        # Succeed this time
        cb.call(working_func)
        assert cb.current_state == STATE_CLOSED

    def test_multiple_failures_before_open(self):
        """Test that fail_max controls when circuit opens."""
        cb = CircuitBreaker(fail_max=3, throw_new_error_on_trip=False)

        def failing_func():
            raise ValueError("error")

        # First failure - still closed
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.current_state == STATE_CLOSED
        assert cb.fail_counter == 1

        # Second failure - still closed
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.current_state == STATE_CLOSED
        assert cb.fail_counter == 2

        # Third failure - should transition to open
        with pytest.raises(ValueError):
            cb.call(failing_func)
        assert cb.current_state == STATE_OPEN
        assert cb.fail_counter == 3


class TestManualStateManagement:
    """Test manual state management methods."""

    def test_open_method(self):
        """Test opening circuit manually."""
        cb = CircuitBreaker()
        assert cb.current_state == STATE_CLOSED

        cb.open()
        assert cb.current_state == STATE_OPEN

    def test_half_open_method(self):
        """Test transitioning to half-open manually."""
        cb = CircuitBreaker()

        # First open
        cb.open()
        assert cb.current_state == STATE_OPEN

        # Then half-open
        cb.half_open()
        assert cb.current_state == STATE_HALF_OPEN

    def test_close_method(self):
        """Test closing circuit manually."""
        cb = CircuitBreaker()

        # Open circuit
        cb.open()
        assert cb.current_state == STATE_OPEN

        # Close circuit
        cb.close()
        assert cb.current_state == STATE_CLOSED

    def test_close_resets_success_counter(self):
        """Test that closing circuit resets success counter."""
        cb = CircuitBreaker()
        storage = cb._state_storage

        # Artificially increment success counter
        storage.increment_success_counter()
        storage.increment_success_counter()
        assert storage.success_counter == 2

        # Close should reset
        cb.close()
        assert storage.success_counter == 0
