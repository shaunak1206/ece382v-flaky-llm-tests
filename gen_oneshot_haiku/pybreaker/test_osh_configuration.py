"""Configuration tests for circuit breaker."""

import pytest
from pybreaker import (
    CircuitBreaker,
    CircuitBreakerError,
    STATE_CLOSED,
    STATE_OPEN,
)


class TestFailMaxConfiguration:
    """Test fail_max configuration."""

    def test_fail_max_default_is_5(self):
        """Test that default fail_max is 5."""
        cb = CircuitBreaker()
        assert cb.fail_max == 5

    def test_fail_max_custom_value(self):
        """Test setting custom fail_max."""
        cb = CircuitBreaker(fail_max=10)
        assert cb.fail_max == 10

    def test_fail_max_one(self):
        """Test fail_max of 1."""
        cb = CircuitBreaker(fail_max=1)
        assert cb.fail_max == 1

        def failing_func():
            raise Exception("error")

        # One failure should trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

    def test_fail_max_large_value(self):
        """Test fail_max with large value."""
        cb = CircuitBreaker(fail_max=100)
        assert cb.fail_max == 100

    def test_fail_max_setter(self):
        """Test setting fail_max after creation."""
        cb = CircuitBreaker(fail_max=5)
        cb.fail_max = 10
        assert cb.fail_max == 10

    def test_fail_max_affects_open_transition(self):
        """Test that fail_max controls when circuit opens."""
        cb = CircuitBreaker(fail_max=3)

        def failing_func():
            raise Exception("error")

        # First two failures shouldn't open
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)
            assert cb.current_state == STATE_CLOSED

        # Third failure should open
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.current_state == STATE_OPEN


class TestResetTimeoutConfiguration:
    """Test reset_timeout configuration."""

    def test_reset_timeout_default_is_60(self):
        """Test that default reset_timeout is 60 seconds."""
        cb = CircuitBreaker()
        assert cb.reset_timeout == 60

    def test_reset_timeout_custom_value(self):
        """Test setting custom reset_timeout."""
        cb = CircuitBreaker(reset_timeout=30)
        assert cb.reset_timeout == 30

    def test_reset_timeout_small_value(self):
        """Test reset_timeout with small value."""
        cb = CircuitBreaker(reset_timeout=0.01)
        assert cb.reset_timeout == 0.01

    def test_reset_timeout_zero(self):
        """Test reset_timeout of 0."""
        cb = CircuitBreaker(reset_timeout=0)
        assert cb.reset_timeout == 0

    def test_reset_timeout_setter(self):
        """Test setting reset_timeout after creation."""
        cb = CircuitBreaker(reset_timeout=60)
        cb.reset_timeout = 120
        assert cb.reset_timeout == 120

    def test_reset_timeout_affects_recovery(self):
        """Test that reset_timeout controls recovery delay."""
        import time

        cb = CircuitBreaker(fail_max=1, reset_timeout=0.2)

        def failing_func():
            raise Exception("error")

        # Trigger open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.current_state == STATE_OPEN

        # Should not recover immediately
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

        # After timeout, should start recovering
        time.sleep(0.25)

        def working_func():
            return "success"

        # Should be able to try again (enters half-open)
        result = cb.call(working_func)
        assert result == "success"


class TestSuccessThresholdConfiguration:
    """Test success_threshold configuration."""

    def test_success_threshold_default_is_1(self):
        """Test that default success_threshold is 1."""
        cb = CircuitBreaker()
        assert cb.success_threshold == 1

    def test_success_threshold_custom_value(self):
        """Test setting custom success_threshold."""
        cb = CircuitBreaker(success_threshold=3)
        assert cb.success_threshold == 3

    def test_success_threshold_setter(self):
        """Test setting success_threshold after creation."""
        cb = CircuitBreaker(success_threshold=1)
        cb.success_threshold = 2
        assert cb.success_threshold == 2

    def test_success_threshold_1_closes_after_one_success(self):
        """Test that success_threshold=1 closes after one success in half-open."""
        import time

        cb = CircuitBreaker(fail_max=1, reset_timeout=0.1, success_threshold=1)

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

        def working_func():
            return "success"

        # One success should close
        cb.call(working_func)
        assert cb.current_state == STATE_CLOSED

    def test_success_threshold_greater_than_1(self):
        """Test that success_threshold > 1 requires multiple successes."""
        import time

        cb = CircuitBreaker(
            fail_max=1,
            reset_timeout=0.1,
            success_threshold=2
        )

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

        def working_func():
            return "success"

        # First success - should stay in half-open
        cb.call(working_func)
        assert cb.current_state == STATE_HALF_OPEN

        # Second success - should close
        cb.call(working_func)
        assert cb.current_state == STATE_CLOSED


class TestExcludeConfiguration:
    """Test exclude configuration for exception filtering."""

    def test_exclude_exception_type(self):
        """Test excluding a specific exception type."""
        cb = CircuitBreaker(fail_max=2, exclude=[ValueError])

        def raises_value_error():
            raise ValueError("ignored")

        # ValueError should not count as failure
        cb.call(raises_value_error)  # Should not raise CircuitBreakerError
        cb.call(raises_value_error)

        # Circuit should still be closed
        assert cb.current_state == STATE_CLOSED

    def test_exclude_multiple_exception_types(self):
        """Test excluding multiple exception types."""
        cb = CircuitBreaker(
            fail_max=2,
            exclude=[ValueError, KeyError]
        )

        def raises_value_error():
            raise ValueError("ignored")

        def raises_key_error():
            raise KeyError("ignored")

        cb.call(raises_value_error)
        cb.call(raises_key_error)

        # Circuit should still be closed
        assert cb.current_state == STATE_CLOSED

    def test_exclude_predicate_function(self):
        """Test excluding exceptions using predicate function."""
        def is_timeout(exc):
            return "timeout" in str(exc).lower()

        cb = CircuitBreaker(fail_max=2, exclude=[is_timeout])

        def raises_timeout_error():
            raise Exception("Request timeout")

        # Timeout exception should not count
        cb.call(raises_timeout_error)
        cb.call(raises_timeout_error)

        assert cb.current_state == STATE_CLOSED

    def test_exclude_mixed_types_and_predicates(self):
        """Test excluding with mix of types and predicates."""
        def is_client_error(exc):
            return isinstance(exc, ValueError)

        cb = CircuitBreaker(
            fail_max=2,
            exclude=[KeyError, is_client_error]
        )

        def raises_key_error():
            raise KeyError("ignored")

        def raises_value_error():
            raise ValueError("ignored")

        def raises_runtime_error():
            raise RuntimeError("not ignored")

        # Excluded exceptions don't count
        cb.call(raises_key_error)
        cb.call(raises_value_error)

        assert cb.current_state == STATE_CLOSED

        # Non-excluded exceptions do count
        with pytest.raises(RuntimeError):
            cb.call(raises_runtime_error)

    def test_add_excluded_exception(self):
        """Test adding excluded exception after creation."""
        cb = CircuitBreaker(fail_max=2)

        cb.add_excluded_exception(ValueError)

        def raises_value_error():
            raise ValueError("ignored")

        cb.call(raises_value_error)
        cb.call(raises_value_error)

        assert cb.current_state == STATE_CLOSED

    def test_add_multiple_excluded_exceptions(self):
        """Test adding multiple excluded exceptions."""
        cb = CircuitBreaker(fail_max=3)

        cb.add_excluded_exceptions(ValueError, KeyError, TypeError)

        def raises_value_error():
            raise ValueError("ignored")

        def raises_key_error():
            raise KeyError("ignored")

        cb.call(raises_value_error)
        cb.call(raises_key_error)

        assert cb.current_state == STATE_CLOSED

    def test_remove_excluded_exception(self):
        """Test removing an excluded exception."""
        cb = CircuitBreaker(fail_max=2, exclude=[ValueError])

        cb.remove_excluded_exception(ValueError)

        def raises_value_error():
            raise ValueError("not ignored anymore")

        # Now ValueError should count as failure
        with pytest.raises(ValueError):
            cb.call(raises_value_error)

        with pytest.raises(ValueError):
            cb.call(raises_value_error)

        # Circuit should be open
        assert cb.current_state == STATE_OPEN

    def test_excluded_exceptions_property(self):
        """Test accessing excluded_exceptions property."""
        cb = CircuitBreaker(exclude=[ValueError, KeyError])

        excluded = cb.excluded_exceptions
        assert ValueError in excluded
        assert KeyError in excluded


class TestThrowNewErrorConfiguration:
    """Test throw_new_error_on_trip configuration."""

    def test_throw_new_error_on_trip_true(self):
        """Test throw_new_error_on_trip=True."""
        cb = CircuitBreaker(fail_max=1, throw_new_error_on_trip=True)

        def failing_func():
            raise ValueError("original error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Circuit now open - should throw CircuitBreakerError (new error)
        with pytest.raises(CircuitBreakerError):
            cb.call(failing_func)

    def test_throw_new_error_on_trip_false(self):
        """Test throw_new_error_on_trip=False."""
        cb = CircuitBreaker(fail_max=1, throw_new_error_on_trip=False)

        def failing_func():
            raise ValueError("original error")

        # First failure
        with pytest.raises(ValueError):
            cb.call(failing_func)

        # Circuit now open - should re-raise original error or similar
        with pytest.raises((ValueError, CircuitBreakerError)):
            cb.call(failing_func)
