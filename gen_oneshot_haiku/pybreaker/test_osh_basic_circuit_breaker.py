"""Basic circuit breaker tests."""

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


class TestCircuitBreakerCreation:
    """Test circuit breaker creation and initialization."""

    def test_create_circuit_breaker_default(self):
        """Test creating circuit breaker with default parameters."""
        cb = CircuitBreaker()
        assert cb is not None
        assert cb.fail_max == 5
        assert cb.reset_timeout == 60

    def test_create_circuit_breaker_with_custom_fail_max(self):
        """Test creating circuit breaker with custom fail_max."""
        cb = CircuitBreaker(fail_max=3)
        assert cb.fail_max == 3

    def test_create_circuit_breaker_with_custom_reset_timeout(self):
        """Test creating circuit breaker with custom reset_timeout."""
        cb = CircuitBreaker(reset_timeout=30)
        assert cb.reset_timeout == 30

    def test_create_circuit_breaker_with_custom_success_threshold(self):
        """Test creating circuit breaker with custom success_threshold."""
        cb = CircuitBreaker(success_threshold=2)
        assert cb.success_threshold == 2

    def test_create_circuit_breaker_with_name(self):
        """Test creating circuit breaker with name."""
        cb = CircuitBreaker(name="my_breaker")
        assert cb.name == "my_breaker"

    def test_circuit_breaker_initial_state_closed(self):
        """Test that circuit breaker starts in closed state."""
        cb = CircuitBreaker()
        assert cb.current_state == STATE_CLOSED

    def test_circuit_breaker_initial_fail_counter_zero(self):
        """Test that fail counter is initially zero."""
        cb = CircuitBreaker()
        assert cb.fail_counter == 0

    def test_circuit_breaker_initial_success_counter_zero(self):
        """Test that success counter is initially zero."""
        cb = CircuitBreaker()
        assert cb.success_counter == 0


class TestCircuitBreakerCall:
    """Test circuit breaker call() method."""

    def test_call_successful_function(self):
        """Test calling a successful function through circuit breaker."""
        cb = CircuitBreaker()

        def successful_func():
            return "success"

        result = cb.call(successful_func)
        assert result == "success"

    def test_call_with_arguments(self):
        """Test calling function with arguments."""
        cb = CircuitBreaker()

        def func_with_args(a, b):
            return a + b

        result = cb.call(func_with_args, 3, 4)
        assert result == 7

    def test_call_with_keyword_arguments(self):
        """Test calling function with keyword arguments."""
        cb = CircuitBreaker()

        def func_with_kwargs(x=0, y=0):
            return x * y

        result = cb.call(func_with_kwargs, x=5, y=6)
        assert result == 30

    def test_call_failing_function(self):
        """Test calling a failing function through circuit breaker."""
        cb = CircuitBreaker(fail_max=3)

        def failing_func():
            raise Exception("Function failed")

        with pytest.raises(Exception):
            cb.call(failing_func)

    def test_call_returns_function_value(self):
        """Test that call() returns the function's return value."""
        cb = CircuitBreaker()

        def func_returns_dict():
            return {"key": "value"}

        result = cb.call(func_returns_dict)
        assert result == {"key": "value"}

    def test_call_increments_fail_counter_on_error(self):
        """Test that fail counter is incremented on error."""
        cb = CircuitBreaker(fail_max=5)

        def failing_func():
            raise Exception("fail")

        initial_count = cb.fail_counter

        with pytest.raises(Exception):
            cb.call(failing_func)

        assert cb.fail_counter == initial_count + 1

    def test_call_resets_fail_counter_on_success(self):
        """Test that fail counter is reset on successful call."""
        cb = CircuitBreaker(fail_max=5)

        def failing_func():
            raise Exception("fail")

        def successful_func():
            return "ok"

        # Cause some failures
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)

        assert cb.fail_counter == 2

        # Success should reset counter
        cb.call(successful_func)
        assert cb.fail_counter == 0


class TestCircuitBreakerDecorator:
    """Test circuit breaker as a decorator."""

    def test_decorator_basic(self):
        """Test using circuit breaker as a decorator."""
        cb = CircuitBreaker()

        @cb
        def my_function():
            return "result"

        result = my_function()
        assert result == "result"

    def test_decorator_with_arguments(self):
        """Test decorator on function with arguments."""
        cb = CircuitBreaker()

        @cb
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_decorator_with_kwargs(self):
        """Test decorator on function with keyword arguments."""
        cb = CircuitBreaker()

        @cb
        def multiply(x=1, y=1):
            return x * y

        result = multiply(x=4, y=5)
        assert result == 20

    def test_decorator_preserves_function_name(self):
        """Test that decorator preserves function name."""
        cb = CircuitBreaker()

        @cb
        def my_decorated_func():
            pass

        assert my_decorated_func.__name__ == "my_decorated_func"

    def test_decorator_with_failure(self):
        """Test decorator when function raises exception."""
        cb = CircuitBreaker(fail_max=2)

        @cb
        def failing_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            failing_func()


class TestCircuitBreakerContext:
    """Test circuit breaker as context manager."""

    def test_calling_context_manager(self):
        """Test using circuit breaker with context manager."""
        cb = CircuitBreaker()

        def func():
            return 42

        # Using calling() context manager
        with cb.calling():
            # The with statement should work
            pass

        # Function should execute normally
        result = cb.call(func)
        assert result == 42


class TestCircuitBreakerProperties:
    """Test circuit breaker properties."""

    def test_fail_max_property_getter(self):
        """Test getting fail_max property."""
        cb = CircuitBreaker(fail_max=7)
        assert cb.fail_max == 7

    def test_fail_max_property_setter(self):
        """Test setting fail_max property."""
        cb = CircuitBreaker(fail_max=5)
        cb.fail_max = 10
        assert cb.fail_max == 10

    def test_reset_timeout_property_getter(self):
        """Test getting reset_timeout property."""
        cb = CircuitBreaker(reset_timeout=45)
        assert cb.reset_timeout == 45

    def test_reset_timeout_property_setter(self):
        """Test setting reset_timeout property."""
        cb = CircuitBreaker(reset_timeout=60)
        cb.reset_timeout = 120
        assert cb.reset_timeout == 120

    def test_success_threshold_property_getter(self):
        """Test getting success_threshold property."""
        cb = CircuitBreaker(success_threshold=3)
        assert cb.success_threshold == 3

    def test_success_threshold_property_setter(self):
        """Test setting success_threshold property."""
        cb = CircuitBreaker(success_threshold=1)
        cb.success_threshold = 5
        assert cb.success_threshold == 5

    def test_name_property_getter(self):
        """Test getting name property."""
        cb = CircuitBreaker(name="test_breaker")
        assert cb.name == "test_breaker"

    def test_name_property_setter(self):
        """Test setting name property."""
        cb = CircuitBreaker()
        cb.name = "renamed_breaker"
        assert cb.name == "renamed_breaker"

    def test_fail_counter_property(self):
        """Test accessing fail_counter property."""
        cb = CircuitBreaker()
        assert isinstance(cb.fail_counter, int)
        assert cb.fail_counter == 0

    def test_success_counter_property(self):
        """Test accessing success_counter property."""
        cb = CircuitBreaker()
        assert isinstance(cb.success_counter, int)
        assert cb.success_counter == 0

    def test_current_state_property(self):
        """Test accessing current_state property."""
        cb = CircuitBreaker()
        assert cb.current_state in [STATE_CLOSED, STATE_OPEN, STATE_HALF_OPEN]

    def test_state_property(self):
        """Test accessing state property."""
        cb = CircuitBreaker()
        state = cb.state
        assert state is not None


class TestCircuitBreakerMemoryStorage:
    """Test circuit breaker with memory storage."""

    def test_memory_storage_creation(self):
        """Test creating CircuitMemoryStorage."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage is not None
        assert storage.state == STATE_CLOSED

    def test_memory_storage_state_getter(self):
        """Test getting state from memory storage."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage.state == STATE_CLOSED

    def test_memory_storage_state_setter(self):
        """Test setting state in memory storage."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.state = STATE_OPEN
        assert storage.state == STATE_OPEN

    def test_memory_storage_counter(self):
        """Test counter in memory storage."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage.counter == 0

    def test_memory_storage_increment_counter(self):
        """Test incrementing counter in memory storage."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.increment_counter()
        assert storage.counter == 1

    def test_memory_storage_reset_counter(self):
        """Test resetting counter in memory storage."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.increment_counter()
        storage.increment_counter()
        assert storage.counter == 2

        storage.reset_counter()
        assert storage.counter == 0

    def test_memory_storage_success_counter(self):
        """Test success counter in memory storage."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        assert storage.success_counter == 0

    def test_memory_storage_increment_success_counter(self):
        """Test incrementing success counter."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.increment_success_counter()
        storage.increment_success_counter()
        assert storage.success_counter == 2

    def test_memory_storage_reset_success_counter(self):
        """Test resetting success counter."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        storage.increment_success_counter()
        assert storage.success_counter == 1

        storage.reset_success_counter()
        assert storage.success_counter == 0

    def test_custom_storage_with_circuit_breaker(self):
        """Test using custom storage with circuit breaker."""
        storage = CircuitMemoryStorage(STATE_CLOSED)
        cb = CircuitBreaker(state_storage=storage)

        assert cb.current_state == STATE_CLOSED
        assert cb.fail_counter == 0
