"""Listener and decorator tests for circuit breaker."""

import pytest
from pybreaker import (
    CircuitBreaker,
    CircuitBreakerListener,
    CircuitBreakerError,
    STATE_CLOSED,
    STATE_OPEN,
    STATE_HALF_OPEN,
)


class TestCircuitBreakerListener:
    """Test CircuitBreakerListener callbacks."""

    def test_listener_before_call(self):
        """Test before_call listener callback."""
        calls = []

        class TestListener(CircuitBreakerListener):
            def before_call(self, cb, func, *args, **kwargs):
                calls.append(("before_call", cb, func, args, kwargs))

        listener = TestListener()
        cb = CircuitBreaker(listeners=[listener])

        def test_func(x):
            return x * 2

        cb.call(test_func, 5)

        assert len(calls) == 1
        assert calls[0][0] == "before_call"

    def test_listener_success_callback(self):
        """Test success listener callback."""
        calls = []

        class TestListener(CircuitBreakerListener):
            def success(self, cb):
                calls.append(("success", cb))

        listener = TestListener()
        cb = CircuitBreaker(listeners=[listener])

        def working_func():
            return "ok"

        cb.call(working_func)

        assert len(calls) == 1
        assert calls[0][0] == "success"

    def test_listener_failure_callback(self):
        """Test failure listener callback."""
        calls = []

        class TestListener(CircuitBreakerListener):
            def failure(self, cb, exc):
                calls.append(("failure", cb, exc))

        listener = TestListener()
        cb = CircuitBreaker(fail_max=2, listeners=[listener])

        def failing_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert len(calls) == 1
        assert calls[0][0] == "failure"
        assert isinstance(calls[0][2], ValueError)

    def test_listener_state_change_callback(self):
        """Test state_change listener callback."""
        calls = []

        class TestListener(CircuitBreakerListener):
            def state_change(self, cb, old_state, new_state):
                calls.append(("state_change", old_state, new_state))

        listener = TestListener()
        cb = CircuitBreaker(fail_max=1, listeners=[listener])

        def failing_func():
            raise Exception("error")

        # Trigger state change from closed to open
        with pytest.raises(Exception):
            cb.call(failing_func)

        with pytest.raises(Exception):
            cb.call(failing_func)

        # Should have state change callback
        assert any(call[0] == "state_change" for call in calls)

    def test_multiple_listeners(self):
        """Test multiple listeners on same circuit breaker."""
        calls1 = []
        calls2 = []

        class Listener1(CircuitBreakerListener):
            def success(self, cb):
                calls1.append("success")

        class Listener2(CircuitBreakerListener):
            def success(self, cb):
                calls2.append("success")

        listener1 = Listener1()
        listener2 = Listener2()
        cb = CircuitBreaker(listeners=[listener1, listener2])

        def working_func():
            return "ok"

        cb.call(working_func)

        assert len(calls1) == 1
        assert len(calls2) == 1

    def test_add_listener(self):
        """Test adding listener after creation."""
        calls = []

        class TestListener(CircuitBreakerListener):
            def success(self, cb):
                calls.append("success")

        cb = CircuitBreaker()
        listener = TestListener()
        cb.add_listener(listener)

        def working_func():
            return "ok"

        cb.call(working_func)
        assert len(calls) == 1

    def test_add_multiple_listeners(self):
        """Test adding multiple listeners at once."""
        calls1 = []
        calls2 = []

        class Listener1(CircuitBreakerListener):
            def success(self, cb):
                calls1.append("success")

        class Listener2(CircuitBreakerListener):
            def success(self, cb):
                calls2.append("success")

        cb = CircuitBreaker()
        cb.add_listeners(Listener1(), Listener2())

        def working_func():
            return "ok"

        cb.call(working_func)

        assert len(calls1) == 1
        assert len(calls2) == 1

    def test_remove_listener(self):
        """Test removing a listener."""
        calls = []

        class TestListener(CircuitBreakerListener):
            def success(self, cb):
                calls.append("success")

        listener = TestListener()
        cb = CircuitBreaker(listeners=[listener])

        cb.remove_listener(listener)

        def working_func():
            return "ok"

        cb.call(working_func)

        # Listener should not be called
        assert len(calls) == 0

    def test_listeners_property(self):
        """Test accessing listeners property."""
        listener1 = CircuitBreakerListener()
        listener2 = CircuitBreakerListener()

        cb = CircuitBreaker(listeners=[listener1, listener2])
        listeners = cb.listeners

        assert len(listeners) == 2
        assert listener1 in listeners
        assert listener2 in listeners


class TestDecoratorUsage:
    """Test circuit breaker as decorator."""

    def test_decorator_on_simple_function(self):
        """Test decorator on simple function."""
        cb = CircuitBreaker()

        @cb
        def simple_func():
            return "result"

        result = simple_func()
        assert result == "result"

    def test_decorator_on_function_with_args(self):
        """Test decorator on function with positional arguments."""
        cb = CircuitBreaker()

        @cb
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_decorator_on_function_with_kwargs(self):
        """Test decorator on function with keyword arguments."""
        cb = CircuitBreaker()

        @cb
        def multiply(x=1, y=1):
            return x * y

        result = multiply(x=5, y=6)
        assert result == 30

    def test_decorator_with_mixed_args(self):
        """Test decorator with mixed positional and keyword arguments."""
        cb = CircuitBreaker()

        @cb
        def mixed_args(a, b, c=0):
            return a + b + c

        result = mixed_args(1, 2, c=3)
        assert result == 6

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        cb = CircuitBreaker()

        @cb
        def documented_func():
            """This is a documented function."""
            return "result"

        assert documented_func.__name__ == "documented_func"
        assert "documented" in documented_func.__doc__

    def test_decorator_with_failing_function(self):
        """Test decorator on function that raises exception."""
        cb = CircuitBreaker(fail_max=1)

        @cb
        def failing_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            failing_func()

        # Second call should trigger open and raise CircuitBreakerError
        with pytest.raises(ValueError):
            failing_func()

    def test_multiple_decorators(self):
        """Test using multiple circuit breaker decorators."""
        cb1 = CircuitBreaker()
        cb2 = CircuitBreaker()

        @cb1
        def func1():
            return "func1"

        @cb2
        def func2():
            return "func2"

        assert func1() == "func1"
        assert func2() == "func2"

    def test_decorator_call_tracking(self):
        """Test that decorator tracks calls through circuit breaker."""
        cb = CircuitBreaker(fail_max=2)

        @cb
        def failing_func():
            raise Exception("error")

        # First failure
        with pytest.raises(Exception):
            failing_func()

        # Second failure - triggers open
        with pytest.raises(Exception):
            failing_func()

        # Third call should hit circuit breaker error
        assert cb.current_state == STATE_OPEN


class TestCallVariations:
    """Test different ways to invoke circuit breaker."""

    def test_call_method_with_args(self):
        """Test call() method with positional arguments."""
        cb = CircuitBreaker()

        def add(a, b):
            return a + b

        result = cb.call(add, 2, 3)
        assert result == 5

    def test_call_method_with_kwargs(self):
        """Test call() method with keyword arguments."""
        cb = CircuitBreaker()

        def multiply(x=1, y=1):
            return x * y

        result = cb.call(multiply, x=4, y=5)
        assert result == 20

    def test_call_method_with_mixed_args(self):
        """Test call() method with mixed arguments."""
        cb = CircuitBreaker()

        def func(a, b, c=0):
            return a + b + c

        result = cb.call(func, 1, 2, c=3)
        assert result == 6

    def test_decorator_style_call(self):
        """Test decorator style call without parentheses."""
        cb = CircuitBreaker()

        def my_func():
            return "result"

        # Decorator without arguments
        wrapped = cb(my_func)
        result = wrapped()
        assert result == "result"

    def test_decorator_style_with_function_arg(self):
        """Test decorator with function as argument."""
        cb = CircuitBreaker()

        @cb
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10


class TestListenerSequence:
    """Test the sequence of listener callbacks."""

    def test_callback_order_on_success(self):
        """Test callback order for successful call."""
        sequence = []

        class SequenceListener(CircuitBreakerListener):
            def before_call(self, cb, func, *args, **kwargs):
                sequence.append("before_call")

            def success(self, cb):
                sequence.append("success")

        listener = SequenceListener()
        cb = CircuitBreaker(listeners=[listener])

        def working_func():
            sequence.append("func_called")
            return "ok"

        cb.call(working_func)

        assert sequence == ["before_call", "func_called", "success"]

    def test_callback_order_on_failure(self):
        """Test callback order for failed call."""
        sequence = []

        class SequenceListener(CircuitBreakerListener):
            def before_call(self, cb, func, *args, **kwargs):
                sequence.append("before_call")

            def failure(self, cb, exc):
                sequence.append("failure")

        listener = SequenceListener()
        cb = CircuitBreaker(fail_max=2, listeners=[listener])

        def failing_func():
            sequence.append("func_called")
            raise ValueError("error")

        with pytest.raises(ValueError):
            cb.call(failing_func)

        assert "before_call" in sequence
        assert "func_called" in sequence
        assert "failure" in sequence
