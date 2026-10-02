"""Tests for tenacity retry decorator and basic Retrying functionality."""

import pytest
from tenacity import (
    retry, stop_after_attempt, stop_after_delay, wait_fixed,
    retry_if_exception_type, RetryError, TryAgain, Retrying
)
import time


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_retry_success_on_first_attempt(self):
        """Test that function succeeds on first attempt."""
        @retry(stop=stop_after_attempt(3))
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_retry_succeeds_after_retries(self):
        """Test that function succeeds after retries."""
        call_count = [0]

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        def sometimes_fails():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        result = sometimes_fails()
        assert result == "success"
        assert call_count[0] == 2

    def test_retry_exhausts_attempts(self):
        """Test that RetryError is raised when attempts exhausted."""
        @retry(stop=stop_after_attempt(2), wait=wait_fixed(0.01))
        def always_fails():
            raise ValueError("always fails")

        with pytest.raises(RetryError) as exc_info:
            always_fails()
        assert exc_info.value.last_attempt.attempt_number == 2

    def test_retry_with_no_parameters(self):
        """Test @retry with no parameters (uses defaults)."""
        call_count = [0]

        @retry()
        def increment():
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("fail")
            return call_count[0]

        result = increment()
        assert result == 2

    def test_retry_error_contains_last_attempt(self):
        """Test that RetryError contains last attempt info."""
        @retry(stop=stop_after_attempt(2))
        def always_fails():
            raise ValueError("test error")

        with pytest.raises(RetryError) as exc_info:
            always_fails()
        assert exc_info.value.last_attempt.failed
        assert isinstance(exc_info.value.last_attempt.exception(), ValueError)

    def test_retry_with_specific_exception_type(self):
        """Test retry only on specific exception types."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(ValueError),
            wait=wait_fixed(0.01)
        )
        def fails_with_valueerror():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry me")
            return "success"

        result = fails_with_valueerror()
        assert result == "success"

    def test_retry_reraise_false(self):
        """Test that reraise=False raises RetryError instead of original."""
        @retry(stop=stop_after_attempt(2), reraise=False)
        def fails():
            raise ValueError("original error")

        with pytest.raises(RetryError):
            fails()

    def test_retry_reraise_true(self):
        """Test that reraise=True re-raises original exception."""
        @retry(stop=stop_after_attempt(2), reraise=True)
        def fails():
            raise ValueError("original error")

        with pytest.raises(ValueError, match="original error"):
            fails()

    def test_retry_try_again_exception(self):
        """Test TryAgain exception for explicit retries."""
        call_count = [0]

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        def explicit_retry():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TryAgain()
            return "success"

        result = explicit_retry()
        assert result == "success"
        assert call_count[0] == 2

    def test_retry_preserves_function_metadata(self):
        """Test that retry decorator preserves function metadata."""
        @retry()
        def add(a, b):
            """Add two numbers."""
            return a + b

        assert add.__name__ == "add"
        assert "Add two numbers" in add.__doc__

    def test_retry_statistics(self):
        """Test that retry statistics are tracked."""
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        def compute():
            return 42

        result = compute()
        stats = compute.retry.statistics
        assert "start_time" in stats
        assert "attempt_number" in stats

    def test_retry_enabled_false(self):
        """Test retry with enabled=False."""
        call_count = [0]

        @retry(stop=stop_after_attempt(3), enabled=False)
        def fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(ValueError):
            fails()
        # Should only be called once (not retried)
        assert call_count[0] == 1


class TestRetryingClass:
    """Tests for Retrying class."""

    def test_retrying_basic_usage(self):
        """Test basic Retrying class usage."""
        retrying = Retrying(stop=stop_after_attempt(3))
        call_count = [0]

        def sometimes_fails():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        result = retrying(sometimes_fails)
        assert result == "success"

    def test_retrying_with_arguments(self):
        """Test Retrying with function arguments."""
        retrying = Retrying(stop=stop_after_attempt(3))

        def add(a, b):
            return a + b

        result = retrying(add, 1, 2)
        assert result == 3

    def test_retrying_with_kwargs(self):
        """Test Retrying with keyword arguments."""
        retrying = Retrying(stop=stop_after_attempt(3))

        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = retrying(greet, name="Alice", greeting="Hi")
        assert result == "Hi, Alice!"

    def test_retrying_wraps(self):
        """Test Retrying.wraps() method."""
        retrying = Retrying(stop=stop_after_attempt(3))

        def add(a, b):
            return a + b

        wrapped = retrying.wraps(add)
        result = wrapped(1, 2)
        assert result == 3

    def test_retrying_copy(self):
        """Test Retrying.copy() method."""
        retrying1 = Retrying(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        retrying2 = retrying1.copy(stop=stop_after_attempt(5))

        # Original should be unchanged
        assert retrying1.stop.max_attempt_number == 3
        assert retrying2.stop.max_attempt_number == 5

    def test_retrying_statistics(self):
        """Test Retrying statistics."""
        retrying = Retrying(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        call_count = [0]

        def fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            retrying(fails)

        stats = retrying.statistics
        assert "attempt_number" in stats
        assert stats["attempt_number"] > 1


class TestRetryConditions:
    """Tests for retry conditions."""

    def test_retry_if_exception_type_matches(self):
        """Test retry_if_exception_type when exception matches."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(ValueError),
            wait=wait_fixed(0.01)
        )
        def fails_with_valueerror():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry")
            return "success"

        result = fails_with_valueerror()
        assert result == "success"
        assert call_count[0] == 2

    def test_retry_if_exception_type_no_match(self):
        """Test retry_if_exception_type when exception doesn't match."""
        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(ValueError),
            wait=wait_fixed(0.01)
        )
        def fails_with_typeerror():
            raise TypeError("don't retry")

        with pytest.raises(TypeError):
            fails_with_typeerror()

    def test_retry_if_exception_type_multiple_types(self):
        """Test retry_if_exception_type with multiple exception types."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type((ValueError, TypeError)),
            wait=wait_fixed(0.01)
        )
        def fails_sometimes():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("retry")
            if call_count[0] == 2:
                raise TypeError("retry")
            return "success"

        result = fails_sometimes()
        assert result == "success"
        assert call_count[0] == 3


class TestStopConditions:
    """Tests for stop conditions."""

    def test_stop_after_attempt(self):
        """Test stop_after_attempt stops after max attempts."""
        call_count = [0]

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError) as exc_info:
            always_fails()
        assert call_count[0] == 3
        assert exc_info.value.last_attempt.attempt_number == 3

    def test_stop_after_delay(self):
        """Test stop_after_delay stops after time limit."""
        @retry(stop=stop_after_delay(0.1), wait=wait_fixed(0.05))
        def always_fails():
            raise ValueError("fail")

        with pytest.raises(RetryError):
            always_fails()

    def test_stop_never(self):
        """Test stop_never never stops (needs other stop condition)."""
        from tenacity import stop_never

        call_count = [0]

        @retry(stop=stop_never | stop_after_attempt(3), wait=wait_fixed(0.01))
        def fails_until_limit():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("fail")
            return "success"

        result = fails_until_limit()
        assert result == "success"


class TestWaitStrategies:
    """Tests for wait strategies."""

    def test_wait_fixed(self):
        """Test wait_fixed waits fixed amount."""
        start = time.time()

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.05))
        def fails_twice():
            if not hasattr(fails_twice, 'called'):
                fails_twice.called = 0
            fails_twice.called += 1
            if fails_twice.called < 3:
                raise ValueError("fail")
            return "success"

        result = fails_twice()
        elapsed = time.time() - start
        assert result == "success"
        # Should wait at least 0.1 seconds (2 retries * 0.05)
        assert elapsed >= 0.08

    def test_wait_strategy_types(self):
        """Test different wait strategy types."""
        from tenacity import wait_none, wait_random, wait_incrementing

        # Test wait_none
        @retry(stop=stop_after_attempt(2), wait=wait_none())
        def quick_fail():
            raise ValueError("fail")

        start = time.time()
        with pytest.raises(RetryError):
            quick_fail()
        elapsed = time.time() - start
        # Should be very fast (no wait)
        assert elapsed < 0.1


class TestBeforeAfterCallbacks:
    """Tests for before/after callbacks."""

    def test_before_callback(self):
        """Test before callback is called before attempt."""
        from tenacity import before_log, before_nothing

        call_log = []

        def custom_before(retry_state):
            call_log.append(("before", retry_state.attempt_number))

        @retry(stop=stop_after_attempt(3), before=custom_before, wait=wait_fixed(0.01))
        def fails_once():
            if not hasattr(fails_once, 'called'):
                fails_once.called = 0
            fails_once.called += 1
            if fails_once.called < 2:
                raise ValueError("fail")
            return "success"

        result = fails_once()
        assert result == "success"
        # before should be called for each attempt
        assert len(call_log) >= 2

    def test_after_callback(self):
        """Test after callback is called after retry decision."""
        from tenacity import after_log, after_nothing

        call_log = []

        def custom_after(retry_state):
            call_log.append(("after", retry_state.attempt_number))

        @retry(stop=stop_after_attempt(3), after=custom_after, wait=wait_fixed(0.01))
        def fails_once():
            if not hasattr(fails_once, 'called'):
                fails_once.called = 0
            fails_once.called += 1
            if fails_once.called < 2:
                raise ValueError("fail")
            return "success"

        result = fails_once()
        assert result == "success"
        # after should be called
        assert len(call_log) >= 1
