"""Tests for tenacity retry conditions and combinations."""

import pytest
from tenacity import (
    retry, stop_after_attempt, wait_fixed,
    retry_if_exception, retry_if_exception_type, retry_if_not_exception_type,
    retry_if_result, retry_if_not_result, retry_if_exception_message,
    retry_any, retry_all, retry_never, retry_always,
    RetryError, TryAgain
)
import re


class TestRetryIfException:
    """Tests for retry_if_exception."""

    def test_retry_if_exception_with_predicate(self):
        """Test retry_if_exception with custom predicate."""
        call_count = [0]

        def is_retryable(exc):
            return "retry" in str(exc).lower()

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception(is_retryable),
            wait=wait_fixed(0.01)
        )
        def fails_with_retryable():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("please retry")
            return "success"

        result = fails_with_retryable()
        assert result == "success"
        assert call_count[0] == 2

    def test_retry_if_exception_non_matching_predicate(self):
        """Test retry_if_exception when predicate doesn't match."""
        def is_retryable(exc):
            return "retry" in str(exc).lower()

        @retry(
            stop=stop_after_attempt(2),
            retry=retry_if_exception(is_retryable),
            wait=wait_fixed(0.01)
        )
        def fails_non_retryable():
            raise ValueError("dont retry")

        with pytest.raises(ValueError):
            fails_non_retryable()


class TestRetryIfExceptionType:
    """Tests for retry_if_exception_type."""

    def test_retry_if_exception_type_single_type(self):
        """Test retry_if_exception_type with single exception type."""
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

    def test_retry_if_exception_type_tuple(self):
        """Test retry_if_exception_type with tuple of types."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type((ValueError, TypeError)),
            wait=wait_fixed(0.01)
        )
        def fails_with_multiple_types():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("retry")
            if call_count[0] == 2:
                raise TypeError("retry")
            return "success"

        result = fails_with_multiple_types()
        assert result == "success"
        assert call_count[0] == 3

    def test_retry_if_exception_type_no_retry_on_mismatch(self):
        """Test that non-matching exception is not retried."""
        @retry(
            stop=stop_after_attempt(2),
            retry=retry_if_exception_type(ValueError),
            wait=wait_fixed(0.01)
        )
        def fails_with_typeerror():
            raise TypeError("dont retry")

        with pytest.raises(TypeError):
            fails_with_typeerror()


class TestRetryIfNotExceptionType:
    """Tests for retry_if_not_exception_type."""

    def test_retry_if_not_exception_type(self):
        """Test retry_if_not_exception_type."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_not_exception_type(ValueError),
            wait=wait_fixed(0.01)
        )
        def fails_with_typeerror():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TypeError("retry")
            return "success"

        result = fails_with_typeerror()
        assert result == "success"

    def test_retry_if_not_exception_type_no_retry_on_match(self):
        """Test no retry when exception type matches."""
        @retry(
            stop=stop_after_attempt(2),
            retry=retry_if_not_exception_type(ValueError),
            wait=wait_fixed(0.01)
        )
        def fails_with_valueerror():
            raise ValueError("dont retry")

        with pytest.raises(ValueError):
            fails_with_valueerror()


class TestRetryIfResult:
    """Tests for retry_if_result."""

    def test_retry_if_result_with_predicate(self):
        """Test retry_if_result with result predicate."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_result(lambda x: x < 10),
            wait=wait_fixed(0.01)
        )
        def returns_incrementing():
            call_count[0] += 1
            return call_count[0] * 5

        result = returns_incrementing()
        assert result >= 10

    def test_retry_if_result_false_condition(self):
        """Test retry_if_result when condition is false."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_result(lambda x: x < 10),
            wait=wait_fixed(0.01)
        )
        def returns_high():
            call_count[0] += 1
            return 100

        result = returns_high()
        assert result == 100
        assert call_count[0] == 1


class TestRetryIfNotResult:
    """Tests for retry_if_not_result."""

    def test_retry_if_not_result(self):
        """Test retry_if_not_result."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_not_result(lambda x: x == "success"),
            wait=wait_fixed(0.01)
        )
        def returns_eventually_success():
            call_count[0] += 1
            if call_count[0] < 2:
                return "waiting"
            return "success"

        result = returns_eventually_success()
        assert result == "success"

    def test_retry_if_not_result_immediate_match(self):
        """Test retry_if_not_result when result matches immediately."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_not_result(lambda x: x == "success"),
            wait=wait_fixed(0.01)
        )
        def returns_success():
            call_count[0] += 1
            return "success"

        result = returns_success()
        assert result == "success"
        assert call_count[0] == 1


class TestRetryIfExceptionMessage:
    """Tests for retry_if_exception_message."""

    def test_retry_if_exception_message_exact(self):
        """Test retry_if_exception_message with exact message."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_message(message="retry me"),
            wait=wait_fixed(0.01)
        )
        def fails_with_message():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry me")
            return "success"

        result = fails_with_message()
        assert result == "success"

    def test_retry_if_exception_message_pattern(self):
        """Test retry_if_exception_message with regex pattern."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_message(match=r"retry.*"),
            wait=wait_fixed(0.01)
        )
        def fails_with_matching_message():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry me please")
            return "success"

        result = fails_with_matching_message()
        assert result == "success"

    def test_retry_if_exception_message_no_match(self):
        """Test no retry when message doesn't match."""
        @retry(
            stop=stop_after_attempt(2),
            retry=retry_if_exception_message(message="retry me"),
            wait=wait_fixed(0.01)
        )
        def fails_wrong_message():
            raise ValueError("dont retry")

        with pytest.raises(ValueError):
            fails_wrong_message()


class TestRetryAnyAll:
    """Tests for retry_any and retry_all."""

    def test_retry_any_first_matches(self):
        """Test retry_any when first condition matches."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_any(
                retry_if_exception_type(ValueError),
                retry_if_exception_type(TypeError)
            ),
            wait=wait_fixed(0.01)
        )
        def fails_with_valueerror():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry")
            return "success"

        result = fails_with_valueerror()
        assert result == "success"

    def test_retry_any_second_matches(self):
        """Test retry_any when second condition matches."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_any(
                retry_if_exception_type(ValueError),
                retry_if_exception_type(TypeError)
            ),
            wait=wait_fixed(0.01)
        )
        def fails_with_typeerror():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TypeError("retry")
            return "success"

        result = fails_with_typeerror()
        assert result == "success"

    def test_retry_all_all_match(self):
        """Test retry_all when all conditions match."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_all(
                retry_if_exception_type(ValueError),
                retry_if_exception(lambda e: "retry" in str(e))
            ),
            wait=wait_fixed(0.01)
        )
        def fails_matching_all():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry please")
            return "success"

        result = fails_matching_all()
        assert result == "success"

    def test_retry_or_operator(self):
        """Test retry_if_exception_type with | operator (retry_any)."""
        call_count = [0]

        retry_cond = retry_if_exception_type(ValueError) | retry_if_exception_type(TypeError)

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_cond,
            wait=wait_fixed(0.01)
        )
        def fails_with_typeerror():
            call_count[0] += 1
            if call_count[0] < 2:
                raise TypeError("retry")
            return "success"

        result = fails_with_typeerror()
        assert result == "success"

    def test_retry_and_operator(self):
        """Test retry with & operator (retry_all)."""
        call_count = [0]

        retry_cond = (
            retry_if_exception_type(ValueError) &
            retry_if_exception(lambda e: "retry" in str(e))
        )

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_cond,
            wait=wait_fixed(0.01)
        )
        def fails_matching_all():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry please")
            return "success"

        result = fails_matching_all()
        assert result == "success"


class TestRetryAlwaysNever:
    """Tests for retry_always and retry_never."""

    def test_retry_always(self):
        """Test retry_always always retries."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(3),
            retry=retry_always,
            wait=wait_fixed(0.01)
        )
        def always_retry():
            call_count[0] += 1
            if call_count[0] < 3:
                return "retry"
            return "final"

        result = always_retry()
        assert result == "final"
        assert call_count[0] == 3

    def test_retry_never(self):
        """Test retry_never never retries."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(5),
            retry=retry_never,
            wait=wait_fixed(0.01)
        )
        def never_retry():
            call_count[0] += 1
            return f"attempt {call_count[0]}"

        result = never_retry()
        assert call_count[0] == 1


class TestComplexRetryLogic:
    """Tests for complex retry logic combinations."""

    def test_complex_retry_combination(self):
        """Test complex retry condition."""
        call_count = [0]

        retry_cond = (
            (retry_if_exception_type(ValueError) | retry_if_exception_type(TypeError)) &
            retry_if_exception(lambda e: "retry" in str(e))
        )

        @retry(
            stop=stop_after_attempt(5),
            retry=retry_cond,
            wait=wait_fixed(0.01)
        )
        def complex_fail():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("retry me")
            if call_count[0] == 2:
                raise TypeError("retry me")
            if call_count[0] == 3:
                raise RuntimeError("don't retry")

        with pytest.raises(RuntimeError):
            complex_fail()
