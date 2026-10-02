"""Tests for tenacity async retry support."""

import pytest
import asyncio
from tenacity import (
    AsyncRetrying, async_retry,
    stop_after_attempt, wait_fixed,
    retry_if_exception_type, RetryError
)


class TestAsyncRetrying:
    """Tests for AsyncRetrying class."""

    @pytest.mark.asyncio
    async def test_async_retrying_basic(self):
        """Test basic AsyncRetrying functionality."""
        retrying = AsyncRetrying(stop=stop_after_attempt(3))
        call_count = [0]

        async def sometimes_fails():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        result = await retrying(sometimes_fails)
        assert result == "success"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_async_retrying_with_args(self):
        """Test AsyncRetrying with function arguments."""
        retrying = AsyncRetrying(stop=stop_after_attempt(3))

        async def add(a, b):
            return a + b

        result = await retrying(add, 1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_async_retrying_with_kwargs(self):
        """Test AsyncRetrying with keyword arguments."""
        retrying = AsyncRetrying(stop=stop_after_attempt(3))

        async def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = await retrying(greet, name="Alice", greeting="Hi")
        assert result == "Hi, Alice!"

    @pytest.mark.asyncio
    async def test_async_retrying_exhausts_attempts(self):
        """Test that AsyncRetrying raises RetryError when exhausted."""
        retrying = AsyncRetrying(stop=stop_after_attempt(2), wait=wait_fixed(0.01))

        async def always_fails():
            raise ValueError("always fails")

        with pytest.raises(RetryError):
            await retrying(always_fails)

    @pytest.mark.asyncio
    async def test_async_retrying_copy(self):
        """Test AsyncRetrying.copy() method."""
        retrying1 = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.01)
        )
        retrying2 = retrying1.copy(stop=stop_after_attempt(5))

        assert retrying1.stop.max_attempt_number == 3
        assert retrying2.stop.max_attempt_number == 5


class TestAsyncRetryDecorator:
    """Tests for @async_retry decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_basic(self):
        """Test basic async_retry decorator."""
        call_count = [0]

        @async_retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        async def sometimes_fails():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        result = await sometimes_fails()
        assert result == "success"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_async_retry_success_first_attempt(self):
        """Test async_retry with success on first attempt."""
        @async_retry(stop=stop_after_attempt(3))
        async def add(a, b):
            return a + b

        result = await add(1, 2)
        assert result == 3

    @pytest.mark.asyncio
    async def test_async_retry_with_exception_type(self):
        """Test async_retry with specific exception type."""
        call_count = [0]

        @async_retry(
            stop=stop_after_attempt(3),
            retry=retry_if_exception_type(ValueError),
            wait=wait_fixed(0.01)
        )
        async def fails_with_valueerror():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("retry me")
            return "success"

        result = await fails_with_valueerror()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_retry_exhausts_attempts(self):
        """Test that async_retry raises RetryError when exhausted."""
        @async_retry(stop=stop_after_attempt(2), wait=wait_fixed(0.01))
        async def always_fails():
            raise ValueError("fail")

        with pytest.raises(RetryError):
            await always_fails()

    @pytest.mark.asyncio
    async def test_async_retry_preserves_metadata(self):
        """Test that async_retry preserves function metadata."""
        @async_retry(stop=stop_after_attempt(3))
        async def add(a, b):
            """Add two numbers."""
            return a + b

        assert add.__name__ == "add"
        assert "Add two numbers" in add.__doc__

    @pytest.mark.asyncio
    async def test_async_retry_with_multiple_args(self):
        """Test async_retry with multiple arguments."""
        call_count = [0]

        @async_retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        async def compute(a, b, c):
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return a + b + c

        result = await compute(1, 2, 3)
        assert result == 6

    @pytest.mark.asyncio
    async def test_async_retry_with_kwargs(self):
        """Test async_retry with keyword arguments."""
        @async_retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        async def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = await greet(name="Alice", greeting="Hi")
        assert result == "Hi, Alice!"


class TestAsyncRetryTiming:
    """Tests for async retry timing behavior."""

    @pytest.mark.asyncio
    async def test_async_retry_respects_wait_time(self):
        """Test that async_retry respects wait strategy."""
        import time
        call_count = [0]

        @async_retry(stop=stop_after_attempt(3), wait=wait_fixed(0.05))
        async def fails_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("fail")
            return "success"

        start = time.time()
        result = await fails_twice()
        elapsed = time.time() - start

        assert result == "success"
        # Should wait at least 0.09 seconds (2 waits * 0.05)
        assert elapsed >= 0.08


class TestAsyncRetryStatistics:
    """Tests for async retry statistics."""

    @pytest.mark.asyncio
    async def test_async_retry_statistics(self):
        """Test that async_retry tracks statistics."""
        call_count = [0]

        @async_retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        async def compute():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        result = await compute()
        stats = compute.retry.statistics

        assert "start_time" in stats
        assert "attempt_number" in stats


class TestAsyncConcurrentRetries:
    """Tests for concurrent async retries."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_async_retries(self):
        """Test multiple concurrent async retries."""
        call_counts = [0, 0, 0]

        @async_retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        async def fails_then_succeeds(idx):
            call_counts[idx] += 1
            if call_counts[idx] < 2:
                raise ValueError("fail")
            return f"success_{idx}"

        results = await asyncio.gather(
            fails_then_succeeds(0),
            fails_then_succeeds(1),
            fails_then_succeeds(2)
        )

        assert results == ["success_0", "success_1", "success_2"]
        # Each function should be called twice
        assert all(count == 2 for count in call_counts)

    @pytest.mark.asyncio
    async def test_async_retrying_concurrent_calls(self):
        """Test AsyncRetrying with concurrent function calls."""
        retrying = AsyncRetrying(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        call_count = [0, 0]

        async def fails_then_succeeds(idx):
            call_count[idx] += 1
            if call_count[idx] < 2:
                raise ValueError("fail")
            return f"success_{idx}"

        results = await asyncio.gather(
            retrying(fails_then_succeeds, 0),
            retrying(fails_then_succeeds, 1)
        )

        assert results == ["success_0", "success_1"]
