"""Tests for tenacity wait strategies with timing verification."""

import pytest
from tenacity import (
    retry, stop_after_attempt, Retrying,
    wait_fixed, wait_random, wait_incrementing, wait_exponential,
    wait_random_exponential, wait_chain, wait_combine, wait_none,
    wait_exponential_jitter, RetryError
)
import time


class TestWaitFixed:
    """Tests for wait_fixed strategy."""

    def test_wait_fixed_constant_delay(self):
        """Test that wait_fixed returns constant delay."""
        wait = wait_fixed(0.05)

        class MockState:
            attempt_number = 1

        delay = wait(MockState())
        assert delay == pytest.approx(0.05, abs=0.01)

    def test_wait_fixed_timing(self):
        """Test actual wait timing with wait_fixed."""
        @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.05))
        def fails_twice():
            if not hasattr(fails_twice, 'called'):
                fails_twice.called = 0
            fails_twice.called += 1
            if fails_twice.called < 3:
                raise ValueError("retry")
            return "success"

        start = time.time()
        result = fails_twice()
        elapsed = time.time() - start

        assert result == "success"
        # Should wait ~0.1 seconds (2 retries * 0.05)
        assert elapsed >= 0.08


class TestWaitNone:
    """Tests for wait_none strategy."""

    def test_wait_none_zero_delay(self):
        """Test that wait_none returns 0 delay."""
        wait = wait_none()

        class MockState:
            attempt_number = 1

        delay = wait(MockState())
        assert delay == 0

    def test_wait_none_timing(self):
        """Test that wait_none is fast."""
        @retry(stop=stop_after_attempt(3), wait=wait_none())
        def fails_twice():
            if not hasattr(fails_twice, 'called'):
                fails_twice.called = 0
            fails_twice.called += 1
            if fails_twice.called < 3:
                raise ValueError("retry")
            return "success"

        start = time.time()
        result = fails_twice()
        elapsed = time.time() - start

        assert result == "success"
        # Should be very fast (no wait)
        assert elapsed < 0.1


class TestWaitRandom:
    """Tests for wait_random strategy."""

    def test_wait_random_range(self):
        """Test that wait_random returns value in range."""
        wait = wait_random(min=0.01, max=0.1)

        class MockState:
            attempt_number = 1

        delays = [wait(MockState()) for _ in range(10)]

        for delay in delays:
            assert 0.01 <= delay <= 0.1

    def test_wait_random_default_range(self):
        """Test wait_random with default range."""
        wait = wait_random()

        class MockState:
            attempt_number = 1

        delay = wait(MockState())
        assert 0 <= delay <= 1

    def test_wait_random_has_jitter(self):
        """Test that wait_random has variation."""
        wait = wait_random(min=0.01, max=0.1)

        class MockState:
            attempt_number = 1

        delays = [wait(MockState()) for _ in range(20)]

        # Check that delays vary (have jitter)
        assert len(set(delays)) > 1


class TestWaitIncrementing:
    """Tests for wait_incrementing strategy."""

    def test_wait_incrementing_increases(self):
        """Test that wait_incrementing increases over attempts."""
        wait = wait_incrementing(start=0, increment=0.05)

        class MockState:
            pass

        state1 = MockState()
        state1.attempt_number = 1
        delay1 = wait(state1)

        state2 = MockState()
        state2.attempt_number = 2
        delay2 = wait(state2)

        state3 = MockState()
        state3.attempt_number = 3
        delay3 = wait(state3)

        assert delay1 < delay2 < delay3

    def test_wait_incrementing_max_limit(self):
        """Test that wait_incrementing respects max."""
        wait = wait_incrementing(start=0, increment=0.05, max=0.1)

        class MockState:
            attempt_number = 10

        delay = wait(MockState())
        assert delay <= 0.1


class TestWaitExponential:
    """Tests for wait_exponential strategy."""

    def test_wait_exponential_increases(self):
        """Test that wait_exponential increases exponentially."""
        wait = wait_exponential(multiplier=1, exp_base=2)

        class MockState:
            pass

        delays = []
        for attempt in range(1, 5):
            state = MockState()
            state.attempt_number = attempt
            delays.append(wait(state))

        # Check exponential growth
        assert delays[0] < delays[1] < delays[2] < delays[3]

    def test_wait_exponential_max_limit(self):
        """Test that wait_exponential respects max."""
        wait = wait_exponential(multiplier=1, max=0.5)

        class MockState:
            attempt_number = 10

        delay = wait(MockState())
        assert delay <= 0.5

    def test_wait_exponential_with_custom_base(self):
        """Test wait_exponential with custom base."""
        wait = wait_exponential(multiplier=1, exp_base=3)

        class MockState:
            pass

        state = MockState()
        state.attempt_number = 2
        delay = wait(state)
        # 1 * 3^(2-1) = 3
        assert delay == pytest.approx(3, abs=0.1)


class TestWaitRandomExponential:
    """Tests for wait_random_exponential strategy."""

    def test_wait_random_exponential_range(self):
        """Test that wait_random_exponential returns random value in range."""
        wait = wait_random_exponential(multiplier=1, max=1.0)

        class MockState:
            attempt_number = 3

        delays = [wait(MockState()) for _ in range(10)]

        for delay in delays:
            assert 0 <= delay <= 1.0

    def test_wait_random_exponential_has_jitter(self):
        """Test that wait_random_exponential has variation."""
        wait = wait_random_exponential(multiplier=0.1, max=1.0)

        class MockState:
            attempt_number = 3

        delays = [wait(MockState()) for _ in range(20)]

        # Check that delays vary
        assert len(set(delays)) > 1

    def test_wait_random_exponential_in_timing_test(self):
        """Test wait_random_exponential in real retry scenario."""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_random_exponential(multiplier=0.01, max=0.2)
        )
        def fails_twice():
            if not hasattr(fails_twice, 'called'):
                fails_twice.called = 0
            fails_twice.called += 1
            if fails_twice.called < 3:
                raise ValueError("retry")
            return "success"

        start = time.time()
        result = fails_twice()
        elapsed = time.time() - start

        assert result == "success"
        # Should have some wait time
        assert elapsed >= 0.01


class TestWaitChain:
    """Tests for wait_chain strategy."""

    def test_wait_chain_sequences_strategies(self):
        """Test that wait_chain sequences strategies."""
        from tenacity import wait_fixed

        wait = wait_chain(wait_fixed(0.01), wait_fixed(0.02), wait_fixed(0.03))

        class MockState:
            pass

        state1 = MockState()
        state1.attempt_number = 1
        delay1 = wait(state1)
        assert delay1 == pytest.approx(0.01, abs=0.001)

        state2 = MockState()
        state2.attempt_number = 2
        delay2 = wait(state2)
        assert delay2 == pytest.approx(0.02, abs=0.001)

        state3 = MockState()
        state3.attempt_number = 3
        delay3 = wait(state3)
        assert delay3 == pytest.approx(0.03, abs=0.001)

        state4 = MockState()
        state4.attempt_number = 4
        delay4 = wait(state4)
        # Should use last strategy
        assert delay4 == pytest.approx(0.03, abs=0.001)

    def test_wait_chain_requires_at_least_one_strategy(self):
        """Test that wait_chain requires at least one strategy."""
        with pytest.raises(ValueError):
            wait_chain()


class TestWaitCombine:
    """Tests for wait_combine strategy."""

    def test_wait_combine_sums_strategies(self):
        """Test that wait_combine sums strategies."""
        wait = wait_combine(wait_fixed(0.01), wait_fixed(0.02))

        class MockState:
            attempt_number = 1

        delay = wait(MockState())
        assert delay == pytest.approx(0.03, abs=0.001)

    def test_wait_combine_with_plus_operator(self):
        """Test wait_combine with + operator."""
        wait = wait_fixed(0.01) + wait_fixed(0.02)

        class MockState:
            attempt_number = 1

        delay = wait(MockState())
        assert delay == pytest.approx(0.03, abs=0.001)

    def test_wait_combine_multiple_strategies(self):
        """Test wait_combine with multiple strategies."""
        wait = wait_fixed(0.01) + wait_fixed(0.02) + wait_fixed(0.03)

        class MockState:
            attempt_number = 1

        delay = wait(MockState())
        assert delay == pytest.approx(0.06, abs=0.001)


class TestWaitExponentialJitter:
    """Tests for wait_exponential_jitter strategy."""

    def test_wait_exponential_jitter_range(self):
        """Test that wait_exponential_jitter returns value in range."""
        wait = wait_exponential_jitter(initial=1, max=10)

        class MockState:
            attempt_number = 2

        delays = [wait(MockState()) for _ in range(10)]

        for delay in delays:
            assert 0 <= delay <= 10


class TestWaitWithRetrying:
    """Tests for wait strategies in Retrying context."""

    def test_wait_with_retrying_object(self):
        """Test wait strategy with Retrying object."""
        retrying = Retrying(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        call_count = [0]

        def sometimes_fails():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("fail")
            return "success"

        start = time.time()
        result = retrying(sometimes_fails)
        elapsed = time.time() - start

        assert result == "success"
        assert elapsed >= 0.009

    def test_multiple_wait_strategies_in_retry(self):
        """Test using different wait strategies in retry."""
        strategies = [
            wait_fixed(0.01),
            wait_random(min=0.005, max=0.02),
            wait_incrementing(start=0, increment=0.01),
        ]

        for strategy in strategies:
            @retry(stop=stop_after_attempt(2), wait=strategy)
            def fails_once():
                if not hasattr(fails_once, 'called'):
                    fails_once.called = 0
                fails_once.called += 1
                if fails_once.called < 2:
                    raise ValueError("fail")
                return "success"

            result = fails_once()
            assert result == "success"
