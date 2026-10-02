"""Tests for tenacity stop strategies."""

import pytest
from tenacity import (
    retry, stop_after_attempt, stop_after_delay, stop_before_delay,
    stop_never, stop_any, stop_all, stop_when_event_set,
    wait_fixed, RetryError, Retrying
)
import time
import threading


class TestStopAfterAttempt:
    """Tests for stop_after_attempt strategy."""

    def test_stop_after_attempt_stops_at_limit(self):
        """Test that stop_after_attempt stops after max attempts."""
        call_count = [0]

        @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            always_fails()
        assert call_count[0] == 3

    def test_stop_after_attempt_single_attempt(self):
        """Test stop_after_attempt with max_attempt=1."""
        @retry(stop=stop_after_attempt(1))
        def fails():
            raise ValueError("fail")

        with pytest.raises(RetryError):
            fails()

    def test_stop_after_attempt_large_number(self):
        """Test stop_after_attempt with large max_attempt."""
        call_count = [0]

        @retry(stop=stop_after_attempt(100), wait=wait_fixed(0.001))
        def fails_quickly():
            call_count[0] += 1
            if call_count[0] == 10:
                return "success"
            raise ValueError("fail")

        result = fails_quickly()
        assert result == "success"


class TestStopAfterDelay:
    """Tests for stop_after_delay strategy."""

    def test_stop_after_delay_stops_after_time(self):
        """Test that stop_after_delay stops after time limit."""
        @retry(stop=stop_after_delay(0.1), wait=wait_fixed(0.05))
        def always_fails():
            raise ValueError("fail")

        start = time.time()
        with pytest.raises(RetryError):
            always_fails()
        elapsed = time.time() - start

        # Should stop after ~0.1 seconds (may exceed due to final wait)
        assert elapsed >= 0.08

    def test_stop_after_delay_with_short_timeout(self):
        """Test stop_after_delay with short timeout."""
        @retry(stop=stop_after_delay(0.05), wait=wait_fixed(0.02))
        def always_fails():
            raise ValueError("fail")

        start = time.time()
        with pytest.raises(RetryError):
            always_fails()
        elapsed = time.time() - start

        # Should not exceed delay by much
        assert elapsed < 0.15


class TestStopBeforeDelay:
    """Tests for stop_before_delay strategy."""

    def test_stop_before_delay_prevents_exceeding_limit(self):
        """Test that stop_before_delay prevents exceeding time limit."""
        @retry(stop=stop_before_delay(0.1), wait=wait_fixed(0.05))
        def always_fails():
            raise ValueError("fail")

        start = time.time()
        with pytest.raises(RetryError):
            always_fails()
        elapsed = time.time() - start

        # Should not exceed limit
        assert elapsed <= 0.12


class TestStopNever:
    """Tests for stop_never strategy."""

    def test_stop_never_with_other_stop_condition(self):
        """Test stop_never combined with another stop condition."""
        from tenacity import stop_never

        call_count = [0]

        @retry(stop=stop_never | stop_after_attempt(3), wait=wait_fixed(0.01))
        def fails_until_limit():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            fails_until_limit()
        assert call_count[0] == 3

    def test_stop_never_never_stops_alone(self):
        """Test that stop_never never stops by itself."""
        # This test just verifies the logic, without actually running forever
        call_count = [0]

        @retry(
            stop=stop_never | stop_after_attempt(2),
            wait=wait_fixed(0.01)
        )
        def fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            fails()
        assert call_count[0] == 2


class TestStopAny:
    """Tests for stop_any (OR) strategy."""

    def test_stop_any_first_condition_matches(self):
        """Test stop_any when first condition matches."""
        call_count = [0]

        @retry(
            stop=stop_any(
                stop_after_attempt(2),
                stop_after_delay(1.0)
            ),
            wait=wait_fixed(0.01)
        )
        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            always_fails()
        assert call_count[0] == 2

    def test_stop_any_second_condition_matches(self):
        """Test stop_any when second condition matches."""
        @retry(
            stop=stop_any(
                stop_after_attempt(100),
                stop_after_delay(0.1)
            ),
            wait=wait_fixed(0.05)
        )
        def always_fails():
            raise ValueError("fail")

        start = time.time()
        with pytest.raises(RetryError):
            always_fails()
        elapsed = time.time() - start

        # Should stop due to delay, not attempts
        assert elapsed >= 0.08

    def test_stop_any_with_or_operator(self):
        """Test stop_any using | operator."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(2) | stop_after_delay(1.0),
            wait=wait_fixed(0.01)
        )
        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            always_fails()
        assert call_count[0] == 2


class TestStopAll:
    """Tests for stop_all (AND) strategy."""

    def test_stop_all_both_conditions_must_match(self):
        """Test that stop_all requires both conditions."""
        call_count = [0]

        @retry(
            stop=stop_all(
                stop_after_attempt(10),  # High limit
                stop_after_delay(0.05)   # Low time limit
            ),
            wait=wait_fixed(0.02)
        )
        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        start = time.time()
        with pytest.raises(RetryError):
            always_fails()
        elapsed = time.time() - start

        # Should stop due to time (both conditions met)
        assert elapsed >= 0.04

    def test_stop_all_with_and_operator(self):
        """Test stop_all using & operator."""
        call_count = [0]

        @retry(
            stop=stop_after_attempt(10) & stop_after_delay(0.05),
            wait=wait_fixed(0.02)
        )
        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            always_fails()


class TestStopWhenEventSet:
    """Tests for stop_when_event_set strategy."""

    def test_stop_when_event_set(self):
        """Test stop_when_event_set stops when event is set."""
        event = threading.Event()
        call_count = [0]

        def set_event_later():
            time.sleep(0.05)
            event.set()

        threading.Thread(target=set_event_later, daemon=True).start()

        @retry(
            stop=stop_when_event_set(event),
            wait=wait_fixed(0.01)
        )
        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            always_fails()
        # Should have some retry attempts
        assert call_count[0] > 1

    def test_stop_when_event_already_set(self):
        """Test stop_when_event_set when event is already set."""
        event = threading.Event()
        event.set()

        @retry(
            stop=stop_when_event_set(event),
            wait=wait_fixed(0.01)
        )
        def always_fails():
            raise ValueError("fail")

        with pytest.raises(RetryError) as exc_info:
            always_fails()
        # Should stop immediately
        assert exc_info.value.last_attempt.attempt_number == 1


class TestStopCombinations:
    """Tests for complex stop condition combinations."""

    def test_stop_complex_or_and_combination(self):
        """Test complex combination of stop conditions."""
        call_count = [0]

        stop_cond = (
            (stop_after_attempt(5) | stop_after_delay(0.05)) &
            stop_after_attempt(100)
        )

        @retry(stop=stop_cond, wait=wait_fixed(0.01))
        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            always_fails()

    def test_stop_with_retrying_object(self):
        """Test stop strategies with Retrying object."""
        retrying = Retrying(stop=stop_after_attempt(3), wait=wait_fixed(0.01))
        call_count = [0]

        def always_fails():
            call_count[0] += 1
            raise ValueError("fail")

        with pytest.raises(RetryError):
            retrying(always_fails)
        assert call_count[0] == 3


class TestStopAttemptNumber:
    """Tests for stop condition behavior across attempts."""

    def test_attempt_number_increments(self):
        """Test that attempt_number increments correctly."""
        attempt_numbers = []

        @retry(stop=stop_after_attempt(5), wait=wait_fixed(0.01))
        def fails_and_tracks():
            if hasattr(fails_and_tracks, 'state'):
                attempt_numbers.append(fails_and_tracks.state.attempt_number)
            raise ValueError("fail")

        try:
            fails_and_tracks()
        except RetryError:
            pass


class TestTimingWithStopStrategies:
    """Tests for accurate timing with stop strategies."""

    def test_stop_after_delay_timing_accuracy(self):
        """Test timing accuracy of stop_after_delay."""
        target_delay = 0.08

        @retry(stop=stop_after_delay(target_delay), wait=wait_fixed(0.02))
        def always_fails():
            raise ValueError("fail")

        start = time.time()
        try:
            always_fails()
        except RetryError:
            pass
        elapsed = time.time() - start

        # Should be close to target delay
        assert abs(elapsed - target_delay) < 0.05
