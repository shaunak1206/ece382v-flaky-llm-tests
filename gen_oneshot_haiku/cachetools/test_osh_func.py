"""Tests for cachetools.func decorators: lru_cache, ttl_cache, etc."""

import pytest
from cachetools import lru_cache, lfu_cache, fifo_cache, rr_cache, ttl_cache
import time


class TestLruCache:
    """Tests for @lru_cache decorator."""

    def test_lru_cache_basic(self):
        """Test basic lru_cache functionality."""
        @lru_cache(maxsize=128)
        def add(a, b):
            return a + b

        result1 = add(1, 2)
        result2 = add(1, 2)
        assert result1 == result2

    def test_lru_cache_default_maxsize(self):
        """Test lru_cache with default maxsize."""
        @lru_cache()
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12

    def test_lru_cache_maxsize_none(self):
        """Test lru_cache with maxsize=None (unlimited)."""
        @lru_cache(maxsize=None)
        def square(x):
            return x * x

        assert square(5) == 25
        assert square(5) == 25

    def test_lru_cache_direct_decoration(self):
        """Test lru_cache as direct decorator without parens."""
        @lru_cache
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_lru_cache_has_cache_info(self):
        """Test that lru_cache has cache_info method."""
        @lru_cache(maxsize=128)
        def add(a, b):
            return a + b

        add(1, 2)
        add(1, 2)
        add(2, 3)
        info = add.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.maxsize == 128

    def test_lru_cache_has_cache_clear(self):
        """Test that lru_cache has cache_clear method."""
        @lru_cache(maxsize=128)
        def add(a, b):
            return a + b

        add(1, 2)
        add.cache_clear()
        info = add.cache_info()
        assert info.hits == 0
        assert info.misses == 0

    def test_lru_cache_has_cache_parameters(self):
        """Test that lru_cache has cache_parameters method."""
        @lru_cache(maxsize=64, typed=True)
        def add(a, b):
            return a + b

        params = add.cache_parameters()
        assert params["maxsize"] == 64
        assert params["typed"] is True

    def test_lru_cache_typed_false(self):
        """Test lru_cache with typed=False."""
        @lru_cache(maxsize=128, typed=False)
        def add(a, b):
            return a + b

        result_int = add(1, 2)
        result_float = add(1.0, 2.0)
        # Same result for equivalent args with typed=False
        assert result_int == result_float

    def test_lru_cache_typed_true(self):
        """Test lru_cache with typed=True."""
        call_count = [0]

        @lru_cache(maxsize=128, typed=True)
        def add(a, b):
            call_count[0] += 1
            return a + b

        result_int = add(1, 2)
        result_float = add(1.0, 2.0)
        # Different cache entries for different types with typed=True
        assert call_count[0] == 2

    def test_lru_cache_with_exception(self):
        """Test that lru_cache doesn't cache exceptions."""
        call_count = [0]

        @lru_cache(maxsize=128)
        def sometimes_fails(x):
            call_count[0] += 1
            if x == 1:
                raise ValueError("x=1 fails")
            return x * 2

        with pytest.raises(ValueError):
            sometimes_fails(1)

        result = sometimes_fails(1)
        # Exception not cached, function called again
        assert call_count[0] == 2


class TestTtlCache:
    """Tests for @ttl_cache decorator."""

    def test_ttl_cache_basic(self):
        """Test basic ttl_cache functionality."""
        @ttl_cache(maxsize=128, ttl=10)
        def add(a, b):
            return a + b

        result1 = add(1, 2)
        result2 = add(1, 2)
        assert result1 == result2

    def test_ttl_cache_expiration(self):
        """Test that ttl_cache items expire."""
        @ttl_cache(maxsize=128, ttl=0.1)
        def get_time():
            return time.time()

        time1 = get_time()
        time2 = get_time()
        assert time1 == time2
        time.sleep(0.2)
        time3 = get_time()
        # After TTL, should recompute
        assert time3 > time1

    def test_ttl_cache_default_ttl(self):
        """Test ttl_cache with default ttl."""
        @ttl_cache(maxsize=128)
        def compute():
            return 42

        assert compute() == 42

    def test_ttl_cache_default_maxsize(self):
        """Test ttl_cache with default maxsize."""
        @ttl_cache(ttl=10)
        def compute():
            return 42

        assert compute() == 42

    def test_ttl_cache_maxsize_none(self):
        """Test ttl_cache with maxsize=None."""
        @ttl_cache(maxsize=None, ttl=10)
        def compute():
            return 42

        assert compute() == 42

    def test_ttl_cache_custom_timer(self):
        """Test ttl_cache with custom timer."""
        timer_values = [0, 1, 2, 3, 4, 5]
        timer_idx = [0]

        def custom_timer():
            val = timer_values[timer_idx[0]]
            timer_idx[0] += 1
            return val

        @ttl_cache(maxsize=128, ttl=2, timer=custom_timer)
        def compute():
            return time.time()

        result = compute()
        assert result is not None

    def test_ttl_cache_has_cache_info(self):
        """Test that ttl_cache has cache_info method."""
        @ttl_cache(maxsize=128, ttl=10)
        def add(a, b):
            return a + b

        add(1, 2)
        add(1, 2)
        info = add.cache_info()
        assert info.hits == 1
        assert info.misses == 1

    def test_ttl_cache_typed_parameter(self):
        """Test ttl_cache with typed parameter."""
        @ttl_cache(maxsize=128, ttl=10, typed=True)
        def add(a, b):
            return a + b

        result_int = add(1, 2)
        result_float = add(1.0, 2.0)
        assert result_int == 3
        assert result_float == 3.0


class TestLfuCache:
    """Tests for @lfu_cache decorator."""

    def test_lfu_cache_basic(self):
        """Test basic lfu_cache functionality."""
        @lfu_cache(maxsize=128)
        def add(a, b):
            return a + b

        result1 = add(1, 2)
        result2 = add(1, 2)
        assert result1 == result2

    def test_lfu_cache_default_maxsize(self):
        """Test lfu_cache with default maxsize."""
        @lfu_cache()
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12

    def test_lfu_cache_maxsize_none(self):
        """Test lfu_cache with maxsize=None."""
        @lfu_cache(maxsize=None)
        def square(x):
            return x * x

        assert square(5) == 25

    def test_lfu_cache_direct_decoration(self):
        """Test lfu_cache as direct decorator."""
        @lfu_cache
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_lfu_cache_has_cache_info(self):
        """Test that lfu_cache has cache_info method."""
        @lfu_cache(maxsize=128)
        def add(a, b):
            return a + b

        add(1, 2)
        add(1, 2)
        add(2, 3)
        info = add.cache_info()
        assert info.hits == 1
        assert info.misses == 2


class TestFifoCache:
    """Tests for @fifo_cache decorator."""

    def test_fifo_cache_basic(self):
        """Test basic fifo_cache functionality."""
        @fifo_cache(maxsize=128)
        def add(a, b):
            return a + b

        result1 = add(1, 2)
        result2 = add(1, 2)
        assert result1 == result2

    def test_fifo_cache_default_maxsize(self):
        """Test fifo_cache with default maxsize."""
        @fifo_cache()
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12

    def test_fifo_cache_maxsize_none(self):
        """Test fifo_cache with maxsize=None."""
        @fifo_cache(maxsize=None)
        def square(x):
            return x * x

        assert square(5) == 25

    def test_fifo_cache_direct_decoration(self):
        """Test fifo_cache as direct decorator."""
        @fifo_cache
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_fifo_cache_has_cache_info(self):
        """Test that fifo_cache has cache_info method."""
        @fifo_cache(maxsize=128)
        def add(a, b):
            return a + b

        add(1, 2)
        add(1, 2)
        add(2, 3)
        info = add.cache_info()
        assert info.hits == 1


class TestRrCache:
    """Tests for @rr_cache decorator."""

    def test_rr_cache_basic(self):
        """Test basic rr_cache functionality."""
        @rr_cache(maxsize=128)
        def add(a, b):
            return a + b

        result1 = add(1, 2)
        result2 = add(1, 2)
        assert result1 == result2

    def test_rr_cache_default_maxsize(self):
        """Test rr_cache with default maxsize."""
        @rr_cache()
        def multiply(a, b):
            return a * b

        assert multiply(3, 4) == 12

    def test_rr_cache_maxsize_none(self):
        """Test rr_cache with maxsize=None."""
        @rr_cache(maxsize=None)
        def square(x):
            return x * x

        assert square(5) == 25

    def test_rr_cache_direct_decoration(self):
        """Test rr_cache as direct decorator."""
        @rr_cache
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_rr_cache_custom_choice(self):
        """Test rr_cache with custom choice function."""
        def first_choice(items):
            return items[0]

        @rr_cache(maxsize=128, choice=first_choice)
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_rr_cache_has_cache_info(self):
        """Test that rr_cache has cache_info method."""
        @rr_cache(maxsize=128)
        def add(a, b):
            return a + b

        add(1, 2)
        add(1, 2)
        add(2, 3)
        info = add.cache_info()
        assert info.hits == 1


class TestFuncDecoratorCombinations:
    """Tests for combining decorator parameters."""

    def test_lru_cache_with_all_parameters(self):
        """Test lru_cache with all parameters specified."""
        @lru_cache(maxsize=64, typed=True)
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3
        params = add.cache_parameters()
        assert params["maxsize"] == 64
        assert params["typed"] is True

    def test_ttl_cache_with_all_parameters(self):
        """Test ttl_cache with all parameters specified."""
        @ttl_cache(maxsize=64, ttl=10, typed=True)
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_cache_info_always_available(self):
        """Test that cache_info is always available."""
        @lru_cache(maxsize=128)
        def func1():
            pass

        @ttl_cache(maxsize=128, ttl=10)
        def func2():
            pass

        @lfu_cache(maxsize=128)
        def func3():
            pass

        assert hasattr(func1, "cache_info")
        assert hasattr(func2, "cache_info")
        assert hasattr(func3, "cache_info")

    def test_cache_clear_always_available(self):
        """Test that cache_clear is always available."""
        @lru_cache(maxsize=128)
        def func1():
            pass

        @ttl_cache(maxsize=128, ttl=10)
        def func2():
            pass

        assert hasattr(func1, "cache_clear")
        assert hasattr(func2, "cache_clear")
