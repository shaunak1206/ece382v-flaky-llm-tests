"""Tests for cachetools.cached decorator and related functionality."""

import pytest
from cachetools import cached, Cache, LRUCache, TTLCache
from cachetools import keys
from threading import Lock, Condition
import time


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_cached_basic_caching(self):
        """Test that cached decorator caches function results."""
        call_count = [0]

        @cached(cache=LRUCache(maxsize=10))
        def fibonacci(n):
            call_count[0] += 1
            if n < 2:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        result1 = fibonacci(5)
        call_count_after_first = call_count[0]
        result2 = fibonacci(5)
        # Should not call function again for same argument
        assert call_count[0] == call_count_after_first
        assert result1 == result2

    def test_cached_with_multiple_arguments(self):
        """Test cached decorator with multiple arguments."""
        @cached(cache=LRUCache(maxsize=10))
        def add(a, b):
            return a + b

        assert add(1, 2) == 3
        assert add(2, 3) == 5
        assert add(1, 2) == 3  # Cached

    def test_cached_with_kwargs(self):
        """Test cached decorator with keyword arguments."""
        @cached(cache=LRUCache(maxsize=10))
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result1 = greet("Alice", greeting="Hi")
        result2 = greet("Alice", greeting="Hi")
        assert result1 == result2

    def test_cached_has_cache_clear(self):
        """Test that cached function has cache_clear method."""
        @cached(cache=LRUCache(maxsize=10))
        def add(a, b):
            return a + b

        assert hasattr(add, "cache_clear")
        add.cache_clear()

    def test_cached_with_lock(self):
        """Test cached decorator with thread lock."""
        lock = Lock()
        call_count = [0]

        @cached(cache=LRUCache(maxsize=10), lock=lock)
        def increment():
            call_count[0] += 1
            return call_count[0]

        result = increment()
        assert result == 1
        result2 = increment()
        # Should be cached
        assert result2 == 1

    def test_cached_with_condition(self):
        """Test cached decorator with condition variable."""
        cond = Condition()
        call_count = [0]

        @cached(cache=LRUCache(maxsize=10), lock=Lock(), condition=cond)
        def compute():
            call_count[0] += 1
            return call_count[0]

        result = compute()
        assert result == 1

    def test_cached_with_custom_key_function(self):
        """Test cached decorator with custom key function."""
        @cached(cache=LRUCache(maxsize=10), key=keys.hashkey)
        def add(a, b):
            return a + b

        result = add(1, 2)
        assert result == 3

    def test_cached_info_enabled(self):
        """Test cache_info() when info=True."""
        @cached(cache=LRUCache(maxsize=10), info=True)
        def add(a, b):
            return a + b

        add(1, 2)
        add(1, 2)
        add(2, 3)
        info = add.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.maxsize == 10

    def test_cached_info_basic_dict(self):
        """Test cache_info() with basic dict cache."""
        @cached(cache={}, info=True)
        def add(a, b):
            return a + b

        add(1, 2)
        add(1, 2)
        info = add.cache_info()
        assert info.hits == 1
        assert info.misses == 1
        # Basic dict cache has None maxsize
        assert info.maxsize is None

    def test_cached_clear_resets_stats(self):
        """Test that cache_clear() resets stats when info=True."""
        @cached(cache=LRUCache(maxsize=10), info=True)
        def add(a, b):
            return a + b

        add(1, 2)
        add(1, 2)
        info_before = add.cache_info()
        add.cache_clear()
        info_after = add.cache_info()
        assert info_after.hits == 0
        assert info_after.misses == 0

    def test_cached_with_none_cache(self):
        """Test cached decorator with None cache (uncached)."""
        call_count = [0]

        @cached(cache=None, info=True)
        def increment():
            call_count[0] += 1
            return call_count[0]

        result1 = increment()
        result2 = increment()
        # Not cached
        assert result1 != result2

    def test_cached_function_metadata_preserved(self):
        """Test that cached decorator preserves function metadata."""
        @cached(cache=LRUCache(maxsize=10))
        def add(a, b):
            """Add two numbers."""
            return a + b

        assert add.__name__ == "add"
        assert "Add two numbers" in add.__doc__

    def test_cached_with_ttl_cache(self):
        """Test cached decorator with TTL cache."""
        @cached(cache=TTLCache(maxsize=10, ttl=0.1))
        def compute():
            return time.time()

        result1 = compute()
        result2 = compute()
        assert result1 == result2
        time.sleep(0.2)
        result3 = compute()
        # After TTL, should compute again
        assert result3 > result1

    def test_cached_with_dict_cache(self):
        """Test cached decorator with plain dict cache."""
        @cached(cache={})
        def add(a, b):
            return a + b

        result1 = add(1, 2)
        result2 = add(1, 2)
        assert result1 == result2

    def test_cached_handles_exceptions(self):
        """Test that exceptions are not cached."""
        call_count = [0]

        @cached(cache=LRUCache(maxsize=10))
        def sometimes_fails():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First call fails")
            return "success"

        with pytest.raises(ValueError):
            sometimes_fails()

        result = sometimes_fails()
        assert result == "success"
        # Function was called twice
        assert call_count[0] == 2

    def test_cached_value_too_large_silently_ignored(self):
        """Test that values too large to cache are silently ignored."""
        call_count = [0]

        def custom_getsizeof(value):
            return 1000 if isinstance(value, str) else 1

        @cached(cache=Cache(maxsize=10, getsizeof=custom_getsizeof))
        def return_large():
            call_count[0] += 1
            return "large value"

        result1 = return_large()
        result2 = return_large()
        # Value too large, so function is called again
        assert call_count[0] == 2


class TestTypedCacheKey:
    """Tests for typed cache key function."""

    def test_typedkey_distinguishes_types(self):
        """Test that typedkey distinguishes between different types."""
        key1 = keys.typedkey(1)
        key2 = keys.typedkey(1.0)
        assert key1 != key2

    def test_typedkey_with_strings_and_ints(self):
        """Test typedkey with strings and ints."""
        key1 = keys.typedkey("1")
        key2 = keys.typedkey(1)
        assert key1 != key2

    def test_typedkey_with_kwargs(self):
        """Test typedkey with keyword arguments."""
        key1 = keys.typedkey(a=1)
        key2 = keys.typedkey(a=1.0)
        assert key1 != key2

    def test_cached_with_typed_key(self):
        """Test cached decorator with typed key."""
        @cached(cache=LRUCache(maxsize=10), key=keys.typedkey)
        def process(x):
            return x * 2

        result_int = process(5)
        result_float = process(5.0)
        # Different types should have different results
        assert result_int == 10
        assert result_float == 10.0


class TestHashKey:
    """Tests for hashkey function."""

    def test_hashkey_basic(self):
        """Test basic hashkey functionality."""
        key1 = keys.hashkey(1, 2, 3)
        key2 = keys.hashkey(1, 2, 3)
        assert key1 == key2

    def test_hashkey_with_kwargs(self):
        """Test hashkey with keyword arguments."""
        key1 = keys.hashkey(a=1, b=2)
        key2 = keys.hashkey(a=1, b=2)
        assert key1 == key2

    def test_hashkey_kwargs_order_independent(self):
        """Test that hashkey kwargs order doesn't matter."""
        key1 = keys.hashkey(a=1, b=2)
        key2 = keys.hashkey(b=2, a=1)
        assert key1 == key2

    def test_hashkey_with_args_and_kwargs(self):
        """Test hashkey with both args and kwargs."""
        key1 = keys.hashkey(1, 2, a=3, b=4)
        key2 = keys.hashkey(1, 2, a=3, b=4)
        assert key1 == key2


class TestMethodKey:
    """Tests for methodkey function."""

    def test_methodkey_ignores_self(self):
        """Test that methodkey ignores the self parameter."""
        obj1 = object()
        obj2 = object()
        key1 = keys.methodkey(obj1, 1, 2, 3)
        key2 = keys.methodkey(obj2, 1, 2, 3)
        assert key1 == key2

    def test_methodkey_with_args(self):
        """Test methodkey with arguments."""
        key1 = keys.methodkey(object(), 1, 2)
        key2 = keys.methodkey(object(), 1, 2)
        assert key1 == key2

    def test_methodkey_with_kwargs(self):
        """Test methodkey with kwargs."""
        key1 = keys.methodkey(object(), a=1, b=2)
        key2 = keys.methodkey(object(), a=1, b=2)
        assert key1 == key2


class TestCachedMethodDecorator:
    """Tests for @cachedmethod decorator."""

    def test_cachedmethod_basic_caching(self):
        """Test basic cachedmethod functionality."""
        from cachetools import cachedmethod

        class MyClass:
            def __init__(self):
                self._cache = LRUCache(maxsize=10)
                self.call_count = 0

            @cachedmethod(lambda self: self._cache)
            def compute(self, x):
                self.call_count += 1
                return x * 2

        obj = MyClass()
        result1 = obj.compute(5)
        result2 = obj.compute(5)
        assert result1 == result2 == 10
        assert obj.call_count == 1

    def test_cachedmethod_separate_caches_per_instance(self):
        """Test that cachedmethod has separate caches per instance."""
        from cachetools import cachedmethod

        class MyClass:
            def __init__(self):
                self._cache = LRUCache(maxsize=10)
                self.call_count = 0

            @cachedmethod(lambda self: self._cache)
            def compute(self, x):
                self.call_count += 1
                return x * 2

        obj1 = MyClass()
        obj2 = MyClass()
        obj1.compute(5)
        obj2.compute(5)
        # Each instance has its own cache
        assert obj1.call_count == 1
        assert obj2.call_count == 1

    def test_cachedmethod_with_cache_clear(self):
        """Test that cachedmethod has cache_clear method."""
        from cachetools import cachedmethod

        class MyClass:
            def __init__(self):
                self._cache = LRUCache(maxsize=10)

            @cachedmethod(lambda self: self._cache)
            def compute(self, x):
                return x * 2

        obj = MyClass()
        obj.compute(5)
        assert hasattr(obj.compute, "cache_clear")
