# cachetools: public API (signatures and documentation)


## module `src/cachetools/__init__.py`

```python
class Cache(collections.abc.MutableMapping)
```
Mutable mapping to serve as a simple cache or cache base class.

```python
# method of Cache
def __init__(self, maxsize, getsizeof=None)
```


```python
# method of Cache
def get(self, key, default=None)
```


```python
# method of Cache
def pop(self, key, default=__marker)
```


```python
# method of Cache
def setdefault(self, key, default=None)
```


```python
# method of Cache
def clear(self)
```


```python
# method of Cache
def maxsize(self)
```
The maximum size of the cache.

```python
# method of Cache
def currsize(self)
```
The current size of the cache.

```python
# method of Cache
def getsizeof(value)
```
Return the size of a cache element's value.

```python
class FIFOCache(Cache)
```
First In First Out (FIFO) cache implementation.

```python
# method of FIFOCache
def __init__(self, maxsize, getsizeof=None)
```


```python
# method of FIFOCache
def popitem(self)
```
Remove and return the `(key, value)` pair first inserted.

```python
# method of FIFOCache
def clear(self)
```


```python
class LFUCache(Cache)
```
Least Frequently Used (LFU) cache implementation.

```python
# method of LFUCache
def __init__(self, maxsize, getsizeof=None)
```


```python
# method of LFUCache
def popitem(self)
```
Remove and return the `(key, value)` pair least frequently used.

```python
# method of LFUCache
def clear(self)
```


```python
class LRUCache(Cache)
```
Least Recently Used (LRU) cache implementation.

```python
# method of LRUCache
def __init__(self, maxsize, getsizeof=None)
```


```python
# method of LRUCache
def popitem(self)
```
Remove and return the `(key, value)` pair least recently used.

```python
# method of LRUCache
def clear(self)
```


```python
class RRCache(Cache)
```
Random Replacement (RR) cache implementation.

```python
# method of RRCache
def __init__(self, maxsize, choice=random.choice, getsizeof=None)
```


```python
# method of RRCache
def choice(self)
```
The `choice` function used by the cache.

```python
# method of RRCache
def popitem(self)
```
Remove and return a random `(key, value)` pair.

```python
# method of RRCache
def clear(self)
```


```python
class TTLCache(_TimedCache)
```
LRU Cache implementation with per-item time-to-live (TTL) value.

```python
# method of TTLCache
def __init__(self, maxsize, ttl, timer=time.monotonic, getsizeof=None)
```


```python
# method of TTLCache
def ttl(self)
```
The time-to-live value of the cache's items.

```python
# method of TTLCache
def expire(self, time=None)
```
Remove expired items from the cache and return an iterable of the
expired `(key, value)` pairs.

```python
# method of TTLCache
def popitem(self)
```
Remove and return the `(key, value)` pair least recently used that
has not already expired.

```python
# method of TTLCache
def clear(self)
```


```python
class TLRUCache(_TimedCache)
```
Time aware Least Recently Used (TLRU) cache implementation.

```python
# method of TLRUCache
def __init__(self, maxsize, ttu, timer=time.monotonic, getsizeof=None)
```


```python
# method of TLRUCache
def ttu(self)
```
The local time-to-use function used by the cache.

```python
# method of TLRUCache
def expire(self, time=None)
```
Remove expired items from the cache and return an iterable of the
expired `(key, value)` pairs.

```python
# method of TLRUCache
def popitem(self)
```
Remove and return the `(key, value)` pair least recently used that
has not already expired.

```python
# method of TLRUCache
def clear(self)
```


```python
def cached(cache, key=keys.hashkey, lock=None, condition=None, info=False)
```
Decorator to wrap a function with a memoizing callable that saves
results in a cache.

```python
def cachedmethod(cache, key=keys.methodkey, lock=None, condition=None, info=False)
```
Decorator to wrap a method with a memoizing callable that saves
results in a cache.


## module `src/cachetools/func.py`

```python
def fifo_cache(maxsize=128, typed=False)
```
Decorator to wrap a function with a memoizing callable that saves
up to `maxsize` results based on a First In First Out (FIFO)
algorithm.

```python
def lfu_cache(maxsize=128, typed=False)
```
Decorator to wrap a function with a memoizing callable that saves
up to `maxsize` results based on a Least Frequently Used (LFU)
algorithm.

```python
def lru_cache(maxsize=128, typed=False)
```
Decorator to wrap a function with a memoizing callable that saves
up to `maxsize` results based on a Least Recently Used (LRU)
algorithm.

```python
def rr_cache(maxsize=128, choice=random.choice, typed=False)
```
Decorator to wrap a function with a memoizing callable that saves
up to `maxsize` results based on a Random Replacement (RR)
algorithm.

```python
def ttl_cache(maxsize=128, ttl=600, timer=time.monotonic, typed=False)
```
Decorator to wrap a function with a memoizing callable that saves
up to `maxsize` results based on a Least Recently Used (LRU)
algorithm with a per-item time-to-live (TTL) value.


## module `src/cachetools/keys.py`

```python
def hashkey(*args, **kwargs)
```
Return a cache key for the specified hashable arguments.

```python
def methodkey(self, *args, **kwargs)
```
Return a cache key for use with cached methods.

```python
def typedkey(*args, **kwargs)
```
Return a typed cache key for the specified hashable arguments.

```python
def typedmethodkey(self, *args, **kwargs)
```
Return a typed cache key for use with cached methods.


# Usage examples from the project README

```python
from cachetools import cached, LRUCache, TTLCache

# speed up calculating Fibonacci numbers with dynamic programming
@cached(cache={})
def fib(n):
   return n if n < 2 else fib(n - 1) + fib(n - 2)

# cache least recently used Python Enhancement Proposals
@cached(cache=LRUCache(maxsize=32))
def get_pep(num):
   url = 'http://www.python.org/dev/peps/pep-%04d/' % num
   with urllib.request.urlopen(url) as s:
       return s.read()

# cache weather data for no longer than ten minutes
@cached(cache=TTLCache(maxsize=1024, ttl=600))
def get_weather(place):
   return owm.weather_at_place(place).get_weather()
```