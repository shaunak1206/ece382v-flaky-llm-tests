# boltons: public API (signatures and documentation)


## module `boltons/cacheutils.py`

```python
class LRI(dict)
```
The ``LRI`` implements the basic *Least Recently Inserted* strategy to
caching. One could also think of this as a ``SizeLimitedDefaultDict``.

*on_miss* is a callable that accepts the missing key (as opposed
to :class:`collections.defaultdict`'s "default_factory", which
accepts no arguments.) Also note that, like the :class:`LRI`,
the ``LRI`` is instrumented with statistics tracking.

>>> cap_cache = LRI(max_size=2)
>>> cap_cache['a'], cap_cache['b'] = 'A', 'B'
>>> from pprint import pprint as pp
>>> pp(dict(cap_cache))
{'a': 'A', 'b': 'B'}
>>> [cap_cache['b'] for i in range(3)][0]
'B'
>>> cap_cache['c'] = 'C'
>>> print(cap_cache.get('a'))
None
>>> cap_cache.hit_count, cap_cache.miss_count, ...

```python
# method of LRI
def __init__(self, max_size=DEFAULT_MAX_SIZE, values=None, on_miss=None)
```


```python
# method of LRI
def get(self, key, default=None)
```


```python
# method of LRI
def pop(self, key, default=_MISSING)
```


```python
# method of LRI
def popitem(self)
```


```python
# method of LRI
def clear(self)
```


```python
# method of LRI
def copy(self)
```


```python
# method of LRI
def setdefault(self, key, default=None)
```


```python
# method of LRI
def update(self, E=(), **F)
```


```python
class LRU(LRI)
```
The ``LRU`` is :class:`dict` subtype implementation of the
*Least-Recently Used* caching strategy.

Args:
    max_size (int): Max number of items to cache. Defaults to ``128``.
    values (iterable): Initial values for the cache. Defaults to ``None``.
    on_miss (callable): a callable which accepts a single argument, the
        key not present in the cache, and returns the value to be cached.

>>> cap_cache = LRU(max_size=2)
>>> cap_cache['a'], cap_cache['b'] = 'A', 'B'
>>> from pprint import pprint as pp
>>> pp(dict(cap_cache))
{'a': 'A', 'b': 'B'}
>>> [cap_cache['b'] for i in range(3)][0]
'B'
>>> cap_cache['c'] = 'C'
>>> print(cap_cache.get('a'))
None

This cache is also instrumented wit ...

```python
def make_cache_key(args, kwargs, typed=False, kwarg_mark=_KWARG_MARK, fasttypes=frozenset([int, str, frozenset, type(None)]))
```
Make a generic key from a function's positional and keyword
arguments, suitable for use in caches. Arguments within *args* and
*kwargs* must be `hashable`_. If *typed* is ``True``, ``3`` and
``3.0`` will be treated as separate keys.

The key is constructed in a way that is flat as possible rather than
as a nested structure that would take more memory.

If there is only a single argument and its data type is known to cache
its hash value, then that argument is returned without a wrapper.  This
saves space and improves lookup speed.

>>> tuple(make_cache_key(('a', 'b'), {'c': ('d')}))
('a', 'b', _KWARG_MARK, ('c', 'd'))

.. _hashable: https://docs.python.org/2/glossary.html#term-hashable

```python
class CachedFunction()
```
This type is used by :func:`cached`, below. Instances of this
class are used to wrap functions in caching logic.

```python
# method of CachedFunction
def __init__(self, func, cache, scoped=True, typed=False, key=None)
```


```python
# method of CachedFunction
def __call__(self, *args, **kwargs)
```


```python
class CachedMethod()
```
Similar to :class:`CachedFunction`, this type is used by
:func:`cachedmethod` to wrap methods in caching logic.

```python
# method of CachedMethod
def __init__(self, func, cache, scoped=True, typed=False, key=None)
```


```python
# method of CachedMethod
def __call__(self, *args, **kwargs)
```


```python
def cached(cache, scoped=True, typed=False, key=None)
```
Cache any function with the cache object of your choosing. Note
that the function wrapped should take only `hashable`_ arguments.

Args:
    cache (Mapping): Any :class:`dict`-like object suitable for
        use as a cache. Instances of the :class:`LRU` and
        :class:`LRI` are good choices, but a plain :class:`dict`
        can work in some cases, as well. This argument can also be
        a callable which accepts no arguments and returns a mapping.
    scoped (bool): Whether the function itself is part of the
        cache key.  ``True`` by default, different functions will
        not read one another's cache entries, but can evict one
        another's results. ``False`` can be usef ...

```python
def cachedmethod(cache, scoped=True, typed=False, key=None)
```
Similar to :func:`cached`, ``cachedmethod`` is used to cache
methods based on their arguments, using any :class:`dict`-like
*cache* object.

Args:
    cache (str/Mapping/callable): Can be the name of an attribute
        on the instance, any Mapping/:class:`dict`-like object, or
        a callable which returns a Mapping.
    scoped (bool): Whether the method itself and the object it is
        bound to are part of the cache keys. ``True`` by default,
        different methods will not read one another's cache
        results. ``False`` can be useful for certain shared cache
        use cases. More advanced behavior can be produced through
        the *key* arguments.
    typed (bool): Wheth ...

```python
class cachedproperty()
```
The ``cachedproperty`` is used similar to :class:`property`, except
that the wrapped method is only called once. This is commonly used
to implement lazy attributes.

After the property has been accessed, the value is stored on the
instance itself, using the same name as the cachedproperty. This
allows the cache to be cleared with :func:`delattr`, or through
manipulating the object's ``__dict__``.

```python
# method of cachedproperty
def __init__(self, func)
```


```python
class ThresholdCounter()
```
A **bounded** dict-like Mapping from keys to counts. The
ThresholdCounter automatically compacts after every (1 /
*threshold*) additions, maintaining exact counts for any keys
whose count represents at least a *threshold* ratio of the total
data. In other words, if a particular key is not present in the
ThresholdCounter, its count represents less than *threshold* of
the total data.

>>> tc = ThresholdCounter(threshold=0.1)
>>> tc.add(1)
>>> tc.items()
[(1, 1)]
>>> tc.update([2] * 10)
>>> tc.get(1)
0
>>> tc.add(5)
>>> 5 in tc
True
>>> len(list(tc.elements()))
11

As you can see above, the API is kept similar to
:class:`collections.Counter`. The most notable feature omissions
being that counte ...

```python
# method of ThresholdCounter
def __init__(self, threshold=0.001)
```


```python
# method of ThresholdCounter
def threshold(self)
```


```python
# method of ThresholdCounter
def add(self, key)
```
Increment the count of *key* by 1, automatically adding it if it
does not exist.

Cache compaction is triggered every *1/threshold* additions.

```python
# method of ThresholdCounter
def elements(self)
```
Return an iterator of all the common elements tracked by the
counter. Yields each key as many times as it has been seen.

```python
# method of ThresholdCounter
def most_common(self, n=None)
```
Get the top *n* keys and counts as tuples. If *n* is omitted,
returns all the pairs.

```python
# method of ThresholdCounter
def get_common_count(self)
```
Get the sum of counts for keys exceeding the configured data
threshold.

```python
# method of ThresholdCounter
def get_uncommon_count(self)
```
Get the sum of counts for keys that were culled because the
associated counts represented less than the configured
threshold. The long-tail counts.

```python
# method of ThresholdCounter
def get_commonality(self)
```
Get a float representation of the effective count accuracy. The
higher the number, the less uniform the keys being added, and
the higher accuracy and efficiency of the ThresholdCounter.

If a stronger measure of data cardinality is required,
consider using hyperloglog.

```python
# method of ThresholdCounter
def iterkeys(self)
```


```python
# method of ThresholdCounter
def keys(self)
```


```python
# method of ThresholdCounter
def itervalues(self)
```


```python
# method of ThresholdCounter
def values(self)
```


```python
# method of ThresholdCounter
def iteritems(self)
```


```python
# method of ThresholdCounter
def items(self)
```


```python
# method of ThresholdCounter
def get(self, key, default=0)
```
Get count for *key*, defaulting to 0.

```python
# method of ThresholdCounter
def update(self, iterable, **kwargs)
```
Like dict.update() but add counts instead of replacing them, used
to add multiple items in one call.

Source can be an iterable of keys to add, or a mapping of keys
to integer counts.

```python
class MinIDMap()
```
Assigns arbitrary weakref-able objects the smallest possible unique
integer IDs, such that no two objects have the same ID at the same
time.

Maps arbitrary hashable objects to IDs.

Based on https://gist.github.com/kurtbrose/25b48114de216a5e55df

```python
# method of MinIDMap
def __init__(self)
```


```python
# method of MinIDMap
def get(self, a)
```


```python
# method of MinIDMap
def drop(self, a)
```


```python
# method of MinIDMap
def iteritems(self)
```



## module `boltons/iterutils.py`

```python
def is_iterable(obj)
```
Similar in nature to :func:`callable`, ``is_iterable`` returns
``True`` if an object is `iterable`_, ``False`` if not.

>>> is_iterable([])
True
>>> is_iterable(object())
False

.. _iterable: https://docs.python.org/2/glossary.html#term-iterable

```python
def is_scalar(obj)
```
A near-mirror of :func:`is_iterable`. Returns ``False`` if an
object is an iterable container type. Strings are considered
scalar as well, because strings are more often treated as whole
values as opposed to iterables of 1-character substrings.

>>> is_scalar(object())
True
>>> is_scalar(range(10))
False
>>> is_scalar('hello')
True

```python
def is_collection(obj)
```
The opposite of :func:`is_scalar`.  Returns ``True`` if an object
is an iterable other than a string.

>>> is_collection(object())
False
>>> is_collection(range(10))
True
>>> is_collection('hello')
False

```python
def split(src, sep=None, maxsplit=None)
```
Splits an iterable based on a separator. Like :meth:`str.split`,
but for all iterables. Returns a list of lists.

>>> split(['hi', 'hello', None, None, 'sup', None, 'soap', None])
[['hi', 'hello'], ['sup'], ['soap']]

See :func:`split_iter` docs for more info.

```python
def split_iter(src, sep=None, maxsplit=None)
```
Splits an iterable based on a separator, *sep*, a max of
*maxsplit* times (no max by default). *sep* can be:

  * a single value
  * an iterable of separators
  * a single-argument callable that returns True when a separator is
    encountered

``split_iter()`` yields lists of non-separator values. A separator will
never appear in the output.

>>> list(split_iter(['hi', 'hello', None, None, 'sup', None, 'soap', None]))
[['hi', 'hello'], ['sup'], ['soap']]

Note that ``split_iter`` is based on :func:`str.split`, so if
*sep* is ``None``, ``split()`` **groups** separators. If empty lists
are desired between two contiguous ``None`` values, simply use
``sep=[None]``:

>>> list(split_iter(['hi', ' ...

```python
def lstrip(iterable, strip_value=None)
```
Strips values from the beginning of an iterable. Stripped items will
match the value of the argument strip_value. Functionality is analogous
to that of the method str.lstrip. Returns a list.

>>> lstrip(['Foo', 'Bar', 'Bam'], 'Foo')
['Bar', 'Bam']

```python
def lstrip_iter(iterable, strip_value=None)
```
Strips values from the beginning of an iterable. Stripped items will
match the value of the argument strip_value. Functionality is analogous
to that of the method str.lstrip. Returns a generator.

>>> list(lstrip_iter(['Foo', 'Bar', 'Bam'], 'Foo'))
['Bar', 'Bam']

```python
def rstrip(iterable, strip_value=None)
```
Strips values from the end of an iterable. Stripped items will
match the value of the argument strip_value. Functionality is analogous
to that of the method str.rstrip. Returns a list.

>>> rstrip(['Foo', 'Bar', 'Bam'], 'Bam')
['Foo', 'Bar']

```python
def rstrip_iter(iterable, strip_value=None)
```
Strips values from the end of an iterable. Stripped items will
match the value of the argument strip_value. Functionality is analogous
to that of the method str.rstrip. Returns a generator.

>>> list(rstrip_iter(['Foo', 'Bar', 'Bam'], 'Bam'))
['Foo', 'Bar']

```python
def strip(iterable, strip_value=None)
```
Strips values from the beginning and end of an iterable. Stripped items
will match the value of the argument strip_value. Functionality is
analogous to that of the method str.strip. Returns a list.

>>> strip(['Fu', 'Foo', 'Bar', 'Bam', 'Fu'], 'Fu')
['Foo', 'Bar', 'Bam']

```python
def strip_iter(iterable, strip_value=None)
```
Strips values from the beginning and end of an iterable. Stripped items
will match the value of the argument strip_value. Functionality is
analogous to that of the method str.strip. Returns a generator.

>>> list(strip_iter(['Fu', 'Foo', 'Bar', 'Bam', 'Fu'], 'Fu'))
['Foo', 'Bar', 'Bam']

```python
def chunked(src, size, count=None, **kw)
```
Returns a list of *count* chunks, each with *size* elements,
generated from iterable *src*. If *src* is not evenly divisible by
*size*, the final chunk will have fewer than *size* elements.
Provide the *fill* keyword argument to provide a pad value and
enable padding, otherwise no padding will take place.

>>> chunked(range(10), 3)
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
>>> chunked(range(10), 3, fill=None)
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]
>>> chunked(range(10), 3, count=2)
[[0, 1, 2], [3, 4, 5]]

See :func:`chunked_iter` for more info.

```python
def chunked_iter(src, size, **kw)
```
Generates *size*-sized chunks from *src* iterable. Unless the
optional *fill* keyword argument is provided, iterables not evenly
divisible by *size* will have a final chunk that is smaller than
*size*.

>>> list(chunked_iter(range(10), 3))
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
>>> list(chunked_iter(range(10), 3, fill=None))
[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]

Note that ``fill=None`` in fact uses ``None`` as the fill value.

```python
def chunk_ranges(input_size, chunk_size, input_offset=0, overlap_size=0, align=False)
```
Generates *chunk_size*-sized chunk ranges for an input with length *input_size*.
Optionally, a start of the input can be set via *input_offset*, and
and overlap between the chunks may be specified via *overlap_size*.
Also, if *align* is set to *True*, any items with *i % (chunk_size-overlap_size) == 0*
are always at the beginning of the chunk.

Returns an iterator of (start, end) tuples, one tuple per chunk.

>>> list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5))
[(10, 15), (15, 20)]
>>> list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5, overlap_size=1))
[(10, 15), (14, 19), (18, 20)]
>>> list(chunk_ranges(input_offset=10, input_size=10, chunk_size=5, overlap_size= ...

```python
def pairwise(src, end=_UNSET)
```
Convenience function for calling :func:`windowed` on *src*, with
*size* set to 2.

>>> pairwise(range(5))
[(0, 1), (1, 2), (2, 3), (3, 4)]
>>> pairwise([])
[]

Unless *end* is set, the number of pairs is always one less than 
the number of elements in the iterable passed in, except on an empty input, 
which will return an empty list.

With *end* set, a number of pairs equal to the length of *src* is returned,
with the last item of the last pair being equal to *end*.

>>> list(pairwise(range(3), end=None))
[(0, 1), (1, 2), (2, None)]

This way, *end* values can be useful as sentinels to signal the end of the iterable.

```python
def pairwise_iter(src, end=_UNSET)
```
Convenience function for calling :func:`windowed_iter` on *src*,
with *size* set to 2.

>>> list(pairwise_iter(range(5)))
[(0, 1), (1, 2), (2, 3), (3, 4)]
>>> list(pairwise_iter([]))
[]

Unless *end* is set, the number of pairs is always one less 
than the number of elements in the iterable passed in, 
or zero, when *src* is empty.

With *end* set, a number of pairs equal to the length of *src* is returned,
with the last item of the last pair being equal to *end*. 

>>> list(pairwise_iter(range(3), end=None))
[(0, 1), (1, 2), (2, None)]    

This way, *end* values can be useful as sentinels to signal the end
of the iterable. For infinite iterators, setting *end* has no effect.

```python
def windowed(src, size, fill=_UNSET)
```
Returns tuples with exactly length *size*. If *fill* is unset 
and the iterable is too short to make a window of length *size*, 
no tuples are returned. See :func:`windowed_iter` for more.

```python
def windowed_iter(src, size, fill=_UNSET)
```
Returns tuples with length *size* which represent a sliding
window over iterable *src*.

>>> list(windowed_iter(range(7), 3))
[(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]

If *fill* is unset, and the iterable is too short to make a window 
of length *size*, then no window tuples are returned.

>>> list(windowed_iter(range(3), 5))
[]

With *fill* set, the iterator always yields a number of windows
equal to the length of the *src* iterable.

>>> windowed(range(4), 3, fill=None)
[(0, 1, 2), (1, 2, 3), (2, 3, None), (3, None, None)]

This way, *fill* values can be useful to signal the end of the iterable.
For infinite iterators, setting *fill* has no effect.

```python
def xfrange(stop, start=None, step=1.0)
```
Same as :func:`frange`, but generator-based instead of returning a
list.

>>> tuple(xfrange(1, 3, step=0.75))
(1.0, 1.75, 2.5)

See :func:`frange` for more details.

```python
def frange(stop, start=None, step=1.0)
```
A :func:`range` clone for float-based ranges.

>>> frange(5)
[0.0, 1.0, 2.0, 3.0, 4.0]
>>> frange(6, step=1.25)
[0.0, 1.25, 2.5, 3.75, 5.0]
>>> frange(100.5, 101.5, 0.25)
[100.5, 100.75, 101.0, 101.25]
>>> frange(5, 0)
[]
>>> frange(5, 0, step=-1.25)
[5.0, 3.75, 2.5, 1.25]

```python
def backoff(start, stop, count=None, factor=2.0, jitter=False)
```
Returns a list of geometrically-increasing floating-point numbers,
suitable for usage with `exponential backoff`_. Exactly like
:func:`backoff_iter`, but without the ``'repeat'`` option for
*count*. See :func:`backoff_iter` for more details.

.. _exponential backoff: https://en.wikipedia.org/wiki/Exponential_backoff

>>> backoff(1, 10)
[1.0, 2.0, 4.0, 8.0, 10.0]

```python
def backoff_iter(start, stop, count=None, factor=2.0, jitter=False)
```
Generates a sequence of geometrically-increasing floats, suitable
for usage with `exponential backoff`_. Starts with *start*,
increasing by *factor* until *stop* is reached, optionally
stopping iteration once *count* numbers are yielded. *factor*
defaults to 2. In general retrying with properly-configured
backoff creates a better-behaved component for a larger service
ecosystem.

.. _exponential backoff: https://en.wikipedia.org/wiki/Exponential_backoff

>>> list(backoff_iter(1.0, 10.0, count=5))
[1.0, 2.0, 4.0, 8.0, 10.0]
>>> list(backoff_iter(1.0, 10.0, count=8))
[1.0, 2.0, 4.0, 8.0, 10.0, 10.0, 10.0, 10.0]
>>> list(backoff_iter(0.25, 100.0, factor=10))
[0.25, 2.5, 25.0, 100.0]

A simplifi ...

```python
def bucketize(src, key=bool, value_transform=None, key_filter=None)
```
Group values in the *src* iterable by the value returned by *key*.

>>> bucketize(range(5))
{False: [0], True: [1, 2, 3, 4]}
>>> is_odd = lambda x: x % 2 == 1
>>> bucketize(range(5), is_odd)
{False: [0, 2, 4], True: [1, 3]}

*key* is :class:`bool` by default, but can either be a callable or a string or a list
if it is a string, it is the name of the attribute on which to bucketize objects.

>>> bucketize([1+1j, 2+2j, 1, 2], key='real')
{1.0: [(1+1j), 1], 2.0: [(2+2j), 2]}

if *key* is a list, it contains the buckets where to put each object

>>> bucketize([1,2,365,4,98],key=[0,1,2,0,2])
{0: [1, 4], 1: [2], 2: [365, 98]}


Value lists are not deduplicated:

>>> bucketize([None, None, None, 'h ...

```python
def partition(src, key=bool, *keys)
```
No relation to :meth:`str.partition`, ``partition`` is like
:func:`bucketize`, but for added convenience returns a collection for
each predicate passed.

``partition`` now accepts multiple *key* functions and will return
``N + 1`` lists for ``N`` predicates. Each value from *src* is placed
into the first list whose predicate evaluates to ``True`` with values
that match none of the predicates placed in the last list.

>>> nonempty, empty = partition(['', '', 'hi', '', 'bye'])
>>> nonempty
['hi', 'bye']

*key* defaults to :class:`bool`, but can be carefully overridden to
use either a function that returns either ``True`` or ``False`` or
a string name of the attribute on which to partition obje ...

```python
def unique(src, key=None)
```
``unique()`` returns a list of unique values, as determined by
*key*, in the order they first appeared in the input iterable,
*src*.

>>> ones_n_zeros = '11010110001010010101010'
>>> ''.join(unique(ones_n_zeros))
'10'

See :func:`unique_iter` docs for more details.

```python
def unique_iter(src, key=None)
```
Yield unique elements from the iterable, *src*, based on *key*,
in the order in which they first appeared in *src*.

>>> repetitious = [1, 2, 3] * 10
>>> list(unique_iter(repetitious))
[1, 2, 3]

By default, *key* is the object itself, but *key* can either be a
callable or, for convenience, a string name of the attribute on
which to uniqueify objects, falling back on identity when the
attribute is not present.

>>> pleasantries = ['hi', 'hello', 'ok', 'bye', 'yes']
>>> list(unique_iter(pleasantries, key=lambda x: len(x)))
['hi', 'hello', 'bye']

```python
def redundant(src, key=None, groups=False)
```
The complement of :func:`unique()`.

By default returns non-unique/duplicate values as a list of the
*first* redundant value in *src*. Pass ``groups=True`` to get
groups of all values with redundancies, ordered by position of the
first redundant value. This is useful in conjunction with some
normalizing *key* function.

>>> redundant([1, 2, 3, 4])
[]
>>> redundant([1, 2, 3, 2, 3, 3, 4])
[2, 3]
>>> redundant([1, 2, 3, 2, 3, 3, 4], groups=True)
[[2, 2], [3, 3, 3]]

An example using a *key* function to do case-insensitive
redundancy detection.

>>> redundant(['hi', 'Hi', 'HI', 'hello'], key=str.lower)
['Hi']
>>> redundant(['hi', 'Hi', 'HI', 'hello'], groups=True, key=str.lower)
[['hi', 'Hi', 'H ...

```python
def one(src, default=None, key=None)
```
Along the same lines as builtins, :func:`all` and :func:`any`, and
similar to :func:`first`, ``one()`` returns the single object in
the given iterable *src* that evaluates to ``True``, as determined
by callable *key*. If unset, *key* defaults to :class:`bool`. If
no such objects are found, *default* is returned. If *default* is
not passed, ``None`` is returned.

If *src* has more than one object that evaluates to ``True``, or
if there is no object that fulfills such condition, return
*default*. It's like an `XOR`_ over an iterable.

>>> one((True, False, False))
True
>>> one((True, False, True))
>>> one((0, 0, 'a'))
'a'
>>> one((0, False, None))
>>> one((True, True), default=False)
False
>>> ...

```python
def first(iterable, default=None, key=None)
```
Return first element of *iterable* that evaluates to ``True``, else
return ``None`` or optional *default*. Similar to :func:`one`.

>>> first([0, False, None, [], (), 42])
42
>>> first([0, False, None, [], ()]) is None
True
>>> first([0, False, None, [], ()], default='ohai')
'ohai'
>>> import re
>>> m = first(re.match(regex, 'abc') for regex in ['b.*', 'a(.*)'])
>>> m.group(1)
'bc'

The optional *key* argument specifies a one-argument predicate function
like that used for *filter()*.  The *key* argument, if supplied, should be
in keyword form. For example, finding the first even number in an iterable:

>>> first([1, 1, 3, 4, 5], key=lambda x: x % 2 == 0)
4

Contributed by Hynek Schlawack, au ...

```python
def flatten_iter(iterable)
```
``flatten_iter()`` yields all the elements from *iterable* while
collapsing any nested iterables.

>>> nested = [[1, 2], [[3], [4, 5]]]
>>> list(flatten_iter(nested))
[1, 2, 3, 4, 5]

```python
def flatten(iterable)
```
``flatten()`` returns a collapsed list of all the elements from
*iterable* while collapsing any nested iterables.

>>> nested = [[1, 2], [[3], [4, 5]]]
>>> flatten(nested)
[1, 2, 3, 4, 5]

```python
def same(iterable, ref=_UNSET)
```
``same()`` returns ``True`` when all values in *iterable* are
equal to one another, or optionally a reference value,
*ref*. Similar to :func:`all` and :func:`any` in that it evaluates
an iterable and returns a :class:`bool`. ``same()`` returns
``True`` for empty iterables.

>>> same([])
True
>>> same([1])
True
>>> same(['a', 'a', 'a'])
True
>>> same(range(20))
False
>>> same([[], []])
True
>>> same([[], []], ref='test')
False

```python
def default_visit(path, key, value)
```


```python
def default_enter(path, key, value)
```


```python
def default_exit(path, key, old_parent, new_parent, new_items)
```


```python
def remap(root, visit=default_visit, enter=default_enter, exit=default_exit, cache: bool=True, **kwargs)
```
The remap ("recursive map") function is used to traverse and
transform nested structures. Lists, tuples, sets, and dictionaries
are just a few of the data structures nested into heterogeneous
tree-like structures that are so common in programming.
Unfortunately, Python's built-in ways to manipulate collections
are almost all flat. List comprehensions may be fast and succinct,
but they do not recurse, making it tedious to apply quick changes
or complex transforms to real-world data.

remap goes where list comprehensions cannot.

Here's an example of removing all Nones from some data:

>>> from pprint import pprint
>>> reviews = {'Star Trek': {'TNG': 10, 'DS9': 8.5, 'ENT': None},
... ...

```python
class PathAccessError(KeyError, IndexError, TypeError)
```
An amalgamation of KeyError, IndexError, and TypeError,
representing what can occur when looking up a path in a nested
object.

```python
# method of PathAccessError
def __init__(self, exc, seg, path)
```


```python
def get_path(root, path, default=_UNSET)
```
Retrieve a value from a nested object via a tuple representing the
lookup path.

>>> root = {'a': {'b': {'c': [[1], [2], [3]]}}}
>>> get_path(root, ('a', 'b', 'c', 2, 0))
3

The path tuple format is intentionally consistent with that of
:func:`remap`, but a single dotted string can also be passed.

One of get_path's chief aims is improved error messaging. EAFP is
great, but the error messages are not.

For instance, ``root['a']['b']['c'][2][1]`` gives back
``IndexError: list index out of range``

What went out of range where? get_path currently raises
``PathAccessError: could not access 2 from path ('a', 'b', 'c', 2,
1), got error: IndexError('list index out of range',)``, a
subclass of Inde ...

```python
def research(root, query=lambda p, k, v: True, reraise=False, enter=default_enter)
```
The :func:`research` function uses :func:`remap` to recurse over
any data nested in *root*, and find values which match a given
criterion, specified by the *query* callable.

Results are returned as a list of ``(path, value)`` pairs. The
paths are tuples in the same format accepted by
:func:`get_path`. This can be useful for comparing values nested
in two or more different structures.

Here's a simple example that finds all integers:

>>> root = {'a': {'b': 1, 'c': (2, 'd', 3)}, 'e': None}
>>> res = research(root, query=lambda p, k, v: isinstance(v, int))
>>> print(sorted(res))
[(('a', 'b'), 1), (('a', 'c', 0), 2), (('a', 'c', 2), 3)]

Note how *query* follows the same, familiar ``path, key, ...

```python
class GUIDerator()
```
The GUIDerator is an iterator that yields a globally-unique
identifier (GUID) on every iteration. The GUIDs produced are
hexadecimal strings.

Testing shows it to be around 12x faster than the uuid module. By
default it is also more compact, partly due to its default 96-bit
(24-hexdigit) length. 96 bits of randomness means that there is a
1 in 2 ^ 32 chance of collision after 2 ^ 64 iterations. If more
or less uniqueness is desired, the *size* argument can be adjusted
accordingly.

Args:
    size (int): character length of the GUID, defaults to 24. Lengths
                between 20 and 36 are considered valid.

The GUIDerator has built-in fork protection that causes it to
detect a fork on n ...

```python
# method of GUIDerator
def __init__(self, size=24)
```


```python
# method of GUIDerator
def reseed(self)
```


```python
class SequentialGUIDerator(GUIDerator)
```
Much like the standard GUIDerator, the SequentialGUIDerator is an
iterator that yields a globally-unique identifier (GUID) on every
iteration. The GUIDs produced are hexadecimal strings.

The SequentialGUIDerator differs in that it picks a starting GUID
value and increments every iteration. This yields GUIDs which are
of course unique, but also ordered and lexicographically sortable.

The SequentialGUIDerator is around 50% faster than the normal
GUIDerator, making it almost 20x as fast as the built-in uuid
module. By default it is also more compact, partly due to its
96-bit (24-hexdigit) default length. 96 bits of randomness means that
there is a 1 in 2 ^ 32 chance of collision after 2 ^ 64 ...

```python
# method of SequentialGUIDerator
def reseed(self)
```


```python
def soft_sorted(iterable, first=None, last=None, key=None, reverse=False)
```
For when you care about the order of some elements, but not about
others.

Use this to float to the top and/or sink to the bottom a specific
ordering, while sorting the rest of the elements according to
normal :func:`sorted` rules.

>>> soft_sorted(['two', 'b', 'one', 'a'], first=['one', 'two'])
['one', 'two', 'a', 'b']
>>> soft_sorted(range(7), first=[6, 15], last=[2, 4], reverse=True)
[6, 5, 3, 1, 0, 2, 4]
>>> import string
>>> ''.join(soft_sorted(string.hexdigits, first='za1', last='b', key=str.lower))
'aA1023456789cCdDeEfFbB'

Args:
   iterable (list): A list or other iterable to sort.
   first (list): A sequence to enforce for elements which should
      appear at the beginning of the r ...

```python
def untyped_sorted(iterable, key=None, reverse=False)
```
A version of :func:`sorted` which will happily sort an iterable of
heterogeneous types and return a new list, similar to legacy Python's
behavior.

>>> untyped_sorted(['abc', 2.0, 1, 2, 'def'])
[1, 2.0, 2, 'abc', 'def']

Note how mutually orderable types are sorted as expected, as in
the case of the integers and floats above.

.. note::

   Results may vary across Python versions and builds, but the
   function will produce a sorted list, except in the case of
   explicitly unorderable objects.


## module `boltons/timeutils.py`

```python
def dt_to_timestamp(dt)
```
Converts from a :class:`~datetime.datetime` object to an integer
timestamp, suitable interoperation with :func:`time.time` and
other `Epoch-based timestamps`.

.. _Epoch-based timestamps: https://en.wikipedia.org/wiki/Unix_time

>>> timestamp = int(time.time())
>>> utc_dt = datetime.fromtimestamp(timestamp, timezone.utc)
>>> timestamp - dt_to_timestamp(utc_dt)
0.0

``dt_to_timestamp`` supports both timezone-aware and naïve
:class:`~datetime.datetime` objects. Note that it assumes naïve
datetime objects are implied UTC, such as those generated with
:meth:`datetime.datetime.utcnow`. If your datetime objects are
local time, such as those generated with
:meth:`datetime.datetime.now`, first conve ...

```python
def isoparse(iso_str)
```
Parses the limited subset of `ISO8601-formatted time`_ strings as
returned by :meth:`datetime.datetime.isoformat`.

>>> epoch_dt = datetime.fromtimestamp(0, timezone.utc).replace(tzinfo=None)
>>> iso_str = epoch_dt.isoformat()
>>> print(iso_str)
1970-01-01T00:00:00
>>> isoparse(iso_str)
datetime.datetime(1970, 1, 1, 0, 0)

>>> utcnow = datetime.now(timezone.utc).replace(tzinfo=None)
>>> utcnow == isoparse(utcnow.isoformat())
True

For further datetime parsing, see the `iso8601`_ package for strict
ISO parsing and `dateutil`_ package for loose parsing and more.

.. _ISO8601-formatted time: https://en.wikipedia.org/wiki/ISO_8601
.. _iso8601: https://pypi.python.org/pypi/iso8601
.. _dateutil: h ...

```python
def parse_timedelta(text)
```
Robustly parses a short text description of a time period into a
:class:`datetime.timedelta`. Supports weeks, days, hours, minutes,
and seconds, with or without decimal points:

Args:
    text (str): Text to parse.
Returns:
    datetime.timedelta
Raises:
    ValueError: on parse failure.

>>> parse_td('1d 2h 3.5m 0s') == timedelta(days=1, seconds=7410)
True

Also supports full words and whitespace.

>>> parse_td('2 weeks 1 day') == timedelta(days=15)
True

Negative times are supported, too:

>>> parse_td('-1.5 weeks 3m 20s') == timedelta(days=-11, seconds=43400)
True

```python
def decimal_relative_time(d, other=None, ndigits=0, cardinalize=True)
```
Get a tuple representing the relative time difference between two
:class:`~datetime.datetime` objects or one
:class:`~datetime.datetime` and now.

Args:
    d (datetime): The first datetime object.
    other (datetime): An optional second datetime object. If
        unset, defaults to the current time as determined
        :meth:`datetime.utcnow`.
    ndigits (int): The number of decimal digits to round to,
        defaults to ``0``.
    cardinalize (bool): Whether to pluralize the time unit if
        appropriate, defaults to ``True``.
Returns:
    (float, str): A tuple of the :class:`float` difference and
       respective unit of time, pluralized if appropriate and
       *cardinalize* is ...

```python
def relative_time(d, other=None, ndigits=0)
```
Get a string representation of the difference between two
:class:`~datetime.datetime` objects or one
:class:`~datetime.datetime` and the current time. Handles past and
future times.

Args:
    d (datetime): The first datetime object.
    other (datetime): An optional second datetime object. If
        unset, defaults to the current time as determined
        :meth:`datetime.utcnow`.
    ndigits (int): The number of decimal digits to round to,
        defaults to ``0``.
Returns:
    A short English-language string.

>>> now = datetime.now(timezone.utc).replace(tzinfo=None)
>>> relative_time(now, ndigits=1)
'0 seconds ago'
>>> relative_time(now - timedelta(days=1, seconds=36000), ndigits=1)
'1 ...

```python
def strpdate(string, format)
```
Parse the date string according to the format in `format`.  Returns a
:class:`date` object.  Internally, :meth:`datetime.strptime` is used to
parse the string and thus conversion specifiers for time fields (e.g. `%H`)
may be provided;  these will be parsed but ignored.

Args:
    string (str): The date string to be parsed.
    format (str): The `strptime`_-style date format string.
Returns:
    datetime.date

.. _`strptime`: https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior

>>> strpdate('2016-02-14', '%Y-%m-%d')
datetime.date(2016, 2, 14)
>>> strpdate('26/12 (2015)', '%d/%m (%Y)')
datetime.date(2015, 12, 26)
>>> strpdate('20151231 23:59:59', '%Y%m%d %H:%M:%S')
datet ...

```python
def daterange(start, stop, step=1, inclusive=False)
```
In the spirit of :func:`range` and :func:`xrange`, the `daterange`
generator that yields a sequence of :class:`~datetime.date`
objects, starting at *start*, incrementing by *step*, until *stop*
is reached.

When *inclusive* is True, the final date may be *stop*, **if**
*step* falls evenly on it. By default, *step* is one day. See
details below for many more details.

Args:
    start (datetime.date): The starting date The first value in
        the sequence.
    stop (datetime.date): The stopping date. By default not
        included in return. Can be `None` to yield an infinite
        sequence.
    step (int): The value to increment *start* by to reach
        *stop*. Can be an :class:`int` ...

```python
class ConstantTZInfo(tzinfo)
```
A :class:`~datetime.tzinfo` subtype whose *offset* remains constant
(no daylight savings).

Args:
    name (str): Name of the timezone.
    offset (datetime.timedelta): Offset of the timezone.

```python
# method of ConstantTZInfo
def __init__(self, name='ConstantTZ', offset=ZERO)
```


```python
# method of ConstantTZInfo
def utcoffset_hours(self)
```


```python
# method of ConstantTZInfo
def utcoffset(self, dt)
```


```python
# method of ConstantTZInfo
def tzname(self, dt)
```


```python
# method of ConstantTZInfo
def dst(self, dt)
```


```python
class LocalTZInfo(tzinfo)
```
The ``LocalTZInfo`` type takes data available in the time module
about the local timezone and makes a practical
:class:`datetime.tzinfo` to represent the timezone settings of the
operating system.

For a more in-depth integration with the operating system, check
out `tzlocal`_. It builds on `pytz`_ and implements heuristics for
many versions of major operating systems to provide the official
``pytz`` tzinfo, instead of the LocalTZ generalization.

.. _tzlocal: https://pypi.python.org/pypi/tzlocal
.. _pytz: https://pypi.python.org/pypi/pytz

```python
# method of LocalTZInfo
def is_dst(self, dt)
```


```python
# method of LocalTZInfo
def utcoffset(self, dt)
```


```python
# method of LocalTZInfo
def dst(self, dt)
```


```python
# method of LocalTZInfo
def tzname(self, dt)
```


```python
class USTimeZone(tzinfo)
```
Copied directly from the Python docs, the ``USTimeZone`` is a
:class:`datetime.tzinfo` subtype used to create the
:data:`Eastern`, :data:`Central`, :data:`Mountain`, and
:data:`Pacific` tzinfo types.

```python
# method of USTimeZone
def __init__(self, hours, reprname, stdname, dstname)
```


```python
# method of USTimeZone
def tzname(self, dt)
```


```python
# method of USTimeZone
def utcoffset(self, dt)
```


```python
# method of USTimeZone
def dst(self, dt)
```


```python
# method of USTimeZone
def fromutc(self, dt)
```



## module `boltons/queueutils.py`

```python
class BasePriorityQueue()
```
The abstract base class for the other PriorityQueues in this
module. Override the ``_backend_type`` class attribute, as well as
the :meth:`_push_entry` and :meth:`_pop_entry` staticmethods for
custom subclass behavior. (Don't forget to use
:func:`staticmethod`).

Args:
    priority_key (callable): A function that takes *priority* as
        passed in by :meth:`add` and returns a real number
        representing the effective priority.

```python
# method of BasePriorityQueue
def __init__(self, **kw)
```


```python
# method of BasePriorityQueue
def add(self, task, priority=None)
```
Add a task to the queue, or change the *task*'s priority if *task*
is already in the queue. *task* can be any hashable object,
and *priority* defaults to ``0``. Higher values representing
higher priority, but this behavior can be controlled by
setting *priority_key* in the constructor.

```python
# method of BasePriorityQueue
def remove(self, task)
```
Remove a task from the priority queue. Raises :exc:`KeyError` if
the *task* is absent.

```python
# method of BasePriorityQueue
def peek(self, default=_REMOVED)
```
Read the next value in the queue without removing it. Returns
*default* on an empty queue, or raises :exc:`KeyError` if
*default* is not set.

```python
# method of BasePriorityQueue
def pop(self, default=_REMOVED)
```
Remove and return the next value in the queue. Returns *default* on
an empty queue, or raises :exc:`KeyError` if *default* is not
set.

```python
class HeapPriorityQueue(BasePriorityQueue)
```
A priority queue inherited from :class:`BasePriorityQueue`,
backed by a list and based on the :func:`heapq.heappop` and
:func:`heapq.heappush` functions in the built-in :mod:`heapq`
module.

```python
class SortedPriorityQueue(BasePriorityQueue)
```
A priority queue inherited from :class:`BasePriorityQueue`, based
on the :func:`bisect.insort` approach for in-order insertion into
a sorted list.


## module `boltons/fileutils.py`

```python
def mkdir_p(path)
```
Creates a directory and any parent directories that may need to
be created along the way, without raising errors for any existing
directories. This function mimics the behavior of the ``mkdir -p``
command available in Linux/BSD environments, but also works on
Windows.

```python
class FilePerms()
```
The :class:`FilePerms` type is used to represent standard POSIX
filesystem permissions:

  * Read
  * Write
  * Execute

Across three classes of user:

  * Owning (u)ser
  * Owner's (g)roup
  * Any (o)ther user

This class assists with computing new permissions, as well as
working with numeric octal ``777``-style and ``rwx``-style
permissions. Currently it only considers the bottom 9 permission
bits; it does not support sticky bits or more advanced permission
systems.

Args:
    user (str): A string in the 'rwx' format, omitting characters
        for which owning user's permissions are not provided.
    group (str): A string in the 'rwx' format, omitting characters
        for which owning ...

```python
# method of FilePerms
def __init__(self, user='', group='', other='')
```


```python
# method of FilePerms
def from_int(cls, i)
```
Create a :class:`FilePerms` object from an integer.

>>> FilePerms.from_int(0o644)  # note the leading zero-oh for octal
FilePerms(user='rw', group='r', other='r')

```python
# method of FilePerms
def from_path(cls, path)
```
Make a new :class:`FilePerms` object based on the permissions
assigned to the file or directory at *path*.

Args:
    path (str): Filesystem path of the target file.

Here's an example that holds true on most systems:

>>> import tempfile
>>> 'r' in FilePerms.from_path(tempfile.gettempdir()).user
True

```python
def atomic_save(dest_path, **kwargs)
```
A convenient interface to the :class:`AtomicSaver` type. Example:

>>> try:
...     with atomic_save("file.txt", text_mode=True) as fo:
...         _ = fo.write('bye')
...         1/0  # will error
...         fo.write('bye')
... except ZeroDivisionError:
...     pass  # at least our file.txt didn't get overwritten

See the :class:`AtomicSaver` documentation for details.

```python
class AtomicSaver()
```
``AtomicSaver`` is a configurable `context manager`_ that provides
a writable :class:`file` which will be moved into place as long as
no exceptions are raised within the context manager's block. These
"part files" are created in the same directory as the destination
path to ensure atomic move operations (i.e., no cross-filesystem
moves occur).

Args:
    dest_path (str): The path where the completed file will be
        written.
    overwrite (bool): Whether to overwrite the destination file if
        it exists at completion time. Defaults to ``True``.
    file_perms (int): Integer representation of file permissions
        for the newly-created file. Defaults are, when the
        destinat ...

```python
# method of AtomicSaver
def __init__(self, dest_path, **kwargs)
```


```python
# method of AtomicSaver
def setup(self)
```
Called on context manager entry (the :keyword:`with` statement),
the ``setup()`` method creates the temporary file in the same
directory as the destination file.

``setup()`` tests for a writable directory with rename permissions
early, as the part file may not be written to immediately (not
using :func:`os.access` because of the potential issues of
effective vs. real privileges).

If the caller i ...

```python
# method of AtomicSaver
def __enter__(self)
```


```python
# method of AtomicSaver
def __exit__(self, exc_type, exc_val, exc_tb)
```


```python
def iter_find_files(directory, patterns, ignored=None, include_dirs=False, max_depth=None)
```
Returns a generator that yields file paths under a *directory*,
matching *patterns* using `glob`_ syntax (e.g., ``*.txt``). Also
supports *ignored* patterns.

Args:
    directory (str): Path that serves as the root of the
        search. Yielded paths will include this as a prefix.
    patterns (str or list): A single pattern or list of
        glob-formatted patterns to find under *directory*.
    ignored (str or list): A single pattern or list of
        glob-formatted patterns to ignore.
    include_dirs (bool): Whether to include directories that match
       patterns, as well. Defaults to ``False``.
    max_depth (int): traverse up to this level of subdirectory.
       I.e., 0 for the s ...


## module `boltons/dictutils.py`

```python
class OrderedMultiDict(dict)
```
A MultiDict is a dictionary that can have multiple values per key
and the OrderedMultiDict (OMD) is a MultiDict that retains
original insertion order. Common use cases include:

  * handling query strings parsed from URLs
  * inverting a dictionary to create a reverse index (values to keys)
  * stacking data from multiple dictionaries in a non-destructive way

The OrderedMultiDict constructor is identical to the built-in
:class:`dict`, and overall the API constitutes an intuitive
superset of the built-in type:

>>> omd = OrderedMultiDict()
>>> omd['a'] = 1
>>> omd['b'] = 2
>>> omd.add('a', 3)
>>> omd.get('a')
3
>>> omd.getlist('a')
[1, 3]

Some non-:class:`dict`-like behaviors also make an a ...

```python
# method of OrderedMultiDict
def __init__(self, *args, **kwargs)
```


```python
# method of OrderedMultiDict
def add(self, k, v)
```
Add a single value *v* under a key *k*. Existing values under *k*
are preserved.

```python
# method of OrderedMultiDict
def addlist(self, k, v)
```
Add an iterable of values underneath a specific key, preserving
any values already under that key.

>>> omd = OrderedMultiDict([('a', -1)])
>>> omd.addlist('a', range(3))
>>> omd
OrderedMultiDict([('a', -1), ('a', 0), ('a', 1), ('a', 2)])

Called ``addlist`` for consistency with :meth:`getlist`, but
tuples and other sequences and iterables work.

```python
# method of OrderedMultiDict
def get(self, k, default=None)
```
Return the value for key *k* if present in the dictionary, else
*default*. If *default* is not given, ``None`` is returned.
This method never raises a :exc:`KeyError`.

To get all values under a key, use :meth:`OrderedMultiDict.getlist`.

```python
# method of OrderedMultiDict
def getlist(self, k, default=_MISSING)
```
Get all values for key *k* as a list, if *k* is in the
dictionary, else *default*. The list returned is a copy and
can be safely mutated. If *default* is not given, an empty
:class:`list` is returned.

```python
# method of OrderedMultiDict
def clear(self)
```
Empty the dictionary.

```python
# method of OrderedMultiDict
def setdefault(self, k, default=_MISSING)
```
If key *k* is in the dictionary, return its value. If not, insert
*k* with a value of *default* and return *default*. *default*
defaults to ``None``. See :meth:`dict.setdefault` for more
information.

```python
# method of OrderedMultiDict
def copy(self)
```
Return a shallow copy of the dictionary.

```python
# method of OrderedMultiDict
def fromkeys(cls, keys, default=None)
```
Create a dictionary from a list of keys, with all the values
set to *default*, or ``None`` if *default* is not set.

```python
# method of OrderedMultiDict
def update(self, E, **F)
```
Add items from a dictionary or iterable (and/or keyword arguments),
overwriting values under an existing key. See
:meth:`dict.update` for more details.

```python
# method of OrderedMultiDict
def update_extend(self, E, **F)
```
Add items from a dictionary, iterable, and/or keyword
arguments without overwriting existing items present in the
dictionary. Like :meth:`update`, but adds to existing keys
instead of overwriting them.

```python
# method of OrderedMultiDict
def pop(self, k, default=_MISSING)
```
Remove all values under key *k*, returning the most-recently
inserted value. Raises :exc:`KeyError` if the key is not
present and no *default* is provided.

```python
# method of OrderedMultiDict
def popall(self, k, default=_MISSING)
```
Remove all values under key *k*, returning them in the form of
a list. Raises :exc:`KeyError` if the key is not present and no
*default* is provided.

```python
# method of OrderedMultiDict
def poplast(self, k=_MISSING, default=_MISSING)
```
Remove and return the most-recently inserted value under the key
*k*, or the most-recently inserted key if *k* is not
provided. If no values remain under *k*, it will be removed
from the OMD.  Raises :exc:`KeyError` if *k* is not present in
the dictionary, or the dictionary is empty.

```python
# method of OrderedMultiDict
def iteritems(self, multi=False)
```
Iterate over the OMD's items in insertion order. By default,
yields only the most-recently inserted value for each key. Set
*multi* to ``True`` to get all inserted items.

```python
# method of OrderedMultiDict
def iterkeys(self, multi=False)
```
Iterate over the OMD's keys in insertion order. By default, yields
each key once, according to the most recent insertion. Set
*multi* to ``True`` to get all keys, including duplicates, in
insertion order.

```python
# method of OrderedMultiDict
def itervalues(self, multi=False)
```
Iterate over the OMD's values in insertion order. By default,
yields the most-recently inserted value per unique key.  Set
*multi* to ``True`` to get all values according to insertion
order.

```python
# method of OrderedMultiDict
def todict(self, multi=False)
```
Gets a basic :class:`dict` of the items in this dictionary. Keys
are the same as the OMD, values are the most recently inserted
values for each key.

Setting the *multi* arg to ``True`` is yields the same
result as calling :class:`dict` on the OMD, except that all the
value lists are copies that can be safely mutated.

```python
# method of OrderedMultiDict
def sorted(self, key=None, reverse=False)
```
Similar to the built-in :func:`sorted`, except this method returns
a new :class:`OrderedMultiDict` sorted by the provided key
function, optionally reversed.

Args:
    key (callable): A callable to determine the sort key of
      each element. The callable should expect an **item**
      (key-value pair tuple).
    reverse (bool): Set to ``True`` to reverse the ordering.

>>> omd = OrderedMultiDic ...

```python
# method of OrderedMultiDict
def sortedvalues(self, key=None, reverse=False)
```
Returns a copy of the :class:`OrderedMultiDict` with the same keys
in the same order as the original OMD, but the values within
each keyspace have been sorted according to *key* and
*reverse*.

Args:
    key (callable): A single-argument callable to determine
      the sort key of each element. The callable should expect
      an **item** (key-value pair tuple).
    reverse (bool): Set to ``True`` ...

```python
# method of OrderedMultiDict
def inverted(self)
```
Returns a new :class:`OrderedMultiDict` with values and keys
swapped, like creating dictionary transposition or reverse
index.  Insertion order is retained and all keys and values
are represented in the output.

>>> omd = OMD([(0, 2), (1, 2)])
>>> omd.inverted().getlist(2)
[0, 1]

Inverting twice yields a copy of the original:

>>> omd.inverted().inverted()
OrderedMultiDict([(0, 2), (1, 2)])

```python
# method of OrderedMultiDict
def counts(self)
```
Returns a mapping from key to number of values inserted under that
key. Like :py:class:`collections.Counter`, but returns a new
:class:`OrderedMultiDict`.

```python
# method of OrderedMultiDict
def keys(self, multi=False)
```
Returns a list containing the output of :meth:`iterkeys`.  See
that method's docs for more details.

```python
# method of OrderedMultiDict
def values(self, multi=False)
```
Returns a list containing the output of :meth:`itervalues`.  See
that method's docs for more details.

```python
# method of OrderedMultiDict
def items(self, multi=False)
```
Returns a list containing the output of :meth:`iteritems`.  See
that method's docs for more details.

```python
# method of OrderedMultiDict
def viewkeys(self)
```
OMD.viewkeys() -> a set-like object providing a view on OMD's keys

```python
# method of OrderedMultiDict
def viewvalues(self)
```
OMD.viewvalues() -> an object providing a view on OMD's values

```python
# method of OrderedMultiDict
def viewitems(self)
```
OMD.viewitems() -> a set-like object providing a view on OMD's items

```python
class OneToOne(dict)
```
Implements a one-to-one mapping dictionary. In addition to
inheriting from and behaving exactly like the builtin
:class:`dict`, all values are automatically added as keys on a
reverse mapping, available as the `inv` attribute. This
arrangement keeps key and value namespaces distinct.

Basic operations are intuitive:

>>> oto = OneToOne({'a': 1, 'b': 2})
>>> print(oto['a'])
1
>>> print(oto.inv[1])
a
>>> len(oto)
2

Overwrites happens in both directions:

>>> oto.inv[1] = 'c'
>>> print(oto.get('a'))
None
>>> len(oto)
2

For a very similar project, with even more one-to-one
functionality, check out `bidict <https://github.com/jab/bidict>`_.

```python
# method of OneToOne
def __init__(self, *a, **kw)
```


```python
# method of OneToOne
def unique(cls, *a, **kw)
```
This alternate constructor for OneToOne will raise an exception
when input values overlap. For instance:

>>> OneToOne.unique({'a': 1, 'b': 1})
Traceback (most recent call last):
...
ValueError: expected unique values, got multiple keys for the following values: ...

This even works across inputs:

>>> a_dict = {'a': 2}
>>> OneToOne.unique(a_dict, b=2)
Traceback (most recent call last):
...
ValueE ...

```python
# method of OneToOne
def clear(self)
```


```python
# method of OneToOne
def copy(self)
```


```python
# method of OneToOne
def pop(self, key, default=_MISSING)
```


```python
# method of OneToOne
def popitem(self)
```


```python
# method of OneToOne
def setdefault(self, key, default=None)
```


```python
# method of OneToOne
def update(self, dict_or_iterable, **kw)
```


```python
class ManyToMany()
```
a dict-like entity that represents a many-to-many relationship
between two groups of objects

behaves like a dict-of-tuples; also has .inv which is kept
up to date which is a dict-of-tuples in the other direction

also, can be used as a directed graph among hashable python objects

```python
# method of ManyToMany
def __init__(self, items=None)
```


```python
# method of ManyToMany
def get(self, key, default=frozenset())
```


```python
# method of ManyToMany
def update(self, iterable)
```
given an iterable of (key, val), add them all

```python
# method of ManyToMany
def add(self, key, val)
```


```python
# method of ManyToMany
def remove(self, key, val)
```


```python
# method of ManyToMany
def replace(self, key, newkey)
```
replace instances of key by newkey

```python
# method of ManyToMany
def iteritems(self)
```


```python
# method of ManyToMany
def keys(self)
```


```python
def subdict(d, keep=None, drop=None)
```
Compute the "subdictionary" of a dict, *d*.

A subdict is to a dict what a subset is a to set. If *A* is a
subdict of *B*, that means that all keys of *A* are present in
*B*.

Returns a new dict with any keys in *drop* removed, and any keys
in *keep* still present, provided they were in the original
dict. *keep* defaults to all keys, *drop* defaults to empty, so
without one of these arguments, calling this function is
equivalent to calling ``dict()``.

>>> from pprint import pprint as pp
>>> pp(subdict({'a': 1, 'b': 2}))
{'a': 1, 'b': 2}
>>> subdict({'a': 1, 'b': 2, 'c': 3}, drop=['b', 'c'])
{'a': 1}
>>> pp(subdict({'a': 1, 'b': 2, 'c': 3}, keep=['a', 'c']))
{'a': 1, 'c': 3}

```python
class FrozenDict(dict)
```
An immutable dict subtype that is hashable and can itself be used
as a :class:`dict` key or :class:`set` entry. What
:class:`frozenset` is to :class:`set`, FrozenDict is to
:class:`dict`.

There was once an attempt to introduce such a type to the standard
library, but it was rejected: `PEP 416 <https://www.python.org/dev/peps/pep-0416/>`_.

Because FrozenDict is a :class:`dict` subtype, it automatically
works everywhere a dict would, including JSON serialization.

```python
# method of FrozenDict
def updated(self, *a, **kw)
```
Make a copy and add items from a dictionary or iterable (and/or
keyword arguments), overwriting values under an existing
key. See :meth:`dict.update` for more details.

```python
# method of FrozenDict
def fromkeys(cls, keys, value=None)
```



## module `boltons/setutils.py`

```python
class IndexedSet(MutableSet)
```
``IndexedSet`` is a :class:`collections.MutableSet` that maintains
insertion order and uniqueness of inserted elements. It's a hybrid
type, mostly like an OrderedSet, but also :class:`list`-like, in
that it supports indexing and slicing.

Args:
    other (iterable): An optional iterable used to initialize the set.

>>> x = IndexedSet(list(range(4)) + list(range(8)))
>>> x
IndexedSet([0, 1, 2, 3, 4, 5, 6, 7])
>>> x - set(range(2))
IndexedSet([2, 3, 4, 5, 6, 7])
>>> x[-1]
7
>>> fcr = IndexedSet('freecreditreport.com')
>>> ''.join(fcr[:fcr.index('.')])
'frecditpo'

Standard set operators and interoperation with :class:`set` are
all supported:

>>> fcr & set('cash4gold.com')
IndexedSet(['c', 'd' ...

```python
# method of IndexedSet
def __init__(self, other=None)
```


```python
# method of IndexedSet
def from_iterable(cls, it)
```
from_iterable(it) -> create a set from an iterable

```python
# method of IndexedSet
def add(self, item)
```
add(item) -> add item to the set

```python
# method of IndexedSet
def remove(self, item)
```
remove(item) -> remove item from the set, raises if not present

```python
# method of IndexedSet
def discard(self, item)
```
discard(item) -> discard item from the set (does not raise)

```python
# method of IndexedSet
def clear(self)
```
clear() -> empty the set

```python
# method of IndexedSet
def isdisjoint(self, other)
```
isdisjoint(other) -> return True if no overlap with other

```python
# method of IndexedSet
def issubset(self, other)
```
issubset(other) -> return True if other contains this set

```python
# method of IndexedSet
def issuperset(self, other)
```
issuperset(other) -> return True if set contains other

```python
# method of IndexedSet
def union(self, *others)
```
union(*others) -> return a new set containing this set and others

```python
# method of IndexedSet
def iter_intersection(self, *others)
```
iter_intersection(*others) -> iterate over elements also in others

```python
# method of IndexedSet
def intersection(self, *others)
```
intersection(*others) -> get a set with overlap of this and others

```python
# method of IndexedSet
def iter_difference(self, *others)
```
iter_difference(*others) -> iterate over elements not in others

```python
# method of IndexedSet
def difference(self, *others)
```
difference(*others) -> get a new set with elements not in others

```python
# method of IndexedSet
def symmetric_difference(self, *others)
```
symmetric_difference(*others) -> XOR set of this and others

```python
# method of IndexedSet
def update(self, *others)
```
update(*others) -> add values from one or more iterables

```python
# method of IndexedSet
def intersection_update(self, *others)
```
intersection_update(*others) -> discard self.difference(*others)

```python
# method of IndexedSet
def difference_update(self, *others)
```
difference_update(*others) -> discard self.intersection(*others)

```python
# method of IndexedSet
def symmetric_difference_update(self, other)
```
symmetric_difference_update(other) -> in-place XOR with other

```python
# method of IndexedSet
def iter_slice(self, start, stop, step=None)
```
iterate over a slice of the set

```python
# method of IndexedSet
def pop(self, index=None)
```
pop(index) -> remove the item at a given index (-1 by default)

```python
# method of IndexedSet
def count(self, val)
```
count(val) -> count number of instances of value (0 or 1)

```python
# method of IndexedSet
def reverse(self)
```
reverse() -> reverse the contents of the set in-place

```python
# method of IndexedSet
def sort(self, **kwargs)
```
sort() -> sort the contents of the set in-place

```python
# method of IndexedSet
def index(self, val)
```
index(val) -> get the index of a value, raises if not present

```python
def complement(wrapped)
```
Given a :class:`set`, convert it to a **complement set**.

Whereas a :class:`set` keeps track of what it contains, a
`complement set
<https://en.wikipedia.org/wiki/Complement_(set_theory)>`_ keeps
track of what it does *not* contain. For example, look what
happens when we intersect a normal set with a complement set::

>>> list(set(range(5)) & complement(set([2, 3])))
[0, 1, 4]

We get the everything in the left that wasn't in the right,
because intersecting with a complement is the same as subtracting
a normal set.

Args:
    wrapped (set): A set or any other iterable which should be
       turned into a complement set.

All set methods and operators are supported by complement sets,
betwee ...


## module `boltons/strutils.py`

```python
def camel2under(camel_string)
```
Converts a camelcased string to underscores. Useful for turning a
class name into a function name.

>>> camel2under('BasicParseTest')
'basic_parse_test'

```python
def under2camel(under_string)
```
Converts an underscored string to camelcased. Useful for turning a
function name into a class name.

>>> under2camel('complex_tokenizer')
'ComplexTokenizer'

```python
def slugify(text, delim='_', lower=True, ascii=False)
```
A basic function that turns text full of scary characters
(i.e., punctuation and whitespace), into a relatively safe
lowercased string separated only by the delimiter specified
by *delim*, which defaults to ``_``.

The *ascii* convenience flag will :func:`asciify` the slug if
you require ascii-only slugs.

>>> slugify('First post! Hi!!!!~1    ')
'first_post_hi_1'

>>> slugify("Kurt Gödel's pretty cool.", ascii=True) ==         b'kurt_goedel_s_pretty_cool'
True

```python
def split_punct_ws(text)
```
While :meth:`str.split` will split on whitespace,
:func:`split_punct_ws` will split on punctuation and
whitespace. This used internally by :func:`slugify`, above.

>>> split_punct_ws('First post! Hi!!!!~1    ')
['First', 'post', 'Hi', '1']

```python
def unit_len(sized_iterable, unit_noun='item')
```
Returns a plain-English description of an iterable's
:func:`len()`, conditionally pluralized with :func:`cardinalize`,
detailed below.

>>> print(unit_len(range(10), 'number'))
10 numbers
>>> print(unit_len('aeiou', 'vowel'))
5 vowels
>>> print(unit_len([], 'worry'))
No worries

```python
def ordinalize(number, ext_only=False)
```
Turns *number* into its cardinal form, i.e., 1st, 2nd,
3rd, 4th, etc. If the last character isn't a digit, it returns the
string value unchanged.

Args:
    number (int or str): Number to be cardinalized.
    ext_only (bool): Whether to return only the suffix. Default ``False``.

>>> print(ordinalize(1))
1st
>>> print(ordinalize(3694839230))
3694839230th
>>> print(ordinalize('hi'))
hi
>>> print(ordinalize(1515))
1515th

```python
def cardinalize(unit_noun, count)
```
Conditionally pluralizes a singular word *unit_noun* if
*count* is not one, preserving case when possible.

>>> vowels = 'aeiou'
>>> print(len(vowels), cardinalize('vowel', len(vowels)))
5 vowels
>>> print(3, cardinalize('Wish', 3))
3 Wishes

```python
def singularize(word)
```
Semi-intelligently converts an English plural *word* to its
singular form, preserving case pattern.

>>> singularize('chances')
'chance'
>>> singularize('Activities')
'Activity'
>>> singularize('Glasses')
'Glass'
>>> singularize('FEET')
'FOOT'

```python
def pluralize(word)
```
Semi-intelligently converts an English *word* from singular form to
plural, preserving case pattern.

>>> pluralize('friend')
'friends'
>>> pluralize('enemy')
'enemies'
>>> pluralize('Sheep')
'Sheep'

```python
def find_hashtags(string)
```
Finds and returns all hashtags in a string, with the hashmark
removed. Supports full-width hashmarks for Asian languages and
does not false-positive on URL anchors.

>>> find_hashtags('#atag http://asite/#ananchor')
['atag']

``find_hashtags`` also works with unicode hashtags.

```python
def a10n(string)
```
That thing where "internationalization" becomes "i18n", what's it
called? Abbreviation? Oh wait, no: ``a10n``. (It's actually a form
of `numeronym`_.)

>>> a10n('abbreviation')
'a10n'
>>> a10n('internationalization')
'i18n'
>>> a10n('')
''

.. _numeronym: http://en.wikipedia.org/wiki/Numeronym

```python
def strip_ansi(text)
```
Strips ANSI escape codes from *text*. Useful for the occasional
time when a log or redirected output accidentally captures console
color codes and the like.

>>> strip_ansi('[0m[1;36mart[46;34m')
'art'

Supports str, bytes and bytearray content as input. Returns the
same type as the input.

There's a lot of ANSI art available for testing on `sixteencolors.net`_.
This function does not interpret or render ANSI art, but you can do so with
`ansi2img`_ or `escapes.js`_.

.. _sixteencolors.net: http://sixteencolors.net
.. _ansi2img: http://www.bedroomlan.org/projects/ansi2img
.. _escapes.js: https://github.com/atdt/escapes.js

```python
def asciify(text, ignore=False)
```
Converts a unicode or bytestring, *text*, into a bytestring with
just ascii characters. Performs basic deaccenting for all you
Europhiles out there.

Also, a gentle reminder that this is a **utility**, primarily meant
for slugification. Whenever possible, make your application work
**with** unicode, not against it.

Args:
    text (str): The string to be asciified.
    ignore (bool): Configures final encoding to ignore remaining
        unasciified string instead of replacing it.

>>> asciify('Beyoncé') == b'Beyonce'
True

```python
def is_ascii(text)
```
Check if a string or bytestring, *text*, is composed of ascii
characters only. Raises :exc:`ValueError` if argument is not text.

Args:
    text (str): The string to be checked.

>>> is_ascii('Beyoncé')
False
>>> is_ascii('Beyonce')
True

```python
def bytes2human(nbytes, ndigits=0)
```
Turns an integer value of *nbytes* into a human readable format. Set
*ndigits* to control how many digits after the decimal point
should be shown (default ``0``).

>>> bytes2human(128991)
'126K'
>>> bytes2human(100001221)
'95M'
>>> bytes2human(0, 2)
'0.00B'
>>> bytes2human(1024)
'1K'
>>> bytes2human(1024 ** 8)
'1Y'

```python
def html2text(html)
```
Strips tags from HTML text, returning markup-free text. Also, does
a best effort replacement of entities like "&nbsp;"

>>> r = html2text(u'<a href="#">Test &amp;<em>(Δ&#x03b7;&#956;&#x03CE;)</em></a>')
>>> r == u'Test &(Δημώ)'
True

```python
def gunzip_bytes(bytestring)
```
The :mod:`gzip` module is great if you have a file or file-like
object, but what if you just have bytes. StringIO is one
possibility, but it's often faster, easier, and simpler to just
use this one-liner. Use this tried-and-true utility function to
decompress gzip from bytes.

>>> gunzip_bytes(_EMPTY_GZIP_BYTES) == b''
True
>>> gunzip_bytes(_NON_EMPTY_GZIP_BYTES).rstrip() == b'bytesahoy!'
True

```python
def gzip_bytes(bytestring, level=6)
```
Turn some bytes into some compressed bytes.

>>> len(gzip_bytes(b'a' * 10000))
46

Args:
    bytestring (bytes): Bytes to be compressed
    level (int): An integer, 1-9, controlling the
      speed/compression. 1 is fastest, least compressed, 9 is
      slowest, but most compressed.

Note that all levels of gzip are pretty fast these days, though
it's not really a competitor in compression, at any level.

```python
def iter_splitlines(text)
```
Like :meth:`str.splitlines`, but returns an iterator of lines
instead of a list. Also similar to :meth:`file.next`, as that also
lazily reads and yields lines from a file.

This function works with a variety of line endings, but as always,
be careful when mixing line endings within a file.

>>> list(iter_splitlines('\nhi\nbye\n'))
['', 'hi', 'bye', '']
>>> list(iter_splitlines('\r\nhi\rbye\r\n'))
['', 'hi', 'bye', '']
>>> list(iter_splitlines(''))
[]

```python
def indent(text, margin, newline='\n', key=bool)
```
The missing counterpart to the built-in :func:`textwrap.dedent`.

Args:
    text (str): The text to indent.
    margin (str): The string to prepend to each line.
    newline (str): The newline used to rejoin the lines (default: ``\n``)
    key (callable): Called on each line to determine whether to
      indent it. Default: :class:`bool`, to ensure that empty lines do
      not get whitespace added.

```python
def is_uuid(obj, version=4)
```
Check the argument is either a valid UUID object or string.

Args:
    obj (object): The test target. Strings and UUID objects supported.
    version (int): The target UUID version, set to 0 to skip version check.

>>> is_uuid('e682ccca-5a4c-4ef2-9711-73f9ad1e15ea')
True
>>> is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9')
False
>>> is_uuid('0221f0d9-d4b9-11e5-a478-10ddb1c2feb9', version=1)
True

```python
def escape_shell_args(args, sep=' ', style=None)
```
Returns an escaped version of each string in *args*, according to
*style*.

Args:
    args (list): A list of arguments to escape and join together
    sep (str): The separator used to join the escaped arguments.
    style (str): The style of escaping to use. Can be one of
      ``cmd`` or ``sh``, geared toward Windows and Linux/BSD/etc.,
      respectively. If *style* is ``None``, then it is picked
      according to the system platform.

See :func:`args2cmd` and :func:`args2sh` for details and example
output for each style.

```python
def args2sh(args, sep=' ')
```
Return a shell-escaped string version of *args*, separated by
*sep*, based on the rules of sh, bash, and other shells in the
Linux/BSD/MacOS ecosystem.

>>> print(args2sh(['aa', '[bb]', "cc'cc", 'dd"dd']))
aa '[bb]' 'cc'"'"'cc' 'dd"dd'

As you can see, arguments with no special characters are not
escaped, arguments with special characters are quoted with single
quotes, and single quotes themselves are quoted with double
quotes. Double quotes are handled like any other special
character.

Based on code from the :mod:`pipes`/:mod:`shlex` modules. Also
note that :mod:`shlex` and :mod:`argparse` have functions to split
and parse strings escaped in this manner.

```python
def args2cmd(args, sep=' ')
```
Return a shell-escaped string version of *args*, separated by
*sep*, using the same rules as the Microsoft C runtime.

>>> print(args2cmd(['aa', '[bb]', "cc'cc", 'dd"dd']))
aa [bb] cc'cc dd\"dd

As you can see, escaping is through backslashing and not quoting,
and double quotes are the only special character. See the comment
in the code for more details. Based on internal code from the
:mod:`subprocess` module.

```python
def parse_int_list(range_string, delim=',', range_delim='-')
```
Returns a sorted list of positive integers based on
*range_string*. Reverse of :func:`format_int_list`.

Args:
    range_string (str): String of comma separated positive
        integers or ranges (e.g. '1,2,4-6,8'). Typical of a custom
        page range string used in printer dialogs.
    delim (char): Defaults to ','. Separates integers and
        contiguous ranges of integers.
    range_delim (char): Defaults to '-'. Indicates a contiguous
        range of integers.

>>> parse_int_list('1,3,5-8,10-11,15')
[1, 3, 5, 6, 7, 8, 10, 11, 15]

```python
def format_int_list(int_list, delim=',', range_delim='-', delim_space=False)
```
Returns a sorted range string from a list of positive integers
(*int_list*). Contiguous ranges of integers are collapsed to min
and max values. Reverse of :func:`parse_int_list`.

Args:
    int_list (list): List of positive integers to be converted
       into a range string (e.g. [1,2,4,5,6,8]).
    delim (char): Defaults to ','. Separates integers and
       contiguous ranges of integers.
    range_delim (char): Defaults to '-'. Indicates a contiguous
       range of integers.
    delim_space (bool): Defaults to ``False``. If ``True``, adds a
       space after all *delim* characters.

>>> format_int_list([1,3,5,6,7,8,10,11,15])
'1,3,5-8,10-11,15'

```python
def complement_int_list(range_string, range_start=0, range_end=None, delim=',', range_delim='-')
```
Returns range string that is the complement of the one provided as
*range_string* parameter.

These range strings are of the kind produce by :func:`format_int_list`, and
parseable by :func:`parse_int_list`.

Args:
    range_string (str): String of comma separated positive integers or
       ranges (e.g. '1,2,4-6,8'). Typical of a custom page range string
       used in printer dialogs.
    range_start (int): A positive integer from which to start the resulting
       range. Value is inclusive. Defaults to ``0``.
    range_end (int): A positive integer from which the produced range is
       stopped. Value is exclusive. Defaults to the maximum value found in
       the provided ``range_string ...

```python
def int_ranges_from_int_list(range_string, delim=',', range_delim='-')
```
Transform a string of ranges (*range_string*) into a tuple of tuples.

Args:
    range_string (str): String of comma separated positive integers or
       ranges (e.g. '1,2,4-6,8'). Typical of a custom page range string
       used in printer dialogs.
    delim (char): Defaults to ','. Separates integers and contiguous ranges
       of integers.
    range_delim (char): Defaults to '-'. Indicates a contiguous range of
       integers.

>>> int_ranges_from_int_list('1,3,5-8,10-11,15')
((1, 1), (3, 3), (5, 8), (10, 11), (15, 15))

>>> int_ranges_from_int_list('1')
((1, 1),)

>>> int_ranges_from_int_list('')
()

```python
class MultiReplace()
```
MultiReplace is a tool for doing multiple find/replace actions in one pass.

Given a mapping of values to be replaced it allows for all of the matching
values to be replaced in a single pass which can save a lot of performance
on very large strings. In addition to simple replace, it also allows for
replacing based on regular expressions.

Keyword Arguments:

:type regex: bool
:param regex: Treat search keys as regular expressions [Default: False]
:type flags: int
:param flags: flags to pass to the regex engine during compile

Dictionary Usage::

    from boltons import strutils
    s = strutils.MultiReplace({
        'foo': 'zoo',
        'cat': 'hat',
        'bat': 'kraken'
    })
    new ...

```python
# method of MultiReplace
def __init__(self, sub_map, **kwargs)
```
Compile any regular expressions that have been passed.

```python
# method of MultiReplace
def sub(self, text)
```
Run substitutions on the input text.

Given an input string, run all substitutions given in the
constructor.

```python
def multi_replace(text, sub_map, **kwargs)
```
Shortcut function to invoke MultiReplace in a single call.

Example Usage::

    from boltons.strutils import multi_replace
    new = multi_replace(
        'The foo bar cat ate a bat',
        {'foo': 'zoo', 'cat': 'hat', 'bat': 'kraken'}
    )
    new == 'The zoo bar hat ate a kraken'

```python
def unwrap_text(text, ending='\n\n')
```
Unwrap text, the natural complement to :func:`textwrap.wrap`.

>>> text = "Short \n lines  \nwrapped\nsmall.\n\nAnother\nparagraph."
>>> unwrap_text(text)
'Short lines wrapped small.\n\nAnother paragraph.'

Args:
   text: A string to unwrap.
   ending (str): The string to join all unwrapped paragraphs
      by. Pass ``None`` to get the list. Defaults to '\n\n' for
      compatibility with Markdown and RST.

```python
def removeprefix(text: str, prefix: str) -> str
```
Remove `prefix` from start of `text` if present.

Backport of `str.removeprefix` for Python versions less than 3.9.

Args:
    text: A string to remove the prefix from.
    prefix: The string to remove from the beginning of `text`.

```python
def human_readable_list(items: typing.Sequence[str], delimiter: str=',', conjunction: str='and', *, oxford: bool=True) -> str
```
Given a list of strings, return a human readable string with
appropriate delimiters and the conjunction word.

Args:
    items: The list of strings to join.
    delimiter (optional): The delimiter to use between items.
    conjunction (optional): The word to use before the last item.
    oxford (optional): Whether to use the Oxford comma/delimiter before
        the conjunction in lists of 3+ items.

Returns:
    str: The human readable string.

```python
def ellipsize(text, max_len=160, *, ellipsis='…')
```
Truncate *text* to at most *max_len* characters, cutting at the
last space before the limit and appending *ellipsis*. The returned
string, ellipsis included, is never longer than *max_len*.

Text short enough to fit is returned unchanged:

>>> ellipsize('Hello, World!', 16)
'Hello, World!'

Longer text is cut at a space, never mid-word, and trailing
punctuation at the cut is stripped:

>>> ellipsize('Beautiful is better than ugly. Explicit is better.', 31)
'Beautiful is better than ugly…'

A ``.`` between two digits is a decimal point, not sentence
punctuation, and numbers are kept whole:

>>> ellipsize('rates around 6.5% this week', 20)
'rates around 6.5%…'

A single token longer than *max_ ...


# Usage examples from the project README

```python
On macOS, it can also be installed via [MacPorts](https://ports.macports.org/port/py-boltons/summary):
```
```python
Then, [thanks to PyPI][boltons_pypi], dozens of boltons are just an import away:
```
```python
However, due to the nature of utilities, application developers might
want to consider other options, including vendorization of individual
modules into a project. Boltons is pure-Python and has no
dependencies. If the whole project is too big, each module is
independent, and can be copied directly into a project. See the
[Integration][integration] section of the docs for more details.

[boltons_pypi]: https://pypi.python.org/pypi/boltons
[integration]: https://boltons.readthedocs.org/en/latest/architecture.html#integration

## Third-party packages

The majority of boltons strive to be "good enough" for a wide range of
basic uses, leaving advanced use cases to Python's [myriad specialized
3rd-party libraries][pypi]. In many cases the respective ``boltons`` module
will describe 3rd-party alternatives worth investigating when use
cases outgrow `boltons`. If you've found a natural "next-step"
library worth mentioning, see the next section!

[pypi]: https://pypi.python.org/pypi

## Gaps

Found something missing in the standard library that should be in
`boltons`? Found something missing in `boltons`? First, take a
moment to read the very brief [architecture statement][architecture]
to make sure the functionality would be a good fit. It covers the
design philosophy, what belongs in boltons, and how modules are
organized.

Then, if you are very motivated, submit [a Pull Request][prs]. Otherwise,
submit a short feature request on [the Issues page][issues], and we will
figure something out.

[architecture]: https://boltons.readthedocs.org/en/latest/architecture.html
[issues]: https://github.com/mahmoud/boltons/issues
[prs]: https://github.com/mahmoud/boltons/pulls

## Development

Install [uv](https://docs.astral.sh/uv/), then set up a dev environment:
```
```python
Run the test suite (includes doctests):
```