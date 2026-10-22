# more_itertools: public API (signatures and documentation)


## module `more_itertools/more.py`

```python
def chunked(iterable, n, strict=False)
```
Break *iterable* into lists of length *n*:

    >>> list(chunked([1, 2, 3, 4, 5, 6], 3))
    [[1, 2, 3], [4, 5, 6]]

By the default, the last yielded list will have fewer than *n* elements
if the length of *iterable* is not divisible by *n*:

    >>> list(chunked([1, 2, 3, 4, 5, 6, 7, 8], 3))
    [[1, 2, 3], [4, 5, 6], [7, 8]]

To use a fill-in value instead, see the :func:`grouper` recipe.

If the length of *iterable* is not divisible by *n* and *strict* is
``True``, then ``ValueError`` will be raised before the last
list is yielded.

```python
def first(iterable, default=_marker)
```
Return the first item of *iterable*, or *default* if *iterable* is
empty.

    >>> first([0, 1, 2, 3])
    0
    >>> first([], 'some default')
    'some default'

If *default* is not provided and there are no items in the iterable,
raise ``ValueError``.

:func:`first` is useful when you have a generator of expensive-to-retrieve
values and want any arbitrary one. It is marginally shorter than
``next(iter(iterable), default)``.

```python
def last(iterable, default=_marker)
```
Return the last item of *iterable*, or *default* if *iterable* is
empty.

    >>> last([0, 1, 2, 3])
    3
    >>> last([], 'some default')
    'some default'

If *default* is not provided and there are no items in the iterable,
raise ``ValueError``.

```python
def nth_or_last(iterable, n, default=_marker)
```
Return the nth or the last item of *iterable*,
or *default* if *iterable* is empty.

    >>> nth_or_last([0, 1, 2, 3], 2)
    2
    >>> nth_or_last([0, 1], 2)
    1
    >>> nth_or_last([], 0, 'some default')
    'some default'

If *default* is not provided and there are no items in the iterable,
raise ``ValueError``.

```python
class peekable()
```
Wrap an iterator to allow lookahead and prepending elements.

Call :meth:`peek` on the result to get the value that will be returned
by :func:`next`. This won't advance the iterator:

    >>> p = peekable(['a', 'b'])
    >>> p.peek()
    'a'
    >>> next(p)
    'a'

Pass :meth:`peek` a default value to return that instead of raising
``StopIteration`` when the iterator is exhausted.

    >>> p = peekable([])
    >>> p.peek('hi')
    'hi'

peekables also offer a :meth:`prepend` method, which "inserts" items
at the head of the iterable:

    >>> p = peekable([1, 2, 3])
    >>> p.prepend(10, 11, 12)
    >>> next(p)
    10
    >>> p.peek()
    11
    >>> list(p)
    [11, 12, 1, 2, 3]

peekables c ...

```python
# method of peekable
def __init__(self, iterable)
```


```python
# method of peekable
def peek(self, default=_marker)
```
Return the item that will be next returned from ``next()``.

Return ``default`` if there are no items left. If ``default`` is not
provided, raise ``StopIteration``.

```python
# method of peekable
def prepend(self, *items)
```
Stack up items to be the next ones returned from ``next()`` or
``self.peek()``. The items will be returned in
first in, first out order::

    >>> p = peekable([1, 2, 3])
    >>> p.prepend(10, 11, 12)
    >>> next(p)
    10
    >>> list(p)
    [11, 12, 1, 2, 3]

It is possible, by prepending items, to "resurrect" a peekable that
previously raised ``StopIteration``.

    >>> p = peekable([])
    >> ...

```python
def consumer(func)
```
Decorator that automatically advances a PEP-342-style "reverse iterator"
to its first yield point so you don't have to call ``next()`` on it
manually.

    >>> @consumer
    ... def tally():
    ...     i = 0
    ...     while True:
    ...         print('Thing number %s is %s.' % (i, (yield)))
    ...         i += 1
    ...
    >>> t = tally()
    >>> t.send('red')
    Thing number 0 is red.
    >>> t.send('fish')
    Thing number 1 is fish.

Without the decorator, you would have to call ``next(t)`` before
``t.send()`` could be used.

```python
def ilen(iterable)
```
Return the number of items in *iterable*.

For example, there are 168 prime numbers below 1,000:

    >>> ilen(sieve(1000))
    168

Equivalent to, but faster than::

    def ilen(iterable):
        count = 0
        for _ in iterable:
            count += 1
        return count

This fully consumes the iterable, so handle with care.

```python
def iterate(func, start)
```
Return ``start``, ``func(start)``, ``func(func(start))``, ...

Produces an infinite iterator. To add a stopping condition,
use :func:`take`, ``takewhile``, or :func:`takewhile_inclusive`:.

>>> take(10, iterate(lambda x: 2*x, 1))
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

>>> collatz = lambda x: 3*x + 1 if x%2==1 else x // 2
>>> list(takewhile_inclusive(lambda x: x!=1, iterate(collatz, 10)))
[10, 5, 16, 8, 4, 2, 1]

```python
def with_iter(context_manager)
```
Wrap an iterable in a ``with`` statement, so it closes once exhausted.

For example, this will close the file when the iterator is exhausted::

    upper_lines = (line.upper() for line in with_iter(open('foo')))

Note that you have to actually exhaust the iterator for opened files to be closed.

Any context manager which returns an iterable is a candidate for
``with_iter``.

```python
class sized_iterator()
```
Wrapper for *iterable* that implements ``__len__``.

>>> it = map(str, range(5))
>>> sized_it = sized_iterator(it, 5)
>>> len(sized_it)
5
>>> list(sized_it)
['0', '1', '2', '3', '4']

This is useful for tools that use :func:`len`, like
`tqdm <https://pypi.org/project/tqdm/>`__ .

The wrapper doesn't validate the provided *length*, so be sure to choose
a value that reflects reality.

```python
# method of sized_iterator
def __init__(self, iterable, length)
```


```python
def one(iterable, too_short=None, too_long=None)
```
Return the first item from *iterable*, which is expected to contain only
that item. Raise an exception if *iterable* is empty or has more than one
item.

:func:`one` is useful for ensuring that an iterable contains only one item.
For example, it can be used to retrieve the result of a database query
that is expected to return a single row.

If *iterable* is empty, ``ValueError`` will be raised. You may specify a
different exception with the *too_short* keyword:

    >>> it = []
    >>> one(it)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: too few items in iterable (expected 1)'
    >>> too_short = IndexError('too few items')
    >>> one(i ...

```python
def raise_(exception, *args)
```


```python
def strictly_n(iterable, n, too_short=None, too_long=None)
```
Validate that *iterable* has exactly *n* items and return them if
it does. If it has fewer than *n* items, call function *too_short*
with the actual number of items. If it has more than *n* items, call function
*too_long* with the number ``n + 1``.

    >>> iterable = ['a', 'b', 'c', 'd']
    >>> n = 4
    >>> list(strictly_n(iterable, n))
    ['a', 'b', 'c', 'd']

Note that the returned iterable must be consumed in order for the check to
be made.

By default, *too_short* and *too_long* are functions that raise
``ValueError``.

    >>> list(strictly_n('ab', 3))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    ValueError: too few items in iterable (got 2 ...

```python
def distinct_permutations(iterable, r=None)
```
Yield successive distinct permutations of the elements in *iterable*.

    >>> sorted(distinct_permutations([1, 0, 1]))
    [(0, 1, 1), (1, 0, 1), (1, 1, 0)]

Equivalent to yielding from ``set(permutations(iterable))``, except
duplicates are not generated and thrown away. For larger input sequences
this is much more efficient.

If the elements of the input iterable are sortable, the output tuples are
produced in sorted order.

Duplicate permutations arise when there are duplicated elements in the
input iterable. The number of items returned is
`n! / (x_1! * x_2! * ... * x_n!)`, where `n` is the total number of
items input, and each `x_i` is the count of a distinct item in the input
sequence. ...

```python
def derangements(iterable, r=None)
```
Yield successive derangements of the elements in *iterable*.

A derangement is a permutation in which no element appears at its original
index. In other words, a derangement is a permutation that has no fixed points.

Suppose Alice, Bob, Carol, and Dave are playing Secret Santa.
The code below outputs all of the different ways to assign gift recipients
such that nobody is assigned to himself or herself:

    >>> for d in derangements(['Alice', 'Bob', 'Carol', 'Dave']):
    ...    print(', '.join(d))
    Bob, Alice, Dave, Carol
    Bob, Carol, Dave, Alice
    Bob, Dave, Alice, Carol
    Carol, Alice, Dave, Bob
    Carol, Dave, Alice, Bob
    Carol, Dave, Bob, Alice
    Dave, Alice, Bob, Carol ...

```python
def intersperse(e, iterable, n=1)
```
Intersperse filler element *e* among the items in *iterable*, leaving
*n* items between each filler element.

    >>> list(intersperse('!', [1, 2, 3, 4, 5]))
    [1, '!', 2, '!', 3, '!', 4, '!', 5]

    >>> list(intersperse(None, [1, 2, 3, 4, 5], n=2))
    [1, 2, None, 3, 4, None, 5]

```python
def unique_to_each(*iterables)
```
Return the elements from each of the input iterables that aren't in the
other input iterables.

For example, suppose you have a set of packages, each with a set of
dependencies::

    {'pkg_1': {'A', 'B'}, 'pkg_2': {'B', 'C'}, 'pkg_3': {'B', 'D'}}

If you remove one package, which dependencies can also be removed?

If ``pkg_1`` is removed, then ``A`` is no longer necessary - it is not
associated with ``pkg_2`` or ``pkg_3``. Similarly, ``C`` is only needed for
``pkg_2``, and ``D`` is only needed for ``pkg_3``::

    >>> unique_to_each({'A', 'B'}, {'B', 'C'}, {'B', 'D'})
    [['A'], ['C'], ['D']]

If there are duplicates in one input iterable that aren't in the others
they will be duplicated i ...

```python
def windowed(seq, n, fillvalue=None, step=1)
```
Return a sliding window of width *n* over the given iterable.

    >>> all_windows = windowed([1, 2, 3, 4, 5], 3)
    >>> list(all_windows)
    [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

When the window is larger than the iterable, *fillvalue* is used in place
of missing values:

    >>> list(windowed([1, 2, 3], 4))
    [(1, 2, 3, None)]

Each window will advance in increments of *step*:

    >>> list(windowed([1, 2, 3, 4, 5, 6], 3, fillvalue='!', step=2))
    [(1, 2, 3), (3, 4, 5), (5, 6, '!')]

To slide into the iterable's items, use :func:`chain` to add filler items
to the left:

    >>> iterable = [1, 2, 3, 4]
    >>> n = 3
    >>> padding = [None] * (n - 1)
    >>> list(windowed(chain(padding, ...

```python
def substrings(iterable)
```
Yield all of the substrings of *iterable*.

    >>> [''.join(s) for s in substrings('more')]
    ['m', 'o', 'r', 'e', 'mo', 'or', 're', 'mor', 'ore', 'more']

Note that non-string iterables can also be subdivided.

    >>> list(substrings([0, 1, 2]))
    [(0,), (1,), (2,), (0, 1), (1, 2), (0, 1, 2)]

Like subslices() but returns tuples instead of lists
and returns the shortest substrings first.

```python
def substrings_indexes(seq, reverse=False)
```
Yield all substrings and their positions in *seq*

The items yielded will be a tuple of the form ``(substr, i, j)``, where
``substr == seq[i:j]``.

This function only works for iterables that support slicing, such as
``str`` objects.

>>> for item in substrings_indexes('more'):
...    print(item)
('m', 0, 1)
('o', 1, 2)
('r', 2, 3)
('e', 3, 4)
('mo', 0, 2)
('or', 1, 3)
('re', 2, 4)
('mor', 0, 3)
('ore', 1, 4)
('more', 0, 4)

Set *reverse* to ``True`` to yield the same items in the opposite order.

```python
class bucket()
```
Wrap *iterable* and return an object that buckets the iterable into
child iterables based on a *key* function.

    >>> iterable = ['a1', 'b1', 'c1', 'a2', 'b2', 'c2', 'b3']
    >>> s = bucket(iterable, key=lambda x: x[0])  # Bucket by 1st character
    >>> sorted(list(s))  # Get the keys
    ['a', 'b', 'c']
    >>> a_iterable = s['a']
    >>> next(a_iterable)
    'a1'
    >>> next(a_iterable)
    'a2'
    >>> list(s['b'])
    ['b1', 'b2', 'b3']

The original iterable will be advanced and its items will be cached until
they are used by the child iterables. This may require significant storage.

By default, attempting to select a bucket to which no items belong  will
exhaust the iterable and ...

```python
# method of bucket
def __init__(self, iterable, key, validator=None)
```


```python
def spy(iterable, n=1)
```
Return a 2-tuple with a list containing the first *n* elements of
*iterable*, and an iterator with the same items as *iterable*.
This allows you to "look ahead" at the items in the iterable without
advancing it.

There is one item in the list by default:

    >>> iterable = 'abcdefg'
    >>> head, iterable = spy(iterable)
    >>> head
    ['a']
    >>> list(iterable)
    ['a', 'b', 'c', 'd', 'e', 'f', 'g']

You may use unpacking to retrieve items instead of lists:

    >>> (head,), iterable = spy('abcdefg')
    >>> head
    'a'
    >>> (first, second), iterable = spy('abcdefg', 2)
    >>> first
    'a'
    >>> second
    'b'

The number of items requested can be larger than the number of ite ...

```python
def interleave(*iterables)
```
Return a new iterable yielding from each iterable in turn,
until the shortest is exhausted.

    >>> list(interleave([1, 2, 3], [4, 5], [6, 7, 8]))
    [1, 4, 6, 2, 5, 7]

For a version that doesn't terminate after the shortest iterable is
exhausted, see :func:`interleave_longest`.

```python
def interleave_longest(*iterables)
```
Return a new iterable yielding from each iterable in turn,
skipping any that are exhausted.

    >>> list(interleave_longest([1, 2, 3], [4, 5], [6, 7, 8]))
    [1, 4, 6, 2, 5, 7, 3, 8]

This function produces the same output as :func:`roundrobin`, but may
perform better for some inputs (in particular when the number of iterables
is large).

```python
def interleave_evenly(iterables, lengths=None)
```
Interleave multiple iterables so that their elements are evenly distributed
throughout the output sequence.

>>> iterables = [1, 2, 3, 4, 5], ['a', 'b']
>>> list(interleave_evenly(iterables))
[1, 2, 'a', 3, 4, 'b', 5]

>>> iterables = [[1, 2, 3], [4, 5], [6, 7, 8]]
>>> list(interleave_evenly(iterables))
[1, 6, 4, 2, 7, 3, 8, 5]

This function requires iterables of known length. Iterables without
``__len__()`` can be used by manually specifying lengths with *lengths*:

>>> from itertools import combinations, repeat
>>> iterables = [combinations(range(4), 2), ['a', 'b', 'c']]
>>> lengths = [4 * (4 - 1) // 2, 3]
>>> list(interleave_evenly(iterables, lengths=lengths))
[(0, 1), (0, 2), 'a', (0, 3 ...

```python
def interleave_randomly(*iterables)
```
Repeatedly select one of the input *iterables* at random and yield the next
item from it.

    >>> iterables = [1, 2, 3], 'abc', (True, False, None)
    >>> list(interleave_randomly(*iterables))  # doctest: +SKIP
    ['a', 'b', 1, 'c', True, False, None, 2, 3]

The relative order of the items in each input iterable will preserved. Note the
sequences of items with this property are not equally likely to be generated.

```python
def collapse(iterable, base_type=None, levels=None)
```
Flatten an iterable with multiple levels of nesting (e.g., a list of
lists of tuples) into non-iterable types.

    >>> iterable = [(1, 2), ([3, 4], [[5], [6]])]
    >>> list(collapse(iterable))
    [1, 2, 3, 4, 5, 6]

Binary and text strings are not considered iterable and
will not be collapsed.

To avoid collapsing other types, specify *base_type*:

    >>> iterable = ['ab', ('cd', 'ef'), ['gh', 'ij']]
    >>> list(collapse(iterable, base_type=tuple))
    ['ab', ('cd', 'ef'), 'gh', 'ij']

Specify *levels* to stop flattening after a certain level:

>>> iterable = [('a', ['b']), ('c', ['d'])]
>>> list(collapse(iterable))  # Fully flattened
['a', 'b', 'c', 'd']
>>> list(collapse(iterable, lev ...

```python
def side_effect(func, iterable, chunk_size=None, before=None, after=None)
```
Invoke *func* on each item in *iterable* (or on each *chunk_size* group
of items) before yielding the item.

`func` must be a function that takes a single argument. Its return value
will be discarded.

*before* and *after* are optional functions that take no arguments. They
will be executed before iteration starts and after it ends, respectively.

`side_effect` can be used for logging, updating progress bars, or anything
that is not functionally "pure."

Emitting a status message:

    >>> from more_itertools import consume
    >>> func = lambda item: print('Received {}'.format(item))
    >>> consume(side_effect(func, range(2)))
    Received 0
    Received 1

Operating on chunks of items: ...

```python
def sliced(seq, n, strict=False)
```
Yield slices of length *n* from the sequence *seq*.

>>> list(sliced((1, 2, 3, 4, 5, 6), 3))
[(1, 2, 3), (4, 5, 6)]

By the default, the last yielded slice will have fewer than *n* elements
if the length of *seq* is not divisible by *n*:

>>> list(sliced((1, 2, 3, 4, 5, 6, 7, 8), 3))
[(1, 2, 3), (4, 5, 6), (7, 8)]

If the length of *seq* is not divisible by *n* and *strict* is
``True``, then ``ValueError`` will be raised before the last
slice is yielded.

This function will only work for iterables that support slicing.
For non-sliceable iterables, see :func:`chunked`.

```python
def split_at(iterable, pred, maxsplit=-1, keep_separator=False)
```
Yield lists of items from *iterable*, where each list is delimited by
an item where callable *pred* returns ``True``.

    >>> list(split_at('abcdcba', lambda x: x == 'b'))
    [['a'], ['c', 'd', 'c'], ['a']]

    >>> list(split_at(range(10), lambda n: n % 2 == 1))
    [[0], [2], [4], [6], [8], []]

At most *maxsplit* splits are done. If *maxsplit* is not specified or -1,
then there is no limit on the number of splits:

    >>> list(split_at(range(10), lambda n: n % 2 == 1, maxsplit=2))
    [[0], [2], [4, 5, 6, 7, 8, 9]]

By default, the delimiting items are not included in the output.
To include them, set *keep_separator* to ``True``.

    >>> list(split_at('abcdcba', lambda x: x == 'b', ke ...

```python
def split_before(iterable, pred, maxsplit=-1)
```
Yield lists of items from *iterable*, where each list ends just before
an item for which callable *pred* returns ``True``:

    >>> list(split_before('OneTwo', lambda s: s.isupper()))
    [['O', 'n', 'e'], ['T', 'w', 'o']]

    >>> list(split_before(range(10), lambda n: n % 3 == 0))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

At most *maxsplit* splits are done. If *maxsplit* is not specified or -1,
then there is no limit on the number of splits:

    >>> list(split_before(range(10), lambda n: n % 3 == 0, maxsplit=2))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]]

```python
def split_after(iterable, pred, maxsplit=-1)
```
Yield lists of items from *iterable*, where each list ends with an
item where callable *pred* returns ``True``:

    >>> list(split_after('one1two2', lambda s: s.isdigit()))
    [['o', 'n', 'e', '1'], ['t', 'w', 'o', '2']]

    >>> list(split_after(range(10), lambda n: n % 3 == 0))
    [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9]]

At most *maxsplit* splits are done. If *maxsplit* is not specified or -1,
then there is no limit on the number of splits:

    >>> list(split_after(range(10), lambda n: n % 3 == 0, maxsplit=2))
    [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9]]

```python
def split_when(iterable, pred, maxsplit=-1)
```
Split *iterable* into pieces based on the output of *pred*.
*pred* should be a function that takes successive pairs of items and
returns ``True`` if the iterable should be split in between them.

For example, to find runs of increasing numbers, split the iterable when
element ``i`` is larger than element ``i + 1``:

    >>> list(split_when([1, 2, 3, 3, 2, 5, 2, 4, 2], lambda x, y: x > y))
    [[1, 2, 3, 3], [2, 5], [2, 4], [2]]

At most *maxsplit* splits are done. If *maxsplit* is not specified or -1,
then there is no limit on the number of splits:

    >>> list(split_when([1, 2, 3, 3, 2, 5, 2, 4, 2],
    ...                 lambda x, y: x > y, maxsplit=2))
    [[1, 2, 3, 3], [2, 5], [2, 4, ...

```python
def split_into(iterable, sizes)
```
Yield a list of sequential items from *iterable* of length 'n' for each
integer 'n' in *sizes*.

    >>> list(split_into([1,2,3,4,5,6], [1,2,3]))
    [[1], [2, 3], [4, 5, 6]]

If the sum of *sizes* is smaller than the length of *iterable*, then the
remaining items of *iterable* will not be returned.

    >>> list(split_into([1,2,3,4,5,6], [2,3]))
    [[1, 2], [3, 4, 5]]

If the sum of *sizes* is larger than the length of *iterable*, fewer items
will be returned in the iteration that overruns the *iterable* and further
lists will be empty:

    >>> list(split_into([1,2,3,4], [1,2,3,4]))
    [[1], [2, 3], [4], []]

When a ``None`` object is encountered in *sizes*, the returned list will
contai ...

```python
def padded(iterable, fillvalue=None, n=None, next_multiple=False)
```
Yield the elements from *iterable*, followed by *fillvalue*, such that
at least *n* items are emitted.

    >>> list(padded([1, 2, 3], '?', 5))
    [1, 2, 3, '?', '?']

If *next_multiple* is ``True``, *fillvalue* will be emitted until the
number of items emitted is a multiple of *n*:

    >>> list(padded([1, 2, 3, 4], n=3, next_multiple=True))
    [1, 2, 3, 4, None, None]

If *n* is ``None``, *fillvalue* will be emitted indefinitely.

To create an *iterable* of exactly size *n*, you can truncate with
:func:`islice`.

    >>> list(islice(padded([1, 2, 3], '?'), 5))
    [1, 2, 3, '?', '?']
    >>> list(islice(padded([1, 2, 3, 4, 5, 6, 7, 8], '?'), 5))
    [1, 2, 3, 4, 5]

```python
def repeat_each(iterable, n=2)
```
Repeat each element in *iterable* *n* times.

>>> list(repeat_each('ABC', 3))
['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']

```python
def repeat_last(iterable, default=None)
```
After the *iterable* is exhausted, keep yielding its last element.

    >>> list(islice(repeat_last(range(3)), 5))
    [0, 1, 2, 2, 2]

If the iterable is empty, yield *default* forever::

    >>> list(islice(repeat_last(range(0), 42), 5))
    [42, 42, 42, 42, 42]

```python
def distribute(n, iterable)
```
Distribute the items from *iterable* among *n* smaller iterables.

    >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
    >>> list(group_1)
    [1, 3, 5]
    >>> list(group_2)
    [2, 4, 6]

If the length of *iterable* is not evenly divisible by *n*, then the
length of the returned iterables will not be identical:

    >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
    >>> [list(c) for c in children]
    [[1, 4, 7], [2, 5], [3, 6]]

If the length of *iterable* is smaller than *n*, then the last returned
iterables will be empty:

    >>> children = distribute(5, [1, 2, 3])
    >>> [list(c) for c in children]
    [[1], [2], [3], [], []]

This function uses :func:`itertools.tee` ...

```python
def stagger(iterable, offsets=(-1, 0, 1), longest=False, fillvalue=None)
```
Yield tuples whose elements are offset from *iterable*.
The amount by which the `i`-th item in each tuple is offset is given by
the `i`-th item in *offsets*.

    >>> list(stagger([0, 1, 2, 3]))
    [(None, 0, 1), (0, 1, 2), (1, 2, 3)]
    >>> list(stagger(range(8), offsets=(0, 2, 4)))
    [(0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 7)]

By default, the sequence will end when the final element of a tuple is the
last item in the iterable. To continue until the first element of a tuple
is the last item in the iterable, set *longest* to ``True``::

    >>> list(stagger([0, 1, 2, 3], longest=True))
    [(None, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, None), (3, None, None)]

By default, ``None`` will be ...

```python
def zip_offset(*iterables, offsets, longest=False, fillvalue=None)
```
``zip`` the input *iterables* together, but offset the `i`-th iterable
by the `i`-th item in *offsets*.

    >>> list(zip_offset('0123', 'abcdef', offsets=(0, 1)))
    [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e')]

This can be used as a lightweight alternative to SciPy or pandas to analyze
data sets in which some series have a lead or lag relationship.

By default, the sequence will end when the shortest iterable is exhausted.
To continue until the longest iterable is exhausted, set *longest* to
``True``.

    >>> list(zip_offset('0123', 'abcdef', offsets=(0, 1), longest=True))
    [('0', 'b'), ('1', 'c'), ('2', 'd'), ('3', 'e'), (None, 'f')]

By default, ``None`` will be used to replace ...

```python
def sort_together(iterables, key_list=(0,), key=None, reverse=False, strict=False)
```
Return the input iterables sorted together, with *key_list* as the
priority for sorting. All iterables are trimmed to the length of the
shortest one.

This can be used like the sorting function in a spreadsheet. If each
iterable represents a column of data, the key list determines which
columns are used for sorting.

By default, all iterables are sorted using the ``0``-th iterable::

    >>> iterables = [(4, 3, 2, 1), ('a', 'b', 'c', 'd')]
    >>> sort_together(iterables)
    [(1, 2, 3, 4), ('d', 'c', 'b', 'a')]

Set a different key list to sort according to another iterable.
Specifying multiple keys dictates how ties are broken::

    >>> iterables = [(3, 1, 2), (0, 1, 0), ('c', 'b', 'a')] ...

```python
def unzip(iterable)
```
The inverse of :func:`zip`, this function disaggregates the elements
of the zipped *iterable*.

The ``i``-th iterable contains the ``i``-th element from each element
of the zipped iterable. The first element is used to determine the
length of the remaining elements.

    >>> iterable = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    >>> letters, numbers = unzip(iterable)
    >>> list(letters)
    ['a', 'b', 'c', 'd']
    >>> list(numbers)
    [1, 2, 3, 4]

This is similar to using ``zip(*iterable)``, but it avoids reading
*iterable* into memory. Note, however, that this function uses
:func:`itertools.tee` and thus may require significant storage.

```python
def divide(n, iterable)
```
Divide the elements from *iterable* into *n* parts, maintaining
order.

    >>> group_1, group_2 = divide(2, [1, 2, 3, 4, 5, 6])
    >>> list(group_1)
    [1, 2, 3]
    >>> list(group_2)
    [4, 5, 6]

If the length of *iterable* is not evenly divisible by *n*, then the
length of the returned iterables will not be identical:

    >>> children = divide(3, [1, 2, 3, 4, 5, 6, 7])
    >>> [list(c) for c in children]
    [[1, 2, 3], [4, 5], [6, 7]]

If the length of the iterable is smaller than n, then the last returned
iterables will be empty:

    >>> children = divide(5, [1, 2, 3])
    >>> [list(c) for c in children]
    [[1], [2], [3], [], []]

This function will exhaust the iterable before r ...

```python
def always_iterable(obj, base_type=(str, bytes))
```
If *obj* is iterable, return an iterator over its items::

    >>> obj = (1, 2, 3)
    >>> list(always_iterable(obj))
    [1, 2, 3]

If *obj* is not iterable, return a one-item iterable containing *obj*::

    >>> obj = 1
    >>> list(always_iterable(obj))
    [1]

If *obj* is ``None``, return an empty iterable:

    >>> obj = None
    >>> list(always_iterable(None))
    []

By default, binary and text strings are not considered iterable::

    >>> obj = 'foo'
    >>> list(always_iterable(obj))
    ['foo']

If *base_type* is set, objects for which ``isinstance(obj, base_type)``
returns ``True`` won't be considered iterable.

    >>> obj = {'a': 1}
    >>> list(always_iterable(obj))  # Iterat ...

```python
def adjacent(predicate, iterable, distance=1)
```
Return an iterable over `(bool, item)` tuples where the `item` is
drawn from *iterable* and the `bool` indicates whether
that item satisfies the *predicate* or is adjacent to an item that does.

For example, to find whether items are adjacent to a ``3``::

    >>> list(adjacent(lambda x: x == 3, range(6)))
    [(False, 0), (False, 1), (True, 2), (True, 3), (True, 4), (False, 5)]

Set *distance* to change what counts as adjacent. For example, to find
whether items are two places away from a ``3``:

    >>> list(adjacent(lambda x: x == 3, range(6), distance=2))
    [(False, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5)]

This is useful for contextualizing the results of a search fu ...

```python
def groupby_transform(iterable, keyfunc=None, valuefunc=None, reducefunc=None)
```
An extension of :func:`itertools.groupby` that can apply transformations
to the grouped data.

* *keyfunc* is a function computing a key value for each item in *iterable*
* *valuefunc* is a function that transforms the individual items from
  *iterable* after grouping
* *reducefunc* is a function that transforms each group of items

>>> iterable = 'aAAbBBcCC'
>>> keyfunc = lambda k: k.upper()
>>> valuefunc = lambda v: v.lower()
>>> reducefunc = lambda g: ''.join(g)
>>> list(groupby_transform(iterable, keyfunc, valuefunc, reducefunc))
[('A', 'aaa'), ('B', 'bbb'), ('C', 'ccc')]

Each optional argument defaults to an identity function if not specified.

:func:`groupby_transform` is useful when ...

```python
class numeric_range(Sequence)
```
An extension of the built-in ``range()`` function whose arguments can
be any orderable numeric type.

With only *stop* specified, *start* defaults to ``0`` and *step*
defaults to ``1``. The output items will match the type of *stop*:

    >>> list(numeric_range(3.5))
    [0.0, 1.0, 2.0, 3.0]

With only *start* and *stop* specified, *step* defaults to ``1``, typed
to match *stop* - *start*. The output items will match the type of
``start + step``:

    >>> from decimal import Decimal
    >>> start = Decimal('2.1')
    >>> stop = Decimal('5.1')
    >>> list(numeric_range(start, stop))
    [Decimal('2.1'), Decimal('3.1'), Decimal('4.1')]

With *start*, *stop*, and *step*  specified the output i ...

```python
# method of numeric_range
def __init__(self, *args)
```


```python
# method of numeric_range
def count(self, value)
```


```python
# method of numeric_range
def index(self, value)
```


```python
def count_cycle(iterable, n=None)
```
Cycle through the items from *iterable* up to *n* times, yielding
the number of completed cycles along with each item. If *n* is omitted the
process repeats indefinitely.

>>> list(count_cycle('AB', 3))
[(0, 'A'), (0, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]

```python
def mark_ends(iterable)
```
Yield 3-tuples of the form ``(is_first, is_last, item)``.

>>> list(mark_ends('ABC'))
[(True, False, 'A'), (False, False, 'B'), (False, True, 'C')]

Use this when looping over an iterable to take special action on its first
and/or last items:

>>> iterable = ['Header', 100, 200, 'Footer']
>>> total = 0
>>> for is_first, is_last, item in mark_ends(iterable):
...     if is_first:
...         continue  # Skip the header
...     if is_last:
...         continue  # Skip the footer
...     total += item
>>> print(total)
300

```python
def locate(iterable, pred=bool, window_size=None)
```
Yield the index of each item in *iterable* for which *pred* returns
``True``.

*pred* defaults to :func:`bool`, which will select truthy items:

    >>> list(locate([0, 1, 1, 0, 1, 0, 0]))
    [1, 2, 4]

Set *pred* to a custom function to, e.g., find the indexes for a particular
item.

    >>> list(locate(['a', 'b', 'c', 'b'], lambda x: x == 'b'))
    [1, 3]

If *window_size* is given, then the *pred* function will be called with
the values in each window. This enables searching for sub-sequences.
Note that *pred* may receive fewer than *window_size* arguments at the end of
the iterable.

    >>> iterable = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    >>> pred = lambda *args: args == (1, 2, 3) ...

```python
def longest_common_prefix(iterables)
```
Yield elements of the longest common prefix among given *iterables*.

>>> ''.join(longest_common_prefix(['abcd', 'abc', 'abf']))
'ab'

```python
def lstrip(iterable, pred)
```
Yield the items from *iterable*, but strip any from the beginning
for which *pred* returns ``True``.

For example, to remove a set of items from the start of an iterable:

    >>> iterable = (None, False, None, 1, 2, None, 3, False, None)
    >>> pred = lambda x: x in {None, False, ''}
    >>> list(lstrip(iterable, pred))
    [1, 2, None, 3, False, None]

This function is analogous to :func:`str.lstrip`, and is essentially
a wrapper for :func:`itertools.dropwhile`.

```python
def rstrip(iterable, pred)
```
Yield the items from *iterable*, but strip any from the end
for which *pred* returns ``True``.

For example, to remove a set of items from the end of an iterable:

    >>> iterable = (None, False, None, 1, 2, None, 3, False, None)
    >>> pred = lambda x: x in {None, False, ''}
    >>> list(rstrip(iterable, pred))
    [None, False, None, 1, 2, None, 3]

This function is analogous to :func:`str.rstrip`.

```python
def strip(iterable, pred)
```
Yield the items from *iterable*, but strip any from the
beginning and end for which *pred* returns ``True``.

For example, to remove a set of items from both ends of an iterable:

    >>> iterable = (None, False, None, 1, 2, None, 3, False, None)
    >>> pred = lambda x: x in {None, False, ''}
    >>> list(strip(iterable, pred))
    [1, 2, None, 3]

This function is analogous to :func:`str.strip`.

```python
class islice_extended()
```
An extension of :func:`itertools.islice` that supports negative values
for *stop*, *start*, and *step*.

    >>> iterator = iter('abcdefgh')
    >>> list(islice_extended(iterator, -4, -1))
    ['e', 'f', 'g']

Slices with negative values require some caching of *iterable*, but this
function takes care to minimize the amount of memory required.

For example, you can use a negative step with an infinite iterator:

    >>> from itertools import count
    >>> list(islice_extended(count(), 110, 99, -2))
    [110, 108, 106, 104, 102, 100]

You can also use slice notation directly:

    >>> iterator = map(str, count())
    >>> it = islice_extended(iterator)[10:20:2]
    >>> list(it)
    ['10', '12' ...

```python
# method of islice_extended
def __init__(self, iterable, *args)
```


```python
def always_reversible(iterable)
```
An extension of :func:`reversed` that supports all iterables, not
just those which implement the ``Reversible`` or ``Sequence`` protocols.

    >>> print(*always_reversible(x for x in range(3)))
    2 1 0

If the iterable is already reversible, this function returns the
result of :func:`reversed()`. If the iterable is not reversible,
this function will cache the remaining items in the iterable and
yield them in reverse order, which may require significant storage.

```python
def consecutive_groups(iterable, ordering=None)
```
Yield groups of consecutive items using :func:`itertools.groupby`.
The *ordering* function determines whether two items are adjacent by
returning their position.

By default, the ordering function is the identity function. This is
suitable for finding runs of numbers:

    >>> iterable = [1, 10, 11, 12, 20, 30, 31, 32, 33, 40]
    >>> for group in consecutive_groups(iterable):
    ...     print(list(group))
    [1]
    [10, 11, 12]
    [20]
    [30, 31, 32, 33]
    [40]

To find runs of adjacent letters, apply :func:`ord` function
to convert letters to ordinals.

    >>> iterable = 'abcdfgilmnop'
    >>> ordering = ord
    >>> for group in consecutive_groups(iterable, ordering):
    ... ...

```python
def difference(iterable, func=sub, *, initial=None)
```
This function is the inverse of :func:`itertools.accumulate`. By default
it will compute the first difference of *iterable* using
:func:`operator.sub`:

    >>> from itertools import accumulate
    >>> iterable = accumulate([0, 1, 2, 3, 4])  # produces 0, 1, 3, 6, 10
    >>> list(difference(iterable))
    [0, 1, 2, 3, 4]

*func* defaults to :func:`operator.sub`, but other functions can be
specified. They will be applied as follows::

    A, B, C, D, ... --> A, func(B, A), func(C, B), func(D, C), ...

For example, to do progressive division:

    >>> iterable = [1, 2, 6, 24, 120]
    >>> func = lambda x, y: x // y
    >>> list(difference(iterable, func))
    [1, 2, 3, 4, 5]

If the *initial* ...

```python
class SequenceView(Sequence)
```
Return a read-only view of the sequence object *target*.

:class:`SequenceView` objects are analogous to Python's built-in
"dictionary view" types. They provide a dynamic view of a sequence's items,
meaning that when the sequence updates, so does the view.

    >>> seq = ['0', '1', '2']
    >>> view = SequenceView(seq)
    >>> view
    SequenceView(['0', '1', '2'])
    >>> seq.append('3')
    >>> view
    SequenceView(['0', '1', '2', '3'])

Sequence views support indexing, slicing, and length queries. They act
like the underlying sequence, except they don't allow assignment:

    >>> view[1]
    '1'
    >>> view[1:-1]
    ['1', '2']
    >>> len(view)
    4

Sequence views are useful as an al ...

```python
# method of SequenceView
def __init__(self, target)
```


```python
class seekable()
```
Wrap an iterator to allow for seeking backward and forward. This
progressively caches the items in the source iterable so they can be
re-visited.

Call :meth:`seek` with an index to seek to that position in the source
iterable.

To "reset" an iterator, seek to ``0``:

    >>> from itertools import count
    >>> it = seekable((str(n) for n in count()))
    >>> next(it), next(it), next(it)
    ('0', '1', '2')
    >>> it.seek(0)
    >>> next(it), next(it), next(it)
    ('0', '1', '2')

You can also seek forward:

    >>> it = seekable((str(n) for n in range(20)))
    >>> it.seek(10)
    >>> next(it)
    '10'
    >>> it.seek(20)  # Seeking past the end of the source isn't a problem
    >>> list( ...

```python
# method of seekable
def __init__(self, iterable, maxlen=None)
```


```python
# method of seekable
def peek(self, default=_marker)
```


```python
# method of seekable
def elements(self)
```


```python
# method of seekable
def seek(self, index)
```


```python
# method of seekable
def relative_seek(self, count)
```


```python
class run_length()
```
:func:`run_length.encode` compresses an iterable with run-length encoding.
It yields groups of repeated items with the count of how many times they
were repeated:

    >>> uncompressed = 'abbcccdddd'
    >>> list(run_length.encode(uncompressed))
    [('a', 1), ('b', 2), ('c', 3), ('d', 4)]

:func:`run_length.decode` decompresses an iterable that was previously
compressed with run-length encoding. It yields the items of the
decompressed iterable:

    >>> compressed = [('a', 1), ('b', 2), ('c', 3), ('d', 4)]
    >>> list(run_length.decode(compressed))
    ['a', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd', 'd']

```python
# method of run_length
def encode(iterable)
```


```python
# method of run_length
def decode(iterable)
```


```python
def exactly_n(iterable, n, predicate=bool)
```
Return ``True`` if exactly ``n`` items in the iterable are ``True``
according to the *predicate* function.

    >>> exactly_n([True, True, False], 2)
    True
    >>> exactly_n([True, True, False], 1)
    False
    >>> exactly_n([0, 1, 2, 3, 4, 5], 3, lambda x: x < 3)
    True

The iterable will be advanced until ``n + 1`` truthy items are encountered,
so avoid calling it on infinite iterables.

```python
def circular_shifts(iterable, steps=1)
```
Yield the circular shifts of *iterable*.

>>> list(circular_shifts(range(4)))
[(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]

Set *steps* to the number of places to rotate to the left
(or to the right if negative).  Defaults to 1.

>>> list(circular_shifts(range(4), 2))
[(0, 1, 2, 3), (2, 3, 0, 1)]

>>> list(circular_shifts(range(4), -1))
[(0, 1, 2, 3), (3, 0, 1, 2), (2, 3, 0, 1), (1, 2, 3, 0)]

```python
def make_decorator(wrapping_func, result_index=0)
```
Return a decorator version of *wrapping_func*, which is a function that
modifies an iterable. *result_index* is the position in that function's
signature where the iterable goes.

This lets you use itertools on the "production end," i.e. at function
definition. This can augment what the function returns without changing the
function's code.

For example, to produce a decorator version of :func:`chunked`:

    >>> from more_itertools import chunked
    >>> chunker = make_decorator(chunked, result_index=0)
    >>> @chunker(3)
    ... def iter_range(n):
    ...     return iter(range(n))
    ...
    >>> list(iter_range(9))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

To only allow truthy items to be r ...

```python
def map_reduce(iterable, keyfunc, valuefunc=None, reducefunc=None)
```
Return a dictionary that maps the items in *iterable* to categories
defined by *keyfunc*, transforms them with *valuefunc*, and
then summarizes them by category with *reducefunc*.

*valuefunc* defaults to the identity function if it is unspecified.
If *reducefunc* is unspecified, no summarization takes place:

    >>> keyfunc = lambda x: x.upper()
    >>> result = map_reduce('abbccc', keyfunc)
    >>> sorted(result.items())
    [('A', ['a']), ('B', ['b', 'b']), ('C', ['c', 'c', 'c'])]

Specifying *valuefunc* transforms the categorized items:

    >>> keyfunc = lambda x: x.upper()
    >>> valuefunc = lambda x: 1
    >>> result = map_reduce('abbccc', keyfunc, valuefunc)
    >>> sorted(result.i ...

```python
def rlocate(iterable, pred=bool, window_size=None)
```
Yield the index of each item in *iterable* for which *pred* returns
``True``, starting from the right and moving left.

*pred* defaults to :func:`bool`, which will select truthy items:

    >>> list(rlocate([0, 1, 1, 0, 1, 0, 0]))  # Truthy at 1, 2, and 4
    [4, 2, 1]

Set *pred* to a custom function to, e.g., find the indexes for a particular
item:

    >>> iterator = iter('abcb')
    >>> pred = lambda x: x == 'b'
    >>> list(rlocate(iterator, pred))
    [3, 1]

If *window_size* is given, then the *pred* function will be called with
that many items. This enables searching for sub-sequences:

    >>> iterable = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    >>> pred = lambda *args: args == (1, 2 ...

```python
def replace(iterable, pred, substitutes, count=None, window_size=1)
```
Yield the items from *iterable*, replacing the items for which *pred*
returns ``True`` with the items from the iterable *substitutes*.

    >>> iterable = [1, 1, 0, 1, 1, 0, 1, 1]
    >>> pred = lambda x: x == 0
    >>> substitutes = (2, 3)
    >>> list(replace(iterable, pred, substitutes))
    [1, 1, 2, 3, 1, 1, 2, 3, 1, 1]

If *count* is given, the number of replacements will be limited:

    >>> iterable = [1, 1, 0, 1, 1, 0, 1, 1, 0]
    >>> pred = lambda x: x == 0
    >>> substitutes = [None]
    >>> list(replace(iterable, pred, substitutes, count=2))
    [1, 1, None, 1, 1, None, 1, 1, 0]

Use *window_size* to control the number of items passed as arguments to
*pred*. This allows for loc ...

```python
def partitions(iterable)
```
Yield all possible order-preserving partitions of *iterable*.

>>> iterable = 'abc'
>>> for part in partitions(iterable):
...     print([''.join(p) for p in part])
['abc']
['a', 'bc']
['ab', 'c']
['a', 'b', 'c']

This is unrelated to :func:`partition`.

```python
def set_partitions(iterable, k=None, min_size=None, max_size=None)
```
Yield the set partitions of *iterable* into *k* parts. Set partitions are
not order-preserving.

>>> iterable = 'abc'
>>> for part in set_partitions(iterable, 2):
...     print([''.join(p) for p in part])
['a', 'bc']
['ab', 'c']
['b', 'ac']


If *k* is not given, every set partition is generated.

>>> iterable = 'abc'
>>> for part in set_partitions(iterable):
...     print([''.join(p) for p in part])
['abc']
['a', 'bc']
['ab', 'c']
['b', 'ac']
['a', 'b', 'c']

if *min_size* and/or *max_size* are given, the minimum and/or maximum size
per block in partition is set.

>>> iterable = 'abc'
>>> for part in set_partitions(iterable, min_size=2):
...     print([''.join(p) for p in part])
['abc']
>>> ...

```python
class time_limited()
```
Yield items from *iterable* until *limit_seconds* have passed.
If the time limit expires before all items have been yielded, the
``timed_out`` parameter will be set to ``True``.

>>> from time import sleep
>>> def generator():
...     yield 1
...     yield 2
...     sleep(0.2)
...     yield 3
>>> iterable = time_limited(0.1, generator())
>>> list(iterable)
[1, 2]
>>> iterable.timed_out
True

Note that the time is checked before each item is yielded, and iteration
stops if  the time elapsed is greater than *limit_seconds*. If your time
limit is 1 second, but it takes 2 seconds to generate the first item from
the iterable, the function will run for 2 seconds and not yield anything.
As a specia ...

```python
# method of time_limited
def __init__(self, limit_seconds, iterable)
```


```python
def only(iterable, default=None, too_long=None)
```
If *iterable* has only one item, return it.
If it has zero items, return *default*.
If it has more than one item, raise the exception given by *too_long*,
which is ``ValueError`` by default.

>>> only([], default='missing')
'missing'
>>> only([1])
1
>>> only([1, 2])  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
ValueError: Expected exactly one item in iterable, but got 1, 2,
 and perhaps more.'
>>> only([1, 2], too_long=TypeError)  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
...
TypeError

Note that :func:`only` attempts to advance *iterable* twice to ensure there
is only one item.  See :func:`spy` or :func:`peekable` to check
iterable ...

```python
def ichunked(iterable, n)
```
Break *iterable* into sub-iterables with *n* elements each.
:func:`ichunked` is like :func:`chunked`, but it yields iterables
instead of lists.

If the sub-iterables are read in order, the elements of *iterable*
won't be stored in memory.
If they are read out of order, :func:`itertools.tee` is used to cache
elements as necessary.

>>> from itertools import count
>>> all_chunks = ichunked(count(), 4)
>>> c_1, c_2, c_3 = next(all_chunks), next(all_chunks), next(all_chunks)
>>> list(c_2)  # c_1's elements have been cached; c_3's haven't been
[4, 5, 6, 7]
>>> list(c_1)
[0, 1, 2, 3]
>>> list(c_3)
[8, 9, 10, 11]

```python
def iequals(*iterables)
```
Return ``True`` if all given *iterables* are equal to each other,
which means that they contain the same elements in the same order.

The function is useful for comparing iterables of different data types
or iterables that do not support equality checks.

>>> iequals("abc", ['a', 'b', 'c'], ('a', 'b', 'c'), iter("abc"))
True

>>> iequals("abc", "acb")
False

Not to be confused with :func:`all_equal`, which checks whether all
elements of iterable are equal to each other.

```python
def distinct_combinations(iterable, r)
```
Yield the distinct combinations of *r* items taken from *iterable*.

    >>> list(distinct_combinations([0, 0, 1], 2))
    [(0, 0), (0, 1)]

Equivalent to ``set(combinations(iterable))``, except duplicates are not
generated and thrown away. For larger input sequences this is much more
efficient.

```python
def filter_except(validator, iterable, *exceptions)
```
Yield the items from *iterable* for which the *validator* function does
not raise one of the specified *exceptions*.

*validator* is called for each item in *iterable*.
It should be a function that accepts one argument and raises an exception
if that item is not valid.

>>> iterable = ['1', '2', 'three', '4', None]
>>> list(filter_except(int, iterable, ValueError, TypeError))
['1', '2', '4']

If an exception other than one given by *exceptions* is raised by
*validator*, it is raised like normal.

```python
def map_except(function, iterable, *exceptions)
```
Transform each item from *iterable* with *function* and yield the
result, unless *function* raises one of the specified *exceptions*.

*function* is called to transform each item in *iterable*.
It should accept one argument.

>>> iterable = ['1', '2', 'three', '4', None]
>>> list(map_except(int, iterable, ValueError, TypeError))
[1, 2, 4]

If an exception other than one given by *exceptions* is raised by
*function*, it is raised like normal.

```python
def map_if(iterable, pred, func, func_else=None)
```
Evaluate each item from *iterable* using *pred*. If the result is
equivalent to ``True``, transform the item with *func* and yield it.
Otherwise, transform the item with *func_else* and yield it.

*pred*, *func*, and *func_else* should each be functions that accept
one argument. By default, *func_else* is the identity function.

>>> from math import sqrt
>>> iterable = list(range(-5, 5))
>>> iterable
[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
>>> list(map_if(iterable, lambda x: x > 3, lambda x: 'toobig'))
[-5, -4, -3, -2, -1, 0, 1, 2, 3, 'toobig']
>>> list(map_if(iterable, lambda x: x >= 0,
... lambda x: f'{sqrt(x):.2f}', lambda x: None))
[None, None, None, None, None, '0.00', '1.00', '1.41', '1.73 ...

```python
def sample(iterable, k, weights=None, *, counts=None, strict=False)
```
Return a *k*-length list of elements chosen (without replacement)
from the *iterable*.

Similar to :func:`random.sample`, but works on inputs that aren't
indexable (such as sets and dictionaries) and on inputs where the
size isn't known in advance (such as generators).

>>> iterable = range(100)
>>> sample(iterable, 5)  # doctest: +SKIP
[81, 60, 96, 16, 4]

For iterables with repeated elements, you may supply *counts* to
indicate the repeats.

>>> iterable = ['a', 'b']
>>> counts = [3, 4]  # Equivalent to 'a', 'a', 'a', 'b', 'b', 'b', 'b'
>>> sample(iterable, k=3, counts=counts)  # doctest: +SKIP
['a', 'a', 'b']

An iterable with *weights* may be given:

>>> iterable = range(100)
>>> weights ...

```python
def is_sorted(iterable, key=None, reverse=False, strict=False)
```
Returns ``True`` if the items of iterable are in sorted order, and
``False`` otherwise. *key* and *reverse* have the same meaning that they do
in the built-in :func:`sorted` function.

>>> is_sorted(['1', '2', '3', '4', '5'], key=int)
True
>>> is_sorted([5, 4, 3, 1, 2], reverse=True)
False

If *strict*, tests for strict sorting, that is, returns ``False`` if equal
elements are found:

>>> is_sorted([1, 2, 2])
True
>>> is_sorted([1, 2, 2], strict=True)
False

The function returns ``False`` after encountering the first out-of-order
item, which means it may produce results that differ from the built-in
:func:`sorted` function for objects with unusual comparison dynamics
(like ``math.nan``). If ...

```python
class AbortThread(BaseException)
```


```python
class callback_iter()
```
Convert a function that uses callbacks to an iterator.

.. deprecated:: 11.0.0
   Will be removed in a future major release.

Let *func* be a function that takes a `callback` keyword argument.
For example:

>>> def func(callback=None):
...     for i, c in [(1, 'a'), (2, 'b'), (3, 'c')]:
...         if callback:
...             callback(i, c)
...     return 4


Use ``with callback_iter(func)`` to get an iterator over the parameters
that are delivered to the callback.

>>> with callback_iter(func) as it:
...     for args, kwargs in it:
...         print(args)
(1, 'a')
(2, 'b')
(3, 'c')

The function will be called in a background thread. The ``done`` property
indicates whether it has completed ...

```python
# method of callback_iter
def __init__(self, func, callback_kwd='callback', wait_seconds=0.1)
```


```python
# method of callback_iter
def __enter__(self)
```


```python
# method of callback_iter
def __exit__(self, exc_type, exc_value, traceback)
```


```python
# method of callback_iter
def done(self)
```


```python
# method of callback_iter
def result(self)
```


```python
def windowed_complete(iterable, n)
```
Yield ``(beginning, middle, end)`` tuples, where:

* Each ``middle`` has *n* items from *iterable*
* Each ``beginning`` has the items before the ones in ``middle``
* Each ``end`` has the items after the ones in ``middle``

>>> iterable = range(7)
>>> n = 3
>>> for beginning, middle, end in windowed_complete(iterable, n):
...     print(beginning, middle, end)
() (0, 1, 2) (3, 4, 5, 6)
(0,) (1, 2, 3) (4, 5, 6)
(0, 1) (2, 3, 4) (5, 6)
(0, 1, 2) (3, 4, 5) (6,)
(0, 1, 2, 3) (4, 5, 6) ()

Note that *n* must be at least 0 and most equal to the length of
*iterable*.

This function will exhaust the iterable and may require significant
storage.

```python
def all_unique(iterable, key=None)
```
Returns ``True`` if all the elements of *iterable* are unique (no two
elements are equal).

    >>> all_unique('ABCB')
    False

If a *key* function is specified, it will be used to make comparisons.

    >>> all_unique('ABCb')
    True
    >>> all_unique('ABCb', str.lower)
    False

The function returns as soon as the first non-unique element is
encountered. Iterables with a mix of hashable and unhashable items can
be used, but the function will be slower for unhashable items.

```python
def nth_product(index, *iterables, repeat=1)
```
Equivalent to ``list(product(*iterables, repeat=repeat))[index]``.

The products of *iterables* can be ordered lexicographically.
:func:`nth_product` computes the product at sort position *index* without
computing the previous products.

    >>> nth_product(8, range(2), range(2), range(2), range(2))
    (1, 0, 0, 0)

The *repeat* keyword argument specifies the number of repetitions
of the iterables.  The above example is equivalent to::

    >>> nth_product(8, range(2), repeat=4)
    (1, 0, 0, 0)

``IndexError`` will be raised if the given *index* is invalid.

```python
def nth_permutation(iterable, r, index)
```
Equivalent to ``list(permutations(iterable, r))[index]```

The subsequences of *iterable* that are of length *r* where order is
important can be ordered lexicographically. :func:`nth_permutation`
computes the subsequence at sort position *index* directly, without
computing the previous subsequences.

    >>> nth_permutation('ghijk', 2, 5)
    ('h', 'i')

``ValueError`` will be raised If *r* is negative.
``IndexError`` will be raised if the given *index* is invalid.

```python
def nth_combination_with_replacement(iterable, r, index)
```
Equivalent to
``list(combinations_with_replacement(iterable, r))[index]``.


The subsequences with repetition of *iterable* that are of length *r* can
be ordered lexicographically. :func:`nth_combination_with_replacement`
computes the subsequence at sort position *index* directly, without
computing the previous subsequences with replacement.

    >>> nth_combination_with_replacement(range(5), 3, 5)
    (0, 1, 1)

``ValueError`` will be raised If *r* is negative.
``IndexError`` will be raised if the given *index* is invalid.

```python
def value_chain(*args)
```
Yield all arguments passed to the function in the same order in which
they were passed. If an argument itself is iterable then iterate over its
values.

    >>> list(value_chain(1, 2, 3, [4, 5, 6]))
    [1, 2, 3, 4, 5, 6]

Binary and text strings are not considered iterable and are emitted
as-is:

    >>> list(value_chain('12', '34', ['56', '78']))
    ['12', '34', '56', '78']

Pre- or postpend a single element to an iterable:

    >>> list(value_chain(1, [2, 3, 4, 5, 6]))
    [1, 2, 3, 4, 5, 6]
    >>> list(value_chain([1, 2, 3, 4, 5], 6))
    [1, 2, 3, 4, 5, 6]

Multiple levels of nesting are not flattened.

```python
def product_index(element, *iterables, repeat=1)
```
Equivalent to ``list(product(*iterables, repeat=repeat)).index(tuple(element))``

The products of *iterables* can be ordered lexicographically.
:func:`product_index` computes the first index of *element* without
computing the previous products.

    >>> product_index([8, 2], range(10), range(5))
    42

The *repeat* keyword argument specifies the number of repetitions
of the iterables::

    >>> product_index([8, 0, 7], range(10), repeat=3)
    807

``ValueError`` will be raised if the given *element* isn't in the product
of *args*.

```python
def combination_index(element, iterable)
```
Equivalent to ``list(combinations(iterable, r)).index(element)``

The subsequences of *iterable* that are of length *r* can be ordered
lexicographically. :func:`combination_index` computes the index of the
first *element*, without computing the previous combinations.

    >>> combination_index('adf', 'abcdefg')
    10

``ValueError`` will be raised if the given *element* isn't one of the
combinations of *iterable*.

```python
def combination_with_replacement_index(element, iterable)
```
Equivalent to
``list(combinations_with_replacement(iterable, r)).index(element)``

The subsequences with repetition of *iterable* that are of length *r* can
be ordered lexicographically. :func:`combination_with_replacement_index`
computes the index of the first *element*, without computing the previous
combinations with replacement.

    >>> combination_with_replacement_index('adf', 'abcdefg')
    20

``ValueError`` will be raised if the given *element* isn't one of the
combinations with replacement of *iterable*.

```python
def permutation_index(element, iterable)
```
Equivalent to ``list(permutations(iterable, r)).index(element)```

The subsequences of *iterable* that are of length *r* where order is
important can be ordered lexicographically. :func:`permutation_index`
computes the index of the first *element* directly, without computing
the previous permutations.

    >>> permutation_index([1, 3, 2], range(5))
    19

``ValueError`` will be raised if the given *element* isn't one of the
permutations of *iterable*.

```python
class countable()
```
Wrap *iterable* and keep a count of how many items have been consumed.

The ``items_seen`` attribute starts at ``0`` and increments as the iterable
is consumed:

    >>> iterable = map(str, range(10))
    >>> it = countable(iterable)
    >>> it.items_seen
    0
    >>> next(it), next(it)
    ('0', '1')
    >>> list(it)
    ['2', '3', '4', '5', '6', '7', '8', '9']
    >>> it.items_seen
    10

```python
# method of countable
def __init__(self, iterable)
```


```python
def chunked_even(iterable, n)
```
Break *iterable* into lists of approximately length *n*.
Items are distributed such the lengths of the lists differ by at most
1 item.

>>> iterable = [1, 2, 3, 4, 5, 6, 7]
>>> n = 3
>>> list(chunked_even(iterable, n))  # List lengths: 3, 2, 2
[[1, 2, 3], [4, 5], [6, 7]]
>>> list(chunked(iterable, n))  # List lengths: 3, 3, 1
[[1, 2, 3], [4, 5, 6], [7]]

```python
def zip_broadcast(*objects, scalar_types=(str, bytes), strict=False)
```
A version of :func:`zip` that "broadcasts" any scalar
(i.e., non-iterable) items into output tuples.

>>> iterable_1 = [1, 2, 3]
>>> iterable_2 = ['a', 'b', 'c']
>>> scalar = '_'
>>> list(zip_broadcast(iterable_1, iterable_2, scalar))
[(1, 'a', '_'), (2, 'b', '_'), (3, 'c', '_')]

The *scalar_types* keyword argument determines what types are considered
scalar. It is set to ``(str, bytes)`` by default. Set it to ``None`` to
treat strings and byte strings as iterable:

>>> list(zip_broadcast('abc', 0, 'xyz', scalar_types=None))
[('a', 0, 'x'), ('b', 0, 'y'), ('c', 0, 'z')]

If the *strict* keyword argument is ``True``, then
``ValueError`` will be raised if any of the iterables have
different l ...

```python
def unique_in_window(iterable, n, key=None)
```
Yield the items from *iterable* that haven't been seen recently.
*n* is the size of the sliding window.

    >>> iterable = [0, 1, 0, 2, 3, 0]
    >>> n = 3
    >>> list(unique_in_window(iterable, n))
    [0, 1, 2, 3, 0]

The *key* function, if provided, will be used to determine uniqueness:

    >>> list(unique_in_window('abAcda', 3, key=lambda x: x.lower()))
    ['a', 'b', 'c', 'd', 'a']

Updates a sliding window no larger than n and yields a value
if the item only occurs once in the updated window.

When `n == 1`, *unique_in_window* is memoryless:

    >>> list(unique_in_window('aab', n=1))
    ['a', 'a', 'b']

The items in *iterable* must be hashable.

```python
def duplicates_everseen(iterable, key=None)
```
Yield duplicate elements after their first appearance.

>>> list(duplicates_everseen('mississippi'))
['s', 'i', 's', 's', 'i', 'p', 'i']
>>> list(duplicates_everseen('AaaBbbCccAaa', str.lower))
['a', 'a', 'b', 'b', 'c', 'c', 'A', 'a', 'a']

This function is analogous to :func:`unique_everseen` and is subject to
the same performance considerations.

If you would like each duplicate to only appear once, much like ``uniq -d``
in the Unix shell or ``Itertools::duplicates`` from the Rust ``itertools``
crate, pass the return value of this function into :func:`unique_everseen`
with the same ``key``.

```python
def duplicates_justseen(iterable, key=None)
```
Yields serially-duplicate elements after their first appearance.

>>> list(duplicates_justseen('mississippi'))
['s', 's', 'p']
>>> list(duplicates_justseen('AaaBbbCccAaa', str.lower))
['a', 'a', 'b', 'b', 'c', 'c', 'a', 'a']

This function is analogous to :func:`unique_justseen`.

```python
def classify_unique(iterable, key=None)
```
Classify each element in terms of its uniqueness.

For each element in the input iterable, return a 3-tuple consisting of:

1. The element itself
2. ``False`` if the element is equal to the one preceding it in the input,
   ``True`` otherwise (i.e. the equivalent of :func:`unique_justseen`)
3. ``False`` if this element has been seen anywhere in the input before,
   ``True`` otherwise (i.e. the equivalent of :func:`unique_everseen`)

>>> list(classify_unique('otto'))    # doctest: +NORMALIZE_WHITESPACE
[('o', True,  True),
 ('t', True,  True),
 ('t', False, False),
 ('o', True,  False)]

This function is analogous to :func:`unique_everseen` and is subject to
the same performance consideration ...

```python
def minmax(iterable_or_value, *others, key=None, default=_marker)
```
Returns both the smallest and largest items from an iterable
or from two or more arguments.

    >>> minmax([3, 1, 5])
    (1, 5)

    >>> minmax(4, 2, 6)
    (2, 6)

If a *key* function is provided, it will be used to transform the input
items for comparison.

    >>> minmax([5, 30], key=str)  # '30' sorts before '5'
    (30, 5)

If a *default* value is provided, it will be returned if there are no
input items.

    >>> minmax([], default=(0, 0))
    (0, 0)

Otherwise ``ValueError`` is raised.

This function makes a single pass over the input elements and takes care to
minimize the number of comparisons made during processing.

Note that unlike the builtin ``max`` function, which always ret ...

```python
def constrained_batches(iterable, max_size, max_count=None, get_len=len, strict=True)
```
Yield batches of items from *iterable* with a combined size limited by
*max_size*.

>>> iterable = [b'12345', b'123', b'12345678', b'1', b'1', b'12', b'1']
>>> list(constrained_batches(iterable, 10))
[(b'12345', b'123'), (b'12345678', b'1', b'1'), (b'12', b'1')]

If a *max_count* is supplied, the number of items per batch is also
limited. It must be greater than zero:

>>> iterable = [b'12345', b'123', b'12345678', b'1', b'1', b'12', b'1']
>>> list(constrained_batches(iterable, 10, max_count = 2))
[(b'12345', b'123'), (b'12345678', b'1'), (b'1', b'12'), (b'1',)]

If a *get_len* function is supplied, use that instead of :func:`len` to
determine item size.

If *strict* is ``True``, raise ``Val ...

```python
def gray_product(*iterables, repeat=1)
```
Like :func:`itertools.product`, but return tuples in an order such
that only one element in the generated tuple changes from one iteration
to the next.

    >>> list(gray_product('AB','CD'))
    [('A', 'C'), ('B', 'C'), ('B', 'D'), ('A', 'D')]

The *repeat* keyword argument specifies the number of repetitions
of the iterables.  For example, ``gray_product('AB', repeat=3)`` is
equivalent to ``gray_product('AB', 'AB', 'AB')``.

This function consumes all of the input iterables before producing output.
If any of the input iterables have fewer than two items, ``ValueError``
is raised.

For information on the algorithm, see
`this section <https://www-cs-faculty.stanford.edu/~knuth/fasc2a.ps.gz>`_ ...

```python
def partial_product(*iterables, repeat=1)
```
Yields tuples containing one item from each iterator, with subsequent
tuples changing a single item at a time by advancing each iterator until it
is exhausted. This sequence guarantees every value in each iterable is
output at least once without generating all possible combinations.

This may be useful, for example, when testing an expensive function.

    >>> list(partial_product('AB', 'C', 'DEF'))
    [('A', 'C', 'D'), ('B', 'C', 'D'), ('B', 'C', 'E'), ('B', 'C', 'F')]

The *repeat* keyword argument specifies the number of repetitions
of the iterables.  For example, ``partial_product('AB', repeat=3)`` is
equivalent to ``partial_product('AB', 'AB', 'AB')``.

```python
def takewhile_inclusive(predicate, iterable)
```
A variant of :func:`takewhile` that yields one additional element.

    >>> list(takewhile_inclusive(lambda x: x < 5, [1, 4, 6, 4, 1]))
    [1, 4, 6]

:func:`takewhile` would return ``[1, 4]``.

```python
def outer_product(func, xs, ys, *args, **kwargs)
```
A generalized outer product that applies a binary function to all
pairs of items. Returns a 2D matrix with ``len(xs)`` rows and ``len(ys)``
columns.
Also accepts ``*args`` and ``**kwargs`` that are passed to ``func``.

Multiplication table:

>>> from operator import mul
>>> list(outer_product(mul, range(1, 4), range(1, 6)))
[(1, 2, 3, 4, 5), (2, 4, 6, 8, 10), (3, 6, 9, 12, 15)]

Cross tabulation:

>>> xs = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
>>> ys = ['X', 'X', 'X', 'Y', 'Z', 'Z', 'Y', 'Y', 'Z', 'Z']
>>> pair_counts = Counter(zip(xs, ys))
>>> count_rows = lambda x, y: pair_counts[x, y]
>>> list(outer_product(count_rows, sorted(set(xs)), sorted(set(ys))))
[(2, 3, 0), (1, 0, 4)] ...

```python
def iter_suppress(iterable, *exceptions)
```
Yield each of the items from *iterable*. If the iteration raises one of
the specified *exceptions*, that exception will be suppressed and iteration
will stop.

>>> from itertools import chain
>>> def breaks_at_five(x):
...     while True:
...         if x >= 5:
...             raise RuntimeError
...         yield x
...         x += 1
>>> it_1 = iter_suppress(breaks_at_five(1), RuntimeError)
>>> it_2 = iter_suppress(breaks_at_five(2), RuntimeError)
>>> list(chain(it_1, it_2))
[1, 2, 3, 4, 2, 3, 4]

```python
def filter_map(func, iterable)
```
Apply *func* to every element of *iterable*, yielding only those which
are not ``None``.

>>> elems = ['1', 'a', '2', 'b', '3']
>>> list(filter_map(lambda s: int(s) if s.isnumeric() else None, elems))
[1, 2, 3]

```python
def powerset_of_sets(iterable, *, baseset=set)
```
Yields all possible subsets of the iterable.

    >>> list(powerset_of_sets([1, 2, 3]))  # doctest: +SKIP
    [set(), {1}, {2}, {3}, {1, 2}, {1, 3}, {2, 3}, {1, 2, 3}]
    >>> list(powerset_of_sets([1, 1, 0]))  # doctest: +SKIP
    [set(), {1}, {0}, {0, 1}]

:func:`powerset_of_sets` takes care to minimize the number
of hash operations performed.

The *baseset* parameter determines what kind of sets are
constructed, either *set* or *frozenset*.

```python
def join_mappings(**field_to_map)
```
Joins multiple mappings together using their common keys.

>>> user_scores = {'elliot': 50, 'claris': 60}
>>> user_times = {'elliot': 30, 'claris': 40}
>>> join_mappings(score=user_scores, time=user_times)
{'elliot': {'score': 50, 'time': 30}, 'claris': {'score': 60, 'time': 40}}

```python
def dft(xarr)
```
Discrete Fourier Transform. *xarr* is a sequence of complex numbers.
Yields the components of the corresponding transformed output vector.

>>> import cmath
>>> xarr = [1, 2-1j, -1j, -1+2j]  # time domain
>>> Xarr = [2, -2-2j, -2j, 4+4j]  # frequency domain
>>> magnitudes, phases = zip(*map(cmath.polar, Xarr))
>>> all(map(cmath.isclose, dft(xarr), Xarr))
True

Inputs are restricted to numeric types that can add and multiply
with a complex number.  This includes int, float, complex, and
Fraction, but excludes Decimal.

See :func:`idft` for the inverse Discrete Fourier Transform.

```python
def idft(Xarr)
```
Inverse Discrete Fourier Transform. *Xarr* is a sequence of
complex numbers. Yields the components of the corresponding
inverse-transformed output vector.

>>> import cmath
>>> xarr = [1, 2-1j, -1j, -1+2j]  # time domain
>>> Xarr = [2, -2-2j, -2j, 4+4j]  # frequency domain
>>> all(map(cmath.isclose, idft(Xarr), xarr))
True

Inputs are restricted to numeric types that can add and multiply
with a complex number.  This includes int, float, complex, and
Fraction, but excludes Decimal.

See :func:`dft` for the Discrete Fourier Transform.

```python
def doublestarmap(func, iterable)
```
Apply *func* to every item of *iterable* by dictionary unpacking
the item into *func*.

The difference between :func:`itertools.starmap` and :func:`doublestarmap`
parallels the distinction between ``func(*a)`` and ``func(**a)``.

>>> iterable = [{'a': 1, 'b': 2}, {'a': 40, 'b': 60}]
>>> list(doublestarmap(lambda a, b: a + b, iterable))
[3, 100]

``TypeError`` will be raised if *func*'s signature doesn't match the
mapping contained in *iterable* or if *iterable* does not contain mappings.

```python
def nth_prime(n, *, approximate=False)
```
Return the nth prime (counting from 0).

>>> nth_prime(0)
2
>>> nth_prime(100)
547

If *approximate* is set to True, will return a prime close
to the nth prime.  The estimation is much faster than computing
an exact result.

>>> nth_prime(200_000_000, approximate=True)  # Exact result is 4222234763
4217820427

```python
def argmin(iterable, *, key=None)
```
Index of the first occurrence of a minimum value in an iterable.

    >>> argmin('efghabcdijkl')
    4
    >>> argmin([3, 2, 1, 0, 4, 2, 1, 0])
    3

For example, look up a label corresponding to the position
of a value that minimizes a cost function::

    >>> def cost(x):
    ...     "Days for a wound to heal given a subject's age."
    ...     return x**2 - 20*x + 150
    ...
    >>> labels =  ['homer', 'marge', 'bart', 'lisa', 'maggie']
    >>> ages =    [  35,      30,      10,      9,      1    ]

    # Fastest healing family member
    >>> labels[argmin(ages, key=cost)]
    'bart'

    # Age with fastest healing
    >>> min(ages, key=cost)
    10

```python
def argmax(iterable, *, key=None)
```
Index of the first occurrence of a maximum value in an iterable.

    >>> argmax('abcdefghabcd')
    7
    >>> argmax([0, 1, 2, 3, 3, 2, 1, 0])
    3

For example, identify the best machine learning model::

    >>> models =   ['svm', 'random forest', 'knn', 'naïve bayes']
    >>> accuracy = [  68,        61,          84,       72      ]

    # Most accurate model
    >>> models[argmax(accuracy)]
    'knn'

    # Best accuracy
    >>> max(accuracy)
    84

```python
def extract(iterable, indices, *, monotonic=False)
```
Yield values at the specified indices.

Example:

    >>> data = 'abcdefghijklmnopqrstuvwxyz'
    >>> list(extract(data, [7, 4, 11, 11, 14]))
    ['h', 'e', 'l', 'l', 'o']

The *iterable* is consumed lazily and can be infinite.

When *monotonic* is false, the *indices* are consumed immediately
and must be finite. When *monotonic* is true, *indices* are consumed
lazily and can be infinite but must be non-decreasing.

Raises ``IndexError`` if an index lies beyond the iterable.
Raises ``ValueError`` for a negative index or for a decreasing
index when *monotonic* is true.

```python
class serialize()
```
Wrap a non-concurrent iterator with a lock to enforce sequential access.

Applies a non-reentrant lock around calls to ``__next__``, allowing
iterator and generator instances to be shared by multiple consumer
threads.

```python
# method of serialize
def __init__(self, iterable)
```


```python
# method of serialize
def send(self, value, /)
```
Send a value to a generator.

Raises AttributeError if not a generator.

```python
# method of serialize
def throw(self, *args)
```
Call throw() on a generator.

Raises AttributeError if not a generator.

```python
# method of serialize
def close(self)
```
Call close() on a generator.

Raises AttributeError if not a generator.

```python
def synchronized(func)
```
Wrap an iterator-returning callable to make its iterators thread-safe.

Existing itertools and more-itertools can be wrapped so that their
iterator instances are serialized.

For example, ``itertools.count`` does not make thread-safe instances,
but that is easily fixed with::

    atomic_counter = synchronized(itertools.count)

Can also be used as a decorator for generator functions definitions
so that the generator instances are serialized::

    @synchronized
    def enumerate_and_timestamp(iterable):
        for count, value in enumerate(iterable):
            yield count, time_ns(), value

```python
def concurrent_tee(iterable, n=2)
```
Variant of itertools.tee() but with guaranteed threading semantics.

Takes a non-threadsafe iterator as an input and creates concurrent
tee objects for other threads to have reliable independent copies of
the data stream.

The new iterators are only thread-safe if consumed within a single thread.
To share just one of the new iterators across multiple threads, wrap it
with :func:`serialize`.

```python
def subfactorial(n)
```
Number of permutations of *n* elements with no fixed points.

The :func:`subfactorial` function computes the length of
:func:`derangements`.  For example, there are 1,854 ways to
rearrange the letters in word "epsilon" without leaving any
letter in its original position:

    >>> from more_itertools import derangements, ilen
    >>> ilen(derangements('epsilon'))
    1854
    >>> subfactorial(len('epsilon'))
    1854

Reference:  https://oeis.org/A000166

```python
def random_ordered_range(*args)
```
Return :func:`range` values in randomly shuffled order.


## module `more_itertools/recipes.py`

```python
def take(n, iterable)
```
Return first *n* items of the *iterable* as a list.

    >>> take(3, range(10))
    [0, 1, 2]

If there are fewer than *n* items in the iterable, all of them are
returned.

    >>> take(10, range(3))
    [0, 1, 2]

```python
def tabulate(function, start=0)
```
Return an iterator over the results of ``func(start)``,
``func(start + 1)``, ``func(start + 2)``...

*func* should be a function that accepts one integer argument.

If *start* is not specified it defaults to 0. It will be incremented each
time the iterator is advanced.

    >>> square = lambda x: x ** 2
    >>> iterator = tabulate(square, -3)
    >>> take(4, iterator)
    [9, 4, 1, 0]

```python
def tail(n, iterable)
```
Return an iterator over the last *n* items of *iterable*.

>>> t = tail(3, 'ABCDEFG')
>>> list(t)
['E', 'F', 'G']

```python
def consume(iterator, n=None)
```
Advance *iterable* by *n* steps. If *n* is ``None``, consume it
entirely.

Efficiently exhausts an iterator without returning values. Defaults to
consuming the whole iterator, but an optional second argument may be
provided to limit consumption.

    >>> i = (x for x in range(10))
    >>> next(i)
    0
    >>> consume(i, 3)
    >>> next(i)
    4
    >>> consume(i)
    >>> next(i)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    StopIteration

If the iterator has fewer items remaining than the provided limit, the
whole iterator will be consumed.

    >>> i = (x for x in range(3))
    >>> consume(i, 5)
    >>> next(i)
    Traceback (most recent call last): ...

```python
def nth(iterable, n, default=None)
```
Returns the nth item or a default value.

>>> l = range(10)
>>> nth(l, 3)
3
>>> nth(l, 20, "zebra")
'zebra'

```python
def all_equal(iterable, key=None)
```
Returns ``True`` if all the elements are equal to each other.

    >>> all_equal('aaaa')
    True
    >>> all_equal('aaab')
    False

A function that accepts a single argument and returns a transformed version
of each input item can be specified with *key*:

    >>> all_equal('AaaA', key=str.casefold)
    True
    >>> all_equal([1, 2, 3], key=lambda x: x < 10)
    True

```python
def quantify(iterable, pred=bool)
```
Return the how many times the predicate is true.

>>> quantify([True, False, True])
2

```python
def pad_none(iterable)
```
Returns the sequence of elements and then returns ``None`` indefinitely.

    >>> take(5, pad_none(range(3)))
    [0, 1, 2, None, None]

Useful for emulating the behavior of the built-in :func:`map` function.

See also :func:`padded`.

```python
def ncycles(iterable, n)
```
Returns the sequence elements *n* times

>>> list(ncycles(["a", "b"], 3))
['a', 'b', 'a', 'b', 'a', 'b']

```python
def dotproduct(vec1, vec2)
```
Returns the dot product of the two iterables.

>>> dotproduct([10, 15, 12], [0.65, 0.80, 1.25])
33.5
>>> 10 * 0.65 + 15 * 0.80 + 12 * 1.25
33.5

In Python 3.12 and later, use ``math.sumprod()`` instead.

```python
def flatten(list_of_lists)
```
Return an iterator flattening one level of nesting in a list of lists.

    >>> list(flatten([[0, 1], [2, 3]]))
    [0, 1, 2, 3]

See also :func:`collapse`, which can flatten multiple levels of nesting.

```python
def repeatfunc(function, times=None, *args)
```
Call *function* with *args* repeatedly, returning an iterable over the
results.

If *times* is specified, the iterable will terminate after that many
repetitions:

    >>> from operator import add
    >>> times = 4
    >>> args = 3, 5
    >>> list(repeatfunc(add, times, *args))
    [8, 8, 8, 8]

If *times* is ``None`` the iterable will not terminate:

    >>> from random import randrange
    >>> times = None
    >>> args = 1, 11
    >>> take(6, repeatfunc(randrange, times, *args))  # doctest:+SKIP
    [2, 4, 8, 1, 8, 4]

```python
def pairwise(iterable)
```
Wrapper for :func:`itertools.pairwise`.

.. deprecated:: 11.0.0
   Will be removed in a future major release.

```python
def grouper(iterable, n, incomplete='fill', fillvalue=None)
```
Group elements from *iterable* into fixed-length groups of length *n*.

>>> list(grouper('ABCDEF', 3))
[('A', 'B', 'C'), ('D', 'E', 'F')]

The keyword arguments *incomplete* and *fillvalue* control what happens for
iterables whose length is not a multiple of *n*.

When *incomplete* is `'fill'`, the last group will contain instances of
*fillvalue*.

>>> list(grouper('ABCDEFG', 3, incomplete='fill', fillvalue='x'))
[('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]

When *incomplete* is `'ignore'`, the last group will not be emitted.

>>> list(grouper('ABCDEFG', 3, incomplete='ignore', fillvalue='x'))
[('A', 'B', 'C'), ('D', 'E', 'F')]

When *incomplete* is `'strict'`, a `ValueError` will be ...

```python
def roundrobin(*iterables)
```
Visit input iterables in a cycle until each is exhausted.

    >>> list(roundrobin('ABC', 'D', 'EF'))
    ['A', 'D', 'E', 'B', 'F', 'C']

This function produces the same output as :func:`interleave_longest`, but
may perform better for some inputs (in particular when the number of
iterables is small).

```python
def partition(pred, iterable)
```
Returns a 2-tuple of iterables derived from the input iterable.
The first yields the items that have ``pred(item) == False``.
The second yields the items that have ``pred(item) == True``.

    >>> is_odd = lambda x: x % 2 != 0
    >>> iterable = range(10)
    >>> even_items, odd_items = partition(is_odd, iterable)
    >>> list(even_items), list(odd_items)
    ([0, 2, 4, 6, 8], [1, 3, 5, 7, 9])

If *pred* is None, :func:`bool` is used.

    >>> iterable = [0, 1, False, True, '', ' ']
    >>> false_items, true_items = partition(None, iterable)
    >>> list(false_items), list(true_items)
    ([0, False, ''], [1, True, ' '])

```python
def powerset(iterable)
```
Yields all possible subsets of the iterable.

    >>> list(powerset([1, 2, 3]))
    [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

:func:`powerset` will operate on iterables that aren't :class:`set`
instances, so repeated elements in the input will produce repeated elements
in the output.

    >>> seq = [1, 1, 0]
    >>> list(powerset(seq))
    [(), (1,), (1,), (0,), (1, 1), (1, 0), (1, 0), (1, 1, 0)]

For a variant that efficiently yields actual :class:`set` instances, see
:func:`powerset_of_sets`.

```python
def unique_everseen(iterable, key=None)
```
Yield unique elements, preserving order. Remember all elements ever seen.

    >>> list(unique_everseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D']
    >>> list(unique_everseen('ABBCcAD', str.casefold))
    ['A', 'B', 'C', 'D']

Raises ``TypeError`` for unhashable items.

Some unhashable objects can be converted to hashable objects
using the *key* parameter:

* For ``list`` objects, try ``key=tuple``.
* For ``set`` objects, try ``key=frozenset``.
* For ``dict`` objects, try ``key=lambda x: frozenset(x.items())``
  or in Python 3.15 and later, set ``key=frozendict``.

Alternatively, consider the ``unique()`` itertool recipe.  It sorts
the data and then uses equality to eliminate duplicates. ...

```python
def unique_justseen(iterable, key=None)
```
Yields elements in order, ignoring serial duplicates

>>> list(unique_justseen('AAAABBBCCDAABBB'))
['A', 'B', 'C', 'D', 'A', 'B']
>>> list(unique_justseen('ABBCcAD', str.lower))
['A', 'B', 'C', 'A', 'D']

```python
def unique(iterable, key=None, reverse=False)
```
Yields unique elements in sorted order.

>>> list(unique([[1, 2], [3, 4], [1, 2]]))
[[1, 2], [3, 4]]

*key* and *reverse* are passed to :func:`sorted`.

>>> list(unique('ABBcCAD', str.casefold))
['A', 'B', 'c', 'D']
>>> list(unique('ABBcCAD', str.casefold, reverse=True))
['D', 'c', 'B', 'A']

The elements in *iterable* need not be hashable, but they must be
comparable for sorting to work.

```python
def iter_except(function, exception, first=None)
```
Yields results from a function repeatedly until an exception is raised.

Converts a call-until-exception interface to an iterator interface.
Like ``iter(function, sentinel)``, but uses an exception instead of a sentinel
to end the loop.

    >>> l = [0, 1, 2]
    >>> list(iter_except(l.pop, IndexError))
    [2, 1, 0]

Multiple exceptions can be specified as a stopping condition:

    >>> l = [1, 2, 3, '...', 4, 5, 6]
    >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
    [7, 6, 5]
    >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
    [4, 3, 2]
    >>> list(iter_except(lambda: 1 + l.pop(), (IndexError, TypeError)))
    []

```python
def first_true(iterable, default=None, pred=None)
```
Returns the first true value in the iterable.

If no true value is found, returns *default*

If *pred* is not None, returns the first item for which
``pred(item) == True`` .

    >>> first_true(range(10))
    1
    >>> first_true(range(10), pred=lambda x: x > 5)
    6
    >>> first_true(range(10), default='missing', pred=lambda x: x > 9)
    'missing'

```python
def random_product(*iterables, repeat=1)
```
Draw an item at random from each of the input iterables.

    >>> random_product('abc', range(4), 'XYZ')  # doctest:+SKIP
    ('c', 3, 'Z')

If *repeat* is provided as a keyword argument, that many items will be
drawn from each iterable.

    >>> random_product('abcd', range(4), repeat=2)  # doctest:+SKIP
    ('a', 2, 'd', 3)

This is equivalent to taking a random selection from
``itertools.product(*args, repeat=repeat)``.

```python
def random_permutation(iterable, r=None)
```
Return a random *r* length permutation of the elements in *iterable*.

If *r* is not specified or is ``None``, then *r* defaults to the length of
*iterable*.

    >>> random_permutation(range(5))  # doctest:+SKIP
    (3, 4, 0, 1, 2)

This is equivalent to taking a random selection from
``itertools.permutations(iterable, r)``.

```python
def random_combination(iterable, r)
```
Return a random *r* length subsequence of the elements in *iterable*.

    >>> random_combination(range(5), 3)  # doctest:+SKIP
    (2, 3, 4)

This is equivalent to taking a random selection from
``itertools.combinations(iterable, r)``.

```python
def random_combination_with_replacement(iterable, r)
```
Return a random *r* length subsequence of elements in *iterable*,
allowing individual elements to be repeated.

    >>> random_combination_with_replacement(range(3), 5) # doctest:+SKIP
    (0, 0, 1, 2, 2)

This is equivalent to taking a random selection from
``itertools.combinations_with_replacement(iterable, r)``.

```python
def nth_combination(iterable, r, index)
```
Equivalent to ``list(combinations(iterable, r))[index]``.

The subsequences of *iterable* that are of length *r* can be ordered
lexicographically. :func:`nth_combination` computes the subsequence at
sort position *index* directly, without computing the previous
subsequences.

    >>> nth_combination(range(5), 3, 5)
    (0, 3, 4)

``ValueError`` will be raised If *r* is negative.
``IndexError`` will be raised if the given *index* is invalid.

```python
def prepend(value, iterable)
```
Yield *value*, followed by the elements in *iterable*.

    >>> value = '0'
    >>> iterable = ['1', '2', '3']
    >>> list(prepend(value, iterable))
    ['0', '1', '2', '3']

To prepend multiple values, see :func:`itertools.chain`
or :func:`value_chain`.

```python
def convolve(signal, kernel)
```
Discrete linear convolution of two iterables.
Equivalent to polynomial multiplication.

For example, multiplying ``(x² -x - 20)`` by ``(x - 3)``
gives ``(x³ -4x² -17x + 60)``.

    >>> list(convolve([1, -1, -20], [1, -3]))
    [1, -4, -17, 60]

Examples of useful kernels:

* The kernel ``[0.25, 0.25, 0.25, 0.25]`` computes a moving average.
  For image data, this blurs the image and reduces noise.
* The kernel ``[1/2, 0, -1/2]`` estimates the first derivative of
  a function evaluated at evenly spaced inputs.
* The kernel ``[1, -2, 1]`` estimates the second derivative of a
  function evaluated at evenly spaced inputs.

Convolutions are mathematically commutative. However, the input iterables ...

```python
def before_and_after(predicate, it)
```
A variant of :func:`takewhile` that allows complete access to the
remainder of the iterator.

     >>> it = iter('ABCdEfGhI')
     >>> all_upper, remainder = before_and_after(str.isupper, it)
     >>> ''.join(all_upper)
     'ABC'
     >>> ''.join(remainder) # takewhile() would lose the 'd'
     'dEfGhI'

Note that the first iterator must be fully consumed before the second
iterator can generate valid results.

```python
def triplewise(iterable)
```
Return overlapping triplets from *iterable*.

>>> list(triplewise('ABCDE'))
[('A', 'B', 'C'), ('B', 'C', 'D'), ('C', 'D', 'E')]

```python
def sliding_window(iterable, n)
```
Return a sliding window of width *n* over *iterable*.

    >>> list(sliding_window(range(6), 4))
    [(0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5)]

If *iterable* has fewer than *n* items, then nothing is yielded:

    >>> list(sliding_window(range(3), 4))
    []

For a variant with more features, see :func:`windowed`.

```python
def subslices(iterable)
```
Return all contiguous non-empty subslices of *iterable*.

    >>> list(subslices('ABC'))
    [['A'], ['A', 'B'], ['A', 'B', 'C'], ['B'], ['B', 'C'], ['C']]

This is similar to :func:`substrings`, but emits items in a different
order.

```python
def polynomial_from_roots(roots)
```
Compute a polynomial's coefficients from its roots.

>>> roots = [5, -4, 3]            # (x - 5) * (x + 4) * (x - 3)
>>> polynomial_from_roots(roots)  # x³ - 4 x² - 17 x + 60
[1, -4, -17, 60]

Note that polynomial coefficients are specified in descending power order.

Supports all numeric types: int, float, complex, Decimal, Fraction.

```python
def iter_index(iterable, value, start=0, stop=None)
```
Yield the index of each place in *iterable* that *value* occurs,
beginning with index *start* and ending before index *stop*.

>>> list(iter_index('AABCADEAF', 'A'))
[0, 1, 4, 7]
>>> list(iter_index('AABCADEAF', 'A', 1))  # start index is inclusive
[1, 4, 7]
>>> list(iter_index('AABCADEAF', 'A', 1, 7))  # stop index is not inclusive
[1, 4]

The behavior for non-scalar *value* arguments matches the built-in Python types.

>>> list(iter_index('ABCDABCD', 'AB'))
[0, 4]
>>> list(iter_index([0, 1, 2, 3, 0, 1, 2, 3], [0, 1]))
[]
>>> list(iter_index([[0, 1], [2, 3], [0, 1], [2, 3]], [0, 1]))
[0, 2]

For ``range`` objects (and other objects whose ``index`` method's behavior doesn't
match that of ``l ...

```python
def sieve(n)
```
Yield the primes less than n.

>>> list(sieve(30))
[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

```python
def transpose(matrix)
```
Swap the rows and columns of the input matrix.

>>> list(transpose([(1, 2, 3), (11, 22, 33)]))
[(1, 11), (2, 22), (3, 33)]

The caller should ensure that the dimensions of the input are compatible.
If the input is empty, no output will be produced.

```python
def reshape(matrix, shape)
```
Change the shape of a *matrix*.

If *shape* is an integer, the matrix must be two dimensional
and the shape is interpreted as the desired number of columns:

    >>> matrix = [(0, 1), (2, 3), (4, 5)]
    >>> cols = 3
    >>> list(reshape(matrix, cols))
    [(0, 1, 2), (3, 4, 5)]

If *shape* is a tuple (or other iterable), the input matrix can have
any number of dimensions. It will first be flattened and then rebuilt
to the desired shape which can also be multidimensional:

    >>> matrix = [(0, 1), (2, 3), (4, 5)]    # Start with a 3 x 2 matrix

    >>> list(reshape(matrix, (2, 3)))        # Make a 2 x 3 matrix
    [(0, 1, 2), (3, 4, 5)]

    >>> list(reshape(matrix, (6,)))          # Make a ...

```python
def matmul(m1, m2)
```
Multiply two matrices.

>>> list(matmul([(7, 5), (3, 5)], [(2, 5), (7, 9)]))
[(49, 80), (41, 60)]

The caller should ensure that the dimensions of the input matrices are
compatible with each other.

Supports all numeric types: int, float, complex, Decimal, Fraction.

```python
def factor(n)
```
Yield the prime factors of n.

>>> list(factor(360))
[2, 2, 2, 3, 3, 5]

Finds small factors with trial division.  Larger factors are
either verified as prime with ``is_prime`` or split into
smaller factors with Pollard's rho algorithm.

```python
def polynomial_eval(coefficients, x)
```
Evaluate a polynomial at a specific value.

Computes with better numeric stability than Horner's method.

Evaluate ``x^3 - 4 * x^2 - 17 * x + 60`` at ``x = 2.5``:

>>> coefficients = [1, -4, -17, 60]
>>> x = 2.5
>>> polynomial_eval(coefficients, x)
8.125

Note that polynomial coefficients are specified in descending power order.

Supports all numeric types: int, float, complex, Decimal, Fraction.

```python
def sum_of_squares(iterable)
```
Return the sum of the squares of the input values.

>>> sum_of_squares([10, 20, 30])
1400

Supports all numeric types: int, float, complex, Decimal, Fraction.

```python
def polynomial_derivative(coefficients)
```
Compute the first derivative of a polynomial.

Evaluate the derivative of ``x³ - 4 x² - 17 x + 60``:

>>> coefficients = [1, -4, -17, 60]
>>> derivative_coefficients = polynomial_derivative(coefficients)
>>> derivative_coefficients
[3, -8, -17]

Note that polynomial coefficients are specified in descending power order.

Supports all numeric types: int, float, complex, Decimal, Fraction.

```python
def totient(n)
```
Return the count of natural numbers up to *n* that are coprime with *n*.

Euler's totient function φ(n) gives the number of totatives.
Totative are integers k in the range 1 ≤ k ≤ n such that gcd(n, k) = 1.

>>> n = 9
>>> totient(n)
6

>>> totatives = [x for x in range(1, n) if gcd(n, x) == 1]
>>> totatives
[1, 2, 4, 5, 7, 8]
>>> len(totatives)
6

Reference:  https://en.wikipedia.org/wiki/Euler%27s_totient_function

```python
def is_prime(n)
```
Return ``True`` if *n* is prime and ``False`` otherwise.

Basic examples:

    >>> is_prime(37)
    True
    >>> is_prime(3 * 13)
    False
    >>> is_prime(18_446_744_073_709_551_557)
    True

Find the next prime over one billion:

    >>> next(filter(is_prime, count(10**9)))
    1000000007

Generate random primes up to 200 bits and up to 60 decimal digits:

    >>> from random import seed, randrange, getrandbits
    >>> seed(18675309)

    >>> next(filter(is_prime, map(getrandbits, repeat(200))))
    893303929355758292373272075469392561129886005037663238028407

    >>> next(filter(is_prime, map(randrange, repeat(10**60))))
    269638077304026462407872868003560484232362454342414618963649 ...

```python
def loops(n)
```
Returns an iterable with *n* elements for efficient looping.
Like ``range(n)`` but doesn't create integers.

>>> i = 0
>>> for _ in loops(5):
...     i += 1
>>> i
5

```python
def multinomial(*counts)
```
Number of distinct arrangements of a multiset.

The expression ``multinomial(3, 4, 2)`` has several equivalent
interpretations:

* In the expansion of ``(a + b + c)⁹``, the coefficient of the
  ``a³b⁴c²`` term is 1260.

* There are 1260 distinct ways to arrange 9 balls consisting of 3 reds, 4
  greens, and 2 blues.

* There are 1260 unique ways to place 9 distinct objects into three bins
  with sizes 3, 4, and 2.

The :func:`multinomial` function computes the length of
:func:`distinct_permutations`.  For example, there are 83,160 distinct
anagrams of the word "abracadabra":

    >>> from more_itertools import distinct_permutations, ilen
    >>> ilen(distinct_permutations('abracadabra')) ...

```python
def running_median(iterable, *, maxlen=None)
```
Cumulative median of values seen so far or values in a sliding window.

Set *maxlen* to a positive integer to specify the maximum size
of the sliding window.  The default of *None* is equivalent to
an unbounded window.

For example:

    >>> list(running_median([5.0, 9.0, 4.0, 12.0, 8.0, 9.0]))
    [5.0, 7.0, 5.0, 7.0, 8.0, 8.5]
    >>> list(running_median([5.0, 9.0, 4.0, 12.0, 8.0, 9.0], maxlen=3))
    [5.0, 7.0, 5.0, 9.0, 8.0, 9.0]

Supports numeric types such as int, float, Decimal, and Fraction,
but not complex numbers which are unorderable.

On version Python 3.13 and prior, max-heaps are simulated with
negative values. The negation causes Decimal inputs to apply context
rounding, makin ...

```python
def running_mean(iterable, *, maxlen=None)
```
Cumulative mean of values seen so far or values in a sliding window.

Set *maxlen* to a positive integer to specify the maximum size
of the sliding window.  The default of *None* is equivalent to
an unbounded window.

For example:

    >>> list(running_mean([40, 30, 50, 46, 39, 44]))
    [40.0, 35.0, 40.0, 41.5, 41.0, 41.5]

    >>> list(running_mean([40, 30, 50, 46, 39, 44], maxlen=3))
    [40.0, 35.0, 40.0, 42.0, 45.0, 43.0]

Supports numeric types such as int, float, complex, Decimal, and Fraction.

No extra effort is made to reduce round-off errors for float inputs.
So the results may be slightly different from `statistics.mean`.

```python
def running_min(iterable, *, maxlen=None)
```
Smallest of values seen so far or values in a sliding window.

Set *maxlen* to a positive integer to specify the maximum size
of the sliding window.  The default of *None* is equivalent to
an unbounded window.

For example:

    >>> list(running_min([4, 3, 7, 0, 8, 1, 6, 2, 9, 5]))
    [4, 3, 3, 0, 0, 0, 0, 0, 0, 0]

    >>> list(running_min([4, 3, 7, 0, 8, 1, 6, 2, 9, 5], maxlen=3))
    [4, 3, 3, 0, 0, 0, 1, 1, 2, 2]

Supports numeric types such as int, float, Decimal, and Fraction,
but not complex numbers which are unorderable.

```python
def running_max(iterable, *, maxlen=None)
```
Largest of values seen so far or values in a sliding window.

Set *maxlen* to a positive integer to specify the maximum size
of the sliding window.  The default of *None* is equivalent to
an unbounded window.

For example:

    >>> list(running_max([4, 3, 7, 0, 8, 1, 6, 2, 9, 5]))
    [4, 4, 7, 7, 8, 8, 8, 8, 9, 9]

    >>> list(running_max([4, 3, 7, 0, 8, 1, 6, 2, 9, 5], maxlen=3))
    [4, 4, 7, 7, 8, 8, 8, 6, 9, 9]

Supports numeric types such as int, float, Decimal, and Fraction,
but not complex numbers which are unorderable.

```python
class Stats()
```


```python
def running_statistics(iterable, *, maxlen=None)
```
Statistics for values seen so far or values in a sliding window.

Set *maxlen* to a positive integer to specify the maximum size
of the sliding window.  The default of *None* is equivalent to
an unbounded window.

Yields instances of a ``Stats`` dataclass with fields for the dataset *size*,
*minimum* value, *median* value, *maximum* value, and the arithmetic *mean*.

Supports numeric types such as int, float, Decimal, and Fraction,
but not complex numbers which are unorderable.

```python
def random_derangement(iterable)
```
Return a random derangement of elements in the iterable.

Equivalent to but much faster than ``choice(list(derangements(iterable)))``.


# Usage examples from the project README

```python
>>> from more_itertools import flatten
>>> iterable = [(0, 1), (2, 3)]
>>> list(flatten(iterable))
[0, 1, 2, 3]
```
```python
>>> from more_itertools import chunked
>>> iterable = [0, 1, 2, 3, 4, 5, 6, 7, 8]
>>> list(chunked(iterable, 3))
[[0, 1, 2], [3, 4, 5], [6, 7, 8]]

>>> from more_itertools import spy
>>> iterable = (x * x for x in range(1, 6))
>>> head, iterable = spy(iterable, n=3)
>>> list(head)
[1, 4, 9]
>>> list(iterable)
[1, 4, 9, 16, 25]
```