# toolz: public API (signatures and documentation)


## module `toolz/itertoolz.py`

```python
def remove(predicate, seq)
```
Return those items of sequence for which predicate(item) is False

>>> def iseven(x):
...     return x % 2 == 0
>>> list(remove(iseven, [1, 2, 3, 4]))
[1, 3]

```python
def accumulate(binop, seq, initial=no_default)
```
Repeatedly apply binary function to a sequence, accumulating results

>>> from operator import add, mul
>>> list(accumulate(add, [1, 2, 3, 4, 5]))
[1, 3, 6, 10, 15]
>>> list(accumulate(mul, [1, 2, 3, 4, 5]))
[1, 2, 6, 24, 120]

Accumulate is similar to ``reduce`` and is good for making functions like
cumulative sum:

>>> from functools import partial, reduce
>>> sum    = partial(reduce, add)
>>> cumsum = partial(accumulate, add)

Accumulate also takes an optional argument that will be used as the first
value. This is similar to reduce.

>>> list(accumulate(add, [1, 2, 3], -1))
[-1, 0, 2, 5]
>>> list(accumulate(add, [], 1))
[1]

See Also:
    itertools.accumulate :  In standard itertools for ...

```python
def groupby(key, seq)
```
Group a collection by a key function

>>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
>>> groupby(len, names)  # doctest: +SKIP
{3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}

>>> iseven = lambda x: x % 2 == 0
>>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
{False: [1, 3, 5, 7], True: [2, 4, 6, 8]}

Non-callable keys imply grouping on a member.

>>> groupby('gender', [{'name': 'Alice', 'gender': 'F'},
...                    {'name': 'Bob', 'gender': 'M'},
...                    {'name': 'Charlie', 'gender': 'M'}]) # doctest:+SKIP
{'F': [{'gender': 'F', 'name': 'Alice'}],
 'M': [{'gender': 'M', 'name': 'Bob'},
       {'gender': 'M', 'name' ...

```python
def merge_sorted(*seqs, **kwargs)
```
Merge and sort a collection of sorted collections

This works lazily and only keeps one value from each iterable in memory.

>>> list(merge_sorted([1, 3, 5], [2, 4, 6]))
[1, 2, 3, 4, 5, 6]

>>> ''.join(merge_sorted('abc', 'abc', 'abc'))
'aaabbbccc'

The "key" function used to sort the input may be passed as a keyword.

>>> list(merge_sorted([2, 3], [1, 3], key=lambda x: x // 3))
[2, 1, 3, 3]

```python
def interleave(seqs)
```
Interleave a sequence of sequences

>>> list(interleave([[1, 2], [3, 4]]))
[1, 3, 2, 4]

>>> ''.join(interleave(('ABC', 'XY')))
'AXBYC'

Both the individual sequences and the sequence of sequences may be infinite

Returns a lazy iterator

```python
def unique(seq, key=None)
```
Return only unique elements of a sequence

>>> tuple(unique((1, 2, 3)))
(1, 2, 3)
>>> tuple(unique((1, 2, 1, 3)))
(1, 2, 3)

Uniqueness can be defined by key keyword

>>> tuple(unique(['cat', 'mouse', 'dog', 'hen'], key=len))
('cat', 'mouse')

```python
def isiterable(x)
```
Is x iterable?

>>> isiterable([1, 2, 3])
True
>>> isiterable('abc')
True
>>> isiterable(5)
False

```python
def isdistinct(seq)
```
All values in sequence are distinct

>>> isdistinct([1, 2, 3])
True
>>> isdistinct([1, 2, 1])
False

>>> isdistinct("Hello")
False
>>> isdistinct("World")
True

```python
def take(n, seq)
```
The first n elements of a sequence

>>> list(take(2, [10, 20, 30, 40, 50]))
[10, 20]

See Also:
    drop
    tail

```python
def tail(n, seq)
```
The last n elements of a sequence

>>> tail(2, [10, 20, 30, 40, 50])
[40, 50]

See Also:
    drop
    take

```python
def drop(n, seq)
```
The sequence following the first n elements

>>> list(drop(2, [10, 20, 30, 40, 50]))
[30, 40, 50]

See Also:
    take
    tail

```python
def take_nth(n, seq)
```
Every nth item in seq

>>> list(take_nth(2, [10, 20, 30, 40, 50]))
[10, 30, 50]

```python
def first(seq)
```
The first element in a sequence

>>> first('ABC')
'A'

```python
def second(seq)
```
The second element in a sequence

>>> second('ABC')
'B'

```python
def nth(n, seq)
```
The nth element in a sequence

>>> nth(1, 'ABC')
'B'

```python
def last(seq)
```
The last element in a sequence

>>> last('ABC')
'C'

```python
def get(ind, seq, default=no_default)
```
Get element in a sequence or dict

Provides standard indexing

>>> get(1, 'ABC')       # Same as 'ABC'[1]
'B'

Pass a list to get multiple values

>>> get([1, 2], 'ABC')  # ('ABC'[1], 'ABC'[2])
('B', 'C')

Works on any value that supports indexing/getitem
For example here we see that it works with dictionaries

>>> phonebook = {'Alice':  '555-1234',
...              'Bob':    '555-5678',
...              'Charlie':'555-9999'}
>>> get('Alice', phonebook)
'555-1234'

>>> get(['Alice', 'Bob'], phonebook)
('555-1234', '555-5678')

Provide a default for missing values

>>> get(['Alice', 'Dennis'], phonebook, None)
('555-1234', None)

See Also:
    pluck

```python
def concat(seqs)
```
Concatenate zero or more iterables, any of which may be infinite.

An infinite sequence will prevent the rest of the arguments from
being included.

We use chain.from_iterable rather than ``chain(*seqs)`` so that seqs
can be a generator.

>>> list(concat([[], [1], [2, 3]]))
[1, 2, 3]

See also:
    itertools.chain.from_iterable  equivalent

```python
def concatv(*seqs)
```
Variadic version of concat

>>> list(concatv([], ["a"], ["b", "c"]))
['a', 'b', 'c']

See also:
    itertools.chain

```python
def mapcat(func, seqs)
```
Apply func to each sequence in seqs, concatenating results.

>>> list(mapcat(lambda s: [c.upper() for c in s],
...             [["a", "b"], ["c", "d", "e"]]))
['A', 'B', 'C', 'D', 'E']

```python
def cons(el, seq)
```
Add el to beginning of (possibly infinite) sequence seq.

>>> list(cons(1, [2, 3]))
[1, 2, 3]

```python
def interpose(el, seq)
```
Introduce element between each pair of elements in seq

>>> list(interpose("a", [1, 2, 3]))
[1, 'a', 2, 'a', 3]

```python
def frequencies(seq)
```
Find number of occurrences of each value in seq

>>> frequencies(['cat', 'cat', 'ox', 'pig', 'pig', 'cat'])  #doctest: +SKIP
{'cat': 3, 'ox': 1, 'pig': 2}

See Also:
    countby
    groupby

```python
def reduceby(key, binop, seq, init=no_default)
```
Perform a simultaneous groupby and reduction

The computation:

>>> result = reduceby(key, binop, seq, init)      # doctest: +SKIP

is equivalent to the following:

>>> def reduction(group):                           # doctest: +SKIP
...     return reduce(binop, group, init)           # doctest: +SKIP

>>> groups = groupby(key, seq)                    # doctest: +SKIP
>>> result = valmap(reduction, groups)              # doctest: +SKIP

But the former does not build the intermediate groups, allowing it to
operate in much less space.  This makes it suitable for larger datasets
that do not fit comfortably in memory

The ``init`` keyword argument is the default initialization of the
reduction. ...

```python
def iterate(func, x)
```
Repeatedly apply a function func onto an original input

Yields x, then func(x), then func(func(x)), then func(func(func(x))), etc..

>>> def inc(x):  return x + 1
>>> counter = iterate(inc, 0)
>>> next(counter)
0
>>> next(counter)
1
>>> next(counter)
2

>>> double = lambda x: x * 2
>>> powers_of_two = iterate(double, 1)
>>> next(powers_of_two)
1
>>> next(powers_of_two)
2
>>> next(powers_of_two)
4
>>> next(powers_of_two)
8

```python
def sliding_window(n, seq)
```
A sequence of overlapping subsequences

>>> list(sliding_window(2, [1, 2, 3, 4]))
[(1, 2), (2, 3), (3, 4)]

This function creates a sliding window suitable for transformations like
sliding means / smoothing

>>> mean = lambda seq: float(sum(seq)) / len(seq)
>>> list(map(mean, sliding_window(2, [1, 2, 3, 4])))
[1.5, 2.5, 3.5]

```python
def partition(n, seq, pad=no_pad)
```
Partition sequence into tuples of length n

>>> list(partition(2, [1, 2, 3, 4]))
[(1, 2), (3, 4)]

If the length of ``seq`` is not evenly divisible by ``n``, the final tuple
is dropped if ``pad`` is not specified, or filled to length ``n`` by pad:

>>> list(partition(2, [1, 2, 3, 4, 5]))
[(1, 2), (3, 4)]

>>> list(partition(2, [1, 2, 3, 4, 5], pad=None))
[(1, 2), (3, 4), (5, None)]

See Also:
    partition_all

```python
def partition_all(n, seq)
```
Partition all elements of sequence into tuples of length at most n

The final tuple may be shorter to accommodate extra elements.

>>> list(partition_all(2, [1, 2, 3, 4]))
[(1, 2), (3, 4)]

>>> list(partition_all(2, [1, 2, 3, 4, 5]))
[(1, 2), (3, 4), (5,)]

See Also:
    partition

```python
def count(seq)
```
Count the number of items in seq

Like the builtin ``len`` but works on lazy sequences.

Not to be confused with ``itertools.count``

See also:
    len

```python
def pluck(ind, seqs, default=no_default)
```
plucks an element or several elements from each item in a sequence.

``pluck`` maps ``itertoolz.get`` over a sequence and returns one or more
elements of each item in the sequence.

This is equivalent to running `map(curried.get(ind), seqs)`

``ind`` can be either a single string/index or a list of strings/indices.
``seqs`` should be sequence containing sequences or dicts.

e.g.

>>> data = [{'id': 1, 'name': 'Cheese'}, {'id': 2, 'name': 'Pies'}]
>>> list(pluck('name', data))
['Cheese', 'Pies']
>>> list(pluck([0, 1], [[1, 2, 3], [4, 5, 7]]))
[(1, 2), (4, 5)]

See Also:
    get
    map

```python
def join(leftkey, leftseq, rightkey, rightseq, left_default=no_default, right_default=no_default)
```
Join two sequences on common attributes

This is a semi-streaming operation.  The LEFT sequence is fully evaluated
and placed into memory.  The RIGHT sequence is evaluated lazily and so can
be arbitrarily large.

(Note: If right_default is defined, then unique keys of rightseq
    will also be stored in memory.)

>>> friends = [('Alice', 'Edith'),
...            ('Alice', 'Zhao'),
...            ('Edith', 'Alice'),
...            ('Zhao', 'Alice'),
...            ('Zhao', 'Edith')]

>>> cities = [('Alice', 'NYC'),
...           ('Alice', 'Chicago'),
...           ('Dan', 'Sydney'),
...           ('Edith', 'Paris'),
...           ('Edith', 'Berlin'),
...           ('Zhao', 'Shanghai')]

>>> # ...

```python
def diff(*seqs, **kwargs)
```
Return those items that differ between sequences

>>> list(diff([1, 2, 3], [1, 2, 10, 100]))
[(3, 10)]

Shorter sequences may be padded with a ``default`` value:

>>> list(diff([1, 2, 3], [1, 2, 10, 100], default=None))
[(3, 10), (None, 100)]

A ``key`` function may also be applied to each item to use during
comparisons:

>>> list(diff(['apples', 'bananas'], ['Apples', 'Oranges'], key=str.lower))
[('bananas', 'Oranges')]

```python
def topk(k, seq, key=None)
```
Find the k largest elements of a sequence

Operates lazily in ``n*log(k)`` time

>>> topk(2, [1, 100, 10, 1000])
(1000, 100)

Use a key function to change sorted order

>>> topk(2, ['Alice', 'Bob', 'Charlie', 'Dan'], key=len)
('Charlie', 'Alice')

See also:
    heapq.nlargest

```python
def peek(seq)
```
Retrieve the next element of a sequence

Returns the first element and an iterable equivalent to the original
sequence, still having the element retrieved.

>>> seq = [0, 1, 2, 3, 4]
>>> first, seq = peek(seq)
>>> first
0
>>> list(seq)
[0, 1, 2, 3, 4]

```python
def peekn(n, seq)
```
Retrieve the next n elements of a sequence

Returns a tuple of the first n elements and an iterable equivalent
to the original, still having the elements retrieved.

>>> seq = [0, 1, 2, 3, 4]
>>> first_two, seq = peekn(2, seq)
>>> first_two
(0, 1)
>>> list(seq)
[0, 1, 2, 3, 4]

```python
def random_sample(prob, seq, random_state=None)
```
Return elements from a sequence with probability of prob

Returns a lazy iterator of random items from seq.

``random_sample`` considers each item independently and without
replacement. See below how the first time it returned 13 items and the
next time it returned 6 items.

>>> seq = list(range(100))
>>> list(random_sample(0.1, seq)) # doctest: +SKIP
[6, 9, 19, 35, 45, 50, 58, 62, 68, 72, 78, 86, 95]
>>> list(random_sample(0.1, seq)) # doctest: +SKIP
[6, 44, 54, 61, 69, 94]

Providing an integer seed for ``random_state`` will result in
deterministic sampling. Given the same seed it will return the same sample
every time.

>>> list(random_sample(0.1, seq, random_state=2016))
[7, 9, 19, 25, 3 ...


## module `toolz/functoolz.py`

```python
def identity(x)
```
Identity function. Return x

>>> identity(3)
3

```python
def apply(*func_and_args, **kwargs)
```
Applies a function and returns the results

>>> def double(x): return 2*x
>>> def inc(x):    return x + 1
>>> apply(double, 5)
10

>>> tuple(map(apply, [double, inc, double], [10, 500, 8000]))
(20, 501, 16000)

```python
def thread_first(val, *forms)
```
Thread value through a sequence of functions/forms

>>> def double(x): return 2*x
>>> def inc(x):    return x + 1
>>> thread_first(1, inc, double)
4

If the function expects more than one input you can specify those inputs
in a tuple.  The value is used as the first input.

>>> def add(x, y): return x + y
>>> def pow(x, y): return x**y
>>> thread_first(1, (add, 4), (pow, 2))  # pow(add(1, 4), 2)
25

So in general
    thread_first(x, f, (g, y, z))
expands to
    g(f(x), y, z)

See Also:
    thread_last

```python
def thread_last(val, *forms)
```
Thread value through a sequence of functions/forms

>>> def double(x): return 2*x
>>> def inc(x):    return x + 1
>>> thread_last(1, inc, double)
4

If the function expects more than one input you can specify those inputs
in a tuple.  The value is used as the last input.

>>> def add(x, y): return x + y
>>> def pow(x, y): return x**y
>>> thread_last(1, (add, 4), (pow, 2))  # pow(2, add(4, 1))
32

So in general
    thread_last(x, f, (g, y, z))
expands to
    g(y, z, f(x))

>>> def iseven(x):
...     return x % 2 == 0
>>> list(thread_last([1, 2, 3], (map, inc), (filter, iseven)))
[2, 4]

See Also:
    thread_first

```python
class curry()
```
Curry a callable function

Enables partial application of arguments through calling a function with an
incomplete set of arguments.

>>> def mul(x, y):
...     return x * y
>>> mul = curry(mul)

>>> double = mul(2)
>>> double(10)
20

Also supports keyword arguments

>>> @curry                  # Can use curry as a decorator
... def f(x, y, a=10):
...     return a * (x + y)

>>> add = f(a=1)
>>> add(2, 3)
5

See Also:
    toolz.curried - namespace of curried functions
                    https://toolz.readthedocs.io/en/latest/curry.html

```python
# method of curry
def __init__(self, *args, **kwargs)
```


```python
# method of curry
def func(self)
```


```python
# method of curry
def args(self)
```


```python
# method of curry
def keywords(self)
```


```python
# method of curry
def func_name(self)
```


```python
# method of curry
def __call__(self, *args, **kwargs)
```


```python
# method of curry
def bind(self, *args, **kwargs)
```


```python
# method of curry
def call(self, *args, **kwargs)
```


```python
def memoize(func, cache=None, key=None)
```
Cache a function's result for speedy future evaluation

Considerations:
    Trades memory for speed.
    Only use on pure functions.

>>> def add(x, y):  return x + y
>>> add = memoize(add)

Or use as a decorator

>>> @memoize
... def add(x, y):
...     return x + y

Use the ``cache`` keyword to provide a dict-like object as an initial cache

>>> @memoize(cache={(1, 2): 3})
... def add(x, y):
...     return x + y

Note that the above works as a decorator because ``memoize`` is curried.

It is also possible to provide a ``key(args, kwargs)`` function that
calculates keys used for the cache, which receives an ``args`` tuple and
``kwargs`` dict as input, and must return a hashable value.  Howev ...

```python
def compose(*funcs)
```
Compose functions to operate in series.

Returns a function that applies other functions in sequence.

Functions are applied from right to left so that
``compose(f, g, h)(x, y)`` is the same as ``f(g(h(x, y)))``.

If no arguments are provided, the identity function (f(x) = x) is returned.

>>> inc = lambda i: i + 1
>>> compose(str, inc)(3)
'4'

See Also:
    compose_left
    pipe

```python
def compose_left(*funcs)
```
Compose functions to operate in series.

Returns a function that applies other functions in sequence.

Functions are applied from left to right so that
``compose_left(f, g, h)(x, y)`` is the same as ``h(g(f(x, y)))``.

If no arguments are provided, the identity function (f(x) = x) is returned.

>>> inc = lambda i: i + 1
>>> compose_left(inc, str)(3)
'4'

See Also:
    compose
    pipe

```python
def pipe(data, *funcs)
```
Pipe a value through a sequence of functions

I.e. ``pipe(data, f, g, h)`` is equivalent to ``h(g(f(data)))``

We think of the value as progressing through a pipe of several
transformations, much like pipes in UNIX

``$ cat data | f | g | h``

>>> double = lambda i: 2 * i
>>> pipe(3, double, str)
'6'

See Also:
    compose
    compose_left
    thread_first
    thread_last

```python
def complement(func)
```
Convert a predicate function to its logical complement.

In other words, return a function that, for inputs that normally
yield True, yields False, and vice-versa.

>>> def iseven(n): return n % 2 == 0
>>> isodd = complement(iseven)
>>> iseven(2)
True
>>> isodd(2)
False

```python
class juxt()
```
Creates a function that calls several functions with the same arguments

Takes several functions and returns a function that applies its arguments
to each of those functions then returns a tuple of the results.

Name comes from juxtaposition: the fact of two things being seen or placed
close together with contrasting effect.

>>> inc = lambda x: x + 1
>>> double = lambda x: x * 2
>>> juxt(inc, double)(10)
(11, 20)
>>> juxt([inc, double])(10)
(11, 20)

```python
# method of juxt
def __init__(self, *funcs)
```


```python
# method of juxt
def __call__(self, *args, **kwargs)
```


```python
def do(func, x)
```
Runs ``func`` on ``x``, returns ``x``

Because the results of ``func`` are not returned, only the side
effects of ``func`` are relevant.

Logging functions can be made by composing ``do`` with a storage function
like ``list.append`` or ``file.write``

>>> from toolz import compose
>>> from toolz.curried import do

>>> log = []
>>> inc = lambda x: x + 1
>>> inc = compose(inc, do(log.append))
>>> inc(1)
2
>>> inc(11)
12
>>> log
[1, 11]

```python
def flip(func, a, b)
```
Call the function call with the arguments flipped

This function is curried.

>>> def div(a, b):
...     return a // b
...
>>> flip(div, 2, 6)
3
>>> div_by_two = flip(div, 2)
>>> div_by_two(4)
2

This is particularly useful for built in functions and functions defined
in C extensions that accept positional only arguments. For example:
isinstance, issubclass.

>>> data = [1, 'a', 'b', 2, 1.5, object(), 3]
>>> only_ints = list(filter(flip(isinstance, int), data))
>>> only_ints
[1, 2, 3]

```python
class excepts()
```
A wrapper around a function to catch exceptions and
dispatch to a handler.

This is like a functional try/except block, in the same way that
ifexprs are functional if/else blocks.

Examples
--------
>>> excepting = excepts(
...     ValueError,
...     lambda a: [1, 2].index(a),
...     lambda _: -1,
... )
>>> excepting(1)
0
>>> excepting(3)
-1

Multiple exceptions and default except clause.

>>> excepting = excepts((IndexError, KeyError), lambda a: a[0])
>>> excepting([])
>>> excepting([1])
1
>>> excepting({})
>>> excepting({0: 1})
1

```python
# method of excepts
def __init__(self, exc, func, handler=return_none)
```


```python
# method of excepts
def __call__(self, *args, **kwargs)
```



## module `toolz/dicttoolz.py`

```python
def merge(*dicts, **kwargs)
```
Merge a collection of dictionaries

>>> merge({1: 'one'}, {2: 'two'})
{1: 'one', 2: 'two'}

Later dictionaries have precedence

>>> merge({1: 2, 3: 4}, {3: 3, 4: 4})
{1: 2, 3: 3, 4: 4}

See Also:
    merge_with

```python
def merge_with(func, *dicts, **kwargs)
```
Merge dictionaries and apply function to combined values

A key may occur in more than one dict, and all values mapped from the key
will be passed to the function as a list, such as func([val1, val2, ...]).

>>> merge_with(sum, {1: 1, 2: 2}, {1: 10, 2: 20})
{1: 11, 2: 22}

>>> merge_with(first, {1: 1, 2: 2}, {2: 20, 3: 30})  # doctest: +SKIP
{1: 1, 2: 2, 3: 30}

See Also:
    merge

```python
def valmap(func, d, factory=dict)
```
Apply function to values of dictionary

>>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
>>> valmap(sum, bills)  # doctest: +SKIP
{'Alice': 65, 'Bob': 45}

See Also:
    keymap
    itemmap

```python
def keymap(func, d, factory=dict)
```
Apply function to keys of dictionary

>>> bills = {"Alice": [20, 15, 30], "Bob": [10, 35]}
>>> keymap(str.lower, bills)  # doctest: +SKIP
{'alice': [20, 15, 30], 'bob': [10, 35]}

See Also:
    valmap
    itemmap

```python
def itemmap(func, d, factory=dict)
```
Apply function to items of dictionary

>>> accountids = {"Alice": 10, "Bob": 20}
>>> itemmap(reversed, accountids)  # doctest: +SKIP
{10: "Alice", 20: "Bob"}

See Also:
    keymap
    valmap

```python
def valfilter(predicate, d, factory=dict)
```
Filter items in dictionary by value

>>> iseven = lambda x: x % 2 == 0
>>> d = {1: 2, 2: 3, 3: 4, 4: 5}
>>> valfilter(iseven, d)
{1: 2, 3: 4}

See Also:
    keyfilter
    itemfilter
    valmap

```python
def keyfilter(predicate, d, factory=dict)
```
Filter items in dictionary by key

>>> iseven = lambda x: x % 2 == 0
>>> d = {1: 2, 2: 3, 3: 4, 4: 5}
>>> keyfilter(iseven, d)
{2: 3, 4: 5}

See Also:
    valfilter
    itemfilter
    keymap

```python
def itemfilter(predicate, d, factory=dict)
```
Filter items in dictionary by item

>>> def isvalid(item):
...     k, v = item
...     return k % 2 == 0 and v < 4

>>> d = {1: 2, 2: 3, 3: 4, 4: 5}
>>> itemfilter(isvalid, d)
{2: 3}

See Also:
    keyfilter
    valfilter
    itemmap

```python
def assoc(d, key, value, factory=dict)
```
Return a new dict with new key value pair

New dict has d[key] set to value. Does not modify the initial dictionary.

>>> assoc({'x': 1}, 'x', 2)
{'x': 2}
>>> assoc({'x': 1}, 'y', 3)   # doctest: +SKIP
{'x': 1, 'y': 3}

```python
def dissoc(d, *keys, **kwargs)
```
Return a new dict with the given key(s) removed.

New dict has d[key] deleted for each supplied key.
Does not modify the initial dictionary.

>>> dissoc({'x': 1, 'y': 2}, 'y')
{'x': 1}
>>> dissoc({'x': 1, 'y': 2}, 'y', 'x')
{}
>>> dissoc({'x': 1}, 'y') # Ignores missing keys
{'x': 1}

```python
def assoc_in(d, keys, value, factory=dict)
```
Return a new dict with new, potentially nested, key value pair

>>> purchase = {'name': 'Alice',
...             'order': {'items': ['Apple', 'Orange'],
...                       'costs': [0.50, 1.25]},
...             'credit card': '5555-1234-1234-1234'}
>>> assoc_in(purchase, ['order', 'costs'], [0.25, 1.00]) # doctest: +SKIP
{'credit card': '5555-1234-1234-1234',
 'name': 'Alice',
 'order': {'costs': [0.25, 1.00], 'items': ['Apple', 'Orange']}}

```python
def update_in(d, keys, func, default=None, factory=dict)
```
Update value in a (potentially) nested dictionary

inputs:
d - dictionary on which to operate
keys - list or tuple giving the location of the value to be changed in d
func - function to operate on that value

If keys == [k0,..,kX] and d[k0]..[kX] == v, update_in returns a copy of the
original dictionary with v replaced by func(v), but does not mutate the
original dictionary.

If k0 is not a key in d, update_in creates nested dictionaries to the depth
specified by the keys, with the innermost value set to func(default).

>>> inc = lambda x: x + 1
>>> update_in({'a': 0}, ['a'], inc)
{'a': 1}

>>> transaction = {'name': 'Alice',
...                'purchase': {'items': ['Apple', 'Orange'],
... ...

```python
def get_in(keys, coll, default=None, no_default=False)
```
Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.

If coll[i0][i1]...[iX] cannot be found, returns ``default``, unless
``no_default`` is specified, then it raises KeyError or IndexError.

``get_in`` is a generalization of ``operator.getitem`` for nested data
structures such as dictionaries and lists.

>>> transaction = {'name': 'Alice',
...                'purchase': {'items': ['Apple', 'Orange'],
...                             'costs': [0.50, 1.25]},
...                'credit card': '5555-1234-1234-1234'}
>>> get_in(['purchase', 'items', 0], transaction)
'Apple'
>>> get_in(['name'], transaction)
'Alice'
>>> get_in(['purchase', 'total'], transaction)
>>> get_in(['purchase', 'items ...


## module `toolz/recipes.py`

```python
def countby(key, seq)
```
Count elements of a collection by a key function

>>> countby(len, ['cat', 'mouse', 'dog'])
{3: 2, 5: 1}

>>> def iseven(x): return x % 2 == 0
>>> countby(iseven, [1, 2, 3])  # doctest:+SKIP
{True: 1, False: 2}

See Also:
    groupby

```python
def partitionby(func, seq)
```
Partition a sequence according to a function

Partition `s` into a sequence of lists such that, when traversing
`s`, every time the output of `func` changes a new list is started
and that and subsequent items are collected into that list.

>>> is_space = lambda c: c == " "
>>> list(partitionby(is_space, "I have space"))
[('I',), (' ',), ('h', 'a', 'v', 'e'), (' ',), ('s', 'p', 'a', 'c', 'e')]

>>> is_large = lambda x: x > 10
>>> list(partitionby(is_large, [1, 2, 1, 99, 88, 33, 99, -1, 5]))
[(1, 2, 1), (99, 88, 33, 99), (-1, 5)]

See also:
    partition
    groupby
    itertools.groupby


## module `toolz/sandbox/core.py`

```python
class EqualityHashKey()
```
Create a hash key that uses equality comparisons between items.

This may be used to create hash keys for otherwise unhashable types:

>>> from toolz import curry
>>> EqualityHashDefault = curry(EqualityHashKey, None)
>>> set(map(EqualityHashDefault, [[], (), [1], [1]]))  # doctest: +SKIP
{=[]=, =()=, =[1]=}

**Caution:** adding N ``EqualityHashKey`` items to a hash container
may require O(N**2) operations, not O(N) as for typical hashable types.
Therefore, a suitable key function such as ``tuple`` or ``frozenset``
is usually preferred over using ``EqualityHashKey`` if possible.

The ``key`` argument to ``EqualityHashKey`` should be a function or
index that returns a hashable object that eff ...

```python
# method of EqualityHashKey
def __init__(self, key, item)
```


```python
def unzip(seq)
```
Inverse of ``zip``

>>> a, b = unzip([('a', 1), ('b', 2)])
>>> list(a)
['a', 'b']
>>> list(b)
[1, 2]

Unlike the naive implementation ``def unzip(seq): zip(*seq)`` this
implementation can handle an infinite sequence ``seq``.

Caveats:

* The implementation uses ``tee``, and so can use a significant amount
  of auxiliary storage if the resulting iterators are consumed at
  different times.

* The inner sequence cannot be infinite. In Python 3 ``zip(*seq)`` can be
  used if ``seq`` is a finite sequence of infinite sequences.


## module `toolz/sandbox/parallel.py`

```python
def fold(binop, seq, default=no_default, map=map, chunksize=128, combine=None)
```
Reduce without guarantee of ordered reduction.

Parameters
----------
binops
    Associative operator. The associative property allows us to
    leverage a parallel map to perform reductions in parallel.


inputs:

``binop``     - associative operator. The associative property allows us to
                leverage a parallel map to perform reductions in parallel.

``seq``       - a sequence to be aggregated
``default``   - an identity element like 0 for ``add`` or 1 for mul

``map``       - an implementation of ``map``. This may be parallel and
                determines how work is distributed.
``chunksize`` - Number of elements of ``seq`` that should be handled
                within a sin ...
