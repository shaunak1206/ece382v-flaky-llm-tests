# tinydb: public API (signatures and documentation)


## module `tinydb/database.py`

```python
class TinyDB(TableBase)
```
The main class of TinyDB.

The ``TinyDB`` class is responsible for creating the storage class instance
that will store this database's documents, managing the database
tables as well as providing access to the default table.

For table management, a simple ``dict`` is used that stores the table class
instances accessible using their table name.

Default table access is provided by forwarding all unknown method calls
and property access operations to the default table by implementing
``__getattr__``.

When creating a new instance, all arguments and keyword arguments (except
for ``storage``) will be passed to the storage class that is provided. If
no storage class is specified, :class:`~tinydb ...

```python
# method of TinyDB
def __init__(self, *args, **kwargs) -> None
```
Create a new instance of TinyDB.

```python
# method of TinyDB
def table(self, name: str, **kwargs) -> Table
```
Get access to a specific table.

If the table hasn't been accessed yet, a new table instance will be
created using the :attr:`~tinydb.database.TinyDB.table_class` class.
Otherwise, the previously created table instance will be returned.

All further options besides the name are passed to the table class which
by default is :class:`~tinydb.table.Table`. Check its documentation
for further parameter ...

```python
# method of TinyDB
def tables(self) -> set[str]
```
Get the names of all tables in the database.

:returns: a set of table names

```python
# method of TinyDB
def drop_tables(self) -> None
```
Drop all tables from the database. **CANNOT BE REVERSED!**

```python
# method of TinyDB
def drop_table(self, name: str) -> None
```
Drop a specific table from the database. **CANNOT BE REVERSED!**

:param name: The name of the table to drop.

```python
# method of TinyDB
def storage(self) -> Storage
```
Get the storage instance used for this TinyDB instance.

:return: This instance's storage
:rtype: Storage

```python
# method of TinyDB
def close(self) -> None
```
Close the database.

This may be needed if the storage instance used for this database
needs to perform cleanup operations like closing file handles.

To ensure this method is called, the TinyDB instance can be used as a
context manager::

    with TinyDB('data.json') as db:
        db.insert({'foo': 'bar'})

Upon leaving this context, the ``close`` method will be called.

```python
# method of TinyDB
def __enter__(self)
```
Use the database as a context manager.

Using the database as a context manager ensures that the
:meth:`~tinydb.database.TinyDB.close` method is called upon leaving
the context.

:return: The current instance

```python
# method of TinyDB
def __exit__(self, *args)
```
Close the storage instance when leaving a context.


## module `tinydb/middlewares.py`

```python
class Middleware()
```
The base class for all Middlewares.

Middlewares hook into the read/write process of TinyDB allowing you to
extend the behaviour by adding caching, logging, ...

Your middleware's ``__init__`` method has to call the parent class
constructor so the middleware chain can be configured properly.

```python
# method of Middleware
def __init__(self, storage_cls) -> None
```


```python
# method of Middleware
def __call__(self, *args, **kwargs)
```
Create the storage instance and store it as self.storage.

Usually a user creates a new TinyDB instance like this::

    TinyDB(storage=StorageClass)

The storage keyword argument is used by TinyDB this way::

    self.storage = storage(*args, **kwargs)

As we can see, ``storage(...)`` runs the constructor and returns the
new storage instance.


Using Middlewares, the user will call:: ...

```python
class CachingMiddleware(Middleware)
```
Add some caching to TinyDB.

This Middleware aims to improve the performance of TinyDB by writing only
the last DB state every :attr:`WRITE_CACHE_SIZE` time and reading always
from cache.

```python
# method of CachingMiddleware
def __init__(self, storage_cls)
```


```python
# method of CachingMiddleware
def read(self)
```


```python
# method of CachingMiddleware
def write(self, data)
```


```python
# method of CachingMiddleware
def flush(self)
```
Flush all unwritten data to disk.

```python
# method of CachingMiddleware
def close(self)
```



## module `tinydb/mypy_plugin.py`

```python
class TinyDBPlugin(Plugin)
```


```python
# method of TinyDBPlugin
def __init__(self, options: Options)
```


```python
# method of TinyDBPlugin
def get_dynamic_class_hook(self, fullname: str) -> CB[DynamicClassDef]
```


```python
def plugin(_version: str)
```



## module `tinydb/operations.py`

```python
def delete(field: str) -> Callable[[MutableMapping], None]
```
Delete a given field from the document.

```python
def add(field: str, n: Union[int, float]) -> Callable[[MutableMapping], None]
```
Add ``n`` to a given field in the document.

```python
def subtract(field: str, n: Union[int, float]) -> Callable[[MutableMapping], None]
```
Subtract ``n`` to a given field in the document.

```python
def set(field: str, val: Any) -> Callable[[MutableMapping], None]
```
Set a given field to ``val``.

```python
def increment(field: str) -> Callable[[MutableMapping], None]
```
Increment a given field in the document by 1.

```python
def decrement(field: str) -> Callable[[MutableMapping], None]
```
Decrement a given field in the document by 1.


## module `tinydb/queries.py`

```python
class QueryLike(Protocol)
```
A typing protocol that acts like a query.

Something that we use as a query must have two properties:

1. It must be callable, accepting a `Mapping` object and returning a
   boolean that indicates whether the value matches the query, and
2. it must have a stable hash that will be used for query caching.

In addition, to mark a query as non-cacheable (e.g. if it involves
some remote lookup) it needs to have a method called ``is_cacheable``
that returns ``False``.

This query protocol is used to make MyPy correctly support the query
pattern that TinyDB uses.

See also https://mypy.readthedocs.io/en/stable/protocols.html#simple-user-defined-protocols

```python
# method of QueryLike
def __call__(self, value: Mapping) -> bool
```


```python
class Query(QueryInstance)
```
TinyDB Queries.

Allows building queries for TinyDB databases. There are two main ways of
using queries:

1) ORM-like usage:

>>> User = Query()
>>> db.search(User.name == 'John Doe')
>>> db.search(User['logged-in'] == True)

2) Classical usage:

>>> db.search(where('value') == True)

Note that ``where(...)`` is a shorthand for ``Query(...)`` allowing for
a more fluent syntax.

Besides the methods documented here you can combine queries using the
binary AND and OR operators:

>>> # Binary AND:
>>> db.search((where('field1').exists()) & (where('field2') == 5))
>>> # Binary OR:
>>> db.search((where('field1').exists()) | (where('field2') == 5))

Queries are executed by calling the resulting obj ...

```python
# method of Query
def __init__(self) -> None
```


```python
# method of Query
def exists(self) -> QueryInstance
```
Test for a dict where a provided key exists.

>>> Query().f1.exists()

```python
# method of Query
def matches(self, regex: str, flags: int=0) -> QueryInstance
```
Run a regex test against a dict value (whole string has to match).

>>> Query().f1.matches(r'^\w+$')

:param regex: The regular expression to use for matching
:param flags: regex flags to pass to ``re.match``

```python
# method of Query
def search(self, regex: str, flags: int=0) -> QueryInstance
```
Run a regex test against a dict value (only substring string has to
match).

>>> Query().f1.search(r'^\w+$')

:param regex: The regular expression to use for matching
:param flags: regex flags to pass to ``re.match``

```python
# method of Query
def test(self, func: Callable[[Mapping], bool], *args) -> QueryInstance
```
Run a user-defined test function against a dict value.

>>> def test_func(val):
...     return val == 42
...
>>> Query().f1.test(test_func)

.. warning::

    The test function provided needs to be deterministic (returning the
    same value when provided with the same arguments), otherwise this
    may mess up the query cache that :class:`~tinydb.table.Table`
    implements.

:param func: The fun ...

```python
# method of Query
def any(self, cond: Union[QueryInstance, list[Any]]) -> QueryInstance
```
Check if a condition is met by any document in a list,
where a condition can also be a sequence (e.g. list).

>>> Query().f1.any(Query().f2 == 1)

Matches::

    {'f1': [{'f2': 1}, {'f2': 0}]}

>>> Query().f1.any([1, 2, 3])

Matches::

    {'f1': [1, 2]}
    {'f1': [3, 4, 5]}

:param cond: Either a query that at least one document has to match or
             a list of which at least one document ...

```python
# method of Query
def all(self, cond: Union['QueryInstance', list[Any]]) -> QueryInstance
```
Check if a condition is met by all documents in a list,
where a condition can also be a sequence (e.g. list).

>>> Query().f1.all(Query().f2 == 1)

Matches::

    {'f1': [{'f2': 1}, {'f2': 1}]}

>>> Query().f1.all([1, 2, 3])

Matches::

    {'f1': [1, 2, 3, 4, 5]}

:param cond: Either a query that all documents have to match or a list
             which has to be contained in the tested document.

```python
# method of Query
def one_of(self, items: list[Any]) -> QueryInstance
```
Check if the value is contained in a list or generator.

>>> Query().f1.one_of(['value 1', 'value 2'])

:param items: The list of items to check with

```python
# method of Query
def fragment(self, document: Mapping) -> QueryInstance
```


```python
# method of Query
def noop(self) -> QueryInstance
```
Always evaluate to ``True``.

Useful for having a base value when composing queries dynamically.

```python
# method of Query
def map(self, fn: Callable[[Any], Any]) -> 'Query'
```
Add a function to the query path. Similar to __getattr__ but for
arbitrary functions.

```python
def where(key: str) -> Query
```
A shorthand for ``Query()[key]``


## module `tinydb/storages.py`

```python
class Storage(ABC)
```
The abstract base class for all Storages.

A Storage (de)serializes the current state of the database and stores it in
some place (memory, file on disk, ...).

```python
# method of Storage
def read(self) -> Optional[dict[str, dict[str, Any]]]
```
Read the current state.

Any kind of deserialization should go here.

Return ``None`` here to indicate that the storage is empty.

```python
# method of Storage
def write(self, data: dict[str, dict[str, Any]]) -> None
```
Write the current state of the database to the storage.

Any kind of serialization should go here.

:param data: The current state of the database.

```python
# method of Storage
def close(self) -> None
```
Optional: Close open file handles, etc.

```python
class JSONStorage(Storage)
```
Store the data in a JSON file.

```python
# method of JSONStorage
def __init__(self, path: str, create_dirs=False, encoding=None, access_mode='r+', **kwargs)
```
Create a new instance.

Also creates the storage file, if it doesn't exist and the access mode
is appropriate for writing.

**Note:** Using an access mode other than `r` or `r+` will probably
lead to data loss or data corruption!

**Note:** **Never** pass untrusted or user-controlled code as ``kwargs``
members like ``cls`` or ``default`` will be called on every write
operation.

:param path: Where ...

```python
# method of JSONStorage
def close(self) -> None
```


```python
# method of JSONStorage
def read(self) -> Optional[dict[str, dict[str, Any]]]
```


```python
# method of JSONStorage
def write(self, data: dict[str, dict[str, Any]])
```


```python
class MemoryStorage(Storage)
```
Store the data as JSON in memory.

```python
# method of MemoryStorage
def __init__(self)
```
Create a new instance.

```python
# method of MemoryStorage
def read(self) -> Optional[dict[str, dict[str, Any]]]
```


```python
# method of MemoryStorage
def write(self, data: dict[str, dict[str, Any]])
```



## module `tinydb/table.py`

```python
class Document(dict)
```
A document stored in the database.

This class provides a way to access both a document's content and
its ID using ``doc.doc_id``.

```python
# method of Document
def __init__(self, value: Mapping, doc_id: int)
```


```python
class Table()
```
Represents a single TinyDB table.

It provides methods for accessing and manipulating documents.

.. admonition:: Query Cache

    As an optimization, a query cache is implemented using a
    :class:`~tinydb.utils.LRUCache`. This class mimics the interface of
    a normal ``dict``, but starts to remove the least-recently used entries
    once a threshold is reached.

    The query cache is updated on every search operation. When writing
    data, the whole cache is discarded as the query results may have
    changed.

.. admonition:: Customization

    For customization, the following class variables can be set:

    - ``document_class`` defines the class that is used to represent
      docu ...

```python
# method of Table
def __init__(self, storage: Storage, name: str, cache_size: int=default_query_cache_capacity, persist_empty: bool=False)
```
Create a table instance.

```python
# method of Table
def name(self) -> str
```
Get the table name.

```python
# method of Table
def storage(self) -> Storage
```
Get the table storage instance.

```python
# method of Table
def insert(self, document: Mapping) -> int
```
Insert a new document into the table.

:param document: the document to insert
:returns: the inserted document's ID

```python
# method of Table
def insert_multiple(self, documents: Iterable[Mapping]) -> list[int]
```
Insert multiple documents into the table.

:param documents: an Iterable of documents to insert
:returns: a list containing the inserted documents' IDs

```python
# method of Table
def all(self) -> list[Document]
```
Get all documents stored in the table.

:returns: a list with all documents.

```python
# method of Table
def search(self, cond: QueryLike) -> list[Document]
```
Search for all documents matching a 'where' cond.

Note: a repeated search with the same ``cond`` is served from an
internal cache. The returned list is a fresh copy, but its
documents are the same objects held by the cache -- treat them as
read-only, since mutating one (a key or a nested value) corrupts
what a later identical search returns.

:param cond: the condition to check against
:returns: ...

```python
# method of Table
def get(self) -> NoReturn
```


```python
# method of Table
def get(self, cond: QueryLike, doc_id: None=..., doc_ids: None=...) -> Optional[Document]
```


```python
# method of Table
def get(self, *, cond: QueryLike, doc_id: None=..., doc_ids: None=...) -> Optional[Document]
```


```python
# method of Table
def get(self, cond: Optional[QueryLike], doc_id: int, doc_ids: Optional[list]=...) -> Optional[Document]
```


```python
# method of Table
def get(self, *, cond: Optional[QueryLike]=..., doc_id: int, doc_ids: Optional[list]=...) -> Optional[Document]
```


```python
# method of Table
def get(self, cond: Optional[QueryLike], doc_id: None, doc_ids: list) -> list[Document]
```


```python
# method of Table
def get(self, cond: Optional[QueryLike], *, doc_id: None=..., doc_ids: list) -> list[Document]
```


```python
# method of Table
def get(self, *, cond: Optional[QueryLike]=..., doc_id: None=..., doc_ids: list) -> list[Document]
```


```python
# method of Table
def get(self, cond: Optional[QueryLike]=None, doc_id: Optional[int]=None, doc_ids: Optional[list]=None)
```
Get exactly one document specified by a query or a document ID.
However, if multiple document IDs are given then returns all
documents in a list.

Returns ``None`` if the document doesn't exist.

:param cond: the condition to check against
:param doc_id: the document's ID
:param doc_ids: the document's IDs(multiple)

:returns: the document(s) or ``None``

```python
# method of Table
def contains(self, cond: Optional[QueryLike]=None, doc_id: Optional[int]=None) -> bool
```
Check whether the database contains a document matching a query or
an ID.

If ``doc_id`` is set, it checks if the db contains the specified ID.

:param cond: the condition use
:param doc_id: the document ID to look for

```python
# method of Table
def update(self, fields: Union[Mapping, Callable[[MutableMapping], None]], cond: Optional[QueryLike]=None, doc_ids: Optional[Iterable[int]]=None) -> list[int]
```
Update all matching documents to have a given set of fields.

When ``doc_ids`` is given, IDs that don't refer to an existing
document are silently skipped, and the returned list only contains
the IDs that were actually updated.

:param fields: the fields that the matching documents will have
               or a callable that mutates matching documents in place
:param cond: which documents to updat ...

```python
# method of Table
def update_multiple(self, updates: Iterable[tuple[Union[Mapping, Callable[[MutableMapping], None]], QueryLike]]) -> list[int]
```
Update all matching documents to have a given set of fields.

:returns: a list containing the updated document's ID

```python
# method of Table
def upsert(self, document: Mapping, cond: Optional[QueryLike]=None) -> list[int]
```
Update documents, if they exist, insert them otherwise.

Note: This will update *all* documents matching the query.
For example, if the query matches 3 documents, all 3 will be updated
with the new data. If no documents match, a new one is inserted.

Document argument can be a tinydb.table.Document object if you want
to specify a doc_id.

:param document: the document to insert or the fields to up ...

```python
# method of Table
def remove(self, cond: Optional[QueryLike]=None, doc_ids: Optional[Iterable[int]]=None) -> list[int]
```
Remove all matching documents.

When ``doc_ids`` is given, IDs that don't refer to an existing
document are silently skipped, and the returned list only contains
the IDs that were actually removed.

:param cond: the condition to check against
:param doc_ids: a list of document IDs
:returns: a list containing the removed documents' ID

```python
# method of Table
def truncate(self) -> None
```
Truncate the table by removing all documents.

```python
# method of Table
def count(self, cond: QueryLike) -> int
```
Count the documents matching a query.

:param cond: the condition use

```python
# method of Table
def clear_cache(self) -> None
```
Clear the query cache.


## module `tinydb/utils.py`

```python
def with_typehint(baseclass: type[T])
```
Add type hints from a specified class to a base class:

>>> class Foo(with_typehint(Bar)):
...     pass

This would add type hints from class ``Bar`` to class ``Foo``.

Note that while PyCharm and Pyright (for VS Code) understand this pattern,
MyPy does not. For that reason TinyDB has a MyPy plugin in
``mypy_plugin.py`` that adds support for this pattern.

```python
class LRUCache(abc.MutableMapping, Generic[K, V])
```
A least-recently used (LRU) cache with a fixed cache size.

This class acts as a dictionary but has a limited size. If the number of
entries in the cache exceeds the cache size, the least-recently accessed
entry will be discarded.

This is implemented using an ``OrderedDict``. On every access the accessed
entry is moved to the front by re-inserting it into the ``OrderedDict``.
When adding an entry and the cache size is exceeded, the last entry will
be discarded.

```python
# method of LRUCache
def __init__(self, capacity=None) -> None
```


```python
# method of LRUCache
def lru(self) -> list[K]
```


```python
# method of LRUCache
def length(self) -> int
```


```python
# method of LRUCache
def clear(self) -> None
```


```python
# method of LRUCache
def get(self, key: K, default: Optional[D]=None) -> Optional[Union[V, D]]
```


```python
# method of LRUCache
def set(self, key: K, value: V)
```


```python
def freeze(obj)
```
Freeze an object by making it immutable and thus hashable.


# Usage examples from the project README

```python
>>> from tinydb import TinyDB, Query
>>> db = TinyDB('/path/to/db.json')
>>> db.insert({'int': 1, 'char': 'a'})
>>> db.insert({'int': 1, 'char': 'b'})
```
```python
>>> User = Query()
>>> # Search for a field value
>>> db.search(User.name == 'John')
[{'name': 'John', 'age': 22}, {'name': 'John', 'age': 37}]

>>> # Combine two queries with logical and
>>> db.search((User.name == 'John') & (User.age <= 30))
[{'name': 'John', 'age': 22}]

>>> # Combine two queries with logical or
>>> db.search((User.name == 'John') | (User.name == 'Bob'))
[{'name': 'John', 'age': 22}, {'name': 'John', 'age': 37}, {'name': 'Bob', 'age': 42}]

>>> # Negate a query with logical not
>>> db.search(~(User.name == 'John'))
[{'name': 'Megan', 'age': 27}, {'name': 'Bob', 'age': 42}]

>>> # Apply transformation to field with `map`
>>> db.search((User.age.map(lambda x: x + x) == 44))
>>> [{'name': 'John', 'age': 22}]

>>> # More possible comparisons:  !=  <  >  <=  >=
>>> # More possible checks: where(...).matches(regex), where(...).test(your_test_func)
```
```python
>>> table = db.table('name')
>>> table.insert({'value': True})
>>> table.all()
[{'value': True}]
```
```python
>>> from tinydb.storages import JSONStorage
>>> from tinydb.middlewares import CachingMiddleware
>>> db = TinyDB('/path/to/db.json', storage=CachingMiddleware(JSONStorage))
```