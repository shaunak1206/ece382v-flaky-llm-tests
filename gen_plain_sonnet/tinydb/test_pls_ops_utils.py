import pytest

from tinydb import TinyDB, where
from tinydb import operations as ops
from tinydb.storages import MemoryStorage
from tinydb.utils import FrozenDict, LRUCache, freeze, with_typehint


# --- operations -------------------------------------------------------------

def test_delete_removes_field_and_missing_field_raises():
    doc = {'a': 1, 'b': 2}
    ops.delete('a')(doc)
    assert doc == {'b': 2}
    with pytest.raises(KeyError):
        ops.delete('nope')(doc)


def test_add_and_subtract():
    doc = {'n': 10, 'f': 1.5, 's': 'ab'}
    ops.add('n', 5)(doc)
    assert doc['n'] == 15
    ops.subtract('n', 3)(doc)
    assert doc['n'] == 12
    ops.add('f', 0.25)(doc)
    ops.add('s', 'cd')(doc)
    assert doc['f'] == 1.75
    assert doc['s'] == 'abcd'


def test_set_creates_or_overwrites_field():
    doc = {'a': 1}
    ops.set('a', 2)(doc)
    ops.set('b', 3)(doc)
    assert doc == {'a': 2, 'b': 3}


def test_increment_and_decrement():
    doc = {'n': 0}
    ops.increment('n')(doc)
    ops.increment('n')(doc)
    ops.decrement('n')(doc)
    assert doc['n'] == 1
    with pytest.raises(KeyError):
        ops.increment('missing')({})


def test_operations_used_through_database_update():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple([{'n': 1, 'tmp': 'x'}, {'n': 2, 'tmp': 'y'}])
    db.update(ops.add('n', 10), where('n') == 1)
    db.update(ops.delete('tmp'), where('n') == 2)
    db.update(ops.set('seen', True))
    assert db.get(doc_id=1) == {'n': 11, 'tmp': 'x', 'seen': True}
    assert db.get(doc_id=2) == {'n': 2, 'seen': True}


# --- LRUCache ---------------------------------------------------------------

def test_lru_cache_basic_mapping_behaviour():
    cache = LRUCache(capacity=3)
    cache['a'] = 1
    cache['b'] = 2
    assert cache['a'] == 1
    assert 'b' in cache
    assert 'zzz' not in cache
    assert len(cache) == 2
    assert cache.length == 2
    assert set(cache) == {'a', 'b'}


def test_lru_cache_evicts_least_recently_used():
    cache = LRUCache(capacity=2)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3
    assert 'a' not in cache
    assert cache.lru == ['b', 'c']

    # a read refreshes an entry, so 'b' is evicted next
    assert cache.get('b') == 2
    cache['d'] = 4
    assert 'c' not in cache
    assert cache.lru == ['b', 'd']


def test_lru_cache_overwrite_and_missing_key_behaviour():
    cache = LRUCache(capacity=2)
    cache['a'] = 1
    cache['b'] = 2
    cache['a'] = 10
    assert len(cache) == 2
    assert cache.lru == ['b', 'a']
    assert cache['a'] == 10
    assert cache.get('x') is None
    assert cache.get('x', 'fallback') == 'fallback'
    with pytest.raises(KeyError):
        cache['x']


def test_lru_cache_unlimited_clear_and_delete():
    cache = LRUCache()
    for i in range(100):
        cache[i] = i
    assert len(cache) == 100
    del cache[0]
    assert len(cache) == 99
    cache.clear()
    assert len(cache) == 0
    assert cache.lru == []


# --- freeze / FrozenDict ----------------------------------------------------

def test_freeze_converts_containers_to_hashable_types():
    frozen = freeze({'a': [1, {'b': 2}], 'c': {3, 4}})
    assert isinstance(frozen, FrozenDict)
    assert frozen['a'] == (1, FrozenDict({'b': 2}))
    assert frozen['c'] == frozenset({3, 4})
    hash(frozen)
    assert freeze((1, [2, 3])) == (1, (2, 3))
    assert freeze(5) == 5
    assert freeze(None) is None


def test_frozen_dict_hash_ignores_insertion_order():
    assert hash(FrozenDict(a=1, b=2)) == hash(FrozenDict(b=2, a=1))
    assert FrozenDict(a=1) == {'a': 1}


def test_frozen_dict_is_immutable():
    fd = FrozenDict(a=1)
    with pytest.raises(TypeError):
        fd['b'] = 2
    with pytest.raises(TypeError):
        del fd['a']
    with pytest.raises(TypeError):
        fd.update({'b': 2})
    with pytest.raises(TypeError):
        fd.pop('a')
    with pytest.raises(TypeError):
        fd.popitem()
    with pytest.raises(TypeError):
        fd.clear()
    with pytest.raises(TypeError):
        fd.setdefault('z', 1)
    assert fd == {'a': 1}


def test_with_typehint_returns_object_at_runtime():
    assert with_typehint(int) is object
