import io
import os
import sys
import tempfile
import threading
import time
import types
import weakref
import shutil
import builtins
import math
import itertools
import pytest
import re
import uuid
import zlib

from boltons import cacheutils
from boltons import iterutils
from boltons import timeutils
from boltons import queueutils
from boltons import fileutils
from boltons import dictutils
from boltons import setutils
from boltons import strutils

# ---- Tests for boltons.cacheutils ----

def test_lri_basic_insert_and_eviction():
    cache = cacheutils.LRI(max_size=2)
    cache['a'] = 1
    cache['b'] = 2
    assert dict(cache) == {'a': 1, 'b': 2}
    cache['c'] = 3
    # 'a' should be evicted as least recently inserted
    assert 'a' not in cache
    assert set(cache) == {'b', 'c'}
    assert cache['b'] == 2

def test_lri_hit_miss_counts_and_on_miss_callable():
    called = []
    def on_miss(k):
        called.append(k)
        return k * 2
    cache = cacheutils.LRI(max_size=3, on_miss=on_miss)
    assert cache.miss_count == 0
    assert cache.hit_count == 0
    # miss triggers on_miss
    assert cache['x'] == 'xx'
    assert cache.miss_count == 1
    # soft miss via get with default
    assert cache.get('y', 42) == 42
    assert cache.soft_miss_count == 1
    # hit increments hit_count
    assert cache['x'] == 'xx'
    assert cache.hit_count == 1

def test_lri_setitem_replaces_value_and_moves_to_front():
    cache = cacheutils.LRI(max_size=3)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3
    cache['a'] = 10  # update value for 'a' and move to front
    keys = [k for k, v in cache._get_flattened_ll()]
    # 'a' should be most recently inserted (last before anchor)
    assert keys[-2] == 'a'
    assert cache['a'] == 10

def test_lri_pop_and_popitem_and_delitem():
    cache = cacheutils.LRI(max_size=3)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3
    val = cache.pop('b')
    assert val == 2
    with pytest.raises(KeyError):
        cache.pop('b')
    val2 = cache.pop('missing', default=42)
    assert val2 == 42
    # popitem removes and returns last inserted (most recent)
    item = cache.popitem()
    assert item[0] == 'c'
    assert 'c' not in cache
    del cache['a']
    assert 'a' not in cache

def test_lri_copy_and_eq_ne_repr():
    cache = cacheutils.LRI(max_size=3)
    cache['a'] = 1
    copy_cache = cache.copy()
    assert copy_cache == cache
    assert not (copy_cache != cache)
    repr_str = repr(cache)
    assert 'LRI' in repr_str
    # eq with regular dict compares equal if keys and values match
    d = dict(cache)
    assert cache == d
    assert not (cache != d)

def test_lru_behavior_moves_to_front_on_getitem():
    cache = cacheutils.LRU(max_size=3)
    cache['a'] = 1
    cache['b'] = 2
    cache['c'] = 3
    # Reading 'a' should move it to front (most recently used)
    cache['a']
    keys = [k for k, v in cache._get_flattened_ll()]
    assert keys[-2] == 'a'  # second last before anchor
    cache['d'] = 4  # causes eviction of least recently used 'b'
    assert 'b' not in cache

def test_make_cache_key_basic_and_typed_behavior():
    # no kwargs, no typed
    key = cacheutils.make_cache_key(('a', 'b'), {})
    assert isinstance(key, cacheutils._HashedKey)
    assert list(key)[:2] == ['a', 'b']
    # single arg is fasttype returns as is
    key2 = cacheutils.make_cache_key(('a',), {})
    assert key2 == 'a'
    # with kwargs
    key3 = cacheutils.make_cache_key(('a',), {'x': 1})
    assert isinstance(key3, cacheutils._HashedKey)
    # with typed
    key4 = cacheutils.make_cache_key(('a', 3), {'b': 5}, typed=True)
    assert isinstance(key4, cacheutils._HashedKey)
    assert all(isinstance(k, (str, type)) for k in key4)
    # different key for 3 vs 3.0 with typed
    k_int = cacheutils.make_cache_key((3,), {}, typed=True)
    k_float = cacheutils.make_cache_key((3.0,), {}, typed=True)
    assert k_int != k_float

def test_cachedfunction_cache_and_repr():
    calls = []
    def f(x): calls.append(x); return x + 1
    cache = {}
    cached_f = cacheutils.CachedFunction(f, cache)
    val = cached_f(10)
    assert val == 11
    assert calls == [10]
    # second call hits cache
    val2 = cached_f(10)
    assert val2 == 11
    assert calls == [10]
    repr_str = repr(cached_f)
    assert 'CachedFunction' in repr_str

def test_cachedmethod_cache_and_repr():
    class Obj:
        def __init__(self):
            self.cache = {}
        @cacheutils.cachedmethod('cache')
        def f(self, x):
            return x*2
    o = Obj()
    # initial call caches result
    ret = o.f(3)
    assert ret == 6
    # cached call returns from cache
    ret = o.f(3)
    assert ret == 6
    # access via class returns descriptor
    attr = Obj.f
    assert isinstance(attr, cacheutils.CachedMethod)
    # repr includes 'bound_to'
    repr1 = repr(o.f)
    assert 'bound_to' in repr1

def test_cached_decorator_basic():
    cache = {}
    @cacheutils.cached(cache)
    def multiply(x, y):
        return x*y
    assert multiply(3,4) == 12
    assert multiply(3,4) == 12  # cache hit

def test_cachedmethod_decorator_with_callable_cache():
    class A:
        def __init__(self):
            self._cache = {}
        def get_cache(self):
            return self._cache
        @cacheutils.cachedmethod(get_cache)
        def triple(self, x):
            return x*3
    a = A()
    assert a.triple(7) == 21
    assert a.triple(7) == 21

def test_cachedproperty_laziness_and_repr():
    class A:
        def __init__(self):
            self.calls = 0
        @cacheutils.cachedproperty
        def val(self):
            self.calls += 1
            return 42
    a = A()
    assert a.val == 42
    assert a.val == 42
    assert a.calls == 1
    # deleting attribute triggers recompute
    del a.val
    assert a.val == 42
    assert a.calls == 2
    repr_str = repr(cacheutils.cachedproperty(lambda self: None))
    assert 'cachedproperty' in repr_str

def test_thresholdcounter_basic_add_and_accuracy():
    tc = cacheutils.ThresholdCounter(threshold=0.1)
    assert tc.threshold == 0.1
    for i in range(10):
        tc.add(1)
    assert tc.get(1) == 10
    assert 1 in tc
    # update with counts dict
    tc.update({2: 5})
    assert tc.get(2) == 5
    # most common returns list of pairs
    m = tc.most_common()
    assert (1, 10) in m
    # elements yield repeated keys
    elems = list(tc.elements())
    assert elems.count(1) == 10
    # total counts
    assert tc.get_common_count() >= 15
    assert tc.get_uncommon_count() >= 0
    # get returns default 0 on missing
    assert tc.get('nope') == 0

def test_thresholdcounter_compaction_happens():
    tc = cacheutils.ThresholdCounter(threshold=0.5)  # small thresh_count = 2
    tc.add('a')
    tc.add('b')
    # After two adds, compaction should happen
    # both keys have count = 1 but beyond current bucket
    assert 'a' in tc or 'b' in tc  # One might be discarded on compaction

def test_thresholdcounter_update_with_kwargs_and_iterable():
    tc = cacheutils.ThresholdCounter(threshold=0.1)
    tc.update(['a', 'a', 'b'], c=3)
    assert tc.get('a') == 2
    assert tc.get('b') == 1
    assert tc.get('c') == 3

def test_minidmap_assigns_increasing_and_recycles_ids():
    m = cacheutils.MinIDMap()
    class Dummy:
        pass
    d1 = Dummy(); d2=Dummy(); d3=Dummy()
    id1 = m.get(d1)
    id2 = m.get(d2)
    id3 = m.get(d3)
    assert set((id1, id2, id3)) == {0,1,2}
    m.drop(d2)
    d4 = Dummy()
    id4 = m.get(d4)
    # id4 should reuse id2's id
    assert id4 == id2

def test_minidmap_contains_iter_len():
    m = cacheutils.MinIDMap()
    class Dummy:
        pass
    d = Dummy()
    m.get(d)
    assert d in m
    assert len(m) == 1
    keys = list(iter(m))
    assert keys == [d]
    items = list(m.iteritems())
    for k, v in items:
        assert k is d
        assert isinstance(v, int)

# ---- Tests for boltons.iterutils ----

def test_is_iterable_and_scalar_and_collection():
    assert iterutils.is_iterable([1,2,3])
    assert not iterutils.is_iterable(42)
    assert iterutils.is_scalar('abc')
    assert not iterutils.is_scalar([1,2])
    assert iterutils.is_collection([1,2])
    assert not iterutils.is_collection('abc')

def test_split_and_split_iter_behavior():
    data = ['a', 'b', None, None, 'c', None]
    result = list(iterutils.split_iter(data))
    assert result == [['a', 'b'], ['c']]
    result2 = list(iterutils.split_iter(data, sep=[None]))
    assert result2 == [['a', 'b'], [], ['c'], []]

def test_lstrip_and_rstrip_and_strip_and_iter_versions():
    seq = ['x', 'x', 'a', 'b', 'x']
    assert iterutils.lstrip(seq, 'x') == ['a', 'b', 'x']
    assert list(iterutils.lstrip_iter(seq, 'x')) == ['a', 'b', 'x']
    assert iterutils.rstrip(seq, 'x') == ['x', 'x', 'a', 'b']
    assert list(iterutils.rstrip_iter(seq, 'x')) == ['x', 'x', 'a', 'b']
    assert iterutils.strip(seq, 'x') == ['a', 'b']
    assert list(iterutils.strip_iter(seq, 'x')) == ['a', 'b']

def test_chunked_and_chunked_iter_basic():
    data = range(10)
    chunks = iterutils.chunked(data, 3)
    assert chunks == [[0,1,2],[3,4,5],[6,7,8],[9]]
    chunks_fill = iterutils.chunked(data, 3, fill=None)
    assert chunks_fill[-1] == [9, None, None]

def test_chunk_ranges_basic_align_false_and_true():
    ranges = list(iterutils.chunk_ranges(10, 5))
    assert ranges == [(0,5),(5,10)]
    ranges_overlap = list(iterutils.chunk_ranges(10,5,overlap_size=2))
    assert all(r[1] >= r[0] for r in ranges_overlap)
    ranges_align = list(iterutils.chunk_ranges(15,5,input_offset=4,align=True))
    assert ranges_align[0][0] == 4

def test_pairwise_and_pairwise_iter_equal_expected():
    data = list(range(5))
    pw = list(iterutils.pairwise(data))
    expected = [(0,1),(1,2),(2,3),(3,4)]
    assert pw == expected
    pw2 = list(iterutils.pairwise_iter(data))
    assert pw2 == expected

def test_windowed_and_windowed_iter_functionality():
    data = [0,1,2,3,4]
    win = iterutils.windowed(data, 3)
    assert win == [(0,1,2),(1,2,3),(2,3,4)]
    win_fill = list(iterutils.windowed_iter(data, 3, fill=None))
    assert len(win_fill) == len(data)

def test_xfrange_and_frange_float_ranges():
    ffloat = list(iterutils.xfrange(1,3,step=0.75))
    assert all(isinstance(x, float) for x in ffloat)
    flist = iterutils.frange(5)
    assert flist == [0.,1.,2.,3.,4.]
    rev = iterutils.frange(5,0,step=-1.25)
    assert rev[0] == 5.0

def test_backoff_iter_and_backoff_list():
    vals = list(iterutils.backoff_iter(1,10,count=5))
    assert vals[-1] == 10.0
    vals2 = iterutils.backoff(1,10)
    assert vals[-1] == list(vals2)[-1]

def test_bucketize_and_partition_and_unique_and_redundant():
    vals = ['a', 'b', 'c', 'a', 'b', 'b']
    buck = iterutils.bucketize(vals)
    assert True in buck
    parts = iterutils.partition(range(3), lambda x: x==0)
    assert len(parts) == 2
    uniq = list(iterutils.unique([1,2,1,3,2]))
    assert set(uniq) == {1,2,3}
    red = iterutils.redundant([1,2,1,3,2])
    assert set(red) == {1,2}

def test_one_and_first_functions_behavior():
    one_true = iterutils.one([False, True, False])
    assert one_true is True
    one_none = iterutils.one([False, False])
    assert one_none is None
    first_val = iterutils.first(['', None, 2])
    assert first_val == 2
    first_def = iterutils.first([], default=42)
    assert first_def == 42

def test_flatten_and_same_functions():
    nested = [[1,2], [[3], [4]]]
    flat = list(iterutils.flatten_iter(nested))
    assert flat == [1,2,3,4]
    assert iterutils.same([]) is True
    assert iterutils.same([1,1,1]) is True
    assert iterutils.same([1,2,1]) is False

def test_remap_basic_and_visit_drop_behavior():
    data = {'a': 1, 'b': 2, 'c': None}
    def visit(path, key, value):
        if value is None:
            return False
        return (key, value)
    result = iterutils.remap(data, visit=visit)
    assert 'c' not in result

def test_get_path_basic_and_exception_and_default():
    data = {'a': {'b': [1,2,3]}}
    assert iterutils.get_path(data, ('a', 'b', 2)) == 3
    with pytest.raises(iterutils.PathAccessError):
        iterutils.get_path(data, ('a', 'b', 4))
    val = iterutils.get_path(data, ('a', 'b', 4), default=99)
    assert val == 99

def test_research_function_finds_ints_and_exceptions():
    data = {'a': 1, 'b': 'x'}
    res = iterutils.research(data, query=lambda p,k,v: isinstance(v,int))
    assert any(v == 1 for p,v in res)
    # test with reraise False swallowing exceptions
    def bad_query(*a):
        raise ValueError('bad')
    res2 = iterutils.research(data, query=bad_query, reraise=False)
    assert isinstance(res2, list)

def test_guid_iterators_unique_and_reseed():
    giter = iterutils.GUIDerator(size=20)
    id1 = next(giter)
    id2 = next(giter)
    assert id1 != id2
    siter = iterutils.SequentialGUIDerator(size=20)
    sid1 = next(siter)
    sid2 = next(siter)
    assert sid1 != sid2

def test_soft_sorted_and_untyped_sorted_behavior():
    data = ['two', 'b', 'one', 'a']
    ss = iterutils.soft_sorted(data, first=['one', 'two'])
    assert ss.index('one') < ss.index('two')
    ut = iterutils.untyped_sorted(['abc', 2.0, 1])
    assert ut[0] == 1

# ---- Tests for boltons.timeutils ----
from datetime import datetime, timedelta, timezone, date

def test_dt_to_timestamp_naive_and_aware():
    now = datetime.utcnow()
    ts = timeutils.dt_to_timestamp(now)
    now_aware = now.replace(tzinfo=timezone.utc)
    ts2 = timeutils.dt_to_timestamp(now_aware)
    # Should be equal within a small delta
    assert abs(ts - ts2) < 1

def test_isoparse_basic_and_fractional():
    dt = timeutils.isoparse('2020-01-01T12:30:45.999999')
    assert dt.microsecond == 999999

def test_parse_timedelta_basic_and_invalid():
    td = timeutils.parse_timedelta('1d 2h 30m')
    assert td.days == 1
    with pytest.raises(ValueError):
        timeutils.parse_timedelta('1z 5q')

def test_decimal_relative_time_and_relative_time_strings():
    d1 = datetime.now(timezone.utc)
    d2 = d1 - timedelta(days=1, seconds=3600)
    dec = timeutils.decimal_relative_time(d2, d1)
    rel = timeutils.relative_time(d2, d1)
    assert isinstance(dec, tuple)
    assert 'ago' in rel or 'from now' in rel

def test_strpdate_and_daterange_valid_and_invalid():
    dt = timeutils.strpdate('2023-06-15', '%Y-%m-%d')
    assert isinstance(dt, date)
    dr = list(timeutils.daterange(date(2023,1,1), date(2023,1,5)))
    assert all(isinstance(d, date) for d in dr)
    with pytest.raises(ValueError):
        list(timeutils.daterange(date(2023,1,1), date(2023,1,10), step=0))

def test_constanttzinfo_and_localtzinfo_repr_and_utcoffset():
    tz = timeutils.ConstantTZInfo(name='FOO', offset=timedelta(hours=3))
    assert 'FOO' in repr(tz)
    ltz = timeutils.LocalTZInfo()
    offs = ltz.utcoffset(datetime.now())
    assert isinstance(offs, timedelta)

def test_ustimezone_dst_and_fromutc_behavior():
    eastern = timeutils.Eastern
    now = datetime.now(timezone.utc)
    offset = eastern.utcoffset(now.astimezone(eastern))
    assert isinstance(offset, timedelta)
    # fromutc should not raise
    dt = datetime(2020, 3, 8, 2, tzinfo=eastern)
    res = eastern.fromutc(dt)
    assert isinstance(res, datetime)

# ---- Tests for boltons.queueutils ----

def test_base_priority_queue_add_peek_pop():
    pq = queueutils.PriorityQueue()
    pq.add('low', 0)
    pq.add('high', 10)
    pq.add('mid', 5)
    assert len(pq) == 3
    assert pq.peek() == 'high'
    p1 = pq.pop()
    assert p1 == 'high'
    assert len(pq) == 2
    # Remove item and ensure it's gone
    pq.remove('mid')
    assert len(pq) == 1
    with pytest.raises(KeyError):
        pq.remove('nonexist')
    # Pop remaining items
    v = pq.pop()
    assert v == 'low'
    with pytest.raises(IndexError):
        pq.pop()

def test_heap_priority_queue_and_sorted_priority_queue_consistency():
    hpq = queueutils.HeapPriorityQueue()
    spq = queueutils.SortedPriorityQueue()
    for q in (hpq, spq):
        q.add('task1', 1)
        q.add('task2', 5)
        q.add('task3', 3)
    assert hpq.pop() == 'task2'
    assert spq.pop() == 'task2'

# ---- Tests for boltons.fileutils ----

def test_mkdir_p_creates_and_no_errors_on_existing(tmp_path):
    new_dir = tmp_path / 'a/b/c'
    cacheutils.mkdir_p(new_dir)
    assert new_dir.exists()
    # calling again doesn't error
    cacheutils.mkdir_p(new_dir)

def test_fileperms_from_int_and_str_behavior():
    perms = fileutils.FilePerms.from_int(0o644)
    assert 'r' in perms.user
    assert int(perms) == 0o644
    perms2 = fileutils.FilePerms('rwx','r','')
    assert 'w' in perms2.user
    with pytest.raises(ValueError):
        fileutils.FilePerms().user = 'invalid'

def test_atomic_saver_context_manager_creates_and_removes(tmp_path):
    dest = tmp_path / 'dest.txt'
    with fileutils.AtomicSaver(dest) as f:
        f.write(b'hello')
    # after context, file exists with content
    assert dest.read_bytes() == b'hello'
    # test with exception rolls back
    s = fileutils.AtomicSaver(dest)
    with pytest.raises(ZeroDivisionError):
        with s as f:
            f.write(b'fail')
            1/0
    assert dest.read_bytes() == b'hello'  # content unchanged

def test_iter_find_files_finds_files(tmp_path):
    file1 = tmp_path / 'file1.py'
    file2 = tmp_path / 'file2.txt'
    file1.write_text('print(1)')
    file2.write_text('hello')
    found = list(fileutils.iter_find_files(tmp_path, '*.py'))
    assert str(file1) in found
    assert str(file2) not in found

def test_copytree_copies_and_ignores(tmp_path):
    src = tmp_path / 'src'
    dst = tmp_path / 'dst'
    src.mkdir()
    f = src / 'file.txt'
    f.write_text('data')
    fileutils.copytree(str(src), str(dst))
    assert (dst / 'file.txt').exists()

def test_dummyfile_behaves_like_file():
    df = fileutils.DummyFile('dummy', 'r')
    with pytest.raises(ValueError):
        df.flush()
    df.close()
    with pytest.raises(ValueError):
        df.write('x')

def test_rotate_file_rotates_files(tmp_path):
    base = tmp_path / 'testfile.txt'
    base.write_text('1')
    for i in range(3):
        n = tmp_path / f'testfile.{i+1}.txt'
        n.write_text(str(i+1))
    fileutils.rotate_file(str(base), keep=3)
    # base file moved to .1, and .1 to .2 etc
    assert (tmp_path / 'testfile.1.txt').read_text() == '1'


# ---- Tests for boltons.dictutils ----

def test_orderedmultidict_basic_and_add_and_getlist():
    omd = dictutils.OrderedMultiDict()
    omd['a'] = 1
    omd.add('a', 2)
    assert omd.get('a') == 2
    assert omd.getlist('a') == [1,2]
    # addlist adds multiple values
    omd.addlist('b', [3,4])
    assert omd.getlist('b') == [3,4]

def test_orderedmultidict_pop_popall_poplast_behavior():
    omd = dictutils.OMD([('a', 1), ('a', 2), ('b', 3)])
    last = omd.poplast('a')
    assert last == 2
    all_a = omd.popall('a')
    assert all_a == [1]
    popped = omd.pop('b')
    assert popped == 3

def test_orderedmultidict_sorted_and_sortedvalues_and_inverted():
    omd = dictutils.OMD([('a', 3), ('b', 1), ('a', 1), ('b', 2)])
    somd = omd.sorted(key=lambda i:i[1])
    assert somd.items(multi=True)[0][1] == 1
    sv = omd.sortedvalues()
    assert sorted(sv.getlist('a')) == [1,3]
    inv = omd.inverted()
    vals = inv.getlist(1)
    assert  'a' in vals or 'b' in vals

def test_onetoone_basic_and_update_and_del():
    oto = dictutils.OneToOne()
    oto['a'] = 1
    oto['b'] = 2
    assert oto.inv[1] == 'a'
    # Overwrite and test
    oto['c'] = 2
    assert 2 in oto.inv
    with pytest.raises(KeyError):
        _ = oto['b']
    del oto['a']
    assert 'a' not in oto
    oto.update({'x': 10})
    assert oto['x'] == 10

def test_manytomany_basic_and_add_and_remove_and_replace():
    mtm = dictutils.ManyToMany()
    mtm.add('a', 1)
    mtm.add('a', 2)
    mtm.add('b', 2)
    assert 1 in mtm['a']
    assert 'a' in mtm.inv[1]
    mtm.remove('a',1)
    assert 1 not in mtm.get('a', set())
    mtm.replace('b', 'c')
    assert 'c' in mtm and 'b' not in mtm

def test_subdict_keep_and_drop():
    d = {'a':1, 'b':2, 'c':3}
    kept = dictutils.subdict(d, keep=['a','c'])
    assert set(kept) == {'a','c'}
    dropped = dictutils.subdict(d, drop=['b'])
    assert 'b' not in dropped

def test_frozendict_immutable_and_hashable():
    fd = dictutils.FrozenDict({'a':1, 'b':2})
    with pytest.raises(TypeError):
        fd['x'] = 42
    assert fd['a'] == 1
    h1 = hash(fd)
    fd2 = dictutils.FrozenDict({'a':1, 'b':2})
    h2 = hash(fd2)
    assert h1 == h2
    assert fd == fd2

# ---- Tests for boltons.setutils ----

def test_indexedset_add_and_remove_and_contains():
    s = setutils.IndexedSet()
    s.add(1)
    s.add(2)
    s.add(2)  # duplicate ignored
    assert 2 in s
    s.remove(1)
    assert 1 not in s
    s.discard(3)  # no error
    assert len(s) == 1

def test_indexedset_index_and_pop_and_slice():
    s = setutils.IndexedSet([0,1,2,3,4])
    assert s.index(3) == 3
    popped = s.pop(2)
    assert popped == 2
    sliced = s[1:3]
    assert isinstance(sliced, setutils.IndexedSet)
    assert 3 in sliced

def test_indexedset_set_operations_and_update_and_clears():
    s = setutils.IndexedSet([1,2,3])
    o = {3,4,5}
    u = s.union(o)
    assert 4 in u
    s.update([4,5])
    assert 5 in s
    s.clear()
    assert len(s) == 0

def test_complement_set_basic_and_operations():
    s = setutils.complement(set([1,2]))
    assert 3 in s
    s2 = set([2,3])
    inter = s & s2
    assert 1 not in inter
    union = s | s2
    assert 1 in union

# ---- Tests for boltons.strutils ----

def test_camel2under_and_under2camel():
    assert strutils.camel2under('CamelCase') == 'camel_case'
    assert strutils.under2camel('snake_case') == 'SnakeCase'

def test_slugify_and_split_punct_ws():
    slug = strutils.slugify("Hello, World!")
    assert 'hello' in slug
    split = strutils.split_punct_ws("Hello, World!")
    assert 'Hello' in split and 'World' in split

def test_ordinalize_and_cardinalize():
    assert strutils.ordinalize(1) == '1st'
    assert strutils.cardinalize('cat', 2) == 'cats'

def test_pluralize_and_singularize_irregulars():
    assert strutils.pluralize('child') == 'children'
    assert strutils.singularize('children') == 'child'
    assert strutils.singularize('Glasses') == 'Glass'

def test_find_hashtags_and_a10n():
    tags = strutils.find_hashtags('#foo #bar #baz')
    assert set(tags) == {'foo', 'bar', 'baz'}
    assert strutils.a10n('internationalization') == 'i18n'

def test_strip_ansi_and_asciify_and_is_ascii_behavior():
    ansi_text = '\x1b[31mred\x1b[0m'
    clean = strutils.strip_ansi(ansi_text)
    assert 'red' in clean
    asciified = strutils.asciify('Beyoncé')
    assert b'Beyonce' == asciified
    assert not strutils.is_ascii('é')
    assert strutils.is_ascii('abc')

def test_html2text_and_gzip_gunzip_bytes():
    html = '<b>bold</b>'
    text = strutils.html2text(html)
    assert 'bold' in text
    compressed = strutils.gzip_bytes(b'hello')
    decompressed = strutils.gunzip_bytes(compressed)
    assert decompressed == b'hello'

def test_iter_splitlines_and_indent_and_is_uuid():
    lines = list(strutils.iter_splitlines('a\nb\nc'))
    assert lines == ['a','b','c']
    indented = strutils.indent('a\nb\n', '->')
    assert indented.startswith('->a')
    u = uuid.uuid4()
    assert strutils.is_uuid(str(u))
    assert not strutils.is_uuid('not-a-uuid')

def test_escape_shell_args_style_sh_and_cmd():
    args = ['foo', 'bar baz', 'quoted"arg']
    sh_escaped = strutils.escape_shell_args(args, style='sh')
    cmd_escaped = strutils.escape_shell_args(args, style='cmd')
    assert ' ' in sh_escaped and '"' in cmd_escaped

def test_parse_int_list_and_format_int_list_and_complement_int_list_and_ranges():
    s = '1,3,5-7'
    lst = strutils.parse_int_list(s)
    fl = strutils.format_int_list(lst)
    comp = strutils.complement_int_list(s, range_end=10)
    ranges = strutils.int_ranges_from_int_list(s)
    assert 3 in lst and '-' in fl
    assert isinstance(comp, str)
    assert isinstance(ranges, tuple)

def test_multireplace_and_multi_replace_substitution():
    repl = strutils.MultiReplace({'foo': 'bar', r'ba.': 'baz'}, regex=True)
    result = repl.sub('foo baq bat')
    assert 'bar' in result or 'baz' in result
    result2 = strutils.multi_replace('foo baq bat', {'foo':'bar', 'ba.':'baz'})
    assert 'bar' in result2 or 'baz' in result2

def test_unwrap_text_removeprefix_human_readable_list_ellipsize():
    text = "This is\nsome text\n\nNew para."
    unwrapped = strutils.unwrap_text(text)
    assert ' ' in unwrapped
    prefixed = strutils.removeprefix('foobar', 'foo')
    assert prefixed == 'bar'
    hrlist = strutils.human_readable_list(['a','b','c'])
    assert 'and' in hrlist
    ellip = strutils.ellipsize('This is some very long text' * 5, max_len=20)
    assert ellip.endswith('…')

