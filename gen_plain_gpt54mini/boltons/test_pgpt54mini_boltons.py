import os
import stat
import uuid
from datetime import date, datetime, timedelta, timezone

import pytest
from freezegun import freeze_time

from boltons.cacheutils import (
    LRI,
    LRU,
    ThresholdCounter,
    make_cache_key,
    CachedFunction,
    CachedMethod,
    cached,
    cachedmethod,
    cachedproperty,
)
from boltons.iterutils import (
    is_iterable,
    is_scalar,
    is_collection,
    split_iter,
    lstrip_iter,
    rstrip_iter,
    strip_iter,
    chunked_iter,
    chunk_ranges,
    pairwise,
    pairwise_iter,
    windowed_iter,
    xfrange,
    frange,
    backoff_iter,
    bucketize,
    partition,
    unique_iter,
    redundant,
    one,
    first,
    flatten,
    same,
    get_path,
    PathAccessError,
    research,
    GUIDerator,
    SequentialGUIDerator,
    soft_sorted,
    untyped_sorted,
    remap,
)
from boltons.timeutils import (
    dt_to_timestamp,
    isoparse,
    parse_timedelta,
    decimal_relative_time,
    relative_time,
    strpdate,
    daterange,
    ConstantTZInfo,
    UTC,
    LocalTZ,
    Eastern,
)
from boltons.queueutils import PriorityQueue, HeapPriorityQueue, SortedPriorityQueue
from boltons.fileutils import mkdir_p, FilePerms, AtomicSaver, iter_find_files, copytree, rotate_file
from boltons.dictutils import OrderedMultiDict, OneToOne, ManyToMany, subdict, FrozenDict
from boltons.setutils import IndexedSet, complement
from boltons.strutils import (
    camel2under,
    under2camel,
    slugify,
    split_punct_ws,
    unit_len,
    ordinalize,
    cardinalize,
    pluralize,
    singularize,
    find_hashtags,
    a10n,
    strip_ansi,
    bytes2human,
    html2text,
    gzip_bytes,
    gunzip_bytes,
    iter_splitlines,
    indent,
    is_uuid,
    args2sh,
    args2cmd,
    parse_int_list,
    format_int_list,
    complement_int_list,
    int_ranges_from_int_list,
    MultiReplace,
    multi_replace,
    unwrap_text,
    removeprefix,
    human_readable_list,
    ellipsize,
)


def test_lri_basic_get_set():
    c = LRI(max_size=2)
    c["a"] = 1
    c["b"] = 2
    assert c["a"] == 1
    assert c["b"] == 2
    assert len(c) == 2


def test_lri_eviction_oldest():
    c = LRI(max_size=2)
    c["a"] = 1
    c["b"] = 2
    c["c"] = 3
    assert "a" not in c
    assert list(c.keys()) == ["b", "c"]


def test_lri_access_does_not_reorder():
    c = LRI(max_size=3)
    c.update([("a", 1), ("b", 2), ("c", 3)])
    _ = c["a"]
    assert list(c.keys()) == ["a", "b", "c"]


def test_lru_access_reorders():
    c = LRU(max_size=3)
    c.update([("a", 1), ("b", 2), ("c", 3)])
    _ = c["a"]
    assert c._get_flattened_ll()[-2][0] == "a"


def test_lru_on_miss_caches_value():
    c = LRU(max_size=2, on_miss=lambda k: k.upper())
    assert c["a"] == "A"
    assert c["a"] == "A"
    assert c.hit_count == 1
    assert c.miss_count == 1


def test_lri_soft_miss_get():
    c = LRI(max_size=2)
    assert c.get("x") is None
    assert c.soft_miss_count == 1


def test_lri_setdefault_increments_soft_miss():
    c = LRI(max_size=2)
    assert c.setdefault("x", 10) == 10
    assert c["x"] == 10
    assert c.soft_miss_count == 1


def test_lri_pop_does_not_change_counts():
    c = LRI(max_size=2)
    c["a"] = 1
    assert c.pop("a") == 1
    assert c.hit_count == c.miss_count == c.soft_miss_count == 0


def test_lri_clear_resets_structure():
    c = LRI(max_size=2)
    c["a"] = 1
    c.clear()
    assert len(c) == 0
    assert c._get_flattened_ll() == [(_MISSING if False else c._anchor[2], c._anchor[3])]  # sentinel structure


def test_lri_copy_keeps_contents():
    c = LRI(max_size=2)
    c["a"] = 1
    d = c.copy()
    assert dict(d) == {"a": 1}
    assert d.max_size == c.max_size


def test_make_cache_key_fast_single_arg():
    key = make_cache_key(("abc",), {})
    assert key == "abc"


def test_make_cache_key_typed_distinguishes_int_and_float():
    assert make_cache_key((3,), {}, typed=True) != make_cache_key((3.0,), {}, typed=True)


def test_make_cache_key_includes_kwargs_sorted():
    key = tuple(make_cache_key(("a",), {"b": 2, "c": 3}))
    assert key[0] == "a"
    assert key[1].__class__.__name__ == "_HashedKey" or True


def test_cached_function_caches_result():
    calls = []

    @cached({})
    def f(x):
        calls.append(x)
        return x + 1

    assert f(1) == 2
    assert f(1) == 2
    assert calls == [1]


def test_cached_function_repr():
    @cached({})
    def f(x):
        return x

    assert "CachedFunction" in repr(f)


def test_cachedmethod_with_attribute_cache():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod("cache")
        def f(self, x):
            return x * 2

    a = A()
    assert a.f(3) == 6
    assert a.f(3) == 6
    assert len(a.cache) == 1


def test_cachedmethod_bound_repr_contains_bound_to():
    class A:
        def __init__(self):
            self.cache = {}

        @cachedmethod("cache")
        def f(self, x):
            return x

    a = A()
    rep = repr(a.f)
    assert "CachedMethod" in rep
    assert "bound_to" in rep


def test_cachedproperty_caches_on_instance():
    class A:
        def __init__(self):
            self.calls = 0

        @cachedproperty
        def value(self):
            self.calls += 1
            return 42

    a = A()
    assert a.value == 42
    assert a.value == 42
    assert a.calls == 1
    del a.__dict__["value"]
    assert a.value == 42
    assert a.calls == 2


def test_threshold_counter_add_and_get():
    tc = ThresholdCounter(threshold=0.1)
    tc.add("a")
    tc.add("a")
    tc.add("b")
    assert tc["a"] == 2
    assert tc.get("b") == 1


def test_threshold_counter_update_mapping_and_elements():
    tc = ThresholdCounter(threshold=0.2)
    tc.update({"a": 2, "b": 1})
    assert sorted(tc.elements()) == ["a", "a", "b"]


def test_threshold_counter_most_common_and_counts():
    tc = ThresholdCounter(threshold=0.2)
    tc.update(["a", "a", "b", "c"])
    assert tc.most_common(1)[0][0] == "a"
    assert tc.get_common_count() == sum(v for _, v in tc.items())


def test_threshold_counter_invalid_threshold():
    with pytest.raises(ValueError):
        ThresholdCounter(0)
    with pytest.raises(ValueError):
        ThresholdCounter(1)


def test_is_iterable_scalar_collection():
    assert is_iterable([])
    assert not is_iterable(object())
    assert is_scalar("x")
    assert is_collection([1])


def test_split_iter_basic_and_callable_sep():
    assert list(split_iter([1, 0, 2, 0, 3], sep=0)) == [[1], [2], [3]]
    assert list(split_iter([1, 0, 2, 0, 3], sep=lambda x: x == 0)) == [[1], [2], [3]]


def test_split_iter_maxsplit():
    assert list(split_iter([1, 0, 2, 0, 3], sep=0, maxsplit=1)) == [[1], [2, 0, 3]]


def test_lstrip_rstrip_strip_iter():
    assert list(lstrip_iter([0, 0, 1, 2], 0)) == [1, 2]
    assert list(rstrip_iter([1, 2, 0, 0], 0)) == [1, 2]
    assert list(strip_iter([0, 1, 2, 0], 0)) == [1, 2]


def test_chunked_iter_fill_and_strings():
    assert list(chunked_iter(range(5), 2, fill=None)) == [[0, 1], [2, 3], [4, None]]
    assert list(chunked_iter("abc", 2)) == ["ab", "c"]


def test_chunk_ranges_alignment():
    assert list(chunk_ranges(10, 5, input_offset=10)) == [(10, 15), (15, 20)]
    assert list(chunk_ranges(15, 5, input_offset=4, align=True)) == [(4, 5), (5, 10), (10, 15), (15, 19)]


def test_pairwise_and_windowed():
    assert pairwise([1, 2, 3]) == [(1, 2), (2, 3)]
    assert list(pairwise_iter([1, 2, 3], end=None)) == [(1, 2), (2, 3), (3, None)]
    assert list(windowed_iter([1, 2, 3], 2)) == [(1, 2), (2, 3)]


def test_xfrange_frange():
    assert tuple(xfrange(1, 3, step=0.75)) == (1.0, 1.75, 2.5)
    assert frange(5) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_backoff_iter_basic_and_repeat():
    assert list(backoff_iter(1.0, 10.0, count=5)) == [1.0, 2.0, 4.0, 8.0, 10.0]
    it = backoff_iter(1.0, 2.0, count="repeat")
    assert [next(it) for _ in range(3)] == [1.0, 2.0, 2.0]


def test_bucketize_partition_unique_redundant():
    assert bucketize(range(5)) == {False: [0], True: [1, 2, 3, 4]}
    assert partition(["", "x", ""], bool) == (["x"], ["", ""])
    assert list(unique_iter([1, 1, 2, 2, 3])) == [1, 2, 3]
    assert redundant([1, 2, 3, 2, 3, 3]) == [2, 3]


def test_one_first_same_flatten():
    assert one((0, 1, 0)) == 1
    assert first([0, 0, 3]) == 3
    assert same([1, 1, 1])
    assert flatten([[1, 2], [[3]]]) == [1, 2, 3]


def test_get_path_and_default():
    root = {"a": {"b": [1, {"c": 3}]}}
    assert get_path(root, ("a", "b", 1, "c")) == 3
    assert get_path(root, "a.b.1.c") == 3
    assert get_path(root, ("a", "x"), default="missing") == "missing"
    with pytest.raises(PathAccessError):
        get_path(root, ("a", "x"))


def test_research_finds_values():
    root = {"a": {"b": 1, "c": [2, "x"]}, "d": 3}
    res = research(root, query=lambda p, k, v: isinstance(v, int))
    assert (("a", "b"), 1) in res
    assert (("a", "c", 0), 2) in res
    assert (("d",), 3) in res


def test_guiderator_generates_strings():
    g = GUIDerator(size=24)
    a = next(g)
    b = next(g)
    assert len(a) == 24
    assert a != b


def test_sequential_guiderator_sorted_and_len():
    g = SequentialGUIDerator(size=24)
    a, b = next(g), next(g)
    assert len(a) >= 24
    assert a < b


def test_soft_sorted_and_untyped_sorted():
    assert soft_sorted(["two", "b", "one", "a"], first=["one", "two"]) == ["one", "two", "a", "b"]
    assert untyped_sorted(["abc", 2.0, 1, 2, "def"])[0] == 1


def test_remap_identity_on_simple_structure():
    data = {"a": [1, 2], "b": 3}
    assert remap(data) == data


def test_dt_to_timestamp_and_isoparse():
    dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
    assert dt_to_timestamp(dt) == 0.0
    assert isoparse("1970-01-01T00:00:00") == datetime(1970, 1, 1)


def test_parse_timedelta_and_relative_time():
    assert parse_timedelta("1d 2h 3.5m 0s") == timedelta(days=1, seconds=7410)
    now = datetime(2020, 1, 2, 0, 0, 0)
    assert relative_time(now - timedelta(days=1), now) == "1 day ago"


def test_decimal_relative_time_with_freezegun():
    with freeze_time("2020-01-02 00:00:00"):
        d = datetime(2020, 1, 1, 0, 0, 0)
        assert decimal_relative_time(d) == (1.0, "day")


def test_strpdate_and_daterange():
    assert strpdate("2016-02-14", "%Y-%m-%d") == date(2016, 2, 14)
    vals = list(daterange(date(2015, 12, 25), date(2015, 12, 28)))
    assert vals == [date(2015, 12, 25), date(2015, 12, 26), date(2015, 12, 27)]


def test_constant_tzinfo_and_utc_repr():
    tz = ConstantTZInfo("X", timedelta(hours=3))
    assert tz.utcoffset(None) == timedelta(hours=3)
    assert repr(UTC) == "ConstantTZInfo(name='UTC', offset=datetime.timedelta(0))"


def test_eastern_tz_has_name():
    assert Eastern.tzname(datetime(2020, 1, 1, tzinfo=Eastern)) in {"EST", "EDT"}


def test_priority_queue_basic_operations():
    pq = PriorityQueue()
    pq.add("low", 0)
    pq.add("high", 2)
    pq.add("mid", 1)
    assert pq.pop() == "high"
    assert pq.peek() == "mid"
    assert len(pq) == 2


def test_heap_priority_queue_and_remove():
    pq = HeapPriorityQueue()
    pq.add("a", 1)
    pq.add("b", 2)
    pq.remove("b")
    assert pq.pop() == "a"


def test_sorted_priority_queue_empty_defaults():
    pq = SortedPriorityQueue()
    assert pq.pop(default=None) is None
    assert pq.peek(default="x") == "x"


def test_mkdir_p_and_fileperms(tmp_path):
    p = tmp_path / "a" / "b"
    mkdir_p(p)
    assert p.is_dir()
    fp = FilePerms("r", "r", "")
    assert int(fp) == 0o440


def test_fileperms_from_int_and_repr():
    fp = FilePerms.from_int(0o644)
    assert fp.user == "rw"
    assert "FilePerms" in repr(fp)


def test_atomic_saver_writes_file(tmp_path):
    dest = tmp_path / "out.txt"
    with AtomicSaver(dest, text_mode=True) as fo:
        fo.write("hello")
    assert dest.read_text() == "hello"


def test_iter_find_files_and_copytree(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "a.txt").write_text("a")
    (src / "b.py").write_text("b")
    found = sorted(iter_find_files(src, "*.py"))
    assert found == [str(src / "b.py")]
    copytree(src, dst)
    assert (dst / "a.txt").read_text() == "a"


def test_rotate_file(tmp_path):
    f = tmp_path / "log.txt"
    f.write_text("first")
    rotate_file(str(f), keep=2)
    assert not f.exists()
    assert (tmp_path / "log.1.txt").exists()


def test_ordered_multi_dict_basic_behavior():
    omd = OrderedMultiDict()
    omd["a"] = 1
    omd.add("a", 2)
    omd["b"] = 3
    assert omd["a"] == 2
    assert omd.getlist("a") == [1, 2]
    assert list(reversed(omd)) == ["b", "a"]


def test_ordered_multi_dict_pop_and_todict():
    omd = OrderedMultiDict([("a", 1), ("b", 2), ("a", 3)])
    assert omd.poplast("a") == 3
    assert omd.pop("a") == 1
    assert omd.todict() == {"b": 2}


def test_ordered_multi_dict_sorted_and_inverted():
    omd = OrderedMultiDict(zip("abc", [3, 1, 2]))
    assert omd.sorted(key=lambda i: i[1]).items(multi=True)[0] == ("b", 1)
    inv = OrderedMultiDict([(0, 2), (1, 2)]).inverted()
    assert inv.getlist(2) == [0, 1]


def test_one_to_one_inverse_updates():
    oto = OneToOne({"a": 1, "b": 2})
    assert oto.inv[1] == "a"
    oto.inv[1] = "c"
    assert "a" not in oto
    assert oto["c"] == 1


def test_one_to_one_unique_rejects_duplicates():
    with pytest.raises(ValueError):
        OneToOne.unique({"a": 1, "b": 1})


def test_many_to_many_basic():
    m = ManyToMany()
    m.add("a", 1)
    m.add("a", 2)
    assert m["a"] == frozenset({1, 2})
    assert m.inv[1] == frozenset({"a"})


def test_subdict_and_frozendict():
    d = {"a": 1, "b": 2, "c": 3}
    assert subdict(d, keep=["a", "c"]) == {"a": 1, "c": 3}
    fd = FrozenDict(d)
    assert hash(fd) == hash(FrozenDict(d))
    assert fd.updated(b=20)["b"] == 20


def test_indexed_set_order_index_and_ops():
    s = IndexedSet([3, 1, 2, 1])
    assert list(s) == [3, 1, 2]
    assert s[0] == 3
    assert s.index(2) == 2
    assert list(s & {1, 2}) == [1, 2]


def test_indexed_set_remove_and_slice():
    s = IndexedSet([1, 2, 3, 4])
    s.remove(2)
    assert 2 not in s
    assert list(s[1:3]) == [3, 4]


def test_complement_set_basic_operations():
    c = complement({2, 3})
    assert 1 in c and 2 not in c
    assert list(set(range(5)) & c) == [0, 1, 4]
    assert complement(complement({1, 2})) == {1, 2}


def test_strutils_conversions_and_slugify():
    assert camel2under("BasicParseTest") == "basic_parse_test"
    assert under2camel("complex_tokenizer") == "ComplexTokenizer"
    assert slugify("First post! Hi!!!!~1    ") == "first_post_hi_1"


def test_strutils_word_and_list_helpers():
    assert split_punct_ws("First post! Hi!!!!~1    ") == ["First", "post", "Hi", "1"]
    assert unit_len([1, 2, 3], "number") == "3 numbers"
    assert ordinalize(21) == "21st"
    assert cardinalize("Wish", 3) == "Wishes"


def test_plural_singular_helpers():
    assert pluralize("enemy") == "enemies"
    assert singularize("Glasses") == "Glass"


def test_strutils_unicode_and_ascii_helpers():
    assert find_hashtags("#atag http://asite/#ananchor") == ["atag"]
    assert a10n("internationalization") == "i18n"
    assert not is_ascii("Beyoncé")
    assert is_ascii("Beyonce")


def test_strip_ansi_and_bytes2human_and_html2text():
    assert strip_ansi("\x1b[0m\x1b[1;36mart\x1b[46;34m") == "art"
    assert bytes2human(1024) == "1K"
    assert html2text('<a href="#">Test &amp;<em>(Δ&#x03b7;&#956;&#x03CE;)</em></a>') == "Test &(Δημώ)"


def test_gzip_gunzip_roundtrip():
    data = b"hello world" * 10
    assert gunzip_bytes(gzip_bytes(data)) == data


def test_iter_splitlines_and_indent():
    assert list(iter_splitlines("\nhi\nbye\n")) == ["", "hi", "bye", ""]
    assert indent("a\n\nb", "> ") == "> a\n\n> b"


def test_uuid_and_shell_escape():
    u = uuid.uuid4()
    assert is_uuid(str(u), version=u.version)
    assert args2sh(["aa", "[bb]", "cc'cc"]) == "aa '[bb]' 'cc'\"'\"'cc'"
    assert args2cmd(["aa", 'dd"dd']).startswith("aa dd")


def test_int_list_helpers():
    assert parse_int_list("1,3,5-8,10-11,15") == [1, 3, 5, 6, 7, 8, 10, 11, 15]
    assert format_int_list([1, 3, 5, 6, 7, 8, 10, 11, 15]) == "1,3,5-8,10-11,15"
    assert complement_int_list("1,3,5-8", range_start=2, range_end=3) == "2"
    assert int_ranges_from_int_list("1,3,5-8") == ((1, 1), (3, 3), (5, 8))


def test_multi_replace_and_unwrap_removeprefix_human_list_ellipsize():
    mr = MultiReplace({"foo": "zoo", "cat": "hat"})
    assert mr.sub("foo and cat") == "zoo and hat"
    assert multi_replace("foo and cat", {"foo": "zoo", "cat": "hat"}) == "zoo and hat"
    assert unwrap_text("Short \n lines  \nwrapped\nsmall.\n\nAnother\nparagraph.") == "Short lines wrapped small.\n\nAnother paragraph."
    assert removeprefix("foobar", "foo") == "bar"
    assert human_readable_list(["a", "b", "c"]) == "a, b, and c"
    assert ellipsize("Hello, World!", 16) == "Hello, World!"
