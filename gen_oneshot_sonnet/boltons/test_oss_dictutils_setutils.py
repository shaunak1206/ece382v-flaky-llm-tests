"""Tests for boltons.dictutils and boltons.setutils."""

import copy
import json
import pickle
from collections import OrderedDict

import pytest

from boltons.dictutils import OMD, FrozenDict, OneToOne, OrderedMultiDict, subdict
from boltons.setutils import IndexedSet, complement

# ---------------------------------------------------------------------------
# OrderedMultiDict
# ---------------------------------------------------------------------------


def test_omd_docstring_walkthrough():
    omd = OrderedMultiDict()
    omd["a"] = 1
    omd["b"] = 2
    omd.add("a", 3)
    assert omd.get("a") == 3
    assert omd.getlist("a") == [1, 3]
    assert list(reversed(omd)) == ["b", "a"]
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2), ('a', 3)])"
    assert omd.poplast("a") == 3
    assert repr(omd) == "OrderedMultiDict([('a', 1), ('b', 2)])"
    assert omd.pop("a") == 1
    assert repr(omd) == "OrderedMultiDict([('b', 2)])"


def test_omd_views_lookup_and_counts():
    omd = OrderedMultiDict([("a", 1), ("b", 2), ("a", 3)])
    assert len(omd) == 2
    assert "a" in omd
    assert omd["a"] == 3
    with pytest.raises(KeyError):
        omd["zzz"]
    assert omd.get("zzz") is None
    assert omd.get("zzz", "d") == "d"
    assert omd.getlist("zzz") == []
    assert omd.getlist("zzz", "dflt") == "dflt"

    assert omd.todict() == {"a": 3, "b": 2}
    assert omd.todict(multi=True) == {"a": [1, 3], "b": [2]}
    assert omd.items() == [("a", 3), ("b", 2)]
    assert omd.items(multi=True) == [("a", 1), ("b", 2), ("a", 3)]
    assert omd.keys() == ["a", "b"]
    assert omd.keys(multi=True) == ["a", "b", "a"]
    assert omd.values() == [3, 2]
    assert omd.values(multi=True) == [1, 2, 3]
    assert omd.counts().todict() == {"a": 2, "b": 1}


def test_omd_update_update_extend_addlist_and_constructors():
    omd = OMD([("a", 1), ("b", 2)])
    omd.update({"a": 9})
    assert omd.getlist("a") == [9]
    omd.update_extend({"a": 10})
    assert omd.getlist("a") == [9, 10]
    omd.update([("b", 5)])
    assert omd.getlist("b") == [5]
    omd.update(OMD([("a", 1), ("a", 2)]))
    assert omd.getlist("a") == [1, 2]

    assert omd.setdefault("z", 5) == 5
    assert omd.setdefault("z", 6) == 5
    assert OMD.fromkeys(["x", "y"], 0).items() == [("x", 0), ("y", 0)]

    fresh = OMD([("a", -1)])
    fresh.addlist("a", range(3))
    assert repr(fresh) == "OrderedMultiDict([('a', -1), ('a', 0), ('a', 1), ('a', 2)])"
    fresh.addlist("b", iter([7, 8]))
    assert fresh.getlist("b") == [7, 8]
    fresh.addlist("c", [])
    assert "c" not in fresh

    with pytest.raises(TypeError):
        OMD([], [])


def test_omd_inverted_sorted_and_sortedvalues():
    omd = OMD([(0, 2), (1, 2)])
    assert omd.inverted().getlist(2) == [0, 1]
    assert omd.inverted().inverted() == omd

    numbers = OrderedMultiDict(zip(range(3), range(3)))
    assert repr(numbers.sorted(reverse=True)) == "OrderedMultiDict([(2, 2), (1, 1), (0, 0)])"
    letters = OrderedMultiDict(zip("hello", "world"))
    assert repr(letters.sorted(key=lambda i: i[1])) == (
        "OrderedMultiDict([('o', 'd'), ('l', 'l'), ('e', 'o'), ('l', 'r'), ('h', 'w')])"
    )

    multi = OrderedMultiDict()
    multi.addlist("even", [6, 2])
    multi.addlist("odd", [1, 5])
    multi.add("even", 4)
    multi.add("odd", 3)
    somd = multi.sortedvalues()
    assert somd.getlist("even") == [2, 4, 6]
    assert somd.keys(multi=True) == multi.keys(multi=True)
    assert multi != somd
    assert repr(somd) == (
        "OrderedMultiDict([('even', 2), ('even', 4), ('odd', 1), ('odd', 3), ('even', 6), ('odd', 5)])"
    )


def test_omd_pop_family():
    omd = OMD([("a", 1), ("b", 2), ("a", 3)])
    assert omd.popall("a") == [1, 3]
    assert "a" not in omd
    assert omd.pop("b") == 2
    with pytest.raises(KeyError):
        omd.pop("b")
    assert omd.pop("b", "dflt") == "dflt"
    with pytest.raises(KeyError):
        omd.popall("nope")
    assert omd.popall("nope", []) == []

    omd = OMD([("a", 1), ("b", 2), ("a", 3)])
    assert omd.poplast() == 3  # most recently inserted value
    assert omd.getlist("a") == [1]
    assert omd.poplast("b") == 2
    assert "b" not in omd
    assert omd.poplast("a") == 1
    with pytest.raises(KeyError):
        omd.poplast()
    assert omd.poplast(default="empty") == "empty"
    with pytest.raises(KeyError):
        omd.poplast("missing")


def test_omd_copy_equality_pickle_and_delete():
    omd = OMD([("a", 1), ("b", 2), ("a", 3)])
    dup = omd.copy()
    assert dup == omd
    assert dup is not omd
    assert dup.items(multi=True) == omd.items(multi=True)
    dup.add("c", 4)
    assert dup != omd
    assert "c" not in omd
    assert omd == {"a": 3, "b": 2}
    assert omd != {"a": 1, "b": 2}

    restored = pickle.loads(pickle.dumps(omd))
    assert type(restored) is OrderedMultiDict
    assert restored.items(multi=True) == [("a", 1), ("b", 2), ("a", 3)]

    del omd["a"]
    assert omd.items(multi=True) == [("b", 2)]
    omd.clear()
    assert len(omd) == 0
    assert omd.items(multi=True) == []


# ---------------------------------------------------------------------------
# OneToOne / FrozenDict / subdict
# ---------------------------------------------------------------------------


def test_one_to_one_reverse_mapping_and_unique_constructor():
    oto = OneToOne({"a": 1, "b": 2})
    assert oto["a"] == 1
    assert oto.inv[1] == "a"
    assert len(oto) == 2
    oto.inv[1] = "c"
    assert oto.get("a") is None
    assert oto["c"] == 1
    assert len(oto) == 2

    assert repr(OneToOne({"a": 1})) == "OneToOne({'a': 1})"
    other = OneToOne({"a": 1})
    other["a"] = 2
    assert dict(other.inv) == {2: "a"}

    with pytest.raises(ValueError):
        OneToOne.unique({"a": 1, "b": 1})
    with pytest.raises(ValueError):
        OneToOne.unique({"a": 2}, b=2)
    assert dict(OneToOne.unique({"a": 1, "b": 2})) == {"a": 1, "b": 2}

    collapsed = OneToOne({"a": 1, "b": 1})
    assert dict(collapsed) == {"b": 1}
    assert collapsed.inv[1] == "b"


def test_one_to_one_mutators_keep_both_directions_in_sync():
    oto = OneToOne({"a": 1, "b": 2})
    assert oto.pop("a") == 1
    assert 1 not in oto.inv
    assert oto.pop("zz", "d") == "d"
    with pytest.raises(KeyError):
        oto.pop("zz")

    oto.update({"x": 5})
    oto.update([("y", 6)], z=7)
    assert oto.inv == {2: "b", 5: "x", 6: "y", 7: "z"}
    del oto["x"]
    assert 5 not in oto.inv
    assert oto.popitem() == ("z", 7)
    assert 7 not in oto.inv

    dup = oto.copy()
    dup["q"] = 99
    assert "q" not in oto
    assert dup.inv[99] == "q"
    assert oto.setdefault("m", 11) == 11
    assert oto.setdefault("m", 12) == 11
    with pytest.raises(TypeError):
        oto["k"] = [1]
    oto.clear()
    assert len(oto) == 0
    assert len(oto.inv) == 0


def test_frozen_dict_is_immutable_hashable_and_serializable():
    fd = FrozenDict({"a": 1, "b": 2})
    assert fd["a"] == 1
    assert len(fd) == 2
    assert hash(fd) == hash(FrozenDict({"b": 2, "a": 1}))
    assert {fd: "frozen"}[FrozenDict(a=1, b=2)] == "frozen"
    assert len({fd, FrozenDict({"a": 1, "b": 2})}) == 1

    mutators = [
        lambda: fd.__setitem__("c", 3),
        lambda: fd.__delitem__("a"),
        lambda: fd.update(c=3),
        lambda: fd.pop("a"),
        lambda: fd.popitem(),
        lambda: fd.clear(),
        lambda: fd.setdefault("c", 3),
    ]
    for mutate in mutators:
        with pytest.raises(TypeError):
            mutate()

    bigger = fd.updated(c=3)
    assert isinstance(bigger, FrozenDict)
    assert dict(bigger) == {"a": 1, "b": 2, "c": 3}
    assert dict(fd) == {"a": 1, "b": 2}
    assert repr(FrozenDict({"a": 1})) == "FrozenDict({'a': 1})"
    assert FrozenDict.fromkeys(["x", "y"], 0) == {"x": 0, "y": 0}

    assert copy.copy(fd) is fd
    roundtrip = pickle.loads(pickle.dumps(fd))
    assert type(roundtrip) is FrozenDict
    assert roundtrip == fd
    assert json.dumps(FrozenDict({"a": 1})) == '{"a": 1}'
    with pytest.raises(TypeError):
        hash(FrozenDict({"a": []}))


def test_subdict():
    assert subdict({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert subdict({"a": 1, "b": 2, "c": 3}, drop=["b", "c"]) == {"a": 1}
    assert subdict({"a": 1, "b": 2, "c": 3}, keep=["a", "c"]) == {"a": 1, "c": 3}
    assert subdict({"a": 1}, keep=["a", "z"]) == {"a": 1}
    assert subdict({"a": 1, "b": 2}, keep=["a", "b"], drop=["b"]) == {"a": 1}
    ordered = OrderedDict([("a", 1), ("b", 2)])
    assert type(subdict(ordered, keep=["b"])) is OrderedDict


# ---------------------------------------------------------------------------
# IndexedSet / complement
# ---------------------------------------------------------------------------


def test_indexed_set_docstring_examples():
    x = IndexedSet(list(range(4)) + list(range(8)))
    assert repr(x) == "IndexedSet([0, 1, 2, 3, 4, 5, 6, 7])"
    assert repr(x - set(range(2))) == "IndexedSet([2, 3, 4, 5, 6, 7])"
    assert x[-1] == 7
    fcr = IndexedSet("freecreditreport.com")
    assert "".join(fcr[: fcr.index(".")]) == "frecditpo"
    assert repr(fcr & set("cash4gold.com")) == "IndexedSet(['c', 'd', 'o', '.', 'm'])"


def test_indexed_set_add_remove_pop_and_index():
    s = IndexedSet("abcde")
    s.remove("c")
    assert list(s) == ["a", "b", "d", "e"]
    assert len(s) == 4
    assert "c" not in s
    assert s.index("d") == 2
    assert s[2] == "d"

    s.add("f")
    s.add("a")  # duplicates are ignored
    assert list(s) == ["a", "b", "d", "e", "f"]
    assert s.pop() == "f"
    assert s.pop(0) == "a"
    assert list(s) == ["b", "d", "e"]

    s.discard("zzz")
    with pytest.raises(KeyError):
        s.remove("zzz")
    with pytest.raises(ValueError):
        s.index("zzz")
    with pytest.raises(IndexError):
        s[10]
    assert s.count("b") == 1
    assert s.count("q") == 0
    s.clear()
    assert len(s) == 0
    assert list(s) == []


def test_indexed_set_set_operations_preserve_order():
    a = IndexedSet([1, 2, 3, 4])
    b = IndexedSet([3, 4, 5])
    assert list(a | b) == [1, 2, 3, 4, 5]
    assert list(a & b) == [3, 4]
    assert list(a - b) == [1, 2]
    assert list(a ^ b) == [1, 2, 5]
    assert list(a.union([9, 8])) == [1, 2, 3, 4, 9, 8]
    assert list(a.intersection([2, 4, 6], [4, 2])) == [2, 4]
    assert a.isdisjoint([7, 8])
    assert not a.isdisjoint(b)
    assert IndexedSet([1, 2]).issubset(a)
    assert a.issuperset(IndexedSet([1, 2]))
    assert IndexedSet([1, 2]) <= a
    assert IndexedSet([1, 2]) == {2, 1}
    assert IndexedSet([1, 2]) == IndexedSet([1, 2])
    assert IndexedSet([1, 2]) != IndexedSet([2, 1])

    c = IndexedSet([1, 2, 3])
    c |= [4]
    assert list(c) == [1, 2, 3, 4]
    c -= [1]
    assert list(c) == [2, 3, 4]
    c &= [3, 4, 5]
    assert list(c) == [3, 4]
    c ^= [4, 9]
    assert list(c) == [3, 9]


def test_indexed_set_sort_reverse_and_slicing():
    s = IndexedSet([3, 1, 2])
    s.sort()
    assert list(s) == [1, 2, 3]
    s.reverse()
    assert list(s) == [3, 2, 1]
    assert s.index(1) == 2
    assert repr(s[0:2]) == "IndexedSet([3, 2])"
    assert list(s[::-1]) == [1, 2, 3]
    assert list(s[1:]) == [2, 1]
    assert list(reversed(s)) == [1, 2, 3]
    assert IndexedSet.from_iterable("abca") == IndexedSet("abc")


def test_complement_sets():
    assert sorted(set(range(5)) & complement({2, 3})) == [0, 1, 4]
    s = set(range(5))
    assert complement(complement(s)) == s

    c = complement({1, 2})
    assert 5 in c
    assert 1 not in c
    assert sorted(c & {2, 3}) == [3]
    union = c | {1}
    assert 1 in union
    assert 2 not in union
    difference = c - {7}
    assert 7 not in difference
    assert 8 in difference
    assert bool(c)

    with pytest.raises(NotImplementedError):
        len(c)
    with pytest.raises(NotImplementedError):
        list(c)
    with pytest.raises(NotImplementedError):
        c.pop()
