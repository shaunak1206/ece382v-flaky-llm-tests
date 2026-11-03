import hashlib
from bisect import bisect_right

import pytest

from uhashring import HashRing

KEYS = ["key-{}".format(i) for i in range(300)]


def test_default_meta_ring_has_160_points_per_node():
    ring = HashRing(["a", "b"])
    assert ring.size == 320
    assert dict(ring.distribution) == {"a": 160, "b": 160}
    assert sorted(ring.get_nodes()) == ["a", "b"]


def test_vnodes_kwarg_controls_points_per_node():
    ring = HashRing(["a", "b", "c"], vnodes=10)
    assert ring.size == 30
    assert dict(ring.distribution) == {"a": 10, "b": 10, "c": 10}


def test_ketama_default_has_160_points_per_equal_weight_node():
    ring = HashRing(["a", "b"], hash_fn="ketama")
    assert dict(ring.distribution) == {"a": 160, "b": 160}


def test_ketama_points_follow_weight_ratio():
    ring = HashRing({"a": 1, "b": 3}, hash_fn="ketama")
    assert dict(ring.distribution) == {"a": 80, "b": 240}


def test_default_hash_is_md5_big_endian_int():
    ring = HashRing()
    expected = int.from_bytes(hashlib.md5(b"foo").digest(), "big")
    assert ring.get_key("foo") == expected


def test_ketama_hash_is_md5_little_endian_word():
    ring = HashRing(hash_fn="ketama")
    expected = int.from_bytes(hashlib.md5(b"foo").digest()[:4], "little")
    assert ring.get_key("foo") == expected


def test_custom_hash_fn_is_used_for_get_key():
    ring = HashRing(hash_fn=lambda key: 42)
    assert ring.get_key("anything") == 42
    assert ring.get_key(None) == 42


def test_non_callable_hash_or_weight_fn_is_rejected():
    with pytest.raises(TypeError):
        HashRing(["a"], hash_fn="md5")
    with pytest.raises(TypeError):
        HashRing(["a"], weight_fn="heavy")


def test_nodes_must_be_str_list_or_dict():
    with pytest.raises(ValueError):
        HashRing(nodes=5)
    with pytest.raises(ValueError):
        HashRing(nodes=("a", "b"))


def test_dict_node_config_must_be_int_or_dict():
    with pytest.raises(ValueError):
        HashRing({"a": "heavy"})


def test_dict_node_config_sets_vnodes_and_weight():
    ring = HashRing({"a": {"vnodes": 10, "weight": 3}})
    assert ring.size == 30
    assert dict(ring.distribution) == {"a": 30}
    assert ring.get_node_weight("x") == 3


def test_int_node_value_sets_weight():
    ring = HashRing({"a": 2, "b": 1})
    assert dict(ring.distribution) == {"a": 320, "b": 160}
    assert ring.size == 480


def test_instance_hostname_and_port_config_round_trip():
    instance = object()
    ring = HashRing(
        {"a": {"hostname": "10.0.0.5", "port": 11211, "instance": instance}}
    )
    assert ring.get_node_hostname("k") == "10.0.0.5"
    assert ring.get_node_port("k") == 11211
    assert ring.get_node_instance("k") is instance
    assert ring["k"] is instance
    assert ring.get_instances() == [instance]


def test_plain_string_nodes_get_default_config():
    ring = HashRing(["a", "b"])
    for key in KEYS[:50]:
        node = ring.get_node(key)
        assert ring.get_node_hostname(key) == node
        assert ring.get_node_port(key) is None
        assert ring[key] is None
        assert ring.get_node_weight(key) == 1
        assert ring.get(key)["nodename"] == node


def test_weight_fn_receives_node_config_as_keywords():
    def weight_fn(**conf):
        return 2 if conf["nodename"] == "big" else 1

    ring = HashRing(["big", "small"], weight_fn=weight_fn)
    assert dict(ring.distribution) == {"big": 320, "small": 160}
    for key in KEYS[:50]:
        expected = 2 if ring.get_node(key) == "big" else 1
        assert ring.get_node_weight(key) == expected


def test_empty_ring_lookups_return_none_and_empty_views():
    ring = HashRing()
    assert ring.size == 0
    assert ring.get_node("x") is None
    assert ring.get("x") is None
    assert ring.get_server("x") is None
    assert ring.get_points() == []
    assert ring.get_instances() == []
    assert list(ring.get_nodes()) == []
    assert list(ring.iterate_nodes("x")) == [None]

    ring["a"] = 1
    del ring["a"]
    assert ring.size == 0
    assert ring.get_node("x") is None


def test_lookup_is_bisect_successor_over_sorted_points():
    ring = HashRing(["a", "b", "c"])
    points = ring.get_points()
    positions = [pos for pos, _ in points]
    assert positions == sorted(positions)
    assert len(points) == ring.size
    for key in KEYS[:100]:
        expected_pos = bisect_right(positions, ring.get_key(key)) % len(positions)
        assert ring.get_node_pos(key) == expected_pos
        assert ring.get_server(key) == points[expected_pos]
        assert ring.get_node(key) == points[expected_pos][1]


def test_node_order_does_not_change_ring():
    first = HashRing(["a", "b", "c"])
    second = HashRing(["c", "a", "b"])
    assert first.get_points() == second.get_points()


def test_adding_node_only_moves_keys_onto_new_node():
    ring = HashRing(["a", "b"])
    before = {key: ring.get_node(key) for key in KEYS}
    ring["c"] = 1
    after = {key: ring.get_node(key) for key in KEYS}
    moved = [key for key in KEYS if before[key] != after[key]]
    assert moved
    assert all(after[key] == "c" for key in moved)
    assert ring.size == 480


def test_removing_node_only_reassigns_its_own_keys():
    ring = HashRing(["a", "b", "c"])
    before = {key: ring.get_node(key) for key in KEYS}
    del ring["b"]
    after = {key: ring.get_node(key) for key in KEYS}
    for key in KEYS:
        if before[key] == "b":
            assert after[key] in ("a", "c")
        else:
            assert after[key] == before[key]
    assert "b" not in ring.get_nodes()
    assert "b" not in dict(ring.distribution)


def test_remove_then_readd_restores_original_mapping():
    ring = HashRing(["a", "b", "c"])
    before = {key: ring.get_node(key) for key in KEYS}
    del ring["b"]
    ring["b"] = 1
    after = {key: ring.get_node(key) for key in KEYS}
    assert after == before
    assert dict(ring.distribution) == {"a": 160, "b": 160, "c": 160}


def test_removing_unknown_node_raises_keyerror():
    ring = HashRing(["a"])
    with pytest.raises(KeyError):
        del ring["missing"]
    with pytest.raises(KeyError):
        ring.remove_node("missing")


def test_iterate_nodes_and_range_yield_unique_nodes():
    ring = HashRing(["a", "b", "c"])
    distinct = list(ring.iterate_nodes("k"))
    assert sorted(distinct) == ["a", "b", "c"]
    assert distinct[0] == ring.get_node("k")

    everything = list(ring.iterate_nodes("k", distinct=False))
    assert len(everything) == ring.size

    two = [conf["nodename"] for conf in ring.range("k", size=2)]
    assert len(two) == 2
    assert len(set(two)) == 2
    assert two[0] == ring.get_node("k")

    capped = [conf["nodename"] for conf in ring.range("k", size=10)]
    assert sorted(capped) == ["a", "b", "c"]


def test_non_string_keys_are_hashed_by_their_str_form():
    ring = HashRing(["a", "b", "c"])
    assert ring.get_key(5) == ring.get_key("5")
    assert ring.get_node(5) == ring.get_node("5")
