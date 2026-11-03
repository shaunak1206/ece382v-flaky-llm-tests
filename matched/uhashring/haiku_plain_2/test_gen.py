import hashlib

import pytest

from uhashring import HashRing
from uhashring.ring_ketama import KetamaRing


def test_empty_ring_lookups_return_none():
    ring = HashRing()
    assert ring.size == 0
    assert list(ring.get_nodes()) == []
    assert ring.get("key") is None
    assert ring.get_node("key") is None
    assert ring.get_node_pos("key") is None
    assert ring.get_server("key") is None
    assert ring["key"] is None
    assert ring.get_points() == []
    assert list(ring.iterate_nodes("key")) == [None]


def test_string_node_is_single_node_ring():
    ring = HashRing("server-1", vnodes=5)
    assert list(ring.get_nodes()) == ["server-1"]
    assert ring.size == 5
    assert ring.get_node("anything") == "server-1"


def test_list_nodes_get_vnodes_each():
    ring = HashRing(["a", "b", "c"], vnodes=10)
    assert sorted(ring.get_nodes()) == ["a", "b", "c"]
    assert ring.size == 30
    assert dict(ring.distribution) == {"a": 10, "b": 10, "c": 10}


def test_default_vnode_counts():
    meta = HashRing(["a"])
    assert meta.distribution["a"] == 160
    assert meta.size == 160

    ketama = HashRing(["a"], hash_fn="ketama")
    assert isinstance(ketama.runtime, KetamaRing)
    # default 40 vnodes, each expanded into 4 replica hashes
    assert ketama.distribution["a"] == 160


def test_dict_config_sets_weight_and_vnodes():
    ring = HashRing({"a": 3, "b": {"vnodes": 4, "weight": 2}}, vnodes=10)
    assert ring.conf["a"]["weight"] == 3
    assert ring.conf["a"]["vnodes"] == 10
    assert ring.conf["b"]["weight"] == 2
    assert ring.conf["b"]["vnodes"] == 4
    assert ring.distribution["a"] == 30
    assert ring.distribution["b"] == 8
    assert ring.size == 38


def test_invalid_node_configuration_raises_value_error():
    with pytest.raises(ValueError):
        HashRing({"a": "heavy"})
    with pytest.raises(ValueError):
        HashRing(42)
    with pytest.raises(ValueError):
        HashRing(("a", "b"))


def test_non_callable_weight_fn_raises_type_error():
    with pytest.raises(TypeError):
        HashRing(["a"], weight_fn="double")


def test_non_callable_hash_fn_raises_type_error():
    with pytest.raises(TypeError):
        HashRing(["a"], hash_fn="md5")


def test_weight_fn_overrides_configured_weight():
    ring = HashRing(
        {"big": 1, "small": 1},
        vnodes=4,
        weight_fn=lambda nodename, **kw: 3 if nodename == "big" else 1,
    )
    assert ring.conf["big"]["weight"] == 3
    assert ring.conf["small"]["weight"] == 1
    assert ring.distribution["big"] == 12
    assert ring.distribution["small"] == 4


def test_lookup_picks_next_point_and_wraps():
    table = {"a-0": 100, "b-0": 200, "k_mid": 150, "k_after": 250, "k_before": 50}
    ring = HashRing(["a", "b"], vnodes=1, hash_fn=lambda key: table[key])
    # 150 falls between 100 and 200, so the next point clockwise is b
    assert ring.get_node("k_mid") == "b"
    assert ring.get_node_pos("k_mid") == 1
    # 250 is past the last point (200), so it wraps to the first point (a)
    assert ring.get_node("k_after") == "a"
    assert ring.get_node_pos("k_after") == 0
    # 50 is before the first point, so it lands on a
    assert ring.get_node("k_before") == "a"


def test_get_points_and_get_server_report_ring_positions():
    table = {"a-0": 100, "b-0": 200, "k_mid": 150}
    ring = HashRing(["a", "b"], vnodes=1, hash_fn=lambda key: table[key])
    assert ring.get_points() == [(100, "a"), (200, "b")]
    assert ring.get_key("k_mid") == 150
    assert ring.get_server("k_mid") == (200, "b")


def test_default_hash_is_md5_of_str_key():
    ring = HashRing(["a"], vnodes=1)
    expected = int(hashlib.md5("foo".encode("utf-8")).hexdigest(), 16)
    assert ring.get_key("foo") == expected


def test_ketama_hash_matches_reference_formula():
    ring = HashRing(["a"], hash_fn="ketama")
    digest = hashlib.md5("foo".encode("utf-8")).digest()
    expected = (digest[3] << 24) | (digest[2] << 16) | (digest[1] << 8) | digest[0]
    assert ring.get_key("foo") == expected


def test_lookup_independent_of_node_order():
    nodes = ["cache-1", "cache-2", "cache-3"]
    forward = HashRing(nodes, vnodes=20)
    backward = HashRing(list(reversed(nodes)), vnodes=20)
    keys = ["user:{}".format(i) for i in range(200)]
    assert [forward.get_node(k) for k in keys] == [backward.get_node(k) for k in keys]
    assert set(forward.get_node(k) for k in keys) == set(nodes)


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(["a", "b", "c"], vnodes=50)
    keys = ["key-{}".format(i) for i in range(500)]
    owners_before = {k: ring.get_node(k) for k in keys}

    ring.add_node("d")

    moved = 0
    for k in keys:
        now = ring.get_node(k)
        if now != owners_before[k]:
            assert now == "d"
            moved += 1
    assert moved > 0


def test_remove_node_only_reassigns_removed_keys():
    ring = HashRing(["a", "b", "c"], vnodes=50)
    keys = ["key-{}".format(i) for i in range(500)]
    owners_before = {k: ring.get_node(k) for k in keys}

    del ring["b"]

    assert "b" not in ring.get_nodes()
    assert ring.size == 100
    for k in keys:
        if owners_before[k] == "b":
            assert ring.get_node(k) in ("a", "c")
        else:
            assert ring.get_node(k) == owners_before[k]


def test_remove_node_alias_and_unknown_node_error():
    ring = HashRing(["a", "b"], vnodes=5)
    ring.remove_node("a")
    assert list(ring.get_nodes()) == ["b"]
    assert ring.get_node("anything") == "b"
    with pytest.raises(KeyError):
        ring.remove_node("zzz")


def test_removing_last_node_empties_ring():
    ring = HashRing(["solo"], vnodes=3)
    del ring["solo"]
    assert ring.size == 0
    assert ring.get("k") is None


def test_setitem_adds_weighted_node_and_readd_is_noop():
    ring = HashRing(["a"], vnodes=10)
    ring["b"] = {"weight": 2}
    assert ring.distribution["b"] == 20
    assert ring.size == 30

    ring.add_node("a")
    assert ring.size == 30
    assert ring.conf["a"]["weight"] == 1


def test_range_yields_distinct_nodes_clockwise():
    ring = HashRing(["a", "b", "c", "d"], vnodes=20)
    result = [node["nodename"] for node in ring.range("some-key")]
    assert sorted(result) == ["a", "b", "c", "d"]
    assert len(result) == len(set(result))

    first_two = [node["nodename"] for node in ring.range("some-key", size=2)]
    assert first_two == result[:2]

    assert list(ring.iterate_nodes("some-key")) == result


def test_range_without_uniqueness_yields_every_point():
    ring = HashRing(["a", "b"], vnodes=10)
    assert len(list(ring.range("k", unique=False))) == ring.size


def test_regenerate_preserves_mapping():
    ring = HashRing(["a", "b"], vnodes=20)
    keys = ["item-{}".format(i) for i in range(50)]
    before = [ring.get_node(k) for k in keys]
    ring.regenerate()
    assert [ring.get_node(k) for k in keys] == before
    assert ring.size == 40


def test_ketama_weighted_distribution():
    ring = HashRing({"a": 1, "b": 3}, hash_fn="ketama", vnodes=10)
    # weight_sum is 4; a gets 5 slots, b gets 15, each slot has 4 replicas
    assert ring.distribution["a"] == 20
    assert ring.distribution["b"] == 60


def test_dict_config_hostname_port_and_instance():
    single = HashRing(
        {"a": {"hostname": "10.0.0.1", "port": 11211, "instance": "client-a"}},
        vnodes=5,
    )
    assert single.get_node_hostname("any-key") == "10.0.0.1"
    assert single.get_node_port("any-key") == 11211
    assert single["any-key"] == "client-a"
    assert single.get_instances() == ["client-a"]

    plain = HashRing(["a"], vnodes=5)
    assert plain.get_node_hostname("any-key") == "a"
    assert plain.get_node_port("any-key") is None
