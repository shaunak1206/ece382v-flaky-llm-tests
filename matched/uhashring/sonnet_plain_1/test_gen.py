# -*- coding: utf-8 -*-
import zlib

import pytest

from uhashring import HashRing


def crc_hash(key):
    return zlib.crc32(str(key).encode("utf-8"))


def test_empty_ring_returns_none_and_has_no_points():
    ring = HashRing()
    assert ring.size == 0
    assert ring.get_node("anything") is None
    assert ring.get_instances() == []
    assert list(ring.get_nodes()) == []
    assert ring.get_points() == []


def test_default_vnodes_for_meta_ring():
    ring = HashRing(["a", "b"])
    assert ring.distribution["a"] == 160
    assert ring.distribution["b"] == 160
    assert 0 < ring.size <= 320
    assert len(ring.get_points()) == ring.size


def test_ketama_default_vnodes_and_distribution():
    ring = HashRing(["a", "b"], hash_fn="ketama")
    # 40 vnodes * 2 nodes * weight 1 // weight sum 2 = 40 groups, 4 replicas each
    assert ring.distribution["a"] == 160
    assert ring.distribution["b"] == 160
    assert 0 < ring.size <= 320


def test_ketama_weighted_distribution():
    ring = HashRing({"a": 1, "b": 3}, hash_fn="ketama")
    assert ring.distribution["a"] == 80
    assert ring.distribution["b"] == 240


def test_string_node_is_accepted_as_single_node():
    ring = HashRing("only", vnodes=5)
    assert list(ring.get_nodes()) == ["only"]
    assert ring.get_node("k1") == "only"
    assert ring.get_node("k2") == "only"


def test_invalid_nodes_type_raises_value_error():
    with pytest.raises(ValueError):
        HashRing(42)
    with pytest.raises(ValueError):
        HashRing(("a", "b"))


def test_non_callable_weight_fn_and_hash_fn_raise_type_error():
    with pytest.raises(TypeError):
        HashRing(["a"], weight_fn="nope")
    with pytest.raises(TypeError):
        HashRing(["a"], hash_fn=123)


def test_invalid_node_conf_type_raises_value_error():
    with pytest.raises(ValueError):
        HashRing({"a": "not-an-int-or-dict"})


def test_int_weights_scale_vnodes():
    ring = HashRing({"a": 1, "b": 3}, vnodes=10)
    assert ring.distribution["a"] == 10
    assert ring.distribution["b"] == 30
    assert ring.conf["b"]["weight"] == 3


def test_dict_conf_exposes_instance_hostname_and_port():
    inst = object()
    nodes = {
        "n1": {"hostname": "host1", "port": 1234, "instance": inst, "vnodes": 4},
    }
    ring = HashRing(nodes)
    assert ring.get_node_instance("key") is inst
    assert ring["key"] is inst
    assert ring.get_node_hostname("key") == "host1"
    assert ring.get_node_port("key") == 1234
    assert ring.get_node_weight("key") == 1
    assert ring.get_instances() == [inst]
    assert ring.get("key")["nodename"] == "n1"


def test_weight_fn_overrides_weight():
    ring = HashRing(["a", "b"], vnodes=7, weight_fn=lambda **conf: 2)
    assert ring.conf["a"]["weight"] == 2
    assert ring.distribution["a"] == 14
    assert ring.distribution["b"] == 14


def test_get_node_is_deterministic_and_returns_known_node():
    ring = HashRing(["a", "b", "c"], vnodes=20)
    names = set(ring.get_nodes())
    for i in range(50):
        key = "key-{}".format(i)
        first = ring.get_node(key)
        assert first in names
        assert ring.get_node(key) == first


def test_custom_hash_fn_and_get_key():
    ring = HashRing(["a", "b"], hash_fn=crc_hash, vnodes=8)
    assert ring.get_key("hello") == crc_hash("hello")
    assert ring.hashi("hello") == crc_hash("hello")
    ketama = HashRing(["a"], hash_fn="ketama")
    assert 0 <= ketama.get_key("hello") < 2 ** 32


def test_wraparound_and_boundaries_with_controlled_hash():
    table = {"a-0": 10, "b-0": 20, "k5": 5, "k10": 10, "k15": 15, "k25": 25}
    ring = HashRing(["a", "b"], hash_fn=table.__getitem__, vnodes=1)
    assert ring.get_points() == [(10, "a"), (20, "b")]
    assert ring.get_node("k5") == "a"
    assert ring.get_node("k10") == "b"
    assert ring.get_node("k15") == "b"
    # beyond the last point wraps to the first
    assert ring.get_node("k25") == "a"
    assert ring.get_node_pos("k5") == 0
    assert ring.get_node_pos("k15") == 1
    assert ring.get_node_pos("k25") == 0
    assert ring.get_server("k15") == (20, "b")


def test_get_points_sorted_and_get_server_matches_ring():
    ring = HashRing(["a", "b", "c"], vnodes=10)
    points = ring.get_points()
    positions = [p for p, _ in points]
    assert positions == sorted(positions)
    assert len(points) == ring.size
    pos, name = ring.get_server("some-key")
    assert ring.ring[pos] == name
    assert name == ring.get_node("some-key")


def test_remove_node_stops_routing_and_keeps_other_mappings():
    ring = HashRing(["a", "b", "c"], vnodes=20)
    keys = ["k{}".format(i) for i in range(100)]
    before = {k: ring.get_node(k) for k in keys}
    del ring["c"]
    assert "c" not in ring.get_nodes()
    assert "c" not in ring.distribution
    for k in keys:
        after = ring.get_node(k)
        assert after != "c"
        if before[k] != "c":
            assert after == before[k]


def test_remove_missing_node_raises_key_error():
    ring = HashRing(["a"], vnodes=3)
    with pytest.raises(KeyError):
        ring.remove_node("missing")
    ketama = HashRing(["a"], hash_fn="ketama")
    with pytest.raises(KeyError):
        ketama.remove_node("missing")


def test_add_node_with_default_and_explicit_conf():
    ring = HashRing(["a"], vnodes=5)
    ring.add_node("b")
    assert "b" in ring.get_nodes()
    assert ring.distribution["b"] == 5
    ring["c"] = {"weight": 2}
    assert ring.conf["c"]["weight"] == 2
    assert ring.distribution["c"] == 10
    assert ring.size > 0
    assert {ring.get_node("k{}".format(i)) for i in range(200)} <= {"a", "b", "c"}


def test_range_unique_returns_each_node_once_and_respects_size():
    ring = HashRing(["a", "b", "c"], vnodes=10)
    all_nodes = [n["nodename"] for n in ring.range("key")]
    assert sorted(all_nodes) == ["a", "b", "c"]
    assert all_nodes[0] == ring.get_node("key")
    limited = [n["nodename"] for n in ring.range("key", size=2)]
    assert len(limited) == 2
    assert len(set(limited)) == 2
    assert limited == all_nodes[:2]


def test_range_not_unique_with_size_allows_repeats():
    ring = HashRing(["a"], vnodes=10)
    result = list(ring.range("key", size=5, unique=False))
    assert len(result) == 5
    assert all(n["nodename"] == "a" for n in result)


def test_iterate_nodes_distinct_and_empty_cases():
    empty = HashRing()
    assert list(empty.iterate_nodes("key")) == [None]
    ring = HashRing(["a", "b", "c"], vnodes=10)
    names = list(ring.iterate_nodes("key"))
    assert sorted(names) == ["a", "b", "c"]
    assert names[0] == ring.get_node("key")


def test_regenerate_preserves_mapping():
    ring = HashRing(["a", "b", "c"], vnodes=10)
    keys = ["k{}".format(i) for i in range(50)]
    before = {k: ring.get_node(k) for k in keys}
    points_before = ring.get_points()
    ring.regenerate()
    assert ring.get_points() == points_before
    assert {k: ring.get_node(k) for k in keys} == before


def test_print_continuum_output(capsys):
    empty = HashRing()
    empty.print_continuum()
    assert "Continuum empty" in capsys.readouterr().out

    ring = HashRing(["a"], vnodes=3)
    ring.print_continuum()
    out = capsys.readouterr().out
    assert "Numpoints in continuum: {}".format(ring.size) in out
    assert "a (" in out


def test_property_aliases_share_underlying_state():
    ring = HashRing(["a", "b"], vnodes=4)
    assert ring.conf is ring.nodes
    assert ring.ring is ring.continuum
    assert ring._ring is ring.ring
    assert ring._nodes is ring.conf
    assert ring._keys == sorted(ring.ring.keys())
    assert ring.size == len(ring.ring)
