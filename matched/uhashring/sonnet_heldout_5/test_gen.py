import collections

import pytest

from uhashring import HashRing


NODES = ["node1", "node2", "node3"]
KEYS = ["key-%d" % i for i in range(200)]


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_node("anything") is None


def test_empty_ring_has_no_nodes():
    ring = HashRing()
    assert list(ring.get_nodes()) == []


def test_single_string_node():
    ring = HashRing("only-node")
    assert list(ring.get_nodes()) == ["only-node"]
    for key in KEYS[:20]:
        assert ring.get_node(key) == "only-node"


def test_invalid_nodes_type_raises():
    with pytest.raises(ValueError):
        HashRing(42)


def test_non_callable_weight_fn_raises():
    with pytest.raises(TypeError):
        HashRing(NODES, weight_fn="not callable")


def test_get_node_returns_member_and_is_deterministic():
    ring = HashRing(NODES)
    for key in KEYS:
        node = ring.get_node(key)
        assert node in NODES
        assert ring.get_node(key) == node


def test_same_nodes_in_two_rings_give_same_mapping():
    ring1 = HashRing(NODES)
    ring2 = HashRing(list(reversed(NODES)))
    for key in KEYS:
        assert ring1.get_node(key) == ring2.get_node(key)


def test_keys_are_spread_over_all_nodes():
    ring = HashRing(NODES)
    counts = collections.Counter(ring.get_node(k) for k in KEYS)
    assert set(counts) == set(NODES)


def test_vnodes_controls_number_of_points():
    ring = HashRing(["a", "b"], vnodes=10)
    points = ring.get_points()
    counts = collections.Counter(p[1] for p in points)
    assert counts["a"] == 10
    assert counts["b"] == 10


def test_weight_multiplies_points():
    ring = HashRing({"a": 1, "b": 3}, vnodes=10)
    counts = collections.Counter(p[1] for p in ring.get_points())
    assert counts["a"] == 10
    assert counts["b"] == 30


def test_dict_config_with_vnodes_and_weight():
    ring = HashRing({"a": {"vnodes": 5, "weight": 2}})
    counts = collections.Counter(p[1] for p in ring.get_points())
    assert counts["a"] == 10


def test_weight_fn_is_applied():
    ring = HashRing(["a", "b"], vnodes=4, weight_fn=lambda **conf: 2)
    counts = collections.Counter(p[1] for p in ring.get_points())
    assert counts["a"] == 8
    assert counts["b"] == 8


def test_instance_and_getitem():
    inst_a, inst_b = object(), object()
    ring = HashRing({"a": {"instance": inst_a}, "b": {"instance": inst_b}})
    for key in KEYS[:30]:
        node = ring.get_node(key)
        expected = inst_a if node == "a" else inst_b
        assert ring.get_node_instance(key) is expected
        assert ring[key] is expected


def test_hostname_and_port_config():
    ring = HashRing({"n1": {"hostname": "host1", "port": 1234}})
    assert ring.get_node_hostname("somekey") == "host1"
    assert ring.get_node_port("somekey") == 1234


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(["a", "b", "c"])
    before = {k: ring.get_node(k) for k in KEYS}
    ring.add_node("d")
    assert "d" in list(ring.get_nodes())
    moved_to_d = 0
    for k in KEYS:
        after = ring.get_node(k)
        if after != before[k]:
            assert after == "d"
            moved_to_d += 1
    assert moved_to_d > 0


def test_remove_node_only_remaps_its_keys():
    ring = HashRing(["a", "b", "c"])
    before = {k: ring.get_node(k) for k in KEYS}
    ring.remove_node("b")
    assert "b" not in list(ring.get_nodes())
    for k in KEYS:
        after = ring.get_node(k)
        assert after != "b"
        if before[k] != "b":
            assert after == before[k]


def test_setitem_and_delitem():
    ring = HashRing(["a", "b"])
    ring["c"] = {"weight": 1}
    assert "c" in list(ring.get_nodes())
    del ring["c"]
    assert "c" not in list(ring.get_nodes())
    for k in KEYS[:50]:
        assert ring.get_node(k) in ("a", "b")


def test_iterate_nodes_distinct_yields_all_nodes_once():
    ring = HashRing(NODES)
    result = list(ring.iterate_nodes("some-key"))
    assert sorted(result) == sorted(NODES)
    assert result[0] == ring.get_node("some-key")


def test_ketama_ring_is_deterministic_and_valid():
    ring1 = HashRing(NODES, hash_fn="ketama")
    ring2 = HashRing(NODES, hash_fn="ketama")
    for key in KEYS[:50]:
        node = ring1.get_node(key)
        assert node in NODES
        assert node == ring2.get_node(key)
