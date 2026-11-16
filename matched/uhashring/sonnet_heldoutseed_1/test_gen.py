import pytest

from uhashring import HashRing


NODES = ["node1", "node2", "node3"]


def make_ring(**kwargs):
    return HashRing(nodes=list(NODES), **kwargs)


def test_get_nodes_lists_all_nodes():
    ring = make_ring()
    assert sorted(ring.get_nodes()) == sorted(NODES)


def test_get_node_returns_member_of_ring():
    ring = make_ring()
    for i in range(100):
        assert ring.get_node("key-%d" % i) in NODES


def test_get_node_is_deterministic():
    ring = make_ring()
    first = [ring.get_node("key-%d" % i) for i in range(200)]
    second = [ring.get_node("key-%d" % i) for i in range(200)]
    assert first == second


def test_two_rings_same_nodes_agree():
    ring1 = make_ring()
    ring2 = HashRing(nodes=list(reversed(NODES)))
    for i in range(200):
        key = "some-key-%d" % i
        assert ring1.get_node(key) == ring2.get_node(key)


def test_keys_are_spread_across_nodes():
    ring = make_ring()
    seen = {ring.get_node("key-%d" % i) for i in range(500)}
    assert seen == set(NODES)


def test_empty_ring_get_node_returns_none():
    ring = HashRing()
    assert ring.get_nodes() == []
    assert ring.get_node("anything") is None


def test_add_node_appears_in_nodes():
    ring = make_ring()
    ring.add_node("node4")
    assert "node4" in ring.get_nodes()
    assert len(ring.get_nodes()) == 4


def test_add_node_only_moves_keys_to_new_node():
    ring = make_ring()
    keys = ["key-%d" % i for i in range(500)]
    before = {k: ring.get_node(k) for k in keys}
    ring.add_node("node4")
    for k in keys:
        after = ring.get_node(k)
        assert after == before[k] or after == "node4"


def test_remove_node_removes_from_nodes():
    ring = make_ring()
    ring.remove_node("node2")
    assert "node2" not in ring.get_nodes()
    for i in range(200):
        assert ring.get_node("key-%d" % i) != "node2"


def test_remove_node_keeps_other_keys_stable():
    ring = make_ring()
    keys = ["key-%d" % i for i in range(500)]
    before = {k: ring.get_node(k) for k in keys}
    ring.remove_node("node2")
    for k in keys:
        if before[k] != "node2":
            assert ring.get_node(k) == before[k]


def test_remove_unknown_node_raises():
    ring = make_ring()
    with pytest.raises(Exception):
        ring.remove_node("does-not-exist")


def test_dict_nodes_with_instances():
    inst_a, inst_b = object(), object()
    ring = HashRing(
        nodes={
            "a": {"instance": inst_a},
            "b": {"instance": inst_b},
        }
    )
    assert sorted(ring.get_nodes()) == ["a", "b"]
    instances = ring.get_instances()
    assert inst_a in instances and inst_b in instances
    for i in range(50):
        key = "key-%d" % i
        node = ring.get_node(key)
        expected = inst_a if node == "a" else inst_b
        assert ring.get_node_instance(key) is expected


def test_weight_influences_distribution():
    ring = HashRing(
        nodes={
            "light": {"weight": 1},
            "heavy": {"weight": 5},
        }
    )
    counts = {"light": 0, "heavy": 0}
    for i in range(3000):
        counts[ring.get_node("key-%d" % i)] += 1
    assert counts["heavy"] > counts["light"]


def test_ketama_hash_is_deterministic_and_valid():
    ring1 = HashRing(nodes=list(NODES), hash_fn="ketama")
    ring2 = HashRing(nodes=list(NODES), hash_fn="ketama")
    for i in range(100):
        key = "key-%d" % i
        node = ring1.get_node(key)
        assert node in NODES
        assert node == ring2.get_node(key)


def test_iterate_nodes_distinct_covers_all_nodes():
    ring = make_ring()
    result = list(ring.iterate_nodes("my-key", distinct=True))
    assert sorted(result) == sorted(NODES)
    assert result[0] == ring.get_node("my-key")
