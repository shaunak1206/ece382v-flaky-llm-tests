import pytest

from uhashring import HashRing


NODES = ["node1", "node2", "node3", "node4"]
KEYS = ["key-%d" % i for i in range(300)]


def test_get_nodes_returns_configured_nodes():
    ring = HashRing(nodes=NODES)
    assert sorted(ring.get_nodes()) == sorted(NODES)


def test_get_node_returns_member_of_nodes():
    ring = HashRing(nodes=NODES)
    for key in KEYS:
        assert ring.get_node(key) in NODES


def test_get_node_is_deterministic_for_same_ring():
    ring = HashRing(nodes=NODES)
    first = [ring.get_node(k) for k in KEYS]
    second = [ring.get_node(k) for k in KEYS]
    assert first == second


def test_two_rings_with_same_nodes_agree():
    ring1 = HashRing(nodes=NODES)
    ring2 = HashRing(nodes=list(reversed(NODES)))
    for key in KEYS:
        assert ring1.get_node(key) == ring2.get_node(key)


def test_single_node_gets_all_keys():
    ring = HashRing(nodes=["only"])
    for key in KEYS:
        assert ring.get_node(key) == "only"


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_nodes() == []
    assert ring.get_node("anything") is None


def test_all_nodes_receive_some_keys():
    ring = HashRing(nodes=NODES)
    seen = {ring.get_node("k%d" % i) for i in range(2000)}
    assert seen == set(NODES)


def test_add_node_registers_node():
    ring = HashRing(nodes=NODES)
    ring.add_node("node5")
    assert "node5" in ring.get_nodes()
    assert len(ring.get_nodes()) == len(NODES) + 1


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(nodes=NODES)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.add_node("node5")
    for key in KEYS:
        after = ring.get_node(key)
        assert after == before[key] or after == "node5"


def test_remove_node_unregisters_node():
    ring = HashRing(nodes=NODES)
    ring.remove_node("node2")
    assert "node2" not in ring.get_nodes()
    for key in KEYS:
        assert ring.get_node(key) != "node2"


def test_remove_node_only_moves_keys_of_removed_node():
    ring = HashRing(nodes=NODES)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.remove_node("node3")
    for key in KEYS:
        if before[key] != "node3":
            assert ring.get_node(key) == before[key]
        else:
            assert ring.get_node(key) in set(NODES) - {"node3"}


def test_remove_unknown_node_raises():
    ring = HashRing(nodes=NODES)
    with pytest.raises(Exception):
        ring.remove_node("does-not-exist")
    assert sorted(ring.get_nodes()) == sorted(NODES)


def test_weight_increases_share_of_keys():
    ring = HashRing(nodes={"light": {"weight": 1}, "heavy": {"weight": 4}})
    counts = {"light": 0, "heavy": 0}
    for i in range(2000):
        counts[ring.get_node("item-%d" % i)] += 1
    assert counts["heavy"] > counts["light"]


def test_node_instance_is_returned():
    inst_a = object()
    inst_b = object()
    ring = HashRing(nodes={"a": {"instance": inst_a}, "b": {"instance": inst_b}})
    instances = {"a": inst_a, "b": inst_b}
    for key in KEYS[:50]:
        assert ring.get_node_instance(key) is instances[ring.get_node(key)]


def test_non_callable_weight_fn_raises_type_error():
    with pytest.raises(TypeError):
        HashRing(nodes=NODES, weight_fn="not callable")


def test_iterate_nodes_distinct_yields_each_node_once():
    ring = HashRing(nodes=NODES)
    result = list(ring.iterate_nodes("some-key"))
    assert sorted(result) == sorted(NODES)
    assert result[0] == ring.get_node("some-key")


def test_ketama_ring_is_deterministic_and_valid():
    ring1 = HashRing(nodes=NODES, hash_fn="ketama")
    ring2 = HashRing(nodes=NODES, hash_fn="ketama")
    for key in KEYS[:100]:
        node = ring1.get_node(key)
        assert node in NODES
        assert node == ring2.get_node(key)
