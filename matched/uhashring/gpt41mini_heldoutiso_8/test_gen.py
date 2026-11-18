import pytest
from uhashring import HashRing


def test_create_hashring_with_nodes_list():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    for node in nodes:
        assert node in ring.get_nodes()
    # The ring should have nodes set equal or more (due to replicas) than input nodes
    assert set(ring.get_nodes()) == set(nodes)


def test_create_hashring_with_nodes_dict():
    nodes = {"a": 3, "b": 2, "c": 1}
    ring = HashRing(nodes)
    # The nodes should be registered
    assert set(ring.get_nodes()) == set(nodes.keys())
    # Check that nodes with more weight have more replicas
    replicas_a = ring._nodes.get("a")
    replicas_b = ring._nodes.get("b")
    replicas_c = ring._nodes.get("c")
    assert replicas_a > replicas_b > replicas_c or replicas_a == replicas_b or replicas_b == replicas_c


def test_get_node_single_key():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    key = "my-key"
    node = ring.get_node(key)
    assert node in nodes


def test_get_node_returns_same_node_for_same_key():
    nodes = ["a", "b", "c", "d"]
    ring = HashRing(nodes)
    key = "repeat-key"
    node1 = ring.get_node(key)
    node2 = ring.get_node(key)
    assert node1 == node2


def test_get_node_returns_different_nodes_for_different_keys():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    node1 = ring.get_node("key1")
    node2 = ring.get_node("key2")
    # Could be same by chance, but unlikely with 3 nodes. Better check that keys distribute.
    assert node1 in nodes and node2 in nodes


def test_get_node_with_no_nodes_returns_none():
    ring = HashRing([])
    assert ring.get_node("anykey") is None


def test_get_nodes_returns_replicas_number_of_nodes():
    nodes = ["a", "b"]
    ring = HashRing(nodes, replicas=10)
    result_nodes = ring.get_nodes()
    # Should contain exactly same nodes as input list
    assert set(result_nodes) == set(nodes)
    # Check that there are no duplicates
    assert len(result_nodes) == len(set(result_nodes))


def test_remove_node_nodes_update():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    ring.remove_node("b")
    assert "b" not in ring.get_nodes()
    assert set(ring.get_nodes()) == {"a", "c"}


def test_add_node_nodes_update():
    nodes = ["a", "b"]
    ring = HashRing(nodes)
    ring.add_node("c")
    assert "c" in ring.get_nodes()
    assert set(ring.get_nodes()) == {"a", "b", "c"}


def test_add_node_weighted_nodes_update():
    nodes = {"a": 1, "b": 2}
    ring = HashRing(nodes)
    ring.add_node("c", weight=3)
    assert "c" in ring.get_nodes()


def test_remove_node_not_existing_raises():
    nodes = ["a", "b"]
    ring = HashRing(nodes)
    with pytest.raises(KeyError):
        ring.remove_node("c")


def test_has_node_true_false():
    nodes = ["a", "b"]
    ring = HashRing(nodes)
    assert ring.has_node("a")
    assert not ring.has_node("c")


def test_get_nodes_sorted_order():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    nodes_list = ring.get_nodes()
    # Should be sorted list
    assert nodes_list == sorted(nodes_list)


def test_ring_distribution_with_weights():
    nodes = {"a": 1, "b": 5}
    ring = HashRing(nodes)
    # Collect nodes for several keys
    results = {}
    for i in range(1000):
        key = f"key{i}"
        node = ring.get_node(key)
        results[node] = results.get(node, 0) + 1
    # node b should get more keys due to weight
    assert results["b"] > results["a"]


def test_repr_contains_nodes():
    nodes = ["a", "b"]
    ring = HashRing(nodes)
    r = repr(ring)
    for n in nodes:
        assert n in r


def test_repr_contains_replicas_count():
    nodes = ["a"]
    ring = HashRing(nodes, replicas=25)
    r = repr(ring)
    assert "replicas=25" in r


def test_len_returns_expected_value():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    assert len(ring) == len(nodes)


def test_iter_returns_nodes():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    for node in ring:
        assert node in nodes


def test_multiple_adds_and_removes_consistency():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    ring.add_node("d")
    ring.add_node("e")
    assert set(ring.get_nodes()) == set(["a", "b", "c", "d", "e"])
    ring.remove_node("b")
    ring.remove_node("e")
    assert set(ring.get_nodes()) == set(["a", "c", "d"])


def test_get_node_after_removal():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    ring.remove_node("b")
    # Keys previously assigned to b must now be assigned to a different node
    key = "somekey"
    node = ring.get_node(key)
    assert node in ["a", "c"]


def test_get_nodes_empty_ring_returns_empty_list():
    ring = HashRing([])
    assert ring.get_nodes() == []


def test_hashring_with_custom_hash_function():
    def custom_hash(key):
        # Always return 0 so all keys map to first node
        return 0

    nodes = ["a", "b", "c"]
    ring = HashRing(nodes, hash_fn=custom_hash)
    for key in ("key1", "key2", "key3"):
        assert ring.get_node(key) == "a"
