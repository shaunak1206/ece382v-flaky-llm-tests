import pytest
from uhashring import HashRing


def test_hashring_init_with_nodes_list():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    # The nodes should be available in ring
    for node in nodes:
        assert node in ring.get_nodes()


def test_hashring_init_with_nodes_dict():
    nodes = {"node1": 3, "node2": 2, "node3": 1}
    ring = HashRing(nodes)
    # The nodes should be available in ring
    for node in nodes.keys():
        assert node in ring.get_nodes()


def test_hashring_init_empty_nodes_raises():
    with pytest.raises(ValueError):
        HashRing([])


def test_get_node_returns_expected_node_string():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    key = "mykey"
    node = ring.get_node(key)
    assert isinstance(node, str)
    assert node in nodes


def test_get_node_with_bytes_key():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    key = b"byteskey"
    node = ring.get_node(key)
    assert isinstance(node, str)
    assert node in nodes


def test_get_node_with_int_key():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    key = 123456
    node = ring.get_node(key)
    assert isinstance(node, str)
    assert node in nodes


def test_get_node_with_different_keys_consistency():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    key = "consistent_key"
    node_first = ring.get_node(key)
    node_second = ring.get_node(key)
    assert node_first == node_second


def test_get_nodes_returns_all_nodes():
    nodes = ["nodeA", "nodeB", "nodeC"]
    ring = HashRing(nodes)
    ring_nodes = ring.get_nodes()
    assert set(ring_nodes) == set(nodes)


def test_get_nodes_returns_unique_nodes():
    nodes = ["node1", "node2", "node2", "node3"]
    ring = HashRing(nodes)
    ring_nodes = ring.get_nodes()
    assert "node2" in ring_nodes
    assert ring_nodes.count("node2") == 1


def test_get_node_with_empty_string_key():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    node = ring.get_node("")
    assert node in nodes


def test_internal_ring_is_balanced_for_different_weights():
    weights = {"node1": 1, "node2": 5, "node3": 10}
    ring = HashRing(weights)
    results = {"node1": 0, "node2": 0, "node3": 0}
    for i in range(1000):
        node = ring.get_node(str(i))
        results[node] += 1
    assert results["node3"] > results["node2"] > results["node1"]


def test_repr_contains_nodes():
    nodes = ["foo", "bar"]
    ring = HashRing(nodes)
    result = repr(ring)
    for node in nodes:
        assert node in result


def test_get_nodes_after_init_with_set():
    nodes = {"alpha", "beta", "gamma"}
    ring = HashRing(nodes)
    ring_nodes = ring.get_nodes()
    assert set(ring_nodes) == nodes


def test_get_node_returns_none_when_no_nodes(monkeypatch):
    # patch HashRing._keys to empty mapping to simulate no nodes
    ring = HashRing(["x"])
    monkeypatch.setattr(ring, "_keys", [])
    monkeypatch.setattr(ring, "_nodes", {})
    with pytest.raises(IndexError):
        ring.get_node("anything")


def test_hashring_with_duplicate_nodes_in_list():
    nodes = ["dup", "dup", "unique"]
    ring = HashRing(nodes)
    ring_nodes = ring.get_nodes()
    # duplicates should be removed internally
    assert ring_nodes.count("dup") == 1
    assert "unique" in ring_nodes


def test_get_node_returns_same_node_for_same_key_with_weights():
    weights = {"a": 1, "b": 2}
    ring = HashRing(weights)
    key = "same_key"
    first = ring.get_node(key)
    second = ring.get_node(key)
    assert first == second


def test_get_nodes_returns_list_even_if_nodes_dict_given():
    weights = {"one": 1, "two": 2}
    ring = HashRing(weights)
    nodes = ring.get_nodes()
    assert isinstance(nodes, list)
    assert set(nodes) == set(weights.keys())
