import pytest
from uhashring import HashRing


def test_hashring_init_empty_nodes():
    ring = HashRing([])
    assert ring.nodes == []
    # The ring should have no replicas
    assert len(ring.replicas) == 0


def test_hashring_single_node():
    node = "node1"
    ring = HashRing([node])
    assert node in ring.nodes
    # The ring should have replicas for the single node, default replicas is 40
    assert len(ring.replicas) == 40


def test_hashring_multiple_nodes():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    for node in nodes:
        assert node in ring.nodes
    # With 3 nodes and default 40 replicas per node, total replicas should be 120
    assert len(ring.replicas) == 120


def test_get_node_returns_node_from_ring():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    # For several keys, get_node should return a node in the ring
    for key in ["mykey", "anotherkey", "xyz"]:
        node = ring.get_node(key)
        assert node in nodes


def test_get_node_consistent_for_same_key():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    key = "mykey"
    node1 = ring.get_node(key)
    node2 = ring.get_node(key)
    assert node1 == node2


def test_get_node_returns_none_if_empty_ring():
    ring = HashRing([])
    assert ring.get_node("anykey") is None


def test_nodes_property_returns_all_nodes():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    assert set(ring.nodes) == set(nodes)


def test_adding_nodes_recreates_ring():
    ring = HashRing(["node1"])
    old_replicas = ring.replicas.copy()
    ring.nodes = ["node1", "node2"]
    assert "node2" in ring.nodes
    # The replicas must have changed because nodes changed
    assert ring.replicas != old_replicas


def test_removing_nodes_recreates_ring():
    ring = HashRing(["node1", "node2"])
    old_replicas = ring.replicas.copy()
    ring.nodes = ["node1"]
    assert ring.nodes == ["node1"]
    # The replicas must have changed because nodes changed
    assert ring.replicas != old_replicas


def test_hashring_with_custom_replicas():
    nodes = ["node1", "node2"]
    replicas = 100
    ring = HashRing(nodes, replicas=replicas)
    # replicas attribute should control number of replicas per node
    assert len(ring.replicas) == replicas * len(nodes)


def test_get_nodes_for_key_returns_correct_nodes():
    """Test get_nodes returns exactly count nodes sorted by ring."""
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    count = 2
    result_nodes = ring.get_nodes("mykey", count)
    assert isinstance(result_nodes, list)
    assert len(result_nodes) == count
    for node in result_nodes:
        assert node in nodes
    # Nodes should be unique
    assert len(set(result_nodes)) == len(result_nodes)


def test_get_nodes_count_larger_than_ring_nodes():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    # Request more nodes than available - should return all unique nodes
    result = ring.get_nodes("mykey", 5)
    assert set(result) == set(nodes)


def test_get_node_rounds_consistently(monkeypatch):
    """Test get_node uses proper hashing to get consistent node."""
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)

    # For some deterministic keys, get_node returns consistent results
    keys = [b"alpha", b"beta", b"gamma"]
    results = [ring.get_node(k) for k in keys]
    results_again = [ring.get_node(k) for k in keys]
    assert results == results_again


def test_repr_contains_nodes_and_replicas():
    nodes = ["nodeA"]
    ring = HashRing(nodes, replicas=10)
    r = repr(ring)
    # The repr string should report the nodes and replicas
    assert "nodeA" in r
    assert "10" in r


def test_nodes_type_accepts_any_hashable():
    # Use tuple and int as nodes instead of strings
    nodes = [(1, 2), 42]
    ring = HashRing(nodes)
    for node in nodes:
        assert node in ring.nodes
    # Calling get_node with string keys works
    node = ring.get_node("key1")
    assert node in nodes


def test_keys_with_non_string_bytes_and_string_inputs():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    # String key
    node1 = ring.get_node("stringkey")
    # Bytes key
    node2 = ring.get_node(b"byteskey")
    assert node1 in nodes
    assert node2 in nodes


def test_hashring_nodes_property_immutable():
    nodes = ["node1"]
    ring = HashRing(nodes)
    with pytest.raises(AttributeError):
        # Attempting to set the nodes list attribute directly should fail (assuming no setter)
        ring.nodes = ["newnode"]


def test_get_many_nodes_empty_ring_returns_empty_list():
    ring = HashRing([])
    result = ring.get_nodes("key", 3)
    assert result == []


def test_get_nodes_with_zero_or_negative_count():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    # count <= 0 should return empty list
    assert ring.get_nodes("key", 0) == []
    assert ring.get_nodes("key", -1) == []


def test_get_node_not_found_returns_none(monkeypatch):
    # Patch _search to simulate no node found scenario
    ring = HashRing(["a", "b"])
    monkeypatch.setattr(ring, "replicas", [])
    assert ring.get_node("key") is None

