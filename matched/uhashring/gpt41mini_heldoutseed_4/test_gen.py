import pytest
from uhashring import HashRing


def test_create_empty_hashring_raises():
    with pytest.raises(ValueError):
        HashRing([])


def test_create_hashring_with_nodes():
    ring = HashRing(["node1", "node2", "node3"])
    assert isinstance(ring, HashRing)


def test_get_node_for_key_returns_node():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    node = ring.get_node("mykey")
    assert node in nodes


def test_get_node_for_key_is_stable():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    node1 = ring.get_node("mykey")
    node2 = ring.get_node("mykey")
    assert node1 == node2


def test_get_node_for_empty_key_returns_node():
    nodes = ["node1"]
    ring = HashRing(nodes)
    node = ring.get_node("")
    assert node == "node1"


def test_get_node_for_key_bytes_and_str_consistent():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    node_str = ring.get_node("mykey")
    node_bytes = ring.get_node(b"mykey")
    assert node_str == node_bytes


def test_repr_contains_nodes():
    nodes = ["nodeA", "nodeB"]
    ring = HashRing(nodes)
    r = repr(ring)
    for n in nodes:
        assert n in r


def test_get_node_round_robin_behavior_nodes_as_int():
    # Ensures nodes can be of any hashable type
    nodes = [10, 20, 30]
    ring = HashRing(nodes)
    node = ring.get_node("key")
    assert node in nodes


def test_add_node_and_get_node_changes_ring(monkeypatch):
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    node1 = ring.get_node("key")
    ring.add_node("node3")
    node2 = ring.get_node("key")
    # Adding a node may change the assigned node, but node2 should be in the updated nodes
    assert node2 in ["node1", "node2", "node3"]
    assert node1 != node2 or len(set(nodes)) == len({node1,node2})  # may or may not change


def test_remove_node_and_get_node_changes_ring():
    nodes = ["nodeA", "nodeB", "nodeC"]
    ring = HashRing(nodes)
    node_before = ring.get_node("key")
    ring.remove_node("nodeB")
    node_after = ring.get_node("key")
    assert "nodeB" not in ring._nodes
    assert node_after in ring._nodes
    # node_after different from removed node
    assert node_after != "nodeB"
    # node_before might or might not be removed node
    if node_before == "nodeB":
        assert node_after != "nodeB"


def test_remove_nonexistent_node_raises():
    ring = HashRing(["n1", "n2"])
    with pytest.raises(KeyError):
        ring.remove_node("not_exist")


def test_add_existing_node_raises():
    ring = HashRing(["n1"])
    with pytest.raises(ValueError):
        ring.add_node("n1")


def test_nodes_property_returns_nodes():
    nodes = ["n1", "n2", "n3"]
    ring = HashRing(nodes)
    result = ring.nodes
    assert isinstance(result, list)
    for n in result:
        assert n in nodes


def test_get_node_no_nodes_raises():
    ring = HashRing(["node1"])
    ring.remove_node("node1")
    with pytest.raises(ValueError):
        ring.get_node("anykey")


def test_large_number_of_nodes_and_keys():
    nodes = [f"node{i}" for i in range(100)]
    ring = HashRing(nodes)
    keys = [f"key{i}" for i in range(1000)]
    for k in keys:
        node = ring.get_node(k)
        assert node in nodes


def test_hashring_nodes_are_unique():
    nodes = ["a", "b", "a"]
    # Expect duplicate nodes filtered or error - check behavior
    ring = HashRing(nodes)
    # The nodes property should have unique nodes
    assert len(set(ring.nodes)) == len(ring.nodes)


def test_repr_output_is_string():
    ring = HashRing(["n1"])
    r = repr(ring)
    assert isinstance(r, str)
    assert "n1" in r


def test_nodes_property_returns_copy():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    ns = ring.nodes
    # modifying returned list shouldn't affect ring.nodes
    ns.append("nodeX")
    # subsequent call doesn't include nodeX
    assert "nodeX" not in ring.nodes


def test_get_node_with_non_string_key_type():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    key = 123456  # non-string key
    node = ring.get_node(key)
    assert node in nodes


def test_get_node_empty_hashring_after_removals():
    ring = HashRing(["n1"])
    ring.remove_node("n1")
    with pytest.raises(ValueError):
        ring.get_node("somekey")
