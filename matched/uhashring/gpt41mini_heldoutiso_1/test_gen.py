import pytest
from uhashring import HashRing

def test_create_ring_with_single_node():
    ring = HashRing(nodes=["node1"])
    assert "node1" in ring.get_nodes()
    assert len(ring.get_nodes()) == 1

def test_create_ring_with_multiple_nodes():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes=nodes)
    for node in nodes:
        assert node in ring.get_nodes()
    assert len(ring.get_nodes()) == 3

def test_get_node_for_key_returns_node_in_nodes():
    nodes = ["nodeA", "nodeB", "nodeC"]
    ring = HashRing(nodes=nodes)
    key = "mykey"
    node = ring.get_node(key)
    assert node in nodes

def test_consistent_node_for_same_key():
    nodes = ["nodeX", "nodeY", "nodeZ"]
    ring = HashRing(nodes=nodes)
    key = "consistent_key"
    node1 = ring.get_node(key)
    node2 = ring.get_node(key)
    assert node1 == node2

def test_different_keys_map_to_different_nodes_possibly():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes=nodes)
    key1 = "key1"
    key2 = "key2"
    # They can be same by hash chance, so only check nodes are valid
    n1 = ring.get_node(key1)
    n2 = ring.get_node(key2)
    assert n1 in nodes
    assert n2 in nodes

def test_add_node_makes_node_available():
    ring = HashRing(nodes=["node1", "node2"])
    ring.add_node("node3")
    nodes = ring.get_nodes()
    assert "node3" in nodes
    assert len(nodes) == 3

def test_remove_node_makes_node_unavailable():
    ring = HashRing(nodes=["node1", "node2", "node3"])
    ring.remove_node("node2")
    nodes = ring.get_nodes()
    assert "node2" not in nodes
    assert len(nodes) == 2

def test_get_node_after_removing_node_changes_if_node_was_target():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes=nodes)
    key = "some_key_for_node2"
    # Ensure key currently maps to node2 if possible
    node_for_key = ring.get_node(key)
    if node_for_key == "node2":
        ring.remove_node("node2")
        new_node = ring.get_node(key)
        assert new_node != "node2"
        assert new_node in ring.get_nodes()

def test_get_node_with_no_nodes_raises_exception():
    ring = HashRing(nodes=[])
    with pytest.raises(Exception):
        ring.get_node("key")

def test_add_node_twice_raises_error_or_no_duplicate():
    ring = HashRing(nodes=["node1"])
    ring.add_node("node2")
    # Adding same node again: either raises or no duplicates
    try:
        ring.add_node("node2")
    except Exception:
        pass
    else:
        # No error means no duplicate in ring
        nodes = ring.get_nodes()
        assert nodes.count("node2") == 1

def test_remove_node_not_in_ring_raises_error():
    ring = HashRing(nodes=["node1"])
    with pytest.raises(Exception):
        ring.remove_node("nodeX")

def test_get_nodes_returns_list_of_nodes():
    ring = HashRing(nodes=["x", "y"])
    nodes = ring.get_nodes()
    assert isinstance(nodes, list)
    assert set(nodes) == {"x", "y"}

def test_keys_are_hashed_consistently():
    ring = HashRing(nodes=["a", "b", "c"])
    key = "somekey"
    n1 = ring.get_node(key)
    n2 = ring.get_node(key)
    assert n1 == n2

def test_ring_with_integer_nodes_converts_to_string():
    ring = HashRing(nodes=[1, 2, 3])
    nodes = ring.get_nodes()
    # Nodes are strings internally
    for n in nodes:
        assert isinstance(n, str)

def test_ring_removal_and_addition_consistency():
    ring = HashRing(nodes=["a", "b"])
    node_for_key1 = ring.get_node("k1")
    ring.add_node("c")
    node_for_key2 = ring.get_node("k1")
    ring.remove_node("c")
    node_for_key3 = ring.get_node("k1")
    # The node before and after adding/removing c could be same or different,
    # but should not raise errors and node must be in nodes.
    assert node_for_key1 in ring.get_nodes()
    assert node_for_key2 in ring.get_nodes()
    assert node_for_key3 in ring.get_nodes()

def test_repr_contains_node_list():
    ring = HashRing(nodes=["alpha", "beta"])
    r = repr(ring)
    assert "alpha" in r and "beta" in r

def test_get_node_with_unicode_key():
    ring = HashRing(nodes=["n1", "n2"])
    key = "ключ"  # "key" in Russian
    node = ring.get_node(key)
    assert node in ring.get_nodes()

def test_get_node_with_empty_string_key():
    ring = HashRing(nodes=["node1", "node2"])
    node = ring.get_node("")
    assert node in ring.get_nodes()

def test_add_node_with_non_string_raises_error_or_str_conversion():
    ring = HashRing(nodes=["a"])
    # If non-string node, it should convert or raise
    try:
        ring.add_node(100)
    except Exception:
        pass
    else:
        assert "100" in ring.get_nodes()

def test_remove_node_after_adding_it():
    ring = HashRing(nodes=["a"])
    ring.add_node("b")
    ring.remove_node("b")
    assert "b" not in ring.get_nodes()

def test_get_node_deterministic_for_bytes_key():
    ring = HashRing(nodes=["a", "b", "c"])
    key = b"byteskey"
    node1 = ring.get_node(key)
    node2 = ring.get_node(key)
    assert node1 == node2
    assert node1 in ring.get_nodes()
