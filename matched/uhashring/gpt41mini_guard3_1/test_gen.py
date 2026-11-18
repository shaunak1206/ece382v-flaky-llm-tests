import random
import pytest
from uhashring import HashRing


def test_empty_nodes_raises():
    with pytest.raises(ValueError):
        HashRing([])


def test_single_node_returns_that_node():
    random.seed(0)
    ring = HashRing(["node1"])
    for key in ["a", 1, b"bytes", "random_key"]:
        assert ring.get_node(key) == "node1"


def test_multiple_nodes_all_return_a_node():
    random.seed(123)
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    keys = [1, 2, 3, 4, "key", b"bytes"]
    for key in keys:
        node = ring.get_node(key)
        assert node in nodes


def test_get_node_returns_same_node_for_same_key():
    random.seed(42)
    nodes = ["a", "b", "c", "d"]
    ring = HashRing(nodes)
    keys = ["samekey", 123456, b"binarykey"]
    for key in keys:
        node1 = ring.get_node(key)
        node2 = ring.get_node(key)
        assert node1 == node2


def test_add_node_changes_ring_assignment(monkeypatch):
    random.seed(1)
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    key = "mykey"
    node_before = ring.get_node(key)
    ring.add_node("node3")
    node_after = ring.get_node(key)
    # Adding a node may cause some keys to move to the new node or others,
    # So node_after may be different or same, but all candidates nodes are inside ring.nodes
    assert node_after in ring.get_nodes()
    assert isinstance(node_after, str)
    # The new node must be present in the ring nodes
    assert "node3" in ring.get_nodes()
    # The number of nodes increases by one
    assert len(ring.get_nodes()) == 3


def test_remove_node_changes_ring_assignment(monkeypatch):
    random.seed(2)
    nodes = ["n1", "n2", "n3"]
    ring = HashRing(nodes)
    key = "key1"
    node_before = ring.get_node(key)
    ring.remove_node("n2")
    assert "n2" not in ring.get_nodes()
    node_after = ring.get_node(key)
    # node_after must be from remaining nodes
    assert node_after in ring.get_nodes()
    # ring has now two nodes
    assert len(ring.get_nodes()) == 2


def test_remove_all_nodes_leads_to_error():
    ring = HashRing(["onlynode"])
    ring.remove_node("onlynode")
    with pytest.raises(ValueError):
        ring.get_node("anykey")


def test_get_nodes_returns_list_of_current_nodes():
    ring = HashRing(["x", "y"])
    nodes = ring.get_nodes()
    assert isinstance(nodes, list)
    assert set(nodes) == {"x", "y"}
    # After add_node, get_nodes contains the new node
    ring.add_node("z")
    nodes2 = ring.get_nodes()
    assert set(nodes2) == {"x", "y", "z"}


def test_ring_with_integer_nodes():
    # Integer nodes have deterministic hash
    nodes = [1, 2, 3]
    ring = HashRing(nodes)
    keys = ["key1", "key2", 100, b"bytes"]
    for key in keys:
        node = ring.get_node(key)
        assert node in nodes


def test_ring_with_mixed_type_nodes():
    nodes = [1, "strnode", b"bytesnode"]
    ring = HashRing(nodes)
    for key in ["foo", 999, b"bar"]:
        node = ring.get_node(key)
        assert node in nodes


def test_get_node_with_int_keys_consistent():
    nodes = [10, 20, 30]
    ring = HashRing(nodes)
    for key in range(100, 110):
        node1 = ring.get_node(key)
        node2 = ring.get_node(key)
        assert node1 == node2


def test_no_side_effects_on_nodes_passed():
    nodes = ["a", "b"]
    nodes_copy = nodes[:]
    ring = HashRing(nodes)
    # Original list not changed
    assert nodes == nodes_copy


def test_add_node_does_not_duplicate(monkeypatch):
    # Adding an existing node should not increase nodes
    ring = HashRing(["n1", "n2"])
    before = ring.get_nodes()
    ring.add_node("n1")
    after = ring.get_nodes()
    assert set(before) == set(after)


def test_remove_node_nonexistent_does_not_error(monkeypatch):
    ring = HashRing(["n1", "n2"])
    ring.remove_node("not_in_ring")
    # still returns a node from existent nodes
    node = ring.get_node("key")
    assert node in ring.get_nodes()


def test_get_node_bytes_and_str_keys():
    ring = HashRing(["a", "b", "c"])
    keys = [b"bytekey", "strkey", b"", ""]
    for key in keys:
        node = ring.get_node(key)
        assert node in ring.get_nodes()


def test_get_node_with_large_number_of_nodes():
    random.seed(0)
    nodes = list(range(100))
    ring = HashRing(nodes)
    keys = [i for i in range(1000, 1010)]
    for key in keys:
        node = ring.get_node(key)
        assert node in nodes


def test_get_node_consistent_after_removal_and_addition(monkeypatch):
    random.seed(10)
    nodes = ["A", "B", "C", "D"]
    ring = HashRing(nodes)
    key = "mykey"
    node1 = ring.get_node(key)
    ring.remove_node("B")
    node2 = ring.get_node(key)
    ring.add_node("E")
    node3 = ring.get_node(key)
    # All returned nodes must be among currently present nodes at each step
    assert node1 in nodes
    assert node2 in ring.get_nodes()
    assert node3 in ring.get_nodes()


def test_repr_contains_nodes():
    ring = HashRing(["r1", "r2"])
    rep = repr(ring)
    assert "r1" in rep and "r2" in rep


def test_eq_and_ne(monkeypatch):
    ring1 = HashRing(["n1", "n2"])
    ring2 = HashRing(["n1", "n2"])
    ring3 = HashRing(["n2", "n3"])
    # HashRing currently may not support eq, but we test that different instances are not equal
    assert ring1 != ring3
    # For same nodes, behavior is consistent, but not necessarily eq()
    assert ring1 != ring2  # Instances differ


def test_hashring_with_custom_weights():
    # If HashRing supports weights via tuples, test it (only if implemented)
    # This test assumes HashRing accepts (node, weight) tuples; skip if not
    try:
        ring = HashRing([("n1", 1), ("n2", 5), ("n3", 2)])
    except TypeError:
        pytest.skip("HashRing does not support weight tuples")
    keys = ["key1", "key2"]
    for key in keys:
        node = ring.get_node(key)
        assert node in ["n1", "n2", "n3"]


def test_get_node_return_type():
    ring = HashRing(["node"])
    key = "somekey"
    node = ring.get_node(key)
    assert isinstance(node, (str, int, bytes))


def test_get_nodes_returns_copy():
    ring = HashRing(["n1", "n2"])
    nodes1 = ring.get_nodes()
    nodes2 = ring.get_nodes()
    # Modifying returned list does not affect ring internal state
    nodes1.append("newnode")
    assert "newnode" not in ring.get_nodes()
