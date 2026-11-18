import pytest
from uhashring import HashRing


def test_hashring_init_with_nodes():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    assert isinstance(ring, HashRing)
    # Expect the nodes we passed to be in the ring (at least keys)
    for node in nodes:
        assert node in ring.get_nodes()


def test_hashring_init_empty_nodes():
    # Creating with empty list or None should raise TypeError or ValueError
    with pytest.raises(TypeError):
        HashRing(None)
    hr = HashRing([])
    # With empty nodes, ring should have no keys
    assert list(hr.get_nodes()) == []


def test_hashring_single_node():
    node = "only"
    ring = HashRing([node])
    assert list(ring.get_nodes()) == [node]
    # Key assigned should map to the single node
    assert ring.get_node("any_key") == node


def test_get_node_consistency():
    nodes = ["x", "y", "z"]
    ring = HashRing(nodes)
    key = "my_key"
    node1 = ring.get_node(key)
    node2 = ring.get_node(key)
    # The node result for the same key should be consistent
    assert node1 == node2
    assert node1 in nodes


def test_get_node_distribution():
    # Basic statistical test if nodes actually get some keys
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    keys = [f"key{i}" for i in range(1000)]
    result_nodes = [ring.get_node(k) for k in keys]
    counts = {node: result_nodes.count(node) for node in nodes}
    # Each node should have at least some keys assigned
    for c in counts.values():
        assert c > 0


def test_get_nodes_returns_all_nodes():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    returned_nodes = set(ring.get_nodes())
    # Should match all nodes given (at least)
    for node in nodes:
        assert node in returned_nodes


def test_get_node_raises_if_no_nodes():
    ring = HashRing([])
    with pytest.raises(ValueError):
        ring.get_node("key")


def test_get_node_with_different_key_types():
    nodes = ["n"]
    ring = HashRing(nodes)
    # Test with int keys
    node = ring.get_node(123)
    assert node == "n"
    # Test with bytes keys
    node = ring.get_node(b"bytes_key")
    assert node == "n"
    # Test with tuple keys
    node = ring.get_node(("tuple", 1))
    assert node == "n"


def test_hashring_nodes_immutable_after_init():
    nodes = ["n1", "n2"]
    ring = HashRing(nodes)
    # Try modifying the original nodes list doesn't affect ring
    nodes.append("n3")
    returned_nodes = set(ring.get_nodes())
    assert "n3" not in returned_nodes


def test_repr_contains_nodes():
    nodes = ["alpha", "beta"]
    ring = HashRing(nodes)
    r = repr(ring)
    for node in nodes:
        assert node in r


def test_nodes_iterable_argument():
    # Accept anything iterable, not just list
    nodes = ("one", "two", "three")
    ring = HashRing(nodes)
    for node in nodes:
        assert node in ring.get_nodes()


def test_nodes_with_duplicates():
    nodes = ["dup", "dup", "unique"]
    ring = HashRing(nodes)
    nodes_set = set(ring.get_nodes())
    # Nodes unique, duplicates collapsed if any
    assert "dup" in nodes_set
    assert "unique" in nodes_set


def test_get_node_returns_none_or_raises_for_empty_ring():
    ring = HashRing([])
    with pytest.raises(ValueError):
        ring.get_node("key")


def test_get_node_on_unicode_keys():
    nodes = ["node"]
    ring = HashRing(nodes)
    node = ring.get_node("ключ")  # Russian for key
    assert node == "node"
    node = ring.get_node("鍵")  # Japanese for key
    assert node == "node"


def test_get_node_always_returns_a_node_for_nonempty_ring():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes)
    # Test with many random keys
    for k in [None, "", 0, 1.5, " ", "\n", "x" * 1000]:
        node = ring.get_node(k)
        assert node in nodes


def test_get_node_with_bytes_and_str_nodes():
    # Nodes can be strings, keys can be bytes or strings
    nodes = ["node"]
    ring = HashRing(nodes)
    assert ring.get_node("keystring") == "node"
    assert ring.get_node(b"keybytes") == "node"


def test_get_node_with_non_string_nodes():
    # If nodes are not strings but hashable, they should work
    node_obj = (1, 2)
    ring = HashRing([node_obj])
    assert ring.get_node("anykey") == node_obj


def test_get_nodes_returns_nodes_of_correct_type():
    node_obj = object()
    ring = HashRing([node_obj])
    nodes = list(ring.get_nodes())
    assert node_obj in nodes


def test_get_nodes_not_affected_by_external_mutations():
    nodes = ["foo", "bar"]
    ring = HashRing(nodes)
    nodes.append("baz")
    got_nodes = list(ring.get_nodes())
    assert "baz" not in got_nodes


def test_large_number_of_nodes():
    nodes = [f"node{i}" for i in range(100)]
    ring = HashRing(nodes)
    # With many nodes, get_node should still return a valid node
    for key in ("testkey", 1234, "abc", ""):
        node = ring.get_node(key)
        assert node in nodes


def test_get_node_different_keys_result_in_different_nodes():
    # While possible some keys hit same node, mostly keys differ
    nodes = [str(i) for i in range(10)]
    ring = HashRing(nodes)
    keys = [f"key{i}" for i in range(100)]
    results = set(ring.get_node(k) for k in keys)
    # Many nodes should be hit generally
    assert len(results) > 1


def test_get_nodes_returns_iterable():
    ring = HashRing(["a", "b"])
    nodes_iterable = ring.get_nodes()
    assert hasattr(nodes_iterable, "__iter__")
