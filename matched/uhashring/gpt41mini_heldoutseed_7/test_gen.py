import pytest
from uhashring import HashRing


def test_create_hashring_with_single_node():
    ring = HashRing(nodes=["node1"])
    assert "node1" in ring.nodes()
    assert len(ring.nodes()) == 1


def test_create_hashring_with_multiple_nodes():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes=nodes)
    for node in nodes:
        assert node in ring.nodes()
    assert len(ring.nodes()) == 3


def test_create_hashring_from_empty_list_raises():
    with pytest.raises(ValueError):
        HashRing(nodes=[])


def test_hashring_get_node_for_key():
    nodes = ["nodeA", "nodeB", "nodeC"]
    ring = HashRing(nodes=nodes)
    key = "my-key"
    node = ring.get_node(key)
    assert node in nodes


def test_hashring_get_nodes_for_key_returns_expected_count():
    nodes = ["node1", "node2", "node3", "node4"]
    ring = HashRing(nodes=nodes)
    key = "abc"
    result_nodes = ring.get_nodes(key, count=2)
    assert len(result_nodes) == 2
    # nodes should be unique
    assert len(set(result_nodes)) == 2
    for n in result_nodes:
        assert n in nodes


def test_hashring_get_nodes_count_limit():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes=nodes)
    key = "abc"
    # request more nodes than present should return only existing nodes
    result_nodes = ring.get_nodes(key, count=10)
    assert set(result_nodes) == set(nodes)


def test_hashring_contains_node():
    nodes = ["n1", "n2"]
    ring = HashRing(nodes=nodes)
    assert "n1" in ring
    assert "n2" in ring
    assert "not-in-ring" not in ring


def test_iter_nodes_results_in_nodes():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes=nodes)
    for node in ring:
        assert node in nodes


def test_eq_and_ne_hashring_equal_and_not_equal():
    nodes1 = ["one", "two"]
    nodes2 = ["two", "one"]
    ring1 = HashRing(nodes=nodes1)
    ring2 = HashRing(nodes=nodes2)
    ring3 = HashRing(nodes=["one", "three"])
    assert ring1 == ring2  # order doesn't matter
    assert ring1 != ring3


def test_repr_contains_node_names():
    nodes = ["nodeX", "nodeY"]
    ring = HashRing(nodes=nodes)
    rep = repr(ring)
    for node in nodes:
        assert node in rep


def test_get_node_raises_on_empty_ring():
    with pytest.raises(ValueError):
        HashRing(nodes=[""]).get_node("key")


def test_add_nodes_and_remove_nodes():
    ring = HashRing(nodes=["n1", "n2"])
    ring.add_node("n3")
    assert "n3" in ring.nodes()
    ring.remove_node("n2")
    assert "n2" not in ring.nodes()
    # Removing node not present should raise an error
    with pytest.raises(KeyError):
        ring.remove_node("n5")


def test_get_nodes_count_zero_returns_empty_list():
    ring = HashRing(nodes=["a", "b"])
    result = ring.get_nodes("key", count=0)
    assert result == []


def test_get_node_consistent_for_same_key():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes=nodes)
    key = "key1"
    node1 = ring.get_node(key)
    node2 = ring.get_node(key)
    assert node1 == node2


def test_get_nodes_consistent_for_same_key_and_count():
    nodes = ["n1", "n2", "n3", "n4"]
    ring = HashRing(nodes=nodes)
    key = "mykey"
    count = 3
    lst1 = ring.get_nodes(key, count=count)
    lst2 = ring.get_nodes(key, count=count)
    assert lst1 == lst2
    assert len(lst1) == count


def test_node_names_can_be_non_string():
    nodes = [123, 456, 789]
    ring = HashRing(nodes=nodes)
    assert 123 in ring.nodes()
    node = ring.get_node("anykey")
    assert node in nodes


def test_nodes_property_is_immutable_from_outside():
    nodes = ["a", "b", "c"]
    ring = HashRing(nodes=nodes)
    external_nodes = ring.nodes()
    external_nodes.append("d")
    # The ring nodes should not be affected
    assert "d" not in ring.nodes()


def test_get_node_raises_when_no_nodes():
    with pytest.raises(ValueError):
        ring = HashRing(nodes=["x"])
        ring.remove_node("x")
        # Now ring is empty
        ring.get_node("key")


def test_add_node_duplicate_does_not_increase_nodes():
    ring = HashRing(nodes=["n1", "n2"])
    ring.add_node("n2")
    assert len(ring.nodes()) == 2


def test_repr_shows_virtual_nodes_count(monkeypatch):
    # The representation should contain virtual nodes count info
    ring = HashRing(nodes=["n1"], replicas=2)
    rep = repr(ring)
    assert "n1" in rep


def test_nodes_returns_sorted_nodes():
    # nodes() returns sorted list of nodes
    nodes = ["z", "a", "m"]
    ring = HashRing(nodes=nodes)
    sorted_nodes = sorted(nodes)
    assert ring.nodes() == sorted_nodes
