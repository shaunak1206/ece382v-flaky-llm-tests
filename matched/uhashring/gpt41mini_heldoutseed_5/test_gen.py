import pytest
from uhashring import HashRing

def test_create_hashring_with_empty_nodes():
    ring = HashRing([])
    assert len(ring.get_nodes()) == 0


def test_create_hashring_with_single_node():
    nodes = ["node1"]
    ring = HashRing(nodes)
    retrieved_nodes = ring.get_nodes()
    assert len(retrieved_nodes) == 1
    assert "node1" in retrieved_nodes


def test_create_hashring_with_multiple_nodes():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    retrieved_nodes = ring.get_nodes()
    assert len(retrieved_nodes) == 3
    for node in nodes:
        assert node in retrieved_nodes


def test_get_node_for_key_consistency():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    key = "my_key"
    node_first = ring.get_node(key)
    node_second = ring.get_node(key)
    assert node_first == node_second


def test_get_node_returns_one_of_the_nodes():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    key = "some_random_key"
    node = ring.get_node(key)
    assert node in nodes


def test_get_nodes_for_key_return_all_nodes_sorted():
    nodes = ["node3", "node1", "node2"]
    ring = HashRing(nodes)
    key = "another_key"
    result_nodes = ring.get_nodes(key)
    # Should return all nodes sorted (default behavior)
    assert sorted(nodes) == result_nodes


def test_get_nodes_with_count_less_than_nodes():
    nodes = ["node1", "node2", "node3", "node4", "node5"]
    ring = HashRing(nodes)
    key = "key_for_nodes"
    count = 3
    result_nodes = ring.get_nodes(key, count=count)
    assert len(result_nodes) == count
    # All must be part of initial nodes
    for node in result_nodes:
        assert node in nodes
    assert len(set(result_nodes)) == count


def test_get_nodes_with_count_greater_than_nodes_returns_all():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    key = "key"
    count = 10
    result_nodes = ring.get_nodes(key, count=count)
    assert set(result_nodes) == set(nodes)
    assert len(result_nodes) == len(nodes)


def test_add_node_and_get_nodes():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    ring.add_node("node3")
    result_nodes = ring.get_nodes()
    for n in nodes + ["node3"]:
        assert n in result_nodes


def test_remove_node_and_get_nodes():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    ring.remove_node("node2")
    result_nodes = ring.get_nodes()
    assert "node2" not in result_nodes
    assert "node1" in result_nodes
    assert "node3" in result_nodes


def test_remove_node_not_in_ring_raises_error():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    with pytest.raises(KeyError):
        ring.remove_node("non_existent_node")


def test_add_existing_node_does_not_duplicate():
    nodes = ["node1"]
    ring = HashRing(nodes)
    ring.add_node("node1")
    nodes_after = ring.get_nodes()
    assert nodes_after.count("node1") == 1


def test_get_node_with_empty_ring_returns_none():
    ring = HashRing([])
    result = ring.get_node("anykey")
    assert result is None


def test_get_nodes_with_empty_ring_returns_empty_list():
    ring = HashRing([])
    result = ring.get_nodes("anykey")
    assert result == []


def test_get_nodes_with_zero_count_returns_empty_list():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    result = ring.get_nodes("key", count=0)
    assert result == []


def test_get_nodes_with_negative_count_raises_value_error():
    nodes = ["node1", "node2"]
    ring = HashRing(nodes)
    with pytest.raises(ValueError):
        ring.get_nodes("key", count=-1)


def test_init_with_duplicate_nodes_does_not_duplicate_in_ring():
    nodes = ["node1", "node1", "node2"]
    ring = HashRing(nodes)
    retrieved = ring.get_nodes()
    assert len(retrieved) == 2
    assert "node1" in retrieved
    assert "node2" in retrieved


def test_get_nodes_count_exact_nodes_count_returns_all_nodes():
    nodes = ["node1", "node2", "node3"]
    ring = HashRing(nodes)
    result = ring.get_nodes("key", count=3)
    assert set(result) == set(nodes)


def test_consistent_distribution_for_different_keys():
    nodes = [f"node{i}" for i in range(10)]
    ring = HashRing(nodes)
    key1 = "key1"
    key2 = "key2"
    node1 = ring.get_node(key1)
    node2 = ring.get_node(key2)
    assert node1 in nodes
    assert node2 in nodes
    # It's okay if node1 and node2 are same or different
    assert isinstance(node1, str)
    assert isinstance(node2, str)

