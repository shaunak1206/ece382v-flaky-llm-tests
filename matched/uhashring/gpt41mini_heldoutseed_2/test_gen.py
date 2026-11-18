import pytest
from uhashring import HashRing


def test_hashring_init_nodes_as_list():
    nodes = ['node1', 'node2', 'node3']
    ring = HashRing(nodes)
    assert set(ring.nodes) == set(nodes)


def test_hashring_init_nodes_as_set():
    nodes = {'node1', 'node2', 'node3'}
    ring = HashRing(nodes)
    assert set(ring.nodes) == nodes


def test_hashring_init_nodes_as_tuple():
    nodes = ('node1', 'node2', 'node3')
    ring = HashRing(nodes)
    assert set(ring.nodes) == set(nodes)


def test_hashring_len_and_contains():
    nodes = ['node1', 'node2', 'node3']
    ring = HashRing(nodes)
    assert len(ring) == 3
    assert 'node1' in ring
    assert 'node4' not in ring


def test_add_node():
    nodes = ['node1', 'node2']
    ring = HashRing(nodes)
    ring.add_node('node3')
    assert 'node3' in ring
    assert len(ring) == 3


def test_add_existing_node_no_duplicate():
    nodes = ['node1', 'node2']
    ring = HashRing(nodes)
    ring.add_node('node2')
    # Adding existing node should not increase count
    assert len(ring) == 2


def test_remove_node():
    nodes = ['node1', 'node2', 'node3']
    ring = HashRing(nodes)
    ring.remove_node('node2')
    assert 'node2' not in ring
    assert len(ring) == 2


def test_remove_nonexistent_node_raises_keyerror():
    nodes = ['node1', 'node2']
    ring = HashRing(nodes)
    with pytest.raises(KeyError):
        ring.remove_node('node3')


def test_get_node_for_key_consistency():
    nodes = ['node1', 'node2', 'node3']
    ring = HashRing(nodes)
    key = 'my_key'
    # The node returned for the same key is always the same
    node1 = ring.get_node(key)
    node2 = ring.get_node(key)
    assert node1 == node2
    assert node1 in ring.nodes


def test_get_node_returns_None_if_no_nodes():
    ring = HashRing([])
    assert ring.get_node('anykey') is None


def test_get_nodes_for_key_count():
    nodes = ['node1', 'node2', 'node3', 'node4']
    ring = HashRing(nodes)
    key = 'key'
    nodes_list = ring.get_nodes(key, 2)
    assert len(nodes_list) == 2
    assert all(n in nodes for n in nodes_list)
    # Requested count bigger than node count returns all nodes unique
    nodes_list_all = ring.get_nodes(key, 10)
    assert set(nodes_list_all) == set(nodes)


def test_get_nodes_order_is_consistent():
    nodes = ['node1', 'node2', 'node3']
    ring = HashRing(nodes)
    key = 'somekey'
    nodes_first = ring.get_nodes(key, 3)
    nodes_second = ring.get_nodes(key, 3)
    assert nodes_first == nodes_second


def test_repr_contains_class_name_and_nodes():
    nodes = ['node1', 'node2']
    ring = HashRing(nodes)
    repr_str = repr(ring)
    assert 'HashRing' in repr_str
    for node in nodes:
        assert node in repr_str


def test_iter_nodes():
    nodes = ['node1', 'node2', 'node3']
    ring = HashRing(nodes)
    iterated = list(iter(ring))
    assert set(iterated) == set(nodes)


def test_init_with_no_nodes_creates_empty_ring():
    ring = HashRing([])
    assert len(ring) == 0


def test_get_node_returns_node_when_only_one_node():
    ring = HashRing(['single'])
    for key in ['a', 'b', 'c', '']:
        assert ring.get_node(key) == 'single'


def test_get_nodes_returns_unique_nodes_when_requesting_multiple():
    nodes = ['node1', 'node2', 'node3']
    ring = HashRing(nodes)
    result = ring.get_nodes('anykey', 3)
    assert len(result) == 3
    assert len(set(result)) == 3
    # Requesting more than available returns all unique nodes
    result2 = ring.get_nodes('anykey', 5)
    assert set(result2) == set(nodes)


def test_add_and_remove_node_affect_get_node():
    nodes = ['node1', 'node2', 'node3']
    ring = HashRing(nodes)
    key = 'testkey'
    node_before = ring.get_node(key)
    ring.add_node('node4')
    node_after_add = ring.get_node(key)
    ring.remove_node(node_before)
    node_after_remove = ring.get_node(key)
    # After removal, old node must not be returned
    assert node_after_remove != node_before or len(ring) == 0
    assert node_after_add in ring.nodes
    assert node_after_remove in ring.nodes


def test_hashring_add_node_invalid_type_raises():
    ring = HashRing([])
    with pytest.raises(TypeError):
        ring.add_node(None)


def test_hashring_remove_node_invalid_type_raises():
    ring = HashRing(['node1'])
    with pytest.raises(KeyError):
        ring.remove_node('nonexistent')


def test_get_nodes_with_zero_or_negative_count_returns_empty_list():
    nodes = ['node1', 'node2']
    ring = HashRing(nodes)
    assert ring.get_nodes('key', 0) == []
    assert ring.get_nodes('key', -1) == []
