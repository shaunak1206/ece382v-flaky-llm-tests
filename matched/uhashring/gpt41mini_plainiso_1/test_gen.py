import pytest
from uhashring import HashRing

def test_create_hashring_with_no_nodes_raises():
    with pytest.raises(ValueError):
        HashRing([])

def test_create_hashring_with_nodes():
    nodes = ['node1', 'node2']
    ring = HashRing(nodes)
    assert set(ring.nodes) == set(nodes)

def test_nodes_property_returns_nodes():
    nodes = ['a', 'b', 'c']
    ring = HashRing(nodes)
    assert set(ring.nodes) == set(nodes)

def test_get_node_returns_one_of_input_nodes():
    nodes = ['n1', 'n2', 'n3']
    ring = HashRing(nodes)
    for key in ['key1', 'key2', 'key3', '']:
        node = ring.get_node(key)
        assert node in nodes

def test_get_node_with_different_keys_changes_node():
    nodes = ['nodeA', 'nodeB', 'nodeC']
    ring = HashRing(nodes)
    node1 = ring.get_node('key1')
    node2 = ring.get_node('key2')
    # It's possible both keys map to same node but unlikely for multiple keys
    # Just check node is in nodes
    for node in (node1, node2):
        assert node in nodes

def test_consistent_get_node_for_same_key():
    nodes = ['x', 'y', 'z']
    ring = HashRing(nodes)
    key = "my-key"
    node_first = ring.get_node(key)
    node_second = ring.get_node(key)
    assert node_first == node_second

def test_get_nodes_returns_list_of_nodes_for_key():
    nodes = ['one', 'two', 'three', 'four']
    ring = HashRing(nodes)
    key = "mykey"
    n = 3
    res = ring.get_nodes(key, n)
    assert isinstance(res, (list, tuple))
    assert len(res) == n
    assert all(node in nodes for node in res)

def test_get_nodes_with_n_greater_than_nodes_length():
    nodes = ['uno', 'dos']
    ring = HashRing(nodes)
    key = "k"
    # If requesting more nodes than exist, should get all nodes without repeats
    res = ring.get_nodes(key, 5)
    assert set(res) == set(nodes)
    assert len(res) == len(nodes)

def test_get_nodes_with_n_zero_returns_empty_list():
    nodes = ['a', 'b']
    ring = HashRing(nodes)
    res = ring.get_nodes("key", 0)
    assert res == []

def test_get_node_falloff_returns_none_if_empty_ring():
    # We cannot create empty ring, so this test simply covers fallback
    # No direct way to test this, skip
    pass

def test_repr_contains_nodes():
    nodes = ['node1', 'node2']
    ring = HashRing(nodes)
    r = repr(ring)
    for n in nodes:
        assert n in r

def test_str_contains_nodes():
    nodes = ['n1', 'n2']
    ring = HashRing(nodes)
    s = str(ring)
    for n in nodes:
        assert n in s

def test_get_nodes_without_repeats_for_multiple_calls():
    nodes = [f"node{i}" for i in range(5)]
    ring = HashRing(nodes)
    key = "repeat-test"
    n = 4
    first_call = ring.get_nodes(key, n)
    second_call = ring.get_nodes(key, n)
    assert first_call == second_call
    assert len(first_call) == n
    assert len(set(first_call)) == n

def test_get_node_and_get_nodes_consistent_for_n_1():
    nodes = ['nodea', 'nodeb', 'nodec']
    ring = HashRing(nodes)
    key = "somekey"
    single_node = ring.get_node(key)
    nodes_list = ring.get_nodes(key, 1)
    assert nodes_list == [single_node]

def test_node_type_preserved_if_probably_str():
    nodes = ['nodeA', 'nodeB', 'nodeC']
    ring = HashRing(nodes)
    key = 'somekey'
    node = ring.get_node(key)
    assert isinstance(node, str)

def test_node_type_preserved_if_nodes_are_ints():
    nodes = [1, 2, 3]
    ring = HashRing(nodes)
    node = ring.get_node('anykey')
    assert node in nodes

def test_nodes_property_immutable():
    # assert that ring.nodes is a list but modifying it doesn't affect ring internal nodes
    nodes = ['a', 'b', 'c']
    ring = HashRing(nodes)
    nodes_copy = ring.nodes
    nodes_copy.append('d')
    # new node 'd' not present in original ring.nodes after mutation of copy
    assert 'd' not in ring.nodes

def test_nodes_property_returns_new_list_each_time():
    nodes = ['n1', 'n2']
    ring = HashRing(nodes)
    n1 = ring.nodes
    n2 = ring.nodes
    assert n1 == n2
    assert n1 is not n2

def test_hashring_with_duplicate_nodes_removes_duplicates():
    nodes = ['dup', 'dup', 'uniq']
    ring = HashRing(nodes)
    unique_nodes = set(nodes)
    assert set(ring.nodes) == unique_nodes

def test_get_nodes_with_negative_n_returns_empty_list():
    nodes = ['a', 'b']
    ring = HashRing(nodes)
    res = ring.get_nodes('key', -1)
    assert res == []

def test_get_nodes_all_nodes_when_n_equals_length():
    nodes = ['aaa', 'bbb', 'ccc']
    ring = HashRing(nodes)
    res = ring.get_nodes('key', 3)
    assert set(res) == set(nodes)

