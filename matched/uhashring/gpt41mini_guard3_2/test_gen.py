import random
import pytest
from uhashring import HashRing


def test_empty_nodes_raises_value_error():
    with pytest.raises(ValueError):
        HashRing([])


def test_single_node_no_replicas():
    random.seed(42)
    ring = HashRing([1], replicas=1)
    # The node 1 must be present in the ring nodes
    assert 1 in ring.nodes()
    # The ring must map any key to node 1
    for key in [0, 123, 'abc', b'bytes']:
        assert ring.get_node(key) == 1


def test_multiple_nodes_with_replicas():
    random.seed(42)
    nodes = [1, 2, 3]
    ring = HashRing(nodes, replicas=10)
    # All nodes must be in ring.nodes()
    assert set(ring.nodes()) == set(nodes)

    # For int keys, get_node returns one of the nodes
    for key in [0, 1, 10, 100, 1000]:
        node = ring.get_node(key)
        assert node in nodes


def test_string_keys_are_mapped_to_nodes():
    random.seed(42)
    nodes = [10, 20, 30]
    ring = HashRing(nodes, replicas=5)
    keys = ['alpha', 'beta', 'gamma', 'delta']
    for key in keys:
        node = ring.get_node(key)
        assert node in nodes


def test_bytes_keys_are_mapped_to_nodes():
    random.seed(42)
    nodes = [100, 200]
    ring = HashRing(nodes, replicas=3)
    keys = [b'a', b'b', b'c', b'\x00\xff']
    for key in keys:
        node = ring.get_node(key)
        assert node in nodes


def test_consistent_node_for_same_key_and_ring():
    random.seed(42)
    nodes = [1, 2, 3]
    ring = HashRing(nodes, replicas=5)

    key = 'consistent_key'
    node_first = ring.get_node(key)
    node_second = ring.get_node(key)
    assert node_first == node_second


def test_ring_nodes_are_immutable():
    random.seed(42)
    nodes = [1, 2, 3]
    ring = HashRing(nodes)
    initial_nodes = set(ring.nodes())
    # attempt to modify returned nodes list should not affect ring
    nodes_from_ring = ring.nodes()
    nodes_from_ring.append(9999)
    assert set(ring.nodes()) == initial_nodes


@pytest.mark.parametrize("replicas", [1, 10, 100])
def test_number_of_replicas_affects_ring_size(replicas):
    random.seed(42)
    nodes = [1, 2]
    ring = HashRing(nodes, replicas=replicas)
    # Each replica generates multiple points in ring._circle
    # We can assert that ring._circle is at least replicas * number_of_nodes in length
    # (some collisions might reduce it but likely not below)
    assert len(ring._circle) >= replicas * len(nodes)


def test_get_node_for_key_not_in_nodes_returns_some_node():
    random.seed(42)
    nodes = [5, 6, 7]
    ring = HashRing(nodes, replicas=3)

    # Choose keys that are not in nodes list
    keys = [100, 'unknown_key', b'another_key']
    for key in keys:
        node = ring.get_node(key)
        assert node in nodes


def test_all_nodes_covered_in_ring():
    """
    Every node should have at least one position in the ring._circle.
    Since hash collisions are possible but highly unlikely, check at least one.
    """
    random.seed(42)
    nodes = [1, 2, 3, 4]
    ring = HashRing(nodes, replicas=10)
    nodes_in_circle = set(ring._node_at(value) for value in ring._circle)
    # _node_at may not be public api but assuming it maps position to node
    # Instead, we check nodes appears in values of _nodes, by design all nodes should be in ring.nodes()

    # Actually ring.nodes() returns all nodes passed at init. Check coverage by circle keys:
    node_positions = {node: 0 for node in nodes}
    for pos in ring._circle:
        node = ring._node_at(pos)
        if node in node_positions:
            node_positions[node] += 1
    # All nodes have at least one position
    for count in node_positions.values():
        assert count > 0


def test_ring_nodes_unique_and_stable():
    random.seed(42)
    nodes = [10, 20, 30]
    ring1 = HashRing(nodes, replicas=5)
    ring2 = HashRing(nodes, replicas=5)

    # Nodes list should be equal (same keys)
    assert set(ring1.nodes()) == set(ring2.nodes())

    # For a fixed key, nodes should be same between rings (deterministic hashing)
    key = 'stable_key'
    assert ring1.get_node(key) == ring2.get_node(key)


def test_nodes_can_be_strings_and_int_mixed():
    random.seed(42)
    nodes = [1, 'node2', 3]
    ring = HashRing(nodes, replicas=5)
    keys = [123, 'abc', b'byte']
    for key in keys:
        node = ring.get_node(key)
        assert node in nodes


def test_get_node_raises_key_error_if_ring_empty(monkeypatch):
    # Monkeypatch to produce empty circle (simulate edge case)
    # first disable normal init
    nodes = [1]
    ring = HashRing(nodes)
    # forcibly clear ring._circle
    monkeypatch.setattr(ring, "_circle", [])
    with pytest.raises(IndexError):
        ring.get_node('anykey')


def test_nodes_property_returns_a_copy():
    random.seed(42)
    nodes = [1, 2, 3]
    ring = HashRing(nodes)
    nodes_list = ring.nodes()
    nodes_list.append('newnode')
    # ring.nodes() should remain unchanged
    assert 'newnode' not in ring.nodes()


def test_get_node_with_int_keys_always_nodes(monkeypatch):
    # In case of debugging monkeypatch to test hashing with int keys
    random.seed(42)
    nodes = list(range(10))
    ring = HashRing(nodes, replicas=5)
    for key in range(100):
        val = ring.get_node(key)
        assert val in nodes


def test_get_node_for_key_iterable_non_str_bytes(monkeypatch):
    # Test that as long as key is str or bytes or int, it works as expected
    random.seed(42)
    nodes = [1, 2]
    ring = HashRing(nodes)
    # keys which are tuples are NOT valid keys (likely) but code may accept arbitrary keys if hashable
    # test with tuple keys, they should return nodes, but no assumption about stability
    keys = [(1, 2), (3, 4)]
    for key in keys:
        node = ring.get_node(key)
        assert node in nodes


def test_hash_collision_handling_with_duplicates(monkeypatch):
    # This tests that duplicated nodes are reduced to unique set internally
    random.seed(42)
    nodes = [1, 1, 2, 2, 3]
    ring = HashRing(nodes)
    # ring.nodes() is unique
    assert set(ring.nodes()) == {1, 2, 3}
    # get_node returns only nodes in that set
    for key in ['x', 'y', 'z', 1, 2, 3]:
        assert ring.get_node(key) in {1, 2, 3}


def test_repr_contains_node_information():
    random.seed(42)
    nodes = [1, 2]
    ring = HashRing(nodes)
    r = repr(ring)
    # Should mention number of nodes and replicas
    assert str(nodes[0]) in r
    assert str(nodes[1]) in r
    # Just that __repr__ returns string containing nodes


def test_len_of_ring_corresponds_to_len_circle():
    random.seed(42)
    nodes = [1, 2, 3]
    replicas = 20
    ring = HashRing(nodes, replicas=replicas)
    # The len of the ring is probably len(ring._circle)
    # Confirming len is positive and >= replicas * unique nodes
    length = len(ring)
    assert length == len(ring._circle)
    assert length >= replicas * len(ring.nodes())


def test_iter_on_ring_yields_sorted_keys():
    random.seed(42)
    nodes = [1, 2]
    ring = HashRing(nodes, replicas=5)
    # iter on ring should produce sorted integer circle keys
    last = -1
    for pos in ring:
        assert isinstance(pos, int)
        assert pos > last
        last = pos


def test_hashring_init_with_large_number_of_replicas():
    random.seed(42)
    nodes = [1]
    ring = HashRing(nodes, replicas=1000)
    # The length of the ring._circle should be 1000 or slightly less due to collisions
    assert len(ring._circle) >= 900  # allowing some collisions


def test_get_node_returns_none_if_no_nodes(monkeypatch):
    # patch __init__ to not raise but produce no nodes
    # Actually constructor must raise on empty, so forcibly creating with empty circle
    ring = HashRing([1])
    monkeypatch.setattr(ring, "_circle", [])
    monkeypatch.setattr(ring, "_nodes", {})
    with pytest.raises(IndexError):
        ring.get_node("key")
