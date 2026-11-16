import uhashring
from uhashring import HashRing
from uhashring.ring import HashRing as RingHashRing


KEYS = ["key-{}".format(i) for i in range(2000)]


def _mapping(ring, keys=KEYS):
    return {key: ring.get_node(key) for key in keys}


def test_package_exports_hashring_class():
    assert "HashRing" in uhashring.__all__
    assert HashRing is RingHashRing


def test_package_all_contains_monkey():
    assert "monkey" in uhashring.__all__


def test_hashring_is_a_class():
    assert isinstance(HashRing, type)


def test_create_ring_with_nodes_does_not_raise():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    assert ring is not None


def test_get_node_returns_one_of_the_nodes():
    nodes = ["node-a", "node-b", "node-c"]
    ring = HashRing(nodes=nodes)
    for key in KEYS[:200]:
        assert ring.get_node(key) in nodes


def test_get_node_is_deterministic_for_same_key():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    for key in KEYS[:200]:
        first = ring.get_node(key)
        assert all(ring.get_node(key) == first for _ in range(5))


def test_same_nodes_produce_same_mapping_across_rings():
    nodes = ["node-a", "node-b", "node-c", "node-d"]
    ring1 = HashRing(nodes=nodes)
    ring2 = HashRing(nodes=list(nodes))
    assert _mapping(ring1) == _mapping(ring2)


def test_single_node_ring_maps_all_keys_to_that_node():
    ring = HashRing(nodes=["only-node"])
    for key in KEYS[:200]:
        assert ring.get_node(key) == "only-node"


def test_empty_ring_returns_none():
    ring = HashRing(nodes=[])
    assert ring.get_node("any-key") is None


def test_all_nodes_receive_some_keys():
    nodes = ["node-a", "node-b", "node-c", "node-d"]
    ring = HashRing(nodes=nodes)
    counts = {node: 0 for node in nodes}
    for key in KEYS:
        counts[ring.get_node(key)] += 1
    for node in nodes:
        assert counts[node] > 0, "node {} received no keys".format(node)


def test_distribution_is_not_extremely_skewed():
    nodes = ["node-a", "node-b", "node-c", "node-d"]
    ring = HashRing(nodes=nodes)
    counts = {node: 0 for node in nodes}
    for key in KEYS:
        counts[ring.get_node(key)] += 1
    expected = len(KEYS) / float(len(nodes))
    for node in nodes:
        assert counts[node] < expected * 2


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    before = _mapping(ring)
    ring.add_node("node-d")
    after = _mapping(ring)
    for key in KEYS:
        if after[key] != before[key]:
            assert after[key] == "node-d"


def test_add_node_makes_new_node_receive_keys():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    ring.add_node("node-d")
    assert any(ring.get_node(key) == "node-d" for key in KEYS)


def test_remove_node_no_longer_returned():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    ring.remove_node("node-b")
    for key in KEYS:
        assert ring.get_node(key) != "node-b"


def test_remove_node_only_moves_keys_from_removed_node():
    ring = HashRing(nodes=["node-a", "node-b", "node-c", "node-d"])
    before = _mapping(ring)
    ring.remove_node("node-c")
    after = _mapping(ring)
    for key in KEYS:
        if before[key] != "node-c":
            assert after[key] == before[key]


def test_removing_all_nodes_yields_none():
    ring = HashRing(nodes=["node-a", "node-b"])
    ring.remove_node("node-a")
    ring.remove_node("node-b")
    assert ring.get_node("some-key") is None


def test_add_then_remove_restores_original_mapping():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    before = _mapping(ring)
    ring.add_node("node-x")
    ring.remove_node("node-x")
    assert _mapping(ring) == before
