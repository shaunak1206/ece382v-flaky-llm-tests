import pytest

from uhashring import HashRing


NODES = ["cache-a:11211", "cache-b:11211", "cache-c:11211", "cache-d:11211"]


def make_keys(count=2000, prefix="key"):
    return [f"{prefix}-{i}" for i in range(count)]


def mapping_for(ring, keys):
    return {key: ring.get_node(key) for key in keys}


def test_constructor_with_nodes_returns_member_for_key():
    ring = HashRing(nodes=NODES)
    assert ring.get_node("user:42") in NODES


def test_get_node_is_deterministic_for_same_key():
    ring = HashRing(nodes=NODES)
    first = ring.get_node("session:abc")
    for _ in range(50):
        assert ring.get_node("session:abc") == first


def test_identical_rings_produce_identical_mappings():
    keys = make_keys(500)
    ring1 = HashRing(nodes=NODES)
    ring2 = HashRing(nodes=list(NODES))
    assert mapping_for(ring1, keys) == mapping_for(ring2, keys)


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_node("anything") is None


def test_single_node_ring_maps_every_key_to_that_node():
    ring = HashRing(nodes=["only-node"])
    for key in make_keys(200):
        assert ring.get_node(key) == "only-node"


def test_all_keys_map_to_known_nodes():
    ring = HashRing(nodes=NODES)
    for key in make_keys(1000):
        assert ring.get_node(key) in NODES


def test_keys_are_distributed_across_all_nodes():
    ring = HashRing(nodes=NODES)
    counts = {node: 0 for node in NODES}
    for key in make_keys(4000):
        counts[ring.get_node(key)] += 1
    for node, count in counts.items():
        assert count > 0, f"{node} received no keys"


def test_distribution_is_reasonably_balanced():
    total = 4000
    ring = HashRing(nodes=NODES)
    counts = {node: 0 for node in NODES}
    for key in make_keys(total):
        counts[ring.get_node(key)] += 1
    ideal = total / len(NODES)
    for node, count in counts.items():
        assert 0.4 * ideal <= count <= 1.6 * ideal, f"{node} got {count} keys"


def test_add_node_makes_new_node_receive_some_keys():
    keys = make_keys(2000)
    ring = HashRing(nodes=NODES[:3])
    ring.add_node(NODES[3])
    owners = {ring.get_node(key) for key in keys}
    assert NODES[3] in owners


def test_add_node_only_moves_keys_to_new_node():
    keys = make_keys(2000)
    ring = HashRing(nodes=NODES[:3])
    before = mapping_for(ring, keys)
    ring.add_node(NODES[3])
    after = mapping_for(ring, keys)
    for key in keys:
        if after[key] != before[key]:
            assert after[key] == NODES[3]


def test_remove_node_no_longer_returns_removed_node():
    keys = make_keys(2000)
    ring = HashRing(nodes=NODES)
    ring.remove_node(NODES[0])
    for key in keys:
        assert ring.get_node(key) != NODES[0]


def test_remove_node_only_moves_keys_that_belonged_to_removed_node():
    keys = make_keys(2000)
    ring = HashRing(nodes=NODES)
    before = mapping_for(ring, keys)
    ring.remove_node(NODES[1])
    after = mapping_for(ring, keys)
    for key in keys:
        if before[key] != NODES[1]:
            assert after[key] == before[key]


def test_add_then_remove_same_node_restores_original_mapping():
    keys = make_keys(1000)
    ring = HashRing(nodes=NODES[:3])
    before = mapping_for(ring, keys)
    ring.add_node(NODES[3])
    ring.remove_node(NODES[3])
    assert mapping_for(ring, keys) == before


def test_removing_all_nodes_leaves_empty_ring():
    ring = HashRing(nodes=["x:1", "y:1"])
    ring.remove_node("x:1")
    ring.remove_node("y:1")
    assert ring.get_node("any-key") is None


def test_unicode_keys_are_supported():
    ring = HashRing(nodes=NODES)
    key = "ключ-🔑-键"
    node = ring.get_node(key)
    assert node in NODES
    assert ring.get_node(key) == node


@pytest.mark.parametrize("key", ["", "a", "x" * 1000, "with spaces and:colons"])
def test_various_key_shapes_return_valid_node(key):
    ring = HashRing(nodes=NODES)
    assert ring.get_node(key) in NODES
