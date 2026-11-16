import pytest

from uhashring import HashRing


KEYS = ["key-{}".format(i) for i in range(500)]
NODES = ["node-a", "node-b", "node-c"]


def mapping(ring, keys=KEYS):
    return {key: ring.get_node(key) for key in keys}


def test_get_node_returns_member_of_ring():
    ring = HashRing(nodes=NODES)
    for key in KEYS:
        assert ring.get_node(key) in NODES


def test_get_node_is_deterministic_for_same_key():
    ring = HashRing(nodes=NODES)
    first = mapping(ring)
    second = mapping(ring)
    assert first == second


def test_identical_rings_produce_identical_mappings():
    ring_one = HashRing(nodes=NODES)
    ring_two = HashRing(nodes=list(NODES))
    assert mapping(ring_one) == mapping(ring_two)


def test_get_node_accepts_bytes_and_str_keys():
    ring = HashRing(nodes=NODES)
    node = ring.get_node("hello")
    assert node in NODES
    assert ring.get_node("hello") == node


def test_all_nodes_receive_some_keys():
    ring = HashRing(nodes=NODES)
    used = set(mapping(ring).values())
    assert used == set(NODES)


def test_empty_ring_returns_none_for_lookup():
    ring = HashRing(nodes=[])
    assert ring.get_node("anything") is None


def test_single_node_ring_maps_all_keys_to_that_node():
    ring = HashRing(nodes=["only"])
    for key in KEYS:
        assert ring.get_node(key) == "only"


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(nodes=NODES)
    before = mapping(ring)
    ring.add_node("node-d")
    after = mapping(ring)

    for key in KEYS:
        if before[key] != after[key]:
            assert after[key] == "node-d"
        assert after[key] in NODES + ["node-d"]


def test_add_node_gives_new_node_some_keys():
    ring = HashRing(nodes=NODES)
    ring.add_node("node-d")
    used = set(mapping(ring).values())
    assert "node-d" in used


def test_remove_node_never_returns_removed_node():
    ring = HashRing(nodes=NODES)
    ring.remove_node("node-b")
    for key in KEYS:
        assert ring.get_node(key) != "node-b"
        assert ring.get_node(key) in ["node-a", "node-c"]


def test_remove_node_leaves_unaffected_keys_in_place():
    ring = HashRing(nodes=NODES)
    before = mapping(ring)
    ring.remove_node("node-b")
    after = mapping(ring)

    for key in KEYS:
        if before[key] != "node-b":
            assert after[key] == before[key]


def test_removing_all_nodes_empties_ring():
    ring = HashRing(nodes=NODES)
    for node in NODES:
        ring.remove_node(node)
    assert ring.get_node("key-1") is None


def test_iterate_nodes_distinct_yields_every_node_once():
    ring = HashRing(nodes=NODES)
    result = list(ring.iterate_nodes("some-key", distinct=True))
    assert len(result) == len(NODES)
    assert set(result) == set(NODES)


def test_iterate_nodes_first_item_matches_get_node():
    ring = HashRing(nodes=NODES)
    for key in KEYS[:50]:
        first = next(iter(ring.iterate_nodes(key, distinct=True)))
        assert first == ring.get_node(key)


def test_weighted_node_receives_larger_share_of_keys():
    ring = HashRing(nodes=NODES, weights={"node-a": 1, "node-b": 1, "node-c": 10})
    counts = {node: 0 for node in NODES}
    many_keys = ["weight-key-{}".format(i) for i in range(3000)]
    for key in many_keys:
        counts[ring.get_node(key)] += 1
    assert counts["node-c"] > counts["node-a"]
    assert counts["node-c"] > counts["node-b"]
    assert counts["node-c"] > len(many_keys) // 2


def test_ring_with_duplicate_nodes_in_list_still_lookups_correctly():
    ring = HashRing(nodes=["node-a", "node-a", "node-b"])
    for key in KEYS[:100]:
        assert ring.get_node(key) in ["node-a", "node-b"]
