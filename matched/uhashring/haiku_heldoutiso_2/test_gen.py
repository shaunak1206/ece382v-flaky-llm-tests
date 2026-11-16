import collections

import pytest

from uhashring import HashRing


KEYS = ["key-{}".format(i) for i in range(2000)]


def _assignments(ring, keys=KEYS):
    return {key: ring.get_node(key) for key in keys}


def test_exports_hashring_and_monkey():
    import uhashring

    assert "HashRing" in uhashring.__all__
    assert "monkey" in uhashring.__all__
    assert uhashring.HashRing is HashRing


def test_empty_ring_returns_none_for_lookup():
    ring = HashRing()
    assert ring.get_node("anything") is None


def test_constructor_with_nodes_routes_keys_to_members():
    nodes = ["node-a", "node-b", "node-c"]
    ring = HashRing(nodes=nodes)
    for key in KEYS[:200]:
        assert ring.get_node(key) in nodes


def test_single_node_ring_routes_all_keys_to_that_node():
    ring = HashRing(nodes=["only"])
    for key in KEYS[:200]:
        assert ring.get_node(key) == "only"


def test_lookup_is_deterministic_across_instances():
    nodes = ["n1", "n2", "n3", "n4"]
    ring1 = HashRing(nodes=nodes)
    ring2 = HashRing(nodes=nodes)
    assert _assignments(ring1) == _assignments(ring2)


def test_same_key_always_maps_to_same_node():
    ring = HashRing(nodes=["x", "y", "z"])
    first = ring.get_node("stable-key")
    for _ in range(50):
        assert ring.get_node("stable-key") == first


def test_add_node_only_takes_keys_it_now_owns():
    ring = HashRing(nodes=["a", "b", "c"])
    before = _assignments(ring)
    ring.add_node("d")
    after = _assignments(ring)

    moved = [k for k in KEYS if before[k] != after[k]]
    assert moved, "adding a node should claim at least some keys"
    for key in moved:
        assert after[key] == "d"


def test_add_node_moves_only_a_fraction_of_keys():
    ring = HashRing(nodes=["a", "b", "c", "d"])
    before = _assignments(ring)
    ring.add_node("e")
    after = _assignments(ring)

    moved = sum(1 for k in KEYS if before[k] != after[k])
    # Ideal movement is about 1/5 of keys; allow generous slack.
    assert moved < len(KEYS) * 0.5


def test_remove_node_reassigns_its_keys_only():
    ring = HashRing(nodes=["a", "b", "c", "d"])
    before = _assignments(ring)
    ring.remove_node("b")
    after = _assignments(ring)

    assert "b" not in set(after.values())
    for key in KEYS:
        if before[key] != "b":
            assert after[key] == before[key]


def test_remove_node_then_lookup_never_returns_removed_node():
    ring = HashRing(nodes=["a", "b"])
    ring.remove_node("a")
    for key in KEYS[:200]:
        assert ring.get_node(key) == "b"


def test_removing_all_nodes_leaves_ring_empty():
    ring = HashRing(nodes=["a", "b"])
    ring.remove_node("a")
    ring.remove_node("b")
    assert ring.get_node("key") is None


def test_add_nodes_matches_repeated_add_node():
    names = ["p", "q", "r"]
    ring_bulk = HashRing()
    ring_bulk.add_nodes(names)

    ring_single = HashRing()
    for name in names:
        ring_single.add_node(name)

    assert _assignments(ring_bulk) == _assignments(ring_single)


def test_higher_weight_receives_more_keys():
    ring = HashRing(nodes=["light", "heavy"], weights={"light": 1, "heavy": 4})
    counts = collections.Counter(ring.get_node(k) for k in KEYS)
    assert counts["heavy"] > counts["light"]


def test_equal_weights_distribute_keys_across_all_nodes():
    names = ["w1", "w2", "w3"]
    ring = HashRing(nodes=names)
    counts = collections.Counter(ring.get_node(k) for k in KEYS)
    assert set(counts) == set(names)
    for name in names:
        assert counts[name] > len(KEYS) * 0.1


def test_iterate_nodes_starts_with_primary_node():
    ring = HashRing(nodes=["a", "b", "c"])
    for key in KEYS[:100]:
        iterator = ring.iterate_nodes(key)
        assert next(iterator) == ring.get_node(key)


def test_iterate_nodes_distinct_yields_unique_members():
    names = ["a", "b", "c", "d"]
    ring = HashRing(nodes=names)
    result = list(ring.iterate_nodes("some-key", distinct=True))
    assert len(result) == len(set(result))
    assert set(result) == set(names)


def test_get_node_pos_returns_member_node_for_key():
    nodes = ["alpha", "beta"]
    ring = HashRing(nodes=nodes)
    for key in KEYS[:100]:
        node, _pos = ring.get_node_pos(key)
        assert node in nodes
        assert node == ring.get_node(key)
