import random

import pytest

from uhashring import HashRing
import uhashring


NODES = ["cache-a:11211", "cache-b:11211", "cache-c:11211"]
KEYS = ["key-{}".format(i) for i in range(500)]


def _mapping(ring, keys):
    return {key: ring.get_node(key) for key in keys}


def test_package_exports_hashring_class():
    assert uhashring.HashRing is HashRing
    assert "HashRing" in uhashring.__all__


def test_construct_with_nodes_routes_keys_to_known_nodes():
    ring = HashRing(nodes=NODES)
    for key in KEYS[:100]:
        assert ring.get_node(key) in NODES


def test_construct_without_nodes_returns_none_for_lookup():
    ring = HashRing()
    assert ring.get_node("anything") is None


def test_get_node_is_deterministic_for_same_key():
    ring = HashRing(nodes=NODES)
    first = ring.get_node("user:42")
    for _ in range(20):
        assert ring.get_node("user:42") == first


def test_identical_rings_produce_identical_mappings():
    ring_one = HashRing(nodes=NODES)
    ring_two = HashRing(nodes=NODES)
    assert _mapping(ring_one, KEYS) == _mapping(ring_two, KEYS)


def test_node_order_does_not_change_mapping():
    ring_one = HashRing(nodes=NODES)
    shuffled = list(NODES)
    random.Random(1234).shuffle(shuffled)
    ring_two = HashRing(nodes=shuffled)
    assert _mapping(ring_one, KEYS) == _mapping(ring_two, KEYS)


def test_keys_are_spread_across_all_nodes():
    ring = HashRing(nodes=NODES)
    used = {ring.get_node(key) for key in KEYS}
    assert used == set(NODES)


def test_add_node_to_empty_ring_makes_it_the_target():
    ring = HashRing()
    ring.add_node("solo:11211")
    for key in KEYS[:50]:
        assert ring.get_node(key) == "solo:11211"


def test_add_node_only_moves_keys_to_the_new_node():
    ring = HashRing(nodes=NODES)
    before = _mapping(ring, KEYS)
    ring.add_node("cache-d:11211")
    after = _mapping(ring, KEYS)

    for key in KEYS:
        if after[key] != before[key]:
            assert after[key] == "cache-d:11211"


def test_add_node_moves_only_a_minority_of_keys():
    ring = HashRing(nodes=NODES)
    before = _mapping(ring, KEYS)
    ring.add_node("cache-d:11211")
    after = _mapping(ring, KEYS)

    moved = sum(1 for key in KEYS if after[key] != before[key])
    assert moved < len(KEYS) * 0.5


def test_remove_node_never_returned_afterwards():
    ring = HashRing(nodes=NODES)
    ring.remove_node("cache-b:11211")
    for key in KEYS:
        assert ring.get_node(key) != "cache-b:11211"


def test_remove_node_only_reassigns_its_own_keys():
    ring = HashRing(nodes=NODES)
    before = _mapping(ring, KEYS)
    ring.remove_node("cache-b:11211")
    after = _mapping(ring, KEYS)

    for key in KEYS:
        if before[key] != "cache-b:11211":
            assert after[key] == before[key]


def test_remove_all_nodes_leaves_ring_empty():
    ring = HashRing(nodes=NODES)
    for node in NODES:
        ring.remove_node(node)
    assert ring.get_node("any-key") is None


def test_add_then_remove_restores_original_mapping():
    ring = HashRing(nodes=NODES)
    before = _mapping(ring, KEYS)
    ring.add_node("cache-d:11211")
    ring.remove_node("cache-d:11211")
    assert _mapping(ring, KEYS) == before


def test_get_node_pos_agrees_with_get_node():
    ring = HashRing(nodes=NODES)
    for key in KEYS[:50]:
        node, _pos = ring.get_node_pos(key)
        assert node == ring.get_node(key)


def test_different_keys_do_not_all_collapse_to_one_node():
    ring = HashRing(nodes=NODES)
    assignments = [ring.get_node("k{}".format(i)) for i in range(200)]
    assert len(set(assignments)) > 1
    assert all(node is not None for node in assignments)


def test_add_duplicate_node_keeps_mapping_stable():
    ring = HashRing(nodes=NODES)
    before = _mapping(ring, KEYS)
    ring.add_node(NODES[0])
    assert _mapping(ring, KEYS) == before

