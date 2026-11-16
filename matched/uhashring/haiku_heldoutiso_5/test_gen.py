import pytest

import uhashring
from uhashring import HashRing


NODES = ["node-a", "node-b", "node-c"]


def make_keys(count):
    return ["key-{}".format(i) for i in range(count)]


def mapping(ring, keys):
    return {key: ring.get_node(key) for key in keys}


def test_package_exports_hashring_class():
    assert uhashring.HashRing is HashRing


def test_all_contains_hashring():
    assert "HashRing" in uhashring.__all__


def test_empty_ring_returns_none_for_any_key():
    ring = HashRing()
    assert ring.get_node("anything") is None


def test_single_node_receives_every_key():
    ring = HashRing(nodes=["only"])
    for key in make_keys(100):
        assert ring.get_node(key) == "only"


def test_get_node_returns_a_member_node():
    ring = HashRing(nodes=NODES)
    for key in make_keys(200):
        assert ring.get_node(key) in NODES


def test_get_node_is_stable_for_repeated_calls():
    ring = HashRing(nodes=NODES)
    key = "stable-key"
    first = ring.get_node(key)
    for _ in range(50):
        assert ring.get_node(key) == first


def test_mapping_is_deterministic_across_instances():
    keys = make_keys(300)
    ring1 = HashRing(nodes=NODES)
    ring2 = HashRing(nodes=NODES)
    assert mapping(ring1, keys) == mapping(ring2, keys)


def test_node_order_does_not_change_mapping():
    keys = make_keys(300)
    ring1 = HashRing(nodes=NODES)
    ring2 = HashRing(nodes=list(reversed(NODES)))
    assert mapping(ring1, keys) == mapping(ring2, keys)


def test_constructor_matches_incremental_add_node():
    keys = make_keys(300)
    built = HashRing(nodes=NODES)
    incremental = HashRing()
    for node in NODES:
        incremental.add_node(node)
    assert mapping(built, keys) == mapping(incremental, keys)


def test_add_node_makes_new_node_reachable():
    ring = HashRing(nodes=NODES)
    ring.add_node("node-new")
    owners = {ring.get_node(key) for key in make_keys(2000)}
    assert "node-new" in owners


def test_remove_node_is_never_returned_afterwards():
    ring = HashRing(nodes=NODES)
    ring.remove_node("node-b")
    for key in make_keys(500):
        assert ring.get_node(key) != "node-b"


def test_remove_node_only_moves_keys_owned_by_removed_node():
    keys = make_keys(1000)
    ring = HashRing(nodes=NODES)
    before = mapping(ring, keys)
    ring.remove_node("node-b")
    after = mapping(ring, keys)
    for key in keys:
        if before[key] != "node-b":
            assert after[key] == before[key]


def test_add_node_only_moves_keys_to_the_new_node():
    keys = make_keys(1000)
    ring = HashRing(nodes=NODES)
    before = mapping(ring, keys)
    ring.add_node("node-d")
    after = mapping(ring, keys)
    for key in keys:
        if after[key] != before[key]:
            assert after[key] == "node-d"


def test_removing_all_nodes_empties_ring():
    ring = HashRing(nodes=NODES)
    for node in NODES:
        ring.remove_node(node)
    assert ring.get_node("key") is None


def test_equal_weights_give_roughly_balanced_distribution():
    ring = HashRing(nodes=NODES)
    counts = {node: 0 for node in NODES}
    total = 3000
    for key in make_keys(total):
        counts[ring.get_node(key)] += 1
    for node in NODES:
        assert total * 0.15 <= counts[node] <= total * 0.50


def test_higher_weight_node_receives_more_keys():
    ring = HashRing(nodes=["light", "heavy"], weights={"light": 1, "heavy": 10})
    counts = {"light": 0, "heavy": 0}
    total = 3000
    for key in make_keys(total):
        counts[ring.get_node(key)] += 1
    assert counts["heavy"] > counts["light"]


def test_custom_vnodes_still_maps_to_members():
    ring = HashRing(nodes=NODES, vnodes=10)
    for key in make_keys(200):
        assert ring.get_node(key) in NODES


def test_different_key_sets_map_independently_of_each_other():
    ring = HashRing(nodes=NODES)
    keys_a = make_keys(100)
    keys_b = ["other-{}".format(i) for i in range(100)]
    results_a_first = mapping(ring, keys_a)
    _ = mapping(ring, keys_b)
    assert mapping(ring, keys_a) == results_a_first
