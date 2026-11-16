import pytest

from uhashring import HashRing
from uhashring.ring import HashRing as RingHashRing


NODES = ["server-a", "server-b", "server-c"]


def _keys(n=300):
    return [f"key-{i}" for i in range(n)]


def _mapping(ring, keys):
    return {k: ring.get_node(k) for k in keys}


def test_package_exports_hashring_class():
    assert HashRing is RingHashRing


def test_init_with_list_of_nodes_routes_every_key_to_a_member():
    ring = HashRing(nodes=NODES)
    for key in _keys(100):
        assert ring.get_node(key) in NODES


def test_get_node_is_deterministic_for_same_key():
    ring = HashRing(nodes=NODES)
    key = "stable-key"
    first = ring.get_node(key)
    for _ in range(50):
        assert ring.get_node(key) == first


def test_same_nodes_produce_same_mapping_across_instances():
    ring1 = HashRing(nodes=NODES)
    ring2 = HashRing(nodes=list(NODES))
    keys = _keys(200)
    assert _mapping(ring1, keys) == _mapping(ring2, keys)


def test_empty_ring_returns_none_for_lookup():
    ring = HashRing()
    assert ring.get_node("anything") is None


def test_add_node_to_empty_ring_makes_it_the_target_for_all_keys():
    ring = HashRing()
    ring.add_node("only-node")
    for key in _keys(50):
        assert ring.get_node(key) == "only-node"


def test_single_node_ring_receives_all_keys():
    ring = HashRing(nodes=["solo"])
    assert all(ring.get_node(k) == "solo" for k in _keys(100))


def test_keys_are_distributed_across_all_nodes():
    ring = HashRing(nodes=NODES)
    seen = set(_mapping(ring, _keys(500)).values())
    assert seen == set(NODES)


def test_add_node_makes_new_node_receive_some_keys():
    ring = HashRing(nodes=NODES)
    ring.add_node("server-d")
    seen = set(_mapping(ring, _keys(500)).values())
    assert "server-d" in seen


def test_add_node_only_moves_keys_onto_the_new_node():
    ring = HashRing(nodes=NODES)
    keys = _keys(500)
    before = _mapping(ring, keys)

    ring.add_node("server-d")
    after = _mapping(ring, keys)

    for key in keys:
        if before[key] != after[key]:
            assert after[key] == "server-d"


def test_remove_node_is_never_returned_afterwards():
    ring = HashRing(nodes=NODES)
    ring.remove_node("server-b")
    for key in _keys(300):
        assert ring.get_node(key) != "server-b"
        assert ring.get_node(key) in {"server-a", "server-c"}


def test_remove_node_keeps_mapping_for_keys_on_other_nodes():
    ring = HashRing(nodes=NODES)
    keys = _keys(500)
    before = _mapping(ring, keys)

    ring.remove_node("server-c")
    after = _mapping(ring, keys)

    for key in keys:
        if before[key] != "server-c":
            assert after[key] == before[key]


def test_add_then_remove_same_node_restores_original_mapping():
    ring = HashRing(nodes=NODES)
    keys = _keys(300)
    before = _mapping(ring, keys)

    ring.add_node("temp-node")
    ring.remove_node("temp-node")

    assert _mapping(ring, keys) == before


def test_removing_all_nodes_leaves_empty_ring():
    ring = HashRing(nodes=["x", "y"])
    ring.remove_node("x")
    ring.remove_node("y")
    assert ring.get_node("key") is None


def test_get_node_pos_agrees_with_get_node():
    ring = HashRing(nodes=NODES)
    for key in _keys(100):
        node, pos = ring.get_node_pos(key)
        assert node == ring.get_node(key)
        assert isinstance(pos, int)


def test_different_node_sets_do_not_share_a_fixed_answer():
    ring_small = HashRing(nodes=["alpha", "beta"])
    ring_large = HashRing(nodes=["alpha", "beta", "gamma", "delta"])
    keys = _keys(300)
    small = set(_mapping(ring_small, keys).values())
    large = set(_mapping(ring_large, keys).values())
    assert small <= {"alpha", "beta"}
    assert large <= {"alpha", "beta", "gamma", "delta"}
    assert large != small
