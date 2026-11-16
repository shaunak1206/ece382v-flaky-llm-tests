"""Unit tests for HashRing in uhashring (exposed via uhashring/__init__.py).

Notes:
- Keys are ints so results do not depend on builtin hash() of str/bytes.
- No random numbers are used, so no generators need seeding.
- Assertions check consistent-hashing properties rather than exact bucket
  assignments, so they do not depend on the hash function's specific output.
- Ring state is local to each test, so tests are independent of order.
"""

import uhashring
from uhashring import HashRing


NODES = ["cache-a", "cache-b", "cache-c", "cache-d"]
KEYS = list(range(1000))


def _mapping(ring, keys=KEYS):
    return {k: ring.get_node(k) for k in keys}


def test_package_exports_hashring_and_monkey():
    assert "HashRing" in uhashring.__all__
    assert "monkey" in uhashring.__all__


def test_package_hashring_is_ring_module_class():
    from uhashring.ring import HashRing as RingHashRing

    assert uhashring.HashRing is RingHashRing


def test_get_node_returns_one_of_the_ring_nodes():
    ring = HashRing(nodes=NODES)
    for key in KEYS:
        assert ring.get_node(key) in NODES


def test_get_node_is_stable_for_repeated_calls():
    ring = HashRing(nodes=NODES)
    first = _mapping(ring)
    second = _mapping(ring)
    assert first == second


def test_two_rings_with_same_nodes_map_keys_identically():
    ring_a = HashRing(nodes=NODES)
    ring_b = HashRing(nodes=NODES)
    assert _mapping(ring_a) == _mapping(ring_b)


def test_node_insertion_order_does_not_change_mapping():
    ring_a = HashRing(nodes=NODES)
    ring_b = HashRing(nodes=list(reversed(NODES)))
    assert _mapping(ring_a) == _mapping(ring_b)


def test_every_node_receives_some_keys():
    ring = HashRing(nodes=NODES)
    owners = set(_mapping(ring).values())
    assert owners == set(NODES)


def test_empty_ring_returns_none_for_lookup():
    ring = HashRing(nodes=[])
    assert ring.get_node(42) is None


def test_single_node_ring_maps_every_key_to_that_node():
    ring = HashRing(nodes=["only"])
    assert all(ring.get_node(k) == "only" for k in KEYS)


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(nodes=NODES)
    before = _mapping(ring)

    ring.add_node("cache-new")
    after = _mapping(ring)

    for key in KEYS:
        if after[key] != before[key]:
            assert after[key] == "cache-new"


def test_add_node_makes_new_node_own_some_keys():
    ring = HashRing(nodes=NODES)
    ring.add_node("cache-new")
    owners = set(_mapping(ring).values())
    assert "cache-new" in owners


def test_remove_node_never_returns_removed_node():
    ring = HashRing(nodes=NODES)
    ring.remove_node("cache-b")
    owners = set(_mapping(ring).values())
    assert "cache-b" not in owners
    assert owners <= set(NODES) - {"cache-b"}


def test_remove_node_only_moves_keys_previously_owned_by_removed_node():
    ring = HashRing(nodes=NODES)
    before = _mapping(ring)

    ring.remove_node("cache-b")
    after = _mapping(ring)

    for key in KEYS:
        if before[key] != "cache-b":
            assert after[key] == before[key]


def test_add_then_remove_same_node_restores_original_mapping():
    ring = HashRing(nodes=NODES)
    before = _mapping(ring)

    ring.add_node("cache-temp")
    ring.remove_node("cache-temp")

    assert _mapping(ring) == before


def test_iterate_nodes_yields_each_node_once_when_distinct():
    ring = HashRing(nodes=NODES)
    for key in KEYS[:50]:
        visited = list(ring.iterate_nodes(key, distinct=True))
        assert sorted(visited) == sorted(NODES)
        assert len(visited) == len(set(visited))


def test_iterate_nodes_first_item_matches_get_node():
    ring = HashRing(nodes=NODES)
    for key in KEYS[:100]:
        first = next(iter(ring.iterate_nodes(key, distinct=True)))
        assert first == ring.get_node(key)


def test_weighted_node_receives_more_keys_than_lighter_node():
    ring = HashRing(nodes={"light": 1, "heavy": 10})
    counts = {"light": 0, "heavy": 0}
    for key in KEYS:
        counts[ring.get_node(key)] += 1
    assert counts["heavy"] > counts["light"]
    assert counts["light"] > 0
</br>
