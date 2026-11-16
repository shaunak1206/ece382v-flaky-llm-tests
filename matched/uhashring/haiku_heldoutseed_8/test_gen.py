import pytest

from uhashring import HashRing


KEYS = ["key-{}".format(i) for i in range(2000)]


def _mapping(ring, keys=KEYS):
    return {key: ring.get_node(key) for key in keys}


def test_get_node_returns_member_of_ring():
    nodes = ["node-a", "node-b", "node-c"]
    ring = HashRing(nodes=nodes)
    for key in KEYS[:200]:
        assert ring.get_node(key) in nodes


def test_same_key_maps_to_same_node_repeatedly():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    first = _mapping(ring)
    second = _mapping(ring)
    assert first == second


def test_identical_rings_produce_identical_mappings():
    ring1 = HashRing(nodes=["node-a", "node-b", "node-c"])
    ring2 = HashRing(nodes=["node-a", "node-b", "node-c"])
    assert _mapping(ring1) == _mapping(ring2)


def test_construction_order_does_not_change_mapping():
    ring1 = HashRing(nodes=["node-a", "node-b", "node-c"])
    ring2 = HashRing(nodes=["node-c", "node-a", "node-b"])
    assert _mapping(ring1) == _mapping(ring2)


def test_empty_ring_returns_none():
    ring = HashRing(nodes=[])
    assert ring.get_node("anything") is None


def test_single_node_receives_all_keys():
    ring = HashRing(nodes=["only-node"])
    for key in KEYS[:200]:
        assert ring.get_node(key) == "only-node"


def test_all_nodes_receive_some_keys():
    nodes = ["node-a", "node-b", "node-c"]
    ring = HashRing(nodes=nodes)
    assigned = set(_mapping(ring).values())
    assert assigned == set(nodes)


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    before = _mapping(ring)

    ring.add_node("node-d")
    after = _mapping(ring)

    for key in KEYS:
        if before[key] != after[key]:
            assert after[key] == "node-d"


def test_add_node_takes_some_keys():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    before = _mapping(ring)

    ring.add_node("node-d")
    after = _mapping(ring)

    assert "node-d" in set(after.values())
    assert any(before[k] != after[k] for k in KEYS)


def test_remove_node_only_moves_keys_from_removed_node():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    before = _mapping(ring)

    ring.remove_node("node-b")
    after = _mapping(ring)

    for key in KEYS:
        if before[key] != "node-b":
            assert after[key] == before[key]
        else:
            assert after[key] != "node-b"


def test_removed_node_no_longer_returned():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    ring.remove_node("node-a")
    for key in KEYS[:500]:
        assert ring.get_node(key) in {"node-b", "node-c"}


def test_removing_all_nodes_yields_none():
    ring = HashRing(nodes=["node-a", "node-b"])
    ring.remove_node("node-a")
    ring.remove_node("node-b")
    assert ring.get_node("some-key") is None


def test_iterate_nodes_first_item_matches_get_node():
    ring = HashRing(nodes=["node-a", "node-b", "node-c"])
    for key in KEYS[:200]:
        first = next(iter(ring.iterate_nodes(key, distinct=True)))
        assert first == ring.get_node(key)


def test_iterate_nodes_distinct_yields_each_node_once():
    nodes = ["node-a", "node-b", "node-c", "node-d"]
    ring = HashRing(nodes=nodes)
    for key in KEYS[:100]:
        yielded = list(ring.iterate_nodes(key, distinct=True))
        assert len(yielded) == len(set(yielded)) == len(nodes)
        assert set(yielded) == set(nodes)


def test_weighted_node_receives_more_keys_than_lighter_node():
    ring = HashRing(
        nodes=["light", "heavy"],
        weights={"light": 1, "heavy": 5},
    )
    counts = {"light": 0, "heavy": 0}
    for node in _mapping(ring).values():
        counts[node] += 1
    assert counts["heavy"] > counts["light"]


def test_unicode_keys_are_supported():
    ring = HashRing(nodes=["node-a", "node-b"])
    keys = [u"clé-{}".format(i) for i in range(50)] + [u"\u4e2d\u6587-{}".format(i) for i in range(50)]
    for key in keys:
        assert ring.get_node(key) in {"node-a", "node-b"}
        assert ring.get_node(key) == ring.get_node(key)


@pytest.mark.parametrize("node_count", [2, 5, 10])
def test_mapping_stable_for_varied_ring_sizes(node_count):
    nodes = ["node-{}".format(i) for i in range(node_count)]
    ring1 = HashRing(nodes=nodes)
    ring2 = HashRing(nodes=list(reversed(nodes)))
    assert _mapping(ring1) == _mapping(ring2)
