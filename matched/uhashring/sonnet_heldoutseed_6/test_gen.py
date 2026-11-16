import pytest

from uhashring import HashRing


NODES = ["node1", "node2", "node3"]
KEYS = ["key-%d" % i for i in range(500)]


def mapping(ring, keys=KEYS):
    return {k: ring.get_node(k) for k in keys}


def test_import_exposes_hashring():
    import uhashring

    assert uhashring.HashRing is HashRing
    assert "HashRing" in uhashring.__all__


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_nodes() == []
    assert ring.get_node("anything") is None


def test_get_nodes_lists_all_nodes():
    ring = HashRing(nodes=NODES)
    assert sorted(ring.get_nodes()) == sorted(NODES)


def test_get_node_returns_a_member_node():
    ring = HashRing(nodes=NODES)
    for key in KEYS:
        assert ring.get_node(key) in NODES


def test_get_node_is_deterministic():
    ring = HashRing(nodes=NODES)
    first = mapping(ring)
    second = mapping(ring)
    assert first == second


def test_two_rings_with_same_nodes_agree():
    ring_a = HashRing(nodes=NODES)
    ring_b = HashRing(nodes=list(reversed(NODES)))
    assert mapping(ring_a) == mapping(ring_b)


def test_single_node_gets_all_keys():
    ring = HashRing(nodes=["only"])
    assert set(mapping(ring).values()) == {"only"}


def test_keys_are_distributed_across_all_nodes():
    ring = HashRing(nodes=NODES)
    used = set(mapping(ring).values())
    assert used == set(NODES)


def test_remove_node_only_remaps_its_keys():
    ring = HashRing(nodes=NODES)
    before = mapping(ring)
    ring.remove_node("node2")
    assert sorted(ring.get_nodes()) == ["node1", "node3"]
    after = mapping(ring)
    for key in KEYS:
        assert after[key] != "node2"
        if before[key] != "node2":
            assert after[key] == before[key]


def test_remove_unknown_node_raises():
    ring = HashRing(nodes=NODES)
    with pytest.raises(KeyError):
        ring.remove_node("does-not-exist")


def test_add_node_only_steals_keys_to_new_node():
    ring = HashRing(nodes=["node1", "node2"])
    before = mapping(ring)
    ring.add_node("node3")
    assert "node3" in ring.get_nodes()
    after = mapping(ring)
    moved_to_new = 0
    for key in KEYS:
        if after[key] != before[key]:
            assert after[key] == "node3"
            moved_to_new += 1
    assert moved_to_new > 0


def test_get_node_pos_returns_int_or_none():
    ring = HashRing(nodes=NODES)
    pos = ring.get_node_pos("some-key")
    assert isinstance(pos, int)
    assert HashRing().get_node_pos("some-key") is None


def test_iterate_nodes_distinct_yields_each_node_once():
    ring = HashRing(nodes=NODES)
    result = list(ring.iterate_nodes("some-key"))
    assert sorted(result) == sorted(NODES)
    assert result[0] == ring.get_node("some-key")


def test_dict_nodes_with_instances():
    inst = {"node1": object(), "node2": object()}
    ring = HashRing(
        nodes={
            "node1": {"instance": inst["node1"]},
            "node2": {"instance": inst["node2"]},
        }
    )
    for key in KEYS[:50]:
        name = ring.get_node(key)
        assert ring.get_node_instance(key) is inst[name]


def test_weight_influences_distribution():
    ring = HashRing(
        nodes={
            "heavy": {"weight": 10},
            "light": {"weight": 1},
        }
    )
    keys = ["k%d" % i for i in range(2000)]
    counts = {"heavy": 0, "light": 0}
    for key in keys:
        counts[ring.get_node(key)] += 1
    assert counts["heavy"] > counts["light"]


def test_ketama_hash_function_is_consistent():
    ring = HashRing(nodes=NODES, hash_fn="ketama")
    first = mapping(ring)
    second = mapping(HashRing(nodes=NODES, hash_fn="ketama"))
    assert first == second
    assert set(first.values()) <= set(NODES)


def test_custom_vnodes_still_maps_keys():
    ring = HashRing(nodes=NODES, vnodes=10)
    result = mapping(ring)
    assert set(result.values()) <= set(NODES)
    assert len(ring.get_points()) > 0
