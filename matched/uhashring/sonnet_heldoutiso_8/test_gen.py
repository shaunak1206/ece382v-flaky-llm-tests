import pytest

from uhashring import HashRing


NODES = ["node1", "node2", "node3"]
KEYS = ["key-%d" % i for i in range(300)]


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_node("anything") is None


def test_get_node_returns_member():
    ring = HashRing(nodes=NODES)
    for key in KEYS[:50]:
        assert ring.get_node(key) in NODES


def test_get_node_is_deterministic():
    ring = HashRing(nodes=NODES)
    first = [ring.get_node(k) for k in KEYS]
    second = [ring.get_node(k) for k in KEYS]
    assert first == second


def test_two_rings_same_nodes_agree():
    ring1 = HashRing(nodes=NODES)
    ring2 = HashRing(nodes=list(reversed(NODES)))
    for key in KEYS:
        assert ring1.get_node(key) == ring2.get_node(key)


def test_single_string_node():
    ring = HashRing(nodes="solo")
    assert ring.get_node("foo") == "solo"
    assert ring.get_node("bar") == "solo"


def test_invalid_nodes_type_raises():
    with pytest.raises(ValueError):
        HashRing(nodes=12)


def test_non_callable_weight_fn_raises():
    with pytest.raises(TypeError):
        HashRing(nodes=NODES, weight_fn="not callable")


def test_get_nodes_lists_all_nodes():
    ring = HashRing(nodes=NODES)
    assert sorted(ring.get_nodes()) == sorted(NODES)


def test_keys_spread_over_all_nodes():
    ring = HashRing(nodes=NODES)
    seen = {ring.get_node(k) for k in KEYS}
    assert seen == set(NODES)


def test_add_node_adds_and_moves_only_some_keys():
    ring = HashRing(nodes=NODES)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.add_node("node4")
    assert "node4" in ring.get_nodes()
    after = {k: ring.get_node(k) for k in KEYS}
    moved = [k for k in KEYS if before[k] != after[k]]
    # every moved key must have moved to the new node
    assert all(after[k] == "node4" for k in moved)
    assert 0 < len(moved) < len(KEYS)


def test_remove_node_only_remaps_its_keys():
    ring = HashRing(nodes=NODES)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.remove_node("node2")
    assert "node2" not in ring.get_nodes()
    for k in KEYS:
        after = ring.get_node(k)
        assert after != "node2"
        if before[k] != "node2":
            assert after == before[k]


def test_remove_unknown_node_raises():
    ring = HashRing(nodes=NODES)
    with pytest.raises(Exception):
        ring.remove_node("does-not-exist")


def test_weighted_nodes_get_more_keys():
    ring = HashRing(nodes={"heavy": 10, "light": 1})
    counts = {"heavy": 0, "light": 0}
    for i in range(2000):
        counts[ring.get_node("k%d" % i)] += 1
    assert counts["heavy"] > counts["light"]


def test_weight_fn_is_used():
    calls = []

    def weight_fn(**conf):
        calls.append(conf["nodename"])
        return 2

    ring = HashRing(nodes=NODES, weight_fn=weight_fn)
    assert set(calls) >= set(NODES)
    assert ring.get_node("x") in NODES


def test_node_instance_from_dict_config():
    inst = object()
    ring = HashRing(nodes={"only": {"instance": inst, "weight": 1}})
    assert ring.get_node_instance("some-key") is inst


def test_get_node_pos_is_int_and_stable():
    ring = HashRing(nodes=NODES)
    pos = ring.get_node_pos("abc")
    assert isinstance(pos, int)
    assert pos == ring.get_node_pos("abc")


def test_iterate_nodes_distinct_covers_all_nodes():
    ring = HashRing(nodes=NODES)
    result = list(ring.iterate_nodes("some-key"))
    assert len(result) == len(NODES)
    assert set(result) == set(NODES)
    # the first node is the one that owns the key
    assert result[0] == ring.get_node("some-key")


def test_ketama_ring_is_deterministic_and_valid():
    ring1 = HashRing(nodes=NODES, hash_fn="ketama")
    ring2 = HashRing(nodes=NODES, hash_fn="ketama")
    for key in KEYS[:100]:
        node = ring1.get_node(key)
        assert node in NODES
        assert node == ring2.get_node(key)


def test_custom_vnodes_still_distributes():
    ring = HashRing(nodes=NODES, vnodes=10)
    seen = {ring.get_node(k) for k in KEYS}
    assert seen.issubset(set(NODES))
    assert len(seen) >= 2
