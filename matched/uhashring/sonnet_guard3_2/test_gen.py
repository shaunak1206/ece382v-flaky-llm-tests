import pytest

from uhashring import HashRing


KEYS = ["key-%d" % i for i in range(200)]
NODES = ["node1", "node2", "node3"]


def test_empty_ring_has_no_nodes():
    ring = HashRing()
    assert ring.get_nodes() == []


def test_get_nodes_returns_configured_names():
    ring = HashRing(nodes=NODES)
    assert sorted(ring.get_nodes()) == sorted(NODES)


def test_single_string_node():
    ring = HashRing(nodes="solo")
    assert ring.get_nodes() == ["solo"]
    for key in KEYS[:20]:
        assert ring.get_node(key) == "solo"


def test_get_node_is_deterministic_and_valid():
    ring = HashRing(nodes=NODES)
    for key in KEYS:
        node = ring.get_node(key)
        assert node in NODES
        assert ring.get_node(key) == node


def test_two_rings_same_config_agree():
    ring1 = HashRing(nodes=NODES)
    ring2 = HashRing(nodes=list(reversed(NODES)))
    for key in KEYS:
        assert ring1.get_node(key) == ring2.get_node(key)


def test_int_keys_work():
    ring = HashRing(nodes=NODES)
    for key in range(50):
        assert ring.get_node(key) in NODES
        assert ring.get_node(key) == ring.get_node(key)


def test_keys_spread_over_all_nodes():
    ring = HashRing(nodes=NODES)
    used = {ring.get_node(k) for k in ["k%d" % i for i in range(1000)]}
    assert used == set(NODES)


def test_remove_node_only_moves_its_keys():
    ring = HashRing(nodes=NODES)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.remove_node("node2")
    assert sorted(ring.get_nodes()) == ["node1", "node3"]
    for k in KEYS:
        after = ring.get_node(k)
        assert after != "node2"
        if before[k] != "node2":
            assert after == before[k]


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(nodes=["node1", "node2"])
    before = {k: ring.get_node(k) for k in KEYS}
    ring.add_node("node3")
    assert sorted(ring.get_nodes()) == NODES
    for k in KEYS:
        after = ring.get_node(k)
        assert after == before[k] or after == "node3"


def test_remove_unknown_node_raises():
    ring = HashRing(nodes=NODES)
    with pytest.raises(KeyError):
        ring.remove_node("does-not-exist")


def test_iterate_nodes_distinct_yields_each_node_once():
    ring = HashRing(nodes=NODES)
    for key in KEYS[:20]:
        result = list(ring.iterate_nodes(key, distinct=True))
        assert sorted(result) == sorted(NODES)
        assert result[0] == ring.get_node(key)


def test_dict_config_with_instances():
    obj_a = object()
    obj_b = object()
    nodes = {
        "a": {"instance": obj_a, "weight": 1},
        "b": {"instance": obj_b, "weight": 2},
    }
    ring = HashRing(nodes=nodes)
    assert sorted(ring.get_nodes()) == ["a", "b"]
    for key in KEYS[:30]:
        name = ring.get_node(key)
        expected = obj_a if name == "a" else obj_b
        assert ring.get_node_instance(key) is expected


def test_weight_increases_share_of_keys():
    nodes = {"light": {"weight": 1}, "heavy": {"weight": 8}}
    ring = HashRing(nodes=nodes)
    counts = {"light": 0, "heavy": 0}
    for i in range(2000):
        counts[ring.get_node("item-%d" % i)] += 1
    assert counts["heavy"] > counts["light"]


def test_custom_vnodes_limits_points():
    ring = HashRing(nodes=NODES, vnodes=10)
    points = ring.get_points()
    assert 0 < len(points) <= 10 * len(NODES)


def test_custom_callable_hash_fn_is_used():
    ring = HashRing(nodes=NODES, hash_fn=lambda s: sum(bytearray(s.encode("utf-8"))))
    for key in KEYS[:30]:
        assert ring.get_node(key) in NODES
        assert ring.get_node(key) == ring.get_node(key)


def test_ketama_ring_is_consistent():
    ring1 = HashRing(nodes=NODES, hash_fn="ketama")
    ring2 = HashRing(nodes=NODES, hash_fn="ketama")
    for key in KEYS[:50]:
        node = ring1.get_node(key)
        assert node in NODES
        assert node == ring2.get_node(key)


def test_invalid_weight_fn_raises_type_error():
    with pytest.raises(TypeError):
        HashRing(nodes=NODES, weight_fn="not callable")
