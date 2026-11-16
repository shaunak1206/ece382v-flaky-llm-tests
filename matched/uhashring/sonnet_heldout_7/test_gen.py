import pytest

from uhashring import HashRing


NODES = ["node1", "node2", "node3"]
KEYS = ["key-%d" % i for i in range(300)]


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_node("anything") is None
    assert list(ring.get_nodes()) == []


def test_get_node_returns_known_node():
    ring = HashRing(nodes=NODES)
    for key in KEYS:
        assert ring.get_node(key) in NODES


def test_get_node_is_deterministic_across_rings():
    ring1 = HashRing(nodes=NODES)
    ring2 = HashRing(nodes=list(NODES))
    for key in KEYS:
        assert ring1.get_node(key) == ring1.get_node(key)
        assert ring1.get_node(key) == ring2.get_node(key)


def test_get_nodes_lists_all_nodes():
    ring = HashRing(nodes=NODES)
    assert sorted(ring.get_nodes()) == sorted(NODES)


def test_single_string_node():
    ring = HashRing(nodes="solo")
    assert list(ring.get_nodes()) == ["solo"]
    assert ring.get_node("foo") == "solo"


def test_invalid_nodes_type_raises():
    with pytest.raises(ValueError):
        HashRing(nodes=12345)


def test_invalid_weight_fn_raises():
    with pytest.raises(TypeError):
        HashRing(nodes=NODES, weight_fn="not callable")


def test_add_node_makes_node_reachable():
    ring = HashRing(nodes=["a"])
    ring.add_node("b")
    assert sorted(ring.get_nodes()) == ["a", "b"]
    assert {ring.get_node(k) for k in KEYS} == {"a", "b"}


def test_remove_node_stops_routing_to_it():
    ring = HashRing(nodes=NODES)
    ring.remove_node("node2")
    assert "node2" not in list(ring.get_nodes())
    for key in KEYS:
        assert ring.get_node(key) in ("node1", "node3")


def test_remove_node_only_remaps_its_keys():
    ring = HashRing(nodes=NODES)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.remove_node("node2")
    for key, node in before.items():
        if node != "node2":
            assert ring.get_node(key) == node


def test_get_points_count_matches_vnodes():
    ring = HashRing(nodes=["a", "b"], vnodes=10)
    points = ring.get_points()
    assert len(points) == 20
    assert {name for _, name in points} == {"a", "b"}


def test_weight_increases_points():
    ring = HashRing(nodes={"a": 1, "b": 3}, vnodes=10)
    points = ring.get_points()
    count_a = sum(1 for _, n in points if n == "a")
    count_b = sum(1 for _, n in points if n == "b")
    assert count_b > count_a


def test_get_server_returns_position_and_node():
    ring = HashRing(nodes=NODES)
    pos, node = ring.get_server("some-key")
    assert node == ring.get_node("some-key")
    assert isinstance(pos, int)


def test_node_instance_and_port_config():
    inst = object()
    ring = HashRing(nodes={"a": {"instance": inst, "port": 1234}})
    assert ring.get_node_instance("k") is inst
    assert ring.get_node_port("k") == 1234
    assert ring["k"] is inst
    assert inst in ring.get_instances()


def test_iterate_nodes_distinct_yields_each_node_once():
    ring = HashRing(nodes=NODES)
    result = list(ring.iterate_nodes("some-key"))
    assert sorted(result) == sorted(NODES)
    assert result[0] == ring.get_node("some-key")


def test_ketama_ring_distributes_keys():
    ring = HashRing(nodes=NODES, hash_fn="ketama")
    ring2 = HashRing(nodes=NODES, hash_fn="ketama")
    seen = set()
    for key in KEYS:
        node = ring.get_node(key)
        assert node in NODES
        assert node == ring2.get_node(key)
        seen.add(node)
    assert seen == set(NODES)


def test_setitem_and_delitem():
    ring = HashRing(nodes=["a"])
    ring["b"] = {"weight": 2}
    assert "b" in list(ring.get_nodes())
    del ring["b"]
    assert "b" not in list(ring.get_nodes())
    assert ring.get_node("x") == "a"
