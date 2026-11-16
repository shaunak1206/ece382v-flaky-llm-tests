import pytest

from uhashring import HashRing


KEYS = ["key-%d" % i for i in range(300)]


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_node("anything") is None
    assert ring.get_nodes() == []


def test_single_node_gets_all_keys():
    ring = HashRing(nodes=["node1"])
    assert all(ring.get_node(k) == "node1" for k in KEYS)


def test_get_node_is_deterministic():
    ring1 = HashRing(nodes=["a", "b", "c"])
    ring2 = HashRing(nodes=["a", "b", "c"])
    for k in KEYS:
        assert ring1.get_node(k) == ring1.get_node(k)
        assert ring1.get_node(k) == ring2.get_node(k)


def test_get_nodes_lists_all_nodes():
    ring = HashRing(nodes=["a", "b", "c"])
    assert sorted(ring.get_nodes()) == ["a", "b", "c"]


def test_string_node_accepted():
    ring = HashRing(nodes="solo")
    assert ring.get_nodes() == ["solo"]
    assert ring.get_node("x") == "solo"


def test_invalid_nodes_type_raises():
    with pytest.raises(ValueError):
        HashRing(nodes=12345)


def test_non_callable_weight_fn_raises():
    with pytest.raises(TypeError):
        HashRing(nodes=["a"], weight_fn="not callable")


def test_default_vnodes_point_count():
    ring = HashRing(nodes=["a", "b"], vnodes=10)
    points = ring.get_points()
    assert len(points) == 20
    assert {name for _, name in points} == {"a", "b"}


def test_weight_multiplies_vnodes():
    ring = HashRing(nodes={"a": 1, "b": 3}, vnodes=10)
    names = [name for _, name in ring.get_points()]
    assert names.count("a") == 10
    assert names.count("b") == 30


def test_dict_conf_with_vnodes_and_weight():
    ring = HashRing(nodes={"a": {"vnodes": 5, "weight": 2}})
    assert len(ring.get_points()) == 10


def test_distribution_covers_all_nodes():
    ring = HashRing(nodes=["a", "b", "c"])
    seen = {ring.get_node(k) for k in KEYS}
    assert seen == {"a", "b", "c"}


def test_add_node_registers_node():
    ring = HashRing(nodes=["a"])
    ring.add_node("b")
    assert sorted(ring.get_nodes()) == ["a", "b"]
    assert {ring.get_node(k) for k in KEYS} == {"a", "b"}


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(nodes=["a", "b", "c"])
    before = {k: ring.get_node(k) for k in KEYS}
    ring.add_node("d")
    for k in KEYS:
        after = ring.get_node(k)
        assert after == before[k] or after == "d"


def test_remove_node_only_moves_its_keys():
    ring = HashRing(nodes=["a", "b", "c"])
    before = {k: ring.get_node(k) for k in KEYS}
    ring.remove_node("b")
    assert sorted(ring.get_nodes()) == ["a", "c"]
    for k in KEYS:
        after = ring.get_node(k)
        assert after != "b"
        if before[k] != "b":
            assert after == before[k]


def test_remove_all_nodes_gives_empty_ring():
    ring = HashRing(nodes=["a"])
    ring.remove_node("a")
    assert ring.get_node("x") is None


def test_ketama_ring_is_deterministic_and_valid():
    nodes = ["a", "b", "c"]
    ring1 = HashRing(nodes=nodes, hash_fn="ketama")
    ring2 = HashRing(nodes=nodes, hash_fn="ketama")
    for k in KEYS:
        assert ring1.get_node(k) in nodes
        assert ring1.get_node(k) == ring2.get_node(k)


def test_node_conf_with_hostname_and_port():
    ring = HashRing(
        nodes={"n1": {"hostname": "example.com", "port": 11211, "weight": 1}}
    )
    assert ring.get_node_hostname("k") == "example.com"
    assert ring.get_node_port("k") == 11211
    assert ring.get_node_weight("k") == 1


def test_weight_fn_is_applied():
    ring = HashRing(
        nodes={"a": 1, "b": 1}, vnodes=4, weight_fn=lambda **conf: 2
    )
    assert len(ring.get_points()) == 16


def test_iterate_nodes_distinct_yields_every_node_once():
    ring = HashRing(nodes=["a", "b", "c"])
    result = list(ring.iterate_nodes("some-key", distinct=True))
    assert sorted(result) == ["a", "b", "c"]
    assert result[0] == ring.get_node("some-key")
