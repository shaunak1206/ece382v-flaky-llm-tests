import pytest

from uhashring import HashRing


NODES = ["node1", "node2", "node3"]
KEYS = ["key-%d" % i for i in range(200)]


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_node("anything") is None
    assert ring.get_points() == []


def test_single_string_node_accepted():
    ring = HashRing(nodes="solo")
    assert ring.get_node("foo") == "solo"
    assert list(ring.get_nodes()) == ["solo"]


def test_invalid_nodes_type_raises():
    with pytest.raises(ValueError):
        HashRing(nodes=12345)


def test_non_callable_weight_fn_raises():
    with pytest.raises(TypeError):
        HashRing(nodes=NODES, weight_fn="not callable")


def test_get_node_returns_member_and_is_stable():
    ring = HashRing(nodes=NODES)
    for key in KEYS:
        node = ring.get_node(key)
        assert node in NODES
        assert ring.get_node(key) == node


def test_two_rings_same_nodes_agree():
    r1 = HashRing(nodes=NODES)
    r2 = HashRing(nodes=list(reversed(NODES)))
    for key in KEYS:
        assert r1.get_node(key) == r2.get_node(key)


def test_getitem_returns_instance():
    ring = HashRing(nodes=NODES)
    for key in KEYS[:20]:
        assert ring[key] == ring.get_node(key)


def test_distribution_uses_all_nodes():
    ring = HashRing(nodes=NODES)
    used = {ring.get_node(k) for k in KEYS}
    assert used == set(NODES)


def test_get_nodes_lists_all():
    ring = HashRing(nodes=NODES)
    assert sorted(ring.get_nodes()) == sorted(NODES)


def test_dict_nodes_with_config():
    obj = object()
    nodes = {
        "a": {"hostname": "host-a", "instance": obj, "port": 8080, "vnodes": 10},
        "b": {"hostname": "host-b", "port": 9090, "vnodes": 10},
    }
    ring = HashRing(nodes=nodes)
    info = {}
    for key in KEYS:
        info[ring.get_node(key)] = ring.get(key)
    assert set(info) <= {"a", "b"}
    if "a" in info:
        assert info["a"]["hostname"] == "host-a"
        assert info["a"]["port"] == 8080
        assert info["a"]["instance"] is obj
    assert len(ring.get_points()) == 20


def test_integer_weight_scales_points():
    ring = HashRing(nodes={"a": 1, "b": 3}, vnodes=10)
    points = ring.get_points()
    count_a = sum(1 for _, n in points if n == "a")
    count_b = sum(1 for _, n in points if n == "b")
    assert count_a == 10
    assert count_b == 30


def test_weight_fn_applied():
    ring = HashRing(nodes=["a", "b"], vnodes=5, weight_fn=lambda **conf: 2)
    assert len(ring.get_points()) == 20
    assert ring.get("somekey")["weight"] == 2


def test_remove_node_only_remaps_its_keys():
    ring = HashRing(nodes=NODES)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.remove_node("node2")
    assert "node2" not in list(ring.get_nodes())
    for key, old in before.items():
        new = ring.get_node(key)
        assert new != "node2"
        if old != "node2":
            assert new == old


def test_remove_unknown_node_raises_keyerror():
    ring = HashRing(nodes=NODES)
    with pytest.raises(KeyError):
        ring.remove_node("missing")


def test_add_node_only_steals_keys_to_new_node():
    ring = HashRing(nodes=NODES)
    before = {k: ring.get_node(k) for k in KEYS}
    ring.add_node("node4")
    assert "node4" in list(ring.get_nodes())
    moved_to_new = 0
    for key, old in before.items():
        new = ring.get_node(key)
        if new != old:
            assert new == "node4"
            moved_to_new += 1
    assert moved_to_new > 0


def test_setitem_and_delitem():
    ring = HashRing(nodes=["a"])
    ring["b"] = {"weight": 2}
    assert sorted(ring.get_nodes()) == ["a", "b"]
    del ring["a"]
    assert list(ring.get_nodes()) == ["b"]
    assert ring.get_node("whatever") == "b"


def test_iterate_nodes_distinct_covers_all_nodes():
    ring = HashRing(nodes=NODES)
    result = list(ring.iterate_nodes("some-key"))
    assert sorted(result) == sorted(NODES)
    assert len(result) == len(set(result))
    assert result[0] == ring.get_node("some-key")


def test_ketama_ring_is_consistent():
    r1 = HashRing(nodes=NODES, hash_fn="ketama")
    r2 = HashRing(nodes=NODES, hash_fn="ketama")
    for key in KEYS[:50]:
        node = r1.get_node(key)
        assert node in NODES
        assert node == r2.get_node(key)


def test_custom_hash_function():
    import hashlib

    def my_hash(key):
        return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)

    ring = HashRing(nodes=NODES, hash_fn=my_hash, vnodes=20)
    for key in KEYS[:50]:
        assert ring.get_node(key) in NODES
        assert ring.get_node(key) == ring.get_node(key)
