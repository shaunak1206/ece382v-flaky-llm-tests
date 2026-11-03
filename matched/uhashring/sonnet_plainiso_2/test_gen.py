import zlib

import pytest

from uhashring import HashRing

NODES = ["node1", "node2", "node3"]
KEYS = ["key-%d" % i for i in range(200)]


def test_empty_ring_returns_none_and_is_empty():
    ring = HashRing()
    assert ring.size == 0
    assert ring.get_node("anything") is None
    assert ring.get("anything") is None
    assert ring.get_server("anything") is None
    assert list(ring.get_nodes()) == []
    assert ring.get_points() == []


def test_iterate_nodes_on_empty_ring_yields_none():
    ring = HashRing()
    assert list(ring.iterate_nodes("k")) == [None]


def test_default_vnodes_and_size_for_list_nodes():
    ring = HashRing(nodes=NODES)
    assert ring.size == 160 * len(NODES)
    assert set(ring.get_nodes()) == set(NODES)
    for name in NODES:
        assert ring.distribution[name] == 160


def test_custom_vnodes_and_weights_from_dict():
    ring = HashRing(nodes={"a": 2, "b": 1}, vnodes=10)
    assert ring.size == 10 * 2 + 10 * 1
    assert ring.distribution["a"] == 20
    assert ring.distribution["b"] == 10


def test_get_node_is_deterministic_and_in_configured_nodes():
    ring = HashRing(nodes=NODES)
    other = HashRing(nodes=NODES)
    for key in KEYS:
        node = ring.get_node(key)
        assert node in NODES
        assert node == ring.get_node(key)
        assert node == other.get_node(key)


def test_single_string_node_gets_all_keys():
    ring = HashRing(nodes="only", vnodes=5)
    assert set(ring.get_nodes()) == {"only"}
    for key in KEYS[:20]:
        assert ring.get_node(key) == "only"


def test_invalid_nodes_type_raises_value_error():
    with pytest.raises(ValueError):
        HashRing(nodes=42)
    with pytest.raises(ValueError):
        HashRing(nodes={"a": "not-an-int-or-dict"})


def test_non_callable_hash_fn_and_weight_fn_raise_type_error():
    with pytest.raises(TypeError):
        HashRing(nodes=NODES, hash_fn="not callable")
    with pytest.raises(TypeError):
        HashRing(nodes=NODES, weight_fn="not callable")


def test_node_config_dict_exposes_port_hostname_instance_weight():
    instance = object()
    nodes = {
        "a": {
            "hostname": "host-a",
            "port": 6379,
            "instance": instance,
            "vnodes": 8,
            "weight": 3,
        }
    }
    ring = HashRing(nodes=nodes)
    assert ring.size == 8 * 3
    key = "some-key"
    assert ring.get_node(key) == "a"
    assert ring.get_node_hostname(key) == "host-a"
    assert ring.get_node_port(key) == 6379
    assert ring.get_node_weight(key) == 3
    assert ring.get_node_instance(key) is instance
    assert ring[key] is instance
    assert ring.get(key)["hostname"] == "host-a"
    assert ring.get_instances() == [instance]


def test_get_instances_skips_nodes_without_instance():
    ring = HashRing(nodes={"a": {"instance": "inst-a"}, "b": {}})
    assert ring.get_instances() == ["inst-a"]


def test_weight_fn_overrides_weight():
    ring = HashRing(nodes=["a", "b"], vnodes=4, weight_fn=lambda **conf: 2)
    assert ring.size == 2 * 4 * 2
    assert ring.get_node_weight("k") == 2


def test_custom_hash_fn_is_used_by_get_key():
    def crc(key):
        return zlib.crc32(str(key).encode("utf-8"))

    ring = HashRing(nodes=NODES, hash_fn=crc, vnodes=20)
    assert ring.get_key("abc") == crc("abc")
    assert ring.hashi("abc") == crc("abc")
    assert ring.get_node("abc") in NODES


def test_get_node_pos_and_get_server_match_get_points():
    ring = HashRing(nodes=NODES, vnodes=10)
    points = ring.get_points()
    assert len(points) == ring.size
    assert [p[0] for p in points] == sorted(p[0] for p in points)
    for key in KEYS[:30]:
        pos = ring.get_node_pos(key)
        assert 0 <= pos < ring.size
        assert ring.get_server(key) == points[pos]
        assert points[pos][1] == ring.get_node(key)


def test_add_node_grows_ring_and_new_node_receives_keys():
    ring = HashRing(nodes=["a", "b"], vnodes=50)
    before = ring.size
    ring.add_node("c")
    assert ring.size == before + 50
    assert "c" in set(ring.get_nodes())
    assigned = set(ring.get_node(k) for k in KEYS)
    assert "c" in assigned


def test_add_node_with_weight_and_setitem_alias():
    ring = HashRing(nodes=["a"], vnodes=10)
    ring["b"] = 3
    assert ring.size == 10 + 30
    assert ring.get_nodes() is not None
    assert ring.conf["b"]["weight"] == 3
    assert ring.nodes is not None


def test_remove_node_stops_routing_and_keeps_other_assignments():
    ring = HashRing(nodes=NODES, vnodes=40)
    before = dict((k, ring.get_node(k)) for k in KEYS)
    ring.remove_node("node2")
    assert "node2" not in set(ring.get_nodes())
    assert ring.size == 40 * 2
    assert "node2" not in ring.distribution
    for key in KEYS:
        after = ring.get_node(key)
        assert after != "node2"
        if before[key] != "node2":
            assert after == before[key]


def test_delitem_and_remove_missing_node_raises_key_error():
    ring = HashRing(nodes=NODES, vnodes=5)
    del ring["node1"]
    assert "node1" not in set(ring.get_nodes())
    with pytest.raises(KeyError):
        ring.remove_node("does-not-exist")


def test_range_unique_and_size_limit():
    ring = HashRing(nodes=NODES, vnodes=20)
    key = "range-key"
    result = list(ring.range(key))
    names = [n["nodename"] for n in result]
    assert len(names) == len(NODES)
    assert set(names) == set(NODES)
    assert names[0] == ring.get_node(key)

    limited = [n["nodename"] for n in ring.range(key, size=2)]
    assert len(limited) == 2
    assert len(set(limited)) == 2
    assert limited == names[:2]


def test_range_non_unique_with_size():
    ring = HashRing(nodes=["a", "b"], vnodes=10)
    result = list(ring.range("k", size=5, unique=False))
    assert len(result) == 5
    assert all(n["nodename"] in ("a", "b") for n in result)


def test_iterate_nodes_distinct_matches_primary_node():
    ring = HashRing(nodes=NODES, vnodes=20)
    key = "iter-key"
    names = list(ring.iterate_nodes(key))
    assert len(names) == len(NODES)
    assert set(names) == set(NODES)
    assert names[0] == ring.get_node(key)


def test_ketama_ring_routes_keys_and_returns_tuples():
    ring = HashRing(nodes=NODES, hash_fn="ketama")
    assert ring.size > 0
    assert ring.size <= 40 * 4 * len(NODES)
    for key in KEYS[:30]:
        pos, name = ring.get_server(key)
        assert name in NODES
        assert name == ring.get_node(key)
        assert isinstance(pos, int)
        assert ring.get_key(key) >= 0


def test_regenerate_preserves_ring_and_print_continuum(capsys):
    ring = HashRing(nodes=["a", "b"], vnodes=5)
    points = ring.get_points()
    ring.regenerate()
    assert ring.get_points() == points

    ring.print_continuum()
    out = capsys.readouterr().out
    assert "Numpoints in continuum: 10" in out

    empty = HashRing()
    empty.print_continuum()
    assert "Continuum empty" in capsys.readouterr().out
