import pytest

from uhashring import HashRing

KEYS = ["key-%d" % i for i in range(500)]


def mapping(ring, keys=KEYS):
    return {k: ring.get_node(k) for k in keys}


def test_empty_ring_returns_none():
    ring = HashRing()
    assert ring.get_node("anything") is None


def test_empty_ring_has_no_nodes():
    ring = HashRing()
    assert list(ring.get_nodes()) == []


def test_get_nodes_lists_all_nodes():
    names = ["node1", "node2", "node3"]
    ring = HashRing(nodes=names)
    assert sorted(ring.get_nodes()) == sorted(names)


def test_get_node_returns_member_of_nodes():
    names = ["node1", "node2", "node3"]
    ring = HashRing(nodes=names)
    for key in KEYS:
        assert ring.get_node(key) in names


def test_get_node_is_deterministic():
    ring = HashRing(nodes=["a", "b", "c"])
    first = mapping(ring)
    second = mapping(ring)
    assert first == second


def test_two_rings_same_nodes_agree():
    ring1 = HashRing(nodes=["a", "b", "c"])
    ring2 = HashRing(nodes=["c", "a", "b"])
    assert mapping(ring1) == mapping(ring2)


def test_single_node_gets_all_keys():
    ring = HashRing(nodes=["only"])
    assert set(mapping(ring).values()) == {"only"}


def test_keys_distributed_across_nodes():
    names = ["n1", "n2", "n3", "n4"]
    ring = HashRing(nodes=names)
    used = set(mapping(ring).values())
    assert used == set(names)


def test_add_node_only_moves_keys_to_new_node():
    ring = HashRing(nodes=["a", "b", "c"])
    before = mapping(ring)
    ring.add_node("d")
    after = mapping(ring)
    assert "d" in ring.get_nodes()
    for key in KEYS:
        if before[key] != after[key]:
            assert after[key] == "d"
    assert any(v == "d" for v in after.values())


def test_remove_node_only_moves_its_keys():
    ring = HashRing(nodes=["a", "b", "c", "d"])
    before = mapping(ring)
    ring.remove_node("d")
    after = mapping(ring)
    assert "d" not in ring.get_nodes()
    assert "d" not in set(after.values())
    for key in KEYS:
        if before[key] != "d":
            assert after[key] == before[key]


def test_dict_nodes_with_instance():
    inst_a, inst_b = object(), object()
    ring = HashRing(nodes={"a": {"instance": inst_a}, "b": {"instance": inst_b}})
    for key in KEYS[:50]:
        node = ring.get_node(key)
        expected = inst_a if node == "a" else inst_b
        assert ring.get_node_instance(key) is expected


def test_weight_affects_distribution():
    ring = HashRing(nodes={"heavy": {"weight": 10}, "light": {"weight": 1}})
    counts = {"heavy": 0, "light": 0}
    for i in range(2000):
        counts[ring.get_node("k%d" % i)] += 1
    assert counts["heavy"] > counts["light"]


def test_add_node_with_weight_kwarg():
    ring = HashRing(nodes=["a"])
    ring.add_node("b", weight=5)
    assert sorted(ring.get_nodes()) == ["a", "b"]
    assert set(mapping(ring).values()) <= {"a", "b"}


def test_custom_vnodes_still_works():
    ring = HashRing(nodes=["a", "b"], vnodes=1)
    assert set(mapping(ring).values()) <= {"a", "b"}


def test_ketama_hash_fn_is_stable():
    ring1 = HashRing(nodes=["a", "b", "c"], hash_fn="ketama")
    ring2 = HashRing(nodes=["a", "b", "c"], hash_fn="ketama")
    assert mapping(ring1) == mapping(ring2)
    assert set(mapping(ring1).values()) <= {"a", "b", "c"}


def test_iterate_nodes_distinct_covers_all_nodes():
    names = ["a", "b", "c"]
    ring = HashRing(nodes=names)
    result = list(ring.iterate_nodes("some-key", distinct=True))
    assert sorted(result) == sorted(names)
    assert result[0] == ring.get_node("some-key")


def test_get_node_pos_returns_int():
    ring = HashRing(nodes=["a", "b", "c"])
    pos = ring.get_node_pos("some-key")
    assert isinstance(pos, int)
    assert pos >= 0
