import uhashring
from uhashring import HashRing
from uhashring import ring as ring_module


KEYS = list(range(1000))


def _mapping(ring, keys=KEYS):
    """Return a dict {key: node} for the given keys."""
    return {key: ring.get_node(key) for key in keys}


def test_package_exports_hashring_class():
    assert "HashRing" in uhashring.__all__
    assert uhashring.HashRing is ring_module.HashRing


def test_get_node_returns_one_of_the_configured_nodes():
    nodes = ["alpha", "beta", "gamma"]
    ring = HashRing(nodes=nodes)
    for key in KEYS:
        assert ring.get_node(key) in nodes


def test_get_node_is_deterministic_for_same_key():
    ring = HashRing(nodes=["alpha", "beta", "gamma"])
    first = _mapping(ring)
    second = _mapping(ring)
    assert first == second


def test_two_rings_with_same_nodes_map_keys_identically():
    ring_a = HashRing(nodes=["alpha", "beta", "gamma"])
    ring_b = HashRing(nodes=["alpha", "beta", "gamma"])
    assert _mapping(ring_a) == _mapping(ring_b)


def test_node_insertion_order_does_not_change_mapping():
    ring_a = HashRing(nodes=["alpha", "beta", "gamma"])
    ring_b = HashRing(nodes=["gamma", "alpha", "beta"])
    assert _mapping(ring_a) == _mapping(ring_b)


def test_every_node_receives_some_keys():
    nodes = ["alpha", "beta", "gamma"]
    ring = HashRing(nodes=nodes)
    used = set(_mapping(ring).values())
    assert used == set(nodes)


def test_single_node_ring_routes_every_key_to_that_node():
    ring = HashRing(nodes=["only"])
    assert all(node == "only" for node in _mapping(ring).values())


def test_add_node_only_moves_keys_to_the_new_node():
    ring = HashRing(nodes=["alpha", "beta", "gamma"])
    before = _mapping(ring)

    ring.add_node("delta")
    after = _mapping(ring)

    for key in KEYS:
        if after[key] != before[key]:
            assert after[key] == "delta"


def test_add_node_moves_only_a_fraction_of_keys():
    ring = HashRing(nodes=["alpha", "beta", "gamma"])
    before = _mapping(ring)

    ring.add_node("delta")
    after = _mapping(ring)

    moved = sum(1 for key in KEYS if after[key] != before[key])
    # Ideal consistent hashing moves about 1/4 of keys; allow a generous bound.
    assert 0 < moved < len(KEYS) * 0.5


def test_remove_node_only_moves_keys_that_were_on_removed_node():
    ring = HashRing(nodes=["alpha", "beta", "gamma"])
    before = _mapping(ring)

    ring.remove_node("beta")
    after = _mapping(ring)

    for key in KEYS:
        if before[key] != "beta":
            assert after[key] == before[key]


def test_removed_node_is_never_returned_afterwards():
    ring = HashRing(nodes=["alpha", "beta", "gamma"])
    ring.remove_node("beta")
    assert "beta" not in set(_mapping(ring).values())


def test_remove_then_readd_restores_original_mapping():
    ring = HashRing(nodes=["alpha", "beta", "gamma"])
    before = _mapping(ring)

    ring.remove_node("beta")
    ring.add_node("beta")

    assert _mapping(ring) == before


def test_rings_do_not_share_state():
    ring_a = HashRing(nodes=["alpha", "beta"])
    ring_b = HashRing(nodes=["alpha", "beta"])
    before_b = _mapping(ring_b)

    ring_a.add_node("gamma")
    ring_a.remove_node("alpha")

    assert _mapping(ring_b) == before_b


def test_heavier_weight_receives_more_keys():
    ring = HashRing(nodes=["light", "heavy"], weights={"light": 1, "heavy": 4})
    counts = {"light": 0, "heavy": 0}
    for node in _mapping(ring).values():
        counts[node] += 1
    assert counts["heavy"] > counts["light"]


def test_equal_weights_give_every_node_some_keys():
    ring = HashRing(nodes=["alpha", "beta"], weights={"alpha": 1, "beta": 1})
    used = set(_mapping(ring).values())
    assert used == {"alpha", "beta"}


def test_small_vnodes_value_still_maps_all_keys_to_valid_nodes():
    nodes = ["alpha", "beta", "gamma"]
    ring = HashRing(nodes=nodes, vnodes=1)
    for node in _mapping(ring).values():
        assert node in nodes


def test_integer_and_string_keys_are_each_stable():
    ring = HashRing(nodes=["alpha", "beta", "gamma"])
    str_keys = ["user-{}".format(i) for i in range(200)]
    assert _mapping(ring, str_keys) == _mapping(ring, str_keys)
    assert _mapping(ring, KEYS[:200]) == _mapping(ring, KEYS[:200])


def test_ring_built_by_add_node_matches_ring_built_by_constructor():
    constructed = HashRing(nodes=["alpha", "beta", "gamma"])
    built = HashRing(nodes=["alpha"])
    built.add_node("beta")
    built.add_node("gamma")
    assert _mapping(constructed) == _mapping(built)
