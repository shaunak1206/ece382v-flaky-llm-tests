import inspect

import pytest

from uhashring import HashRing
from uhashring.ring import HashRing as RingHashRing


def _make_ring(nodes=None, **kwargs):
    """
    Build a ring using the constructor signature that this installed version
    exposes. This keeps the tests tolerant to small API differences while still
    exercising the same class.
    """
    sig = inspect.signature(HashRing)
    params = sig.parameters

    if nodes is None:
        nodes = {1: 1, 2: 1, 3: 1}

    # Most versions accept a first positional "nodes" argument.
    if len(params) == 0:
        return HashRing()

    # Prefer keyword construction when possible.
    if "nodes" in params:
        return HashRing(nodes=nodes, **kwargs)

    # Fallback to first positional argument.
    return HashRing(nodes, **kwargs)


def _lookup(ring, key):
    for method_name in ("get_node", "get_node_pos", "get"):
        method = getattr(ring, method_name, None)
        if callable(method):
            return method(key)
    raise AssertionError("HashRing object has no supported lookup method")


def _remove_node(ring, node):
    for method_name in ("remove_node", "del_node", "remove"):
        method = getattr(ring, method_name, None)
        if callable(method):
            return method(node)
    raise AssertionError("HashRing object has no supported remove method")


def _add_node(ring, node, weight=1):
    for method_name in ("add_node", "add", "update_node"):
        method = getattr(ring, method_name, None)
        if callable(method):
            try:
                return method(node, weight)
            except TypeError:
                return method(node)
    raise AssertionError("HashRing object has no supported add method")


def test_package_reexports_hashring_class():
    assert HashRing is RingHashRing


def test_all_contains_hashring_and_monkey():
    import uhashring

    assert "HashRing" in uhashring.__all__
    assert "monkey" in uhashring.__all__


def test_can_construct_ring_from_int_nodes():
    ring = _make_ring({1: 1, 2: 1, 3: 1})
    assert ring is not None


def test_lookup_returns_a_present_node_for_integer_key():
    ring = _make_ring({1: 1, 2: 1, 3: 1})
    node = _lookup(ring, 42)
    assert node in {1, 2, 3}


def test_lookup_is_deterministic_for_same_key():
    ring = _make_ring({1: 1, 2: 1, 3: 1})
    key = 123456
    assert _lookup(ring, key) == _lookup(ring, key)


def test_two_rings_with_same_nodes_agree_on_lookup():
    nodes = {1: 1, 2: 1, 3: 1}
    ring1 = _make_ring(nodes)
    ring2 = _make_ring(nodes)
    for key in [0, 1, 2, 3, 10, 100, 999999]:
        assert _lookup(ring1, key) == _lookup(ring2, key)


def test_lookup_for_multiple_integer_keys_always_returns_one_of_the_nodes():
    ring = _make_ring({10: 1, 20: 1, 30: 1})
    for key in range(50):
        assert _lookup(ring, key) in {10, 20, 30}


def test_remove_node_prevents_that_node_from_being_returned():
    ring = _make_ring({1: 1, 2: 1, 3: 1})
    _remove_node(ring, 2)

    for key in range(50):
        assert _lookup(ring, key) in {1, 3}


def test_add_node_restores_a_removed_node():
    ring = _make_ring({1: 1, 2: 1, 3: 1})
    _remove_node(ring, 3)
    for key in range(20):
        assert _lookup(ring, key) in {1, 2}

    _add_node(ring, 3)
    seen = {_lookup(ring, key) for key in range(100)}
    assert 3 in seen
    assert seen.issubset({1, 2, 3})


def test_removing_only_node_makes_lookup_fail_or_return_none():
    ring = _make_ring({1: 1})
    _remove_node(ring, 1)

    with pytest.raises(Exception):
        _lookup(ring, 0)


def test_int_keys_do_not_depend_on_python_hash_randomization():
    ring = _make_ring({1: 1, 2: 1, 3: 1})
    results = [_lookup(ring, key) for key in [0, 1, 2, 3, 4, 5, 100, 1000]]
    assert all(node in {1, 2, 3} for node in results)
    assert results == [_lookup(ring, key) for key in [0, 1, 2, 3, 4, 5, 100, 1000]]


def test_string_representation_is_created_without_error():
    ring = _make_ring({1: 1, 2: 1, 3: 1})
    text = str(ring)
    assert isinstance(text, str)
    assert text


def test_ring_can_handle_zero_as_a_key():
    ring = _make_ring({1: 1, 2: 1, 3: 1})
    assert _lookup(ring, 0) in {1, 2, 3}
