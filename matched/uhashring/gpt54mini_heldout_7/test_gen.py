import importlib
import inspect

import pytest

from uhashring import HashRing
import uhashring as uhashring_module


def make_ring(nodes=None, **kwargs):
    if nodes is None:
        nodes = ["a", "b", "c"]
    try:
        return HashRing(nodes, **kwargs)
    except TypeError:
        try:
            return HashRing(nodes=nodes, **kwargs)
        except TypeError:
            return HashRing(**kwargs)


def get_node(ring, key):
    if hasattr(ring, "get_node"):
        return ring.get_node(key)
    if hasattr(ring, "get"):
        return ring.get(key)
    raise AssertionError("HashRing has neither get_node nor get")


def get_nodes(ring, key, count=None):
    if hasattr(ring, "get_nodes"):
        if count is None:
            return ring.get_nodes(key)
        return ring.get_nodes(key, count)
    raise AssertionError("HashRing has no get_nodes")


def test_init_module_reexports_hashring():
    assert uhashring_module.HashRing is HashRing


def test_all_contains_hashring():
    assert "HashRing" in uhashring_module.__all__


def test_hashring_is_class():
    assert inspect.isclass(HashRing)


def test_hashring_importable_from_package():
    mod = importlib.import_module("uhashring")
    assert mod.HashRing is HashRing


def test_hashring_can_be_instantiated_with_list_nodes():
    ring = make_ring(["node1", "node2"])
    assert isinstance(ring, HashRing)


def test_hashring_can_be_instantiated_with_mapping_nodes():
    ring = make_ring({"node1": 1, "node2": 2})
    assert isinstance(ring, HashRing)


def test_hashring_get_node_returns_known_node_for_string_key():
    ring = make_ring(["a", "b", "c"])
    node = get_node(ring, "user-123")
    assert node in {"a", "b", "c"}


def test_hashring_get_node_is_deterministic_for_same_key():
    ring = make_ring(["a", "b", "c"])
    assert get_node(ring, "same-key") == get_node(ring, "same-key")


def test_hashring_get_node_different_keys_return_valid_nodes():
    ring = make_ring(["a", "b", "c"])
    nodes = {get_node(ring, "alpha"), get_node(ring, "beta"), get_node(ring, "gamma")}
    assert nodes.issubset({"a", "b", "c"})
    assert len(nodes) >= 1


def test_hashring_get_nodes_returns_requested_number_when_supported():
    ring = make_ring(["a", "b", "c", "d"])
    result = get_nodes(ring, "some-key", 2)
    assert isinstance(result, (list, tuple))
    assert len(result) == 2
    assert set(result).issubset({"a", "b", "c", "d"})


def test_hashring_get_nodes_defaults_to_all_or_multiple_nodes():
    ring = make_ring(["a", "b", "c"])
    result = get_nodes(ring, "some-key")
    assert isinstance(result, (list, tuple))
    assert len(result) >= 1
    assert set(result).issubset({"a", "b", "c"})


def test_hashring_with_single_node_always_returns_that_node():
    ring = make_ring(["only-node"])
    for key in ["k1", "k2", "another-key", "123"]:
        assert get_node(ring, key) == "only-node"


def test_hashring_accepts_bytes_like_keys_if_supported():
    ring = make_ring(["a", "b", "c"])
    key = b"bytes-key"
    node = get_node(ring, key)
    assert node in {"a", "b", "c"}


def test_hashring_supports_unicode_keys():
    ring = make_ring(["a", "b", "c"])
    node = get_node(ring, "ключ-✓-ñ")
    assert node in {"a", "b", "c"}


def test_hashring_repr_includes_class_name():
    ring = make_ring(["a", "b"])
    assert "HashRing" in repr(ring)


def test_hashring_has_expected_public_methods_or_equivalents():
    ring = make_ring(["a", "b"])
    assert any(hasattr(ring, name) for name in ("get_node", "get", "__getitem__"))
    assert hasattr(ring, "__class__")
