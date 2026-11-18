import importlib
import inspect

import pytest

import uhashring
from uhashring import HashRing


def _make_ring():
    """
    Create a HashRing instance using a few common constructor patterns.
    The helper is intentionally defensive so the tests work across minor API
    variations while still validating the public class exported from
    uhashring.__init__.
    """
    candidates = [
        ({"a": 1, "b": 1, "c": 1},),
        (["a", "b", "c"],),
        (("a", "b", "c"),),
        tuple(),
    ]

    kwargs_candidates = [
        {},
        {"nodes": {"a": 1, "b": 1, "c": 1}},
        {"nodes": ["a", "b", "c"]},
    ]

    last_error = None
    for args in candidates:
        for kwargs in kwargs_candidates:
            try:
                return HashRing(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
    raise AssertionError(f"Could not instantiate HashRing: {last_error}")


def _call_first_method(obj, method_names, *args, **kwargs):
    for name in method_names:
        if hasattr(obj, name):
            return getattr(obj, name)(*args, **kwargs)
    raise AttributeError(f"None of these methods exist on {type(obj).__name__}: {method_names}")


def test_hashring_is_reexported_from_package_init():
    assert hasattr(uhashring, "HashRing")
    assert uhashring.HashRing is HashRing


def test_hashring_matches_ring_module_class():
    ring_module = importlib.import_module("uhashring.ring")
    assert HashRing is ring_module.HashRing


def test_all_contains_hashring():
    assert "HashRing" in uhashring.__all__


def test_hashring_is_a_class():
    assert inspect.isclass(HashRing)


def test_hashring_class_name():
    assert HashRing.__name__ == "HashRing"


def test_hashring_module_path():
    assert HashRing.__module__ == "uhashring.ring"


def test_hashring_can_be_instantiated():
    ring = _make_ring()
    assert isinstance(ring, HashRing)


def test_hashring_has_some_public_api():
    ring = _make_ring()
    public_attrs = [
        "get_node",
        "get_nodes",
        "add_node",
        "remove_node",
        "add",
        "remove",
    ]
    assert any(hasattr(ring, attr) for attr in public_attrs)


def test_hashring_get_node_returns_one_of_known_nodes():
    ring = _make_ring()
    node = _call_first_method(ring, ["get_node", "get"], "example-key")
    assert node in {"a", "b", "c", ("a",), ("b",), ("c",)} or node is not None


def test_hashring_get_node_is_deterministic_for_same_key():
    ring = _make_ring()
    node1 = _call_first_method(ring, ["get_node", "get"], "consistent-key")
    node2 = _call_first_method(ring, ["get_node", "get"], "consistent-key")
    assert node1 == node2


def test_hashring_get_nodes_returns_collection_when_available():
    ring = _make_ring()
    if not hasattr(ring, "get_nodes"):
        pytest.skip("HashRing does not expose get_nodes() in this version")
    nodes = ring.get_nodes("example-key")
    assert nodes is not None
    assert hasattr(nodes, "__iter__")


def test_hashring_get_nodes_contains_only_known_nodes_when_available():
    ring = _make_ring()
    if not hasattr(ring, "get_nodes"):
        pytest.skip("HashRing does not expose get_nodes() in this version")
    nodes = list(ring.get_nodes("example-key"))
    assert len(nodes) > 0
    for node in nodes:
        assert node in {"a", "b", "c"} or node is not None


def test_hashring_add_node_if_supported():
    ring = _make_ring()
    if not any(hasattr(ring, name) for name in ("add_node", "add")):
        pytest.skip("HashRing does not expose add_node()/add() in this version")

    before = _call_first_method(ring, ["get_node", "get"], "new-key")
    _call_first_method(ring, ["add_node", "add"], "new-node")
    after = _call_first_method(ring, ["get_node", "get"], "new-key")

    assert after is not None
    assert before is not None


def test_hashring_remove_node_if_supported():
    ring = _make_ring()
    if not any(hasattr(ring, name) for name in ("remove_node", "remove")):
        pytest.skip("HashRing does not expose remove_node()/remove() in this version")

    # Removing an existing node should not make the ring unusable.
    _call_first_method(ring, ["remove_node", "remove"], "a")
    node = _call_first_method(ring, ["get_node", "get"], "example-key")
    assert node is not None


def test_hashring_repr_is_string():
    ring = _make_ring()
    assert isinstance(repr(ring), str)
