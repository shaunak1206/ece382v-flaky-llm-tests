import importlib
import inspect

import pytest

import uhashring


def test_hashring_is_exported_from_package():
    assert hasattr(uhashring, "HashRing")
    assert uhashring.HashRing is not None


def test_hashring_is_same_object_as_ring_hashring():
    ring_mod = importlib.import_module("uhashring.ring")
    assert uhashring.HashRing is ring_mod.HashRing


def test_hashring_name_is_hashring():
    assert uhashring.HashRing.__name__ == "HashRing"


def test_hashring_module_is_uhashring_ring():
    assert uhashring.HashRing.__module__ == "uhashring.ring"


def test_package_all_exists():
    assert hasattr(uhashring, "__all__")


def test_package_all_contains_hashring():
    assert "HashRing" in uhashring.__all__


def test_package_all_contains_monkey():
    assert "monkey" in uhashring.__all__


def test_package_all_has_expected_order():
    assert uhashring.__all__ == ["HashRing", "monkey"]


def test_direct_from_import_returns_same_class():
    namespace = {}
    exec("from uhashring import HashRing", namespace, namespace)
    assert namespace["HashRing"] is uhashring.HashRing


def test_import_module_attribute_matches_from_import():
    mod = importlib.import_module("uhashring")
    assert mod.HashRing is uhashring.HashRing


def test_reload_keeps_hashring_identity():
    reloaded = importlib.reload(uhashring)
    assert reloaded.HashRing is uhashring.HashRing


def test_hashring_is_a_class():
    assert inspect.isclass(uhashring.HashRing)


def test_hashring_is_defined_in_uhashring_package_namespace():
    assert "HashRing" in vars(uhashring)
    assert vars(uhashring)["HashRing"] is uhashring.HashRing


def test_package_can_be_imported_twice_with_same_hashring():
    mod1 = importlib.import_module("uhashring")
    mod2 = importlib.import_module("uhashring")
    assert mod1.HashRing is mod2.HashRing


def test_ring_module_exports_same_hashring():
    ring_mod = importlib.import_module("uhashring.ring")
    assert getattr(ring_mod, "HashRing") is uhashring.HashRing
