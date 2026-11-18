import importlib
import inspect
import types

import pytest

import uhashring


def test_hashring_is_exported_from_package():
    from uhashring import HashRing

    assert HashRing is uhashring.HashRing


def test_hashring_matches_ring_module_class():
    ring_module = importlib.import_module("uhashring.ring")

    assert uhashring.HashRing is ring_module.HashRing


def test_all_contains_hashring():
    assert "HashRing" in uhashring.__all__


def test_all_contains_monkey():
    assert "monkey" in uhashring.__all__


def test_all_is_list():
    assert isinstance(uhashring.__all__, list)


def test_module_has_hashring_attribute():
    assert hasattr(uhashring, "HashRing")


def test_hashring_attribute_is_class():
    assert inspect.isclass(uhashring.HashRing)


def test_hashring_can_be_imported_via_from_import():
    namespace = {}
    exec("from uhashring import HashRing", namespace)
    assert namespace["HashRing"] is uhashring.HashRing


def test_reimport_package_keeps_same_hashring_object():
    reloaded = importlib.reload(uhashring)
    assert reloaded.HashRing is uhashring.HashRing


def test_multiple_imports_return_same_hashring_class_object():
    first = uhashring.HashRing
    second = importlib.import_module("uhashring").HashRing
    third = __import__("uhashring").HashRing

    assert first is second is third


def test_hashring_in_package_namespace_is_not_a_module():
    assert not isinstance(uhashring.HashRing, types.ModuleType)


def test_hashring_name_and_qualname_are_reasonable():
    assert uhashring.HashRing.__name__ == "HashRing"
    assert "HashRing" in uhashring.HashRing.__qualname__


def test_star_import_exports_hashring():
    namespace = {}
    exec("from uhashring import *", namespace)

    assert "HashRing" in namespace
    assert namespace["HashRing"] is uhashring.HashRing
