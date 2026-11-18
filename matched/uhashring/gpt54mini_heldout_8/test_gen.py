import importlib
import inspect

import uhashring
from uhashring import HashRing


def test_hashring_is_exported_from_package():
    assert hasattr(uhashring, "HashRing")
    assert uhashring.HashRing is HashRing


def test_hashring_is_same_as_ring_module_class():
    ring_module = importlib.import_module("uhashring.ring")
    assert HashRing is ring_module.HashRing


def test_hashring_is_a_class():
    assert inspect.isclass(HashRing)


def test_hashring_name_is_preserved():
    assert HashRing.__name__ == "HashRing"


def test_hashring_module_is_ring_module():
    assert HashRing.__module__ == "uhashring.ring"


def test_package_all_exists():
    assert hasattr(uhashring, "__all__")
    assert isinstance(uhashring.__all__, list)


def test_package_all_contains_hashring():
    assert "HashRing" in uhashring.__all__


def test_package_all_contains_monkey():
    assert "monkey" in uhashring.__all__


def test_dir_contains_hashring():
    assert "HashRing" in dir(uhashring)


def test_from_import_returns_same_object_on_reimport():
    mod = importlib.import_module("uhashring")
    assert mod.HashRing is HashRing


def test_hashring_can_be_subclassed():
    class MyHashRing(HashRing):
        pass

    assert issubclass(MyHashRing, HashRing)
    assert MyHashRing.__mro__[1] is HashRing


def test_hashring_attribute_access_through_package():
    package = importlib.import_module("uhashring")
    assert getattr(package, "HashRing") is HashRing
