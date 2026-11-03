import importlib
import sys

import uhashring
import uhashring.ring as ring_module


def test_package_exports_hashring_in_all():
    assert "HashRing" in uhashring.__all__


def test_package_exports_monkey_in_all():
    assert "monkey" in uhashring.__all__


def test_package_has_hashring_attribute():
    assert hasattr(uhashring, "HashRing")


def test_package_hashring_is_same_class_as_ring_module():
    assert uhashring.HashRing is ring_module.HashRing


def test_package_hashring_is_class():
    assert isinstance(uhashring.HashRing, type)


def test_hashring_class_name_is_preserved():
    assert uhashring.HashRing.__name__ == "HashRing"


def test_hashring_class_module_points_to_ring_module():
    assert uhashring.HashRing.__module__ == "uhashring.ring"


def test_from_package_import_returns_same_class():
    from uhashring import HashRing

    assert HashRing is uhashring.HashRing


def test_wildcard_import_exports_hashring():
    namespace = {}
    exec("from uhashring import *", namespace)
    assert namespace["HashRing"] is uhashring.HashRing


def test_wildcard_import_exports_monkey_name():
    namespace = {}
    exec("from uhashring import *", namespace)
    assert "monkey" in namespace


def test_reimporting_package_preserves_hashring_identity():
    before = uhashring.HashRing
    module = importlib.reload(uhashring)
    assert module.HashRing is before


def test_reimporting_package_preserves_all_contents():
    module = importlib.reload(uhashring)
    assert module.__all__ == ["HashRing", "monkey"]


def test_importing_package_does_not_replace_ring_module():
    before = sys.modules["uhashring.ring"]
    importlib.reload(uhashring)
    after = sys.modules["uhashring.ring"]
    assert after is before


def test_hashring_access_via_package_and_module_are_consistent_after_reload():
    importlib.reload(uhashring)
    assert uhashring.HashRing is ring_module.HashRing
