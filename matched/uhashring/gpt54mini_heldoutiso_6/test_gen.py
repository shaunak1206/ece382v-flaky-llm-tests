import importlib
import sys
import types

import pytest


def test_package_exports_hashring():
    import uhashring

    assert hasattr(uhashring, "HashRing")
    assert uhashring.HashRing is importlib.import_module("uhashring.ring").HashRing


def test_from_package_import_hashring_matches_module_attribute():
    from uhashring import HashRing

    import uhashring

    assert HashRing is uhashring.HashRing


def test_all_is_defined_on_package():
    import uhashring

    assert isinstance(uhashring.__all__, list)
    assert uhashring.__all__ == ["HashRing", "monkey"]


def test_hashring_is_listed_in_all():
    import uhashring

    assert "HashRing" in uhashring.__all__


def test_monkey_is_listed_in_all():
    import uhashring

    assert "monkey" in uhashring.__all__


def test_package_does_not_define_monkey_attribute():
    import uhashring

    assert not hasattr(uhashring, "monkey")


def test_package_module_name_is_uhashring():
    import uhashring

    assert uhashring.__name__ == "uhashring"


def test_package_has_single_hashring_binding():
    import uhashring

    attrs = [name for name in dir(uhashring) if name == "HashRing"]
    assert attrs == ["HashRing"]


def test_reimporting_package_preserves_hashring_identity():
    import uhashring

    module = importlib.reload(uhashring)
    assert module.HashRing is uhashring.HashRing


def test_importlib_import_module_returns_same_package_object():
    import uhashring

    module = importlib.import_module("uhashring")
    assert module is uhashring


def test_hashring_attribute_origin_is_ring_module():
    import uhashring
    import uhashring.ring as ring

    assert uhashring.HashRing is ring.HashRing


def test_package_can_be_imported_multiple_times_without_side_effects():
    first = importlib.import_module("uhashring")
    second = importlib.import_module("uhashring")
    assert first is second


def test_sys_modules_contains_package_after_import():
    import uhashring

    assert sys.modules["uhashring"] is uhashring


def test_sys_modules_contains_ring_after_import():
    import uhashring

    assert "uhashring.ring" in sys.modules
    assert isinstance(sys.modules["uhashring.ring"], types.ModuleType)


def test_hashring_binding_is_callable_or_class_like():
    import uhashring

    assert callable(uhashring.HashRing)


def test_package_all_is_not_mutated_by_access():
    import uhashring

    before = list(uhashring.__all__)
    _ = uhashring.HashRing
    after = list(uhashring.__all__)
    assert before == after == ["HashRing", "monkey"]
