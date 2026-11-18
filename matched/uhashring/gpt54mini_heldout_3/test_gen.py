import importlib

import pytest


def test_package_exports_hashring_name():
    import uhashring

    assert "HashRing" in uhashring.__all__


def test_package_exports_monkey_name():
    import uhashring

    assert "monkey" in uhashring.__all__


def test_hashring_is_available_as_package_attribute():
    import uhashring

    assert hasattr(uhashring, "HashRing")


def test_hashring_is_class_or_type_like():
    from uhashring import HashRing

    assert isinstance(HashRing, type)


def test_hashring_import_from_package_matches_attribute():
    import uhashring
    from uhashring import HashRing

    assert HashRing is uhashring.HashRing


def test_hashring_import_from_package_matches_ring_module():
    from uhashring import HashRing
    from uhashring.ring import HashRing as RingHashRing

    assert HashRing is RingHashRing


def test_import_module_then_access_hashring():
    module = importlib.import_module("uhashring")

    assert module.HashRing is not None


def test_from_import_does_not_create_duplicate_hashring_class():
    from uhashring import HashRing as imported_hashring
    import uhashring

    assert imported_hashring is uhashring.HashRing


def test_hashring_class_name_is_preserved():
    from uhashring import HashRing

    assert HashRing.__name__ == "HashRing"


def test_hashring_module_is_ring_module():
    from uhashring import HashRing

    assert HashRing.__module__ == "uhashring.ring"


def test_package_has_all_attribute():
    import uhashring

    assert isinstance(uhashring.__all__, list)


def test_all_contains_only_expected_symbols():
    import uhashring

    assert uhashring.__all__ == ["HashRing", "monkey"]


def test_package_can_be_reimported_without_changing_hashring_identity():
    import uhashring

    first = uhashring.HashRing
    second = importlib.reload(uhashring).HashRing

    assert first is second


def test_hashring_is_same_object_after_direct_ring_reload():
    import uhashring
    import uhashring.ring as ring_module

    before = uhashring.HashRing
    importlib.reload(ring_module)

    # Re-import package to ensure the package-level alias still points to the
    # same class object exported by the ring module.
    reloaded_package = importlib.reload(uhashring)

    assert reloaded_package.HashRing is ring_module.HashRing
    assert before is reloaded_package.HashRing


def test_star_import_includes_hashring():
    namespace = {}
    exec("from uhashring import *", namespace)

    assert "HashRing" in namespace
    assert namespace["HashRing"].__name__ == "HashRing"


def test_star_import_includes_monkey_name_if_present():
    namespace = {}
    exec("from uhashring import *", namespace)

    assert "monkey" in namespace
