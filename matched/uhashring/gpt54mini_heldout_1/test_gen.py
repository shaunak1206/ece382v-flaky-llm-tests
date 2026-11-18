import importlib
import sys

import pytest


def test_package_import_exposes_hashring():
    import uhashring

    assert hasattr(uhashring, "HashRing")
    assert uhashring.HashRing is not None


def test_hashring_reexport_matches_ring_module():
    import uhashring
    from uhashring.ring import HashRing as RingHashRing

    assert uhashring.HashRing is RingHashRing


def test_from_package_import_hashring_matches_package_attribute():
    import uhashring
    from uhashring import HashRing as ImportedHashRing

    assert ImportedHashRing is uhashring.HashRing


def test_hashring_class_is_defined_in_ring_module():
    from uhashring import HashRing

    assert HashRing.__module__ == "uhashring.ring"


def test_hashring_class_name_is_hashring():
    from uhashring import HashRing

    assert HashRing.__name__ == "HashRing"


def test_package_all_is_defined():
    import uhashring

    assert hasattr(uhashring, "__all__")
    assert isinstance(uhashring.__all__, list)


def test_package_all_contains_hashring():
    import uhashring

    assert "HashRing" in uhashring.__all__


def test_package_all_contains_monkey():
    import uhashring

    assert "monkey" in uhashring.__all__


def test_package_does_not_define_monkey_attribute_by_default():
    import uhashring

    assert not hasattr(uhashring, "monkey")


def test_reload_keeps_same_hashring_object():
    import uhashring

    first = uhashring.HashRing
    reloaded = importlib.reload(uhashring)

    assert reloaded.HashRing is first


def test_importing_package_via_sys_modules_returns_same_module():
    import uhashring

    assert sys.modules["uhashring"] is uhashring


def test_star_import_fails_due_to_missing_monkey_name():
    namespace = {}
    with pytest.raises((AttributeError, ImportError, NameError)):
        exec("from uhashring import *", namespace)
