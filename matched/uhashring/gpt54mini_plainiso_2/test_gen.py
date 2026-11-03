import importlib
import sys


def test_package_import_exposes_hashring():
    import uhashring

    assert hasattr(uhashring, "HashRing")


def test_package_all_is_defined():
    import uhashring

    assert hasattr(uhashring, "__all__")


def test_package_all_is_exact_expected_list():
    import uhashring

    assert uhashring.__all__ == ["HashRing", "monkey"]


def test_package_all_is_a_list():
    import uhashring

    assert isinstance(uhashring.__all__, list)


def test_hashring_is_reexported_from_ring_module():
    import uhashring
    from uhashring.ring import HashRing as RingHashRing

    assert uhashring.HashRing is RingHashRing


def test_from_import_hashring_matches_package_attribute():
    from uhashring import HashRing
    import uhashring

    assert HashRing is uhashring.HashRing


def test_hashring_module_is_uhashring_ring():
    import uhashring

    assert uhashring.HashRing.__module__ == "uhashring.ring"


def test_package_dict_contains_hashring():
    import uhashring

    assert "HashRing" in uhashring.__dict__
    assert uhashring.__dict__["HashRing"] is uhashring.HashRing


def test_import_module_returns_same_loaded_package_object():
    module1 = importlib.import_module("uhashring")
    module2 = sys.modules["uhashring"]

    assert module1 is module2


def test_reload_preserves_hashring_identity():
    import uhashring

    original = uhashring.HashRing
    reloaded = importlib.reload(uhashring)

    assert reloaded.HashRing is original


def test_reimporting_package_does_not_create_new_hashring_object():
    import uhashring

    first = uhashring.HashRing
    second = importlib.import_module("uhashring").HashRing

    assert first is second


def test_hashring_is_accessible_via_sys_modules_package_entry():
    import uhashring

    assert sys.modules["uhashring"].HashRing is uhashring.HashRing


def test_package_dir_includes_hashring():
    import uhashring

    assert "HashRing" in dir(uhashring)


def test_package_name_and_file_are_consistent():
    import uhashring

    assert uhashring.__name__ == "uhashring"
    assert uhashring.__file__.endswith("__init__.py")
