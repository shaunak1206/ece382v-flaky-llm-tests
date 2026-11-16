"""Pytest unit tests for the pypfm package (pypfm/__init__.py and PFMLoader).

Note: only pypfm/__init__.py was supplied. PFMLoader is imported from it, so these
tests check the package surface and PFMLoader's identity and type without assuming
its constructor or method signatures. The tests use no randomness, no clock, and
no hashing of str/bytes, and any module attribute changes go through monkeypatch.
"""

import inspect
import re

import pypfm
import pypfm.pypfm as pypfm_module
from pypfm import PFMLoader


def test_version_is_string():
    assert isinstance(pypfm.__version__, str)


def test_version_value_is_1_0_0():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_three_part_numeric():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pypfm.__version__) is not None


def test_version_parts_convert_to_int():
    major, minor, patch = (int(part) for part in pypfm.__version__.split("."))
    assert (major, minor, patch) == (1, 0, 0)


def test_package_name_is_pypfm():
    assert pypfm.__name__ == "pypfm"


def test_package_is_a_package_with_path():
    assert hasattr(pypfm, "__path__")


def test_pfmloader_is_exported_from_package():
    assert hasattr(pypfm, "PFMLoader")


def test_pfmloader_from_import_is_not_none():
    assert PFMLoader is not None


def test_pfmloader_is_a_class():
    assert inspect.isclass(PFMLoader)


def test_package_pfmloader_is_submodule_pfmloader():
    assert pypfm.PFMLoader is pypfm_module.PFMLoader


def test_pfmloader_defined_in_pypfm_submodule():
    assert PFMLoader.__module__ == "pypfm.pypfm"


def test_pfmloader_has_public_attributes():
    public = [name for name in dir(PFMLoader) if not name.startswith("_")]
    assert len(public) > 0


def test_dir_of_package_lists_public_names():
    names = dir(pypfm)
    assert "PFMLoader" in names
    assert "__version__" in names


def test_monkeypatch_version_is_restored_by_undo(monkeypatch):
    monkeypatch.setattr(pypfm, "__version__", "9.9.9")
    assert pypfm.__version__ == "9.9.9"
    monkeypatch.undo()
    assert pypfm.__version__ == "1.0.0"


def test_monkeypatch_replaces_pfmloader_on_package(monkeypatch):
    class DummyLoader:
        pass

    original = pypfm.PFMLoader
    monkeypatch.setattr(pypfm, "PFMLoader", DummyLoader)
    assert pypfm.PFMLoader is DummyLoader
    monkeypatch.undo()
    assert pypfm.PFMLoader is original
