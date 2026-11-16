"""Unit tests for the public surface of the pypfm package.

These tests cover the package entry point (pypfm/__init__.py) and the
PFMLoader symbol it re-exports. They check the import contract, the version
string, and the class identity between the package and its submodule.
"""

import importlib
import inspect
import os
import re
import types

import pytest

import pypfm
import pypfm.pypfm as pypfm_module
from pypfm import PFMLoader


SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def test_package_imports_successfully():
    assert isinstance(pypfm, types.ModuleType)


def test_pfmloader_is_importable_from_package():
    assert PFMLoader is not None


def test_pfmloader_is_a_class():
    assert inspect.isclass(PFMLoader)


def test_pfmloader_is_same_object_as_submodule_attribute():
    assert PFMLoader is pypfm_module.PFMLoader


def test_pfmloader_is_same_object_as_package_attribute():
    assert pypfm.PFMLoader is PFMLoader


def test_pfmloader_module_is_within_pypfm_package():
    assert PFMLoader.__module__.startswith("pypfm")


def test_pfmloader_can_be_subclassed():
    class SubLoader(PFMLoader):
        pass

    assert issubclass(SubLoader, PFMLoader)


def test_pfmloader_appears_in_star_import_namespace():
    namespace = {}
    exec("from pypfm import *", namespace)
    assert "PFMLoader" in namespace
    assert namespace["PFMLoader"] is PFMLoader


def test_version_attribute_exists():
    assert hasattr(pypfm, "__version__")


def test_version_is_string():
    assert isinstance(pypfm.__version__, str)


def test_version_is_not_empty():
    assert pypfm.__version__.strip() != ""


def test_version_matches_semver_format():
    assert SEMVER_PATTERN.match(pypfm.__version__) is not None


def test_version_value_is_1_0_0():
    assert pypfm.__version__ == "1.0.0"


def test_package_file_is_init_module():
    assert os.path.basename(pypfm.__file__) == "__init__.py"


def test_submodule_is_importable_by_name():
    module = importlib.import_module("pypfm.pypfm")
    assert module is pypfm_module


def test_reimporting_package_yields_same_loader_class():
    reloaded = importlib.reload(pypfm)
    assert reloaded.PFMLoader is PFMLoader


def test_pfmloader_is_not_a_module():
    assert not isinstance(PFMLoader, types.ModuleType)
