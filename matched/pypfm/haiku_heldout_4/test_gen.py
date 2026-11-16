import importlib
import inspect
import re
import sys

import pypfm
from pypfm import PFMLoader


# Tests for pypfm/__init__.py and the PFMLoader re-export.
# The source of pypfm/pypfm.py was not provided, so these tests only check
# the public surface visible from pypfm/__init__.py.


def test_version_attribute_exists():
    assert hasattr(pypfm, "__version__")


def test_version_is_string():
    assert isinstance(pypfm.__version__, str)


def test_version_is_not_empty():
    assert pypfm.__version__.strip() != ""


def test_version_value_is_1_0_0():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_semantic_version_format():
    assert re.fullmatch(r"\d+\.\d+\.\d+", pypfm.__version__) is not None


def test_version_has_three_numeric_components():
    parts = pypfm.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_pfmloader_exported_from_package():
    assert hasattr(pypfm, "PFMLoader")


def test_pfmloader_is_a_class():
    assert inspect.isclass(PFMLoader)


def test_pfmloader_name_is_correct():
    assert PFMLoader.__name__ == "PFMLoader"


def test_pfmloader_defined_in_pypfm_submodule():
    assert PFMLoader.__module__.startswith("pypfm")


def test_pypfm_submodule_is_importable():
    module = importlib.import_module("pypfm.pypfm")
    assert module is not None


def test_pfmloader_matches_submodule_attribute():
    submodule = importlib.import_module("pypfm.pypfm")
    assert submodule.PFMLoader is PFMLoader


def test_pfmloader_identity_stable_across_imports():
    first = importlib.import_module("pypfm")
    second = sys.modules["pypfm"]
    assert first.PFMLoader is second.PFMLoader


def test_star_import_exposes_pfmloader():
    namespace = {}
    exec("from pypfm import *", namespace)
    assert "PFMLoader" in namespace
    assert namespace["PFMLoader"] is PFMLoader


def test_pfmloader_signature_is_introspectable():
    signature = inspect.signature(PFMLoader)
    assert isinstance(signature, inspect.Signature)
