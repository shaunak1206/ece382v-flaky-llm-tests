import importlib
import sys

import pytest

import pypfm


def test_package_has_version_attribute():
    assert hasattr(pypfm, "__version__")


def test_package_version_is_expected_string():
    assert pypfm.__version__ == "1.0.0"


def test_package_version_is_a_string():
    assert isinstance(pypfm.__version__, str)


def test_package_exposes_pfml_loader():
    assert hasattr(pypfm, "PFMLoader")


def test_pfml_loader_is_importable_from_package():
    from pypfm import PFMLoader

    assert PFMLoader is pypfm.PFMLoader


def test_pfml_loader_matches_submodule_symbol():
    from pypfm.pypfm import PFMLoader as submodule_loader

    assert pypfm.PFMLoader is submodule_loader


def test_package_module_origin_is_pypfm_init():
    assert pypfm.__name__ == "pypfm"


def test_reimporting_package_preserves_version():
    module = importlib.import_module("pypfm")
    assert module.__version__ == "1.0.0"


def test_reimporting_package_preserves_pfml_loader_identity():
    first = pypfm.PFMLoader
    module = importlib.reload(pypfm)
    assert module.PFMLoader is first


def test_from_import_works_in_exec_namespace():
    namespace = {}
    exec("from pypfm import PFMLoader", namespace)
    assert "PFMLoader" in namespace
    assert namespace["PFMLoader"] is pypfm.PFMLoader


def test_star_import_brings_in_pfml_loader_and_version():
    namespace = {}
    exec("from pypfm import *", namespace)
    assert namespace["PFMLoader"] is pypfm.PFMLoader
    assert namespace["__version__"] == "1.0.0"


def test_package_does_not_define_unexpected_builtin_version_type():
    assert not isinstance(pypfm.__version__, bytes)


def test_package_can_be_imported_when_already_loaded():
    sys.modules.pop("pypfm", None)
    module = importlib.import_module("pypfm")
    assert module.__version__ == "1.0.0"
    assert module.PFMLoader is importlib.import_module("pypfm.pypfm").PFMLoader
