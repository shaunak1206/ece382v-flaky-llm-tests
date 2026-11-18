import importlib
import sys
import types

import pytest


def test_package_import_exposes_pfml_loader():
    import pypfm

    assert hasattr(pypfm, "PFMLoader")


def test_package_import_exposes_version():
    import pypfm

    assert hasattr(pypfm, "__version__")
    assert pypfm.__version__ == "1.0.0"


def test_pfml_loader_is_class():
    import pypfm

    assert isinstance(pypfm.PFMLoader, type)


def test_pfml_loader_name_is_correct():
    import pypfm

    assert pypfm.PFMLoader.__name__ == "PFMLoader"


def test_pfml_loader_module_is_pypfm_submodule():
    import pypfm

    assert pypfm.PFMLoader.__module__ == "pypfm.pypfm"


def test_pfml_loader_is_same_object_as_submodule_export():
    import pypfm
    from pypfm.pypfm import PFMLoader as submodule_loader

    assert pypfm.PFMLoader is submodule_loader


def test_import_star_brings_pfml_loader_into_namespace():
    namespace = {}
    exec("from pypfm import *", namespace)

    assert "PFMLoader" in namespace
    assert namespace["PFMLoader"].__name__ == "PFMLoader"


def test_import_star_does_not_override_builtin_name_by_default():
    namespace = {"__builtins__": __builtins__}
    exec("from pypfm import *", namespace)

    assert "PFMLoader" in namespace
    assert "__version__" not in namespace


def test_package_module_can_be_reimported_and_same_version():
    import pypfm

    reloaded = importlib.reload(pypfm)

    assert reloaded.__version__ == "1.0.0"
    assert reloaded.PFMLoader is pypfm.PFMLoader


def test_module_name_is_pypfm():
    import pypfm

    assert pypfm.__name__ == "pypfm"


def test_package_has_expected_minimum_attributes():
    import pypfm

    assert hasattr(pypfm, "PFMLoader")
    assert hasattr(pypfm, "__version__")


def test_importing_package_does_not_create_unexpected_modules():
    before = set(sys.modules.keys())

    import pypfm  # noqa: F401

    after = set(sys.modules.keys())
    new_modules = after - before

    assert "pypfm" in sys.modules
    assert "pypfm.pypfm" in sys.modules
    assert "pypfm" in new_modules or "pypfm.pypfm" in new_modules
