import importlib
import inspect

import pytest

import pypfm


def test_package_exports_pfml_loader():
    assert hasattr(pypfm, "PFMLoader")
    assert inspect.isclass(pypfm.PFMLoader)


def test_package_version_constant():
    assert hasattr(pypfm, "__version__")
    assert pypfm.__version__ == "1.0.0"


def test_pfml_loader_is_reexported_from_submodule():
    submodule = importlib.import_module("pypfm.pypfm")
    assert pypfm.PFMLoader is submodule.PFMLoader


def test_pfml_loader_module_name():
    assert pypfm.PFMLoader.__module__ == "pypfm.pypfm"


def test_package_allows_direct_import_of_pfml_loader():
    from pypfm import PFMLoader

    assert PFMLoader is pypfm.PFMLoader


def test_importing_package_is_idempotent():
    reloaded = importlib.import_module("pypfm")
    assert reloaded is pypfm
    assert reloaded.PFMLoader is pypfm.PFMLoader
    assert reloaded.__version__ == pypfm.__version__


def test_pfml_loader_has_callable_constructor():
    init = pypfm.PFMLoader.__init__
    assert callable(init)


def test_pfml_loader_constructor_signature_has_self():
    sig = inspect.signature(pypfm.PFMLoader.__init__)
    params = list(sig.parameters.values())
    assert params[0].name == "self"
    assert params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def test_pfml_loader_class_has_module_docstring_or_none():
    # This is a stable module-level property and does not depend on runtime state.
    assert pypfm.PFMLoader.__doc__ is None or isinstance(pypfm.PFMLoader.__doc__, str)


def test_package_does_not_override_builtins_import():
    # Sanity check for the package namespace behavior: the export should be the class object.
    assert pypfm.PFMLoader == getattr(pypfm, "PFMLoader")


def test_submodule_import_provides_same_loader_object():
    submodule = importlib.import_module("pypfm.pypfm")
    assert getattr(submodule, "PFMLoader") is pypfm.PFMLoader


def test_version_string_looks_like_semverish_constant():
    parts = pypfm.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
