import importlib
import inspect
import sys
from pathlib import Path

import pytest

import pypfm
from pypfm import PFMLoader


def test_package_import_exposes_version():
    assert hasattr(pypfm, "__version__")
    assert pypfm.__version__ == "1.0.0"


def test_package_import_exposes_pfm_loader():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is PFMLoader


def test_pfml_loader_is_a_class():
    assert inspect.isclass(PFMLoader)


def test_pfml_loader_module_is_pypfm_submodule():
    assert PFMLoader.__module__.startswith("pypfm")
    assert PFMLoader.__module__ == "pypfm.pypfm"


def test_package_init_exports_same_loader_as_submodule():
    submodule = importlib.import_module("pypfm.pypfm")
    assert hasattr(submodule, "PFMLoader")
    assert submodule.PFMLoader is PFMLoader


def test_package_has_expected_dunder_attributes():
    assert pypfm.__name__ == "pypfm"
    assert pypfm.__package__ == "pypfm"
    assert isinstance(pypfm.__file__, str)


def test_version_is_string_like_semver():
    version = pypfm.__version__
    assert isinstance(version, str)
    parts = version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_pfml_loader_has_a_helpful_repr():
    rep = repr(PFMLoader)
    assert "PFMLoader" in rep
    assert "class" in rep or "type" in rep


def test_pfml_loader_signature_is_introspectable():
    sig = inspect.signature(PFMLoader)
    assert isinstance(sig, inspect.Signature)


def test_pfml_loader_can_be_instantiated_without_required_args_if_signature_allows():
    sig = inspect.signature(PFMLoader)
    required_params = [
        p
        for p in sig.parameters.values()
        if p.default is inspect._empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    if required_params:
        pytest.skip("PFMLoader requires constructor arguments; signature-based smoke test only")

    instance = PFMLoader()
    assert isinstance(instance, PFMLoader)


def test_pfml_loader_instance_has_no_unexpected_module_mismatch():
    sig = inspect.signature(PFMLoader)
    required_params = [
        p
        for p in sig.parameters.values()
        if p.default is inspect._empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]

    if required_params:
        pytest.skip("PFMLoader requires constructor arguments; cannot validate instance module")

    instance = PFMLoader()
    assert instance.__class__ is PFMLoader
    assert instance.__class__.__module__ == "pypfm.pypfm"


def test_importing_package_does_not_shadow_builtin_imports():
    assert "builtins" in sys.modules
    assert "pypfm" in sys.modules
    assert sys.modules["pypfm"] is pypfm


def test_package_root_is_inside_project_tree():
    package_path = Path(pypfm.__file__).resolve()
    assert package_path.name == "__init__.py"
    assert package_path.parent.name == "pypfm"
    assert package_path.parent.parent.exists()
