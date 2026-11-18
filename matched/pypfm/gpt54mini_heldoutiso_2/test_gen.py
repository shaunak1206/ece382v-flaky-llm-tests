import importlib
import sys
import types
import inspect

import pypfm


def test_version_is_exact_string():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_a_string():
    assert isinstance(pypfm.__version__, str)


def test_pfml_loader_is_exported_from_package():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is not None


def test_pfml_loader_is_a_class():
    assert inspect.isclass(pypfm.PFMLoader)


def test_pfml_loader_name_matches():
    assert pypfm.PFMLoader.__name__ == "PFMLoader"


def test_pfml_loader_module_is_pypfm_pypfm():
    assert pypfm.PFMLoader.__module__ == "pypfm.pypfm"


def test_from_import_returns_same_object():
    from pypfm import PFMLoader as imported

    assert imported is pypfm.PFMLoader


def test_submodule_attribute_matches_package_export():
    submodule = importlib.import_module("pypfm.pypfm")
    assert submodule.PFMLoader is pypfm.PFMLoader


def test_package_allows_direct_attribute_access():
    assert getattr(pypfm, "PFMLoader") is pypfm.PFMLoader


def test_reload_uses_current_submodule_pfml_loader_object(monkeypatch):
    original_submodule = sys.modules.get("pypfm.pypfm")
    fake_submodule = types.ModuleType("pypfm.pypfm")

    sentinel = object()
    fake_submodule.PFMLoader = sentinel

    monkeypatch.setitem(sys.modules, "pypfm.pypfm", fake_submodule)
    reloaded = importlib.reload(pypfm)

    assert reloaded.PFMLoader is sentinel

    if original_submodule is not None:
        monkeypatch.setitem(sys.modules, "pypfm.pypfm", original_submodule)
    importlib.reload(pypfm)


def test_reload_with_new_submodule_object_updates_export(monkeypatch):
    original_submodule = sys.modules.get("pypfm.pypfm")

    fake_submodule_1 = types.ModuleType("pypfm.pypfm")
    sentinel_1 = object()
    fake_submodule_1.PFMLoader = sentinel_1

    monkeypatch.setitem(sys.modules, "pypfm.pypfm", fake_submodule_1)
    importlib.reload(pypfm)
    assert pypfm.PFMLoader is sentinel_1

    fake_submodule_2 = types.ModuleType("pypfm.pypfm")
    sentinel_2 = object()
    fake_submodule_2.PFMLoader = sentinel_2

    monkeypatch.setitem(sys.modules, "pypfm.pypfm", fake_submodule_2)
    importlib.reload(pypfm)
    assert pypfm.PFMLoader is sentinel_2

    if original_submodule is not None:
        monkeypatch.setitem(sys.modules, "pypfm.pypfm", original_submodule)
    importlib.reload(pypfm)
