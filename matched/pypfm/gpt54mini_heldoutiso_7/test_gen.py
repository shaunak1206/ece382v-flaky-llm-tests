import importlib
import inspect
import pkgutil
import types

import pypfm


def test_package_defines_version_string():
    assert hasattr(pypfm, "__version__")
    assert isinstance(pypfm.__version__, str)


def test_package_version_matches_expected():
    assert pypfm.__version__ == "1.0.0"


def test_package_exports_pfmloader():
    assert hasattr(pypfm, "PFMLoader")


def test_pfmloader_is_class():
    assert inspect.isclass(pypfm.PFMLoader)


def test_pfmloader_is_reexported_from_submodule():
    submodule = importlib.import_module("pypfm.pypfm")
    assert pypfm.PFMLoader is submodule.PFMLoader


def test_pfmloader_appears_in_package_namespace():
    assert "PFMLoader" in dir(pypfm)


def test_version_appears_in_package_namespace():
    assert "__version__" in dir(pypfm)


def test_importing_pfml_loader_from_package_works():
    from pypfm import PFMLoader

    assert PFMLoader is pypfm.PFMLoader


def test_star_import_includes_pfml_loader_and_version():
    namespace = {}
    exec("from pypfm import *", namespace)

    assert namespace["PFMLoader"] is pypfm.PFMLoader
    assert namespace["__version__"] == pypfm.__version__


def test_package_module_name_is_pypfm():
    assert pypfm.__name__ == "pypfm"


def test_package_file_points_to_init_py():
    assert pypfm.__file__.endswith("__init__.py")


def test_package_is_a_module_not_a_package_object():
    assert isinstance(pypfm, types.ModuleType)


def test_package_can_be_reloaded():
    reloaded = importlib.reload(pypfm)
    assert reloaded is pypfm
    assert reloaded.__version__ == "1.0.0"
    assert reloaded.PFMLoader is pypfm.PFMLoader


def test_submodule_is_importable():
    submodule = importlib.import_module("pypfm.pypfm")
    assert submodule.__name__ == "pypfm.pypfm"


def test_package_contains_at_least_one_module():
    modules = [m.name for m in pkgutil.iter_modules(pypfm.__path__)]
    assert "pypfm" in modules


def test_pfml_loader_symbol_identity_stable_across_imports():
    first = pypfm.PFMLoader
    second = importlib.import_module("pypfm").PFMLoader
    assert first is second
