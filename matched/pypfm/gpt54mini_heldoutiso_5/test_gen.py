import importlib
import inspect

import pypfm
from pypfm import PFMLoader


def test_package_has_version_attribute():
    assert hasattr(pypfm, "__version__")
    assert isinstance(pypfm.__version__, str)


def test_package_version_matches_expected_value():
    assert pypfm.__version__ == "1.0.0"


def test_package_exposes_pfmloder_name():
    assert hasattr(pypfm, "PFMLoader")


def test_pfmloder_imported_from_package_is_same_object():
    assert PFMLoader is pypfm.PFMLoader


def test_pfmloder_is_class():
    assert inspect.isclass(PFMLoader)


def test_pfmloder_is_same_as_submodule_export():
    submodule = importlib.import_module("pypfm.pypfm")
    assert PFMLoader is submodule.PFMLoader


def test_pfmloder_module_attribute_points_to_submodule():
    assert PFMLoader.__module__ == "pypfm.pypfm"


def test_package_import_from_star_exports_pfmloder_and_version():
    namespace = {}
    exec("from pypfm import *", namespace)
    assert "PFMLoader" in namespace
    assert "pypfm" not in namespace
    assert namespace["PFMLoader"] is PFMLoader
    assert namespace["__version__"] == "1.0.0"


def test_reimporting_package_preserves_pfmloder_identity():
    reloaded = importlib.reload(pypfm)
    assert reloaded.PFMLoader is PFMLoader
    assert reloaded.__version__ == "1.0.0"


def test_pfmloder_has_a_call_or_init_interface():
    # The loader should be instantiable or at least behave like a normal class.
    assert hasattr(PFMLoader, "__init__")
    assert callable(PFMLoader)
    assert inspect.signature(PFMLoader.__init__) is not None


def test_package_allows_direct_attribute_access_after_reload():
    reloaded = importlib.reload(pypfm)
    assert getattr(reloaded, "PFMLoader") is PFMLoader
    assert getattr(reloaded, "__version__") == "1.0.0"
