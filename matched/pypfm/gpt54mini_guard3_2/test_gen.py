import importlib
import inspect

import pypfm
from pypfm import PFMLoader


def test_import_module_exposes_pfml_loader():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is PFMLoader


def test_from_import_matches_module_attribute():
    assert PFMLoader is pypfm.PFMLoader


def test_pfml_loader_is_defined_in_pypfm_pypfm_module():
    import pypfm.pypfm as pypfm_impl

    assert pypfm_impl.PFMLoader is PFMLoader


def test_version_attribute_exists():
    assert hasattr(pypfm, "__version__")


def test_version_is_expected_string():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_string():
    assert isinstance(pypfm.__version__, str)


def test_pfml_loader_is_a_class_or_callable():
    assert inspect.isclass(PFMLoader) or callable(PFMLoader)


def test_pfml_loader_name_is_stable():
    assert PFMLoader.__name__ == "PFMLoader"


def test_pfml_loader_module_name_is_pypfm_pypfm():
    assert PFMLoader.__module__ == "pypfm.pypfm"


def test_reimporting_package_preserves_exported_loader_identity():
    reloaded = importlib.reload(pypfm)
    assert reloaded.PFMLoader is PFMLoader
    assert reloaded.__version__ == "1.0.0"
