import importlib
import inspect
import re
import sys
import types

import pytest

import pypfm
from pypfm import PFMLoader


SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def test_version_is_exact_expected_value():
    assert pypfm.__version__ == "1.0.0"


def test_version_is_string():
    assert isinstance(pypfm.__version__, str)


def test_version_matches_semver_format():
    assert SEMVER_PATTERN.match(pypfm.__version__) is not None


def test_version_attribute_is_stable_across_reload():
    before = pypfm.__version__
    reloaded = importlib.reload(pypfm)
    assert reloaded.__version__ == before == "1.0.0"


def test_pfmloader_is_exported_at_package_level():
    assert hasattr(pypfm, "PFMLoader")
    assert pypfm.PFMLoader is PFMLoader


def test_pfmloader_is_a_class():
    assert inspect.isclass(PFMLoader)


def test_pfmloader_is_not_none():
    assert PFMLoader is not None


def test_pfmloader_defined_in_submodule_pypfm():
    assert PFMLoader.__module__ == "pypfm.pypfm"


def test_package_init_reexports_same_object_as_submodule():
    submodule = sys.modules["pypfm.pypfm"]
    assert submodule.PFMLoader is pypfm.PFMLoader


def test_submodule_is_loaded_on_package_import():
    assert "pypfm.pypfm" in sys.modules
    assert isinstance(sys.modules["pypfm.pypfm"], types.ModuleType)


def test_package_module_file_lives_in_pypfm_directory():
    import os

    assert pypfm.__file__ is not None
    assert os.path.basename(os.path.dirname(pypfm.__file__)) == "pypfm"


def test_star_import_exposes_pfmloader():
    namespace = {}
    exec("from pypfm import *", namespace)
    assert "PFMLoader" in namespace
    assert namespace["PFMLoader"] is PFMLoader


def test_package_dir_lists_pfmloader_and_version():
    names = dir(pypfm)
    assert "PFMLoader" in names
    assert "__version__" in names


def test_pfmloader_has_public_interface():
    public = [name for name in dir(PFMLoader) if not name.startswith("_")]
    assert len(public) > 0


def test_pfmloader_class_is_type_instance():
    assert isinstance(PFMLoader, type)


def test_importing_pfmloader_twice_returns_same_class():
    from pypfm import PFMLoader as first
    from pypfm.pypfm import PFMLoader as second
    assert first is second


def test_accessing_missing_attribute_on_package_raises():
    with pytest.raises(AttributeError):
        getattr(pypfm, "this_attribute_does_not_exist_xyz")
