import importlib
import sys

import pytest


def _fresh_import(monkeypatch, fake_version_info=None):
    """
    Import timemarker from scratch, optionally patching sys.version_info first.
    Returns the imported module.
    """
    for name in list(sys.modules):
        if name == "timemarker" or name.startswith("timemarker."):
            sys.modules.pop(name, None)

    if fake_version_info is not None:
        monkeypatch.setattr(sys, "version_info", fake_version_info, raising=False)

    importlib.invalidate_caches()
    return importlib.import_module("timemarker")


def test_import_timemarker_succeeds():
    import timemarker  # noqa: F401


def test_time_marker_is_exported_at_package_level():
    import timemarker

    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is not None


def test_time_marker_reexport_matches_timer_module():
    import timemarker
    from timemarker.timer import TimeMarker as timer_time_marker

    assert timemarker.TimeMarker is timer_time_marker


def test_from_import_time_marker_works():
    from timemarker import TimeMarker

    assert TimeMarker is not None


def test_module_exposes_version_info_name():
    import timemarker

    assert hasattr(timemarker, "version_info")
    assert timemarker.version_info is sys.version_info


def test_import_with_python_3_6_version_passes(monkeypatch):
    mod = _fresh_import(monkeypatch, fake_version_info=(3, 6, 0))

    assert mod.TimeMarker is not None


def test_import_with_python_3_5_version_fails(monkeypatch):
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        _fresh_import(monkeypatch, fake_version_info=(3, 5, 9))


def test_import_with_python_3_6_1_version_passes(monkeypatch):
    mod = _fresh_import(monkeypatch, fake_version_info=(3, 6, 1))

    assert hasattr(mod, "TimeMarker")


def test_import_with_python_3_7_version_passes(monkeypatch):
    mod = _fresh_import(monkeypatch, fake_version_info=(3, 7, 0))

    assert hasattr(mod, "TimeMarker")


def test_reimport_after_evicting_modules_keeps_same_object_identity():
    import timemarker

    first = timemarker.TimeMarker

    for name in list(sys.modules):
        if name == "timemarker" or name.startswith("timemarker."):
            sys.modules.pop(name, None)

    timemarker2 = importlib.import_module("timemarker")
    assert timemarker2.TimeMarker is first


def test_reload_timemarker_keeps_same_time_marker_object():
    import timemarker

    first = timemarker.TimeMarker
    reloaded = importlib.reload(timemarker)

    assert reloaded.TimeMarker is first
