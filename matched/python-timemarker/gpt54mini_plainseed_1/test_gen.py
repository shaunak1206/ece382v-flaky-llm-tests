import importlib
import sys

import pytest


def _fresh_import_timemarker(monkeypatch, version_info):
    """
    Import timemarker with a temporary sys.version_info and without cached modules.
    """
    monkeypatch.setattr(sys, "version_info", version_info)
    sys.modules.pop("timemarker", None)
    sys.modules.pop("timemarker.timer", None)
    return importlib.import_module("timemarker")


def test_import_succeeds_on_supported_version(monkeypatch):
    mod = _fresh_import_timemarker(monkeypatch, (3, 6, 0))
    assert hasattr(mod, "TimeMarker")


def test_import_succeeds_on_later_patch_version(monkeypatch):
    mod = _fresh_import_timemarker(monkeypatch, (3, 6, 1))
    assert hasattr(mod, "TimeMarker")


def test_import_succeeds_on_newer_minor_version(monkeypatch):
    mod = _fresh_import_timemarker(monkeypatch, (3, 8, 0))
    assert hasattr(mod, "TimeMarker")


def test_import_succeeds_on_newer_major_version(monkeypatch):
    mod = _fresh_import_timemarker(monkeypatch, (4, 0, 0))
    assert hasattr(mod, "TimeMarker")


def test_import_fails_on_python_35(monkeypatch):
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        _fresh_import_timemarker(monkeypatch, (3, 5, 10))


def test_import_fails_on_python_27(monkeypatch):
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        _fresh_import_timemarker(monkeypatch, (2, 7, 18))


def test_import_fails_on_python_30(monkeypatch):
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        _fresh_import_timemarker(monkeypatch, (3, 0, 0))


def test_import_fails_on_python_34(monkeypatch):
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        _fresh_import_timemarker(monkeypatch, (3, 4, 10))


def test_time_marker_is_reexported_from_timer_module(monkeypatch):
    mod = _fresh_import_timemarker(monkeypatch, (3, 8, 0))
    timer_mod = importlib.import_module("timemarker.timer")
    assert mod.TimeMarker is timer_mod.TimeMarker


def test_from_import_returns_same_time_marker_class(monkeypatch):
    mod = _fresh_import_timemarker(monkeypatch, (3, 8, 0))
    from timemarker import TimeMarker

    assert TimeMarker is mod.TimeMarker


def test_time_marker_attribute_is_accessible_after_import(monkeypatch):
    mod = _fresh_import_timemarker(monkeypatch, (3, 6, 0))
    assert getattr(mod, "TimeMarker") is not None


def test_time_marker_name_matches_exported_symbol(monkeypatch):
    mod = _fresh_import_timemarker(monkeypatch, (3, 9, 0))
    assert mod.TimeMarker.__name__ == "TimeMarker"


def test_reimport_after_supported_version_keeps_same_symbol(monkeypatch):
    mod1 = _fresh_import_timemarker(monkeypatch, (3, 6, 0))
    mod2 = _fresh_import_timemarker(monkeypatch, (3, 6, 0))
    assert mod1.TimeMarker is mod2.TimeMarker


def test_import_after_failed_version_can_succeed_with_supported_version(monkeypatch):
    with pytest.raises(AssertionError):
        _fresh_import_timemarker(monkeypatch, (3, 5, 9))

    mod = _fresh_import_timemarker(monkeypatch, (3, 6, 0))
    assert hasattr(mod, "TimeMarker")
