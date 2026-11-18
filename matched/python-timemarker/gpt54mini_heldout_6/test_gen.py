import importlib
import inspect
import sys

import pytest

import timemarker
import timemarker.timer


def test_package_exports_timemarker_name():
    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is not None


def test_from_import_gets_same_object_as_package_attribute():
    from timemarker import TimeMarker as imported_time_marker

    assert imported_time_marker is timemarker.TimeMarker


def test_package_reexport_matches_timer_module():
    assert timemarker.TimeMarker is timemarker.timer.TimeMarker


def test_timer_module_is_loaded_when_package_is_imported():
    assert "timemarker.timer" in sys.modules
    assert sys.modules["timemarker.timer"] is timemarker.timer


def test_init_source_contains_version_guard():
    source = inspect.getsource(timemarker)
    assert "assert version_info >= (3, 6)" in source
    assert "Currently only python >= 3.6 is supported" in source


def test_reload_succeeds_for_supported_python_version(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 8, 0))
    module = importlib.reload(timemarker)
    assert module.TimeMarker is timemarker.timer.TimeMarker


def test_reload_succeeds_for_python_36(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 6, 0))
    module = importlib.reload(timemarker)
    assert module.TimeMarker is timemarker.timer.TimeMarker


def test_reload_succeeds_for_python_37(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 7, 12))
    module = importlib.reload(timemarker)
    assert module.TimeMarker is timemarker.timer.TimeMarker


def test_reload_raises_for_python_35(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 5, 9))
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        importlib.reload(timemarker)


def test_reload_raises_for_python_34(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 4, 10))
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        importlib.reload(timemarker)
