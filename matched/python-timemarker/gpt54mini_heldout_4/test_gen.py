import importlib
import sys

import pytest

import timemarker
from timemarker import TimeMarker


def test_package_exports_timemarker_class():
    assert hasattr(timemarker, "TimeMarker")
    assert TimeMarker is timemarker.TimeMarker


def test_time_marker_is_a_class():
    assert isinstance(TimeMarker, type)


def test_time_marker_module_is_timer():
    assert TimeMarker.__module__ == "timemarker.timer"


def test_time_marker_name():
    assert TimeMarker.__name__ == "TimeMarker"


def test_time_marker_qualname_ends_with_class_name():
    assert TimeMarker.__qualname__.endswith("TimeMarker")


def test_import_from_package_and_module_match():
    timer_module = importlib.import_module("timemarker.timer")
    assert TimeMarker is timer_module.TimeMarker


def test_package_reexport_matches_timer_module_after_reload():
    timer_module = importlib.import_module("timemarker.timer")
    reloaded = importlib.reload(timer_module)
    assert timemarker.TimeMarker is reloaded.TimeMarker


def test_dir_contains_time_marker():
    assert "TimeMarker" in dir(timemarker)


def test_package_has_no_unexpected_time_marker_alias_object():
    assert getattr(timemarker, "TimeMarker") is TimeMarker


def test_imported_name_is_same_object_as_package_attribute():
    from timemarker import TimeMarker as imported_time_marker

    assert imported_time_marker is timemarker.TimeMarker


def test_reload_package_preserves_time_marker_identity():
    reloaded = importlib.reload(timemarker)
    assert reloaded.TimeMarker is TimeMarker


def test_version_check_allows_supported_version(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 6, 0), raising=False)
    reloaded = importlib.reload(timemarker)
    assert reloaded.TimeMarker is TimeMarker


def test_version_check_rejects_unsupported_version(monkeypatch):
    monkeypatch.setattr(sys, "version_info", (3, 5, 9), raising=False)

    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        importlib.reload(timemarker)

    # Restore the package for the rest of the test session.
    monkeypatch.setattr(sys, "version_info", sys.version_info, raising=False)
    importlib.reload(timemarker)
