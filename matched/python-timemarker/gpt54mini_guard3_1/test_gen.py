import importlib
import sys
from collections import namedtuple

import pytest


VersionInfo = namedtuple("VersionInfo", "major minor micro releaselevel serial")


def _clear_timemarker_modules():
    for name in list(sys.modules):
        if name == "timemarker" or name.startswith("timemarker."):
            sys.modules.pop(name, None)


def _fresh_import_timemarker(monkeypatch, version_info=None):
    _clear_timemarker_modules()
    if version_info is not None:
        monkeypatch.setattr(sys, "version_info", version_info, raising=False)
    return importlib.import_module("timemarker")


def test_import_exposes_timemarker_attribute(monkeypatch):
    tm = _fresh_import_timemarker(monkeypatch)
    assert hasattr(tm, "TimeMarker")
    assert tm.TimeMarker is not None


def test_from_import_matches_module_attribute(monkeypatch):
    tm = _fresh_import_timemarker(monkeypatch)
    from timemarker import TimeMarker

    assert TimeMarker is tm.TimeMarker


def test_time_marker_is_same_as_timer_module_class(monkeypatch):
    tm = _fresh_import_timemarker(monkeypatch)
    timer = importlib.import_module("timemarker.timer")

    assert tm.TimeMarker is timer.TimeMarker


def test_time_marker_name_is_stable(monkeypatch):
    tm = _fresh_import_timemarker(monkeypatch)

    assert tm.TimeMarker.__name__ == "TimeMarker"


def test_time_marker_qualname_contains_class_name(monkeypatch):
    tm = _fresh_import_timemarker(monkeypatch)

    assert "TimeMarker" in tm.TimeMarker.__qualname__


def test_time_marker_module_points_to_timer(monkeypatch):
    tm = _fresh_import_timemarker(monkeypatch)

    assert tm.TimeMarker.__module__ == "timemarker.timer"


def test_package_dir_includes_time_marker(monkeypatch):
    tm = _fresh_import_timemarker(monkeypatch)

    assert "TimeMarker" in dir(tm)


def test_package_namespace_includes_version_info(monkeypatch):
    tm = _fresh_import_timemarker(monkeypatch)

    assert "version_info" in tm.__dict__


def test_reload_keeps_time_marker_identity(monkeypatch):
    tm1 = _fresh_import_timemarker(monkeypatch)
    time_marker_1 = tm1.TimeMarker

    tm2 = importlib.reload(tm1)

    assert tm2.TimeMarker is time_marker_1


def test_import_succeeds_on_supported_version_36(monkeypatch):
    version = VersionInfo(3, 6, 0, "final", 0)
    tm = _fresh_import_timemarker(monkeypatch, version)

    assert tm.TimeMarker is not None


def test_import_succeeds_on_supported_version_38(monkeypatch):
    version = VersionInfo(3, 8, 10, "final", 0)
    tm = _fresh_import_timemarker(monkeypatch, version)

    assert tm.TimeMarker.__name__ == "TimeMarker"


@pytest.mark.parametrize(
    "version",
    [
        VersionInfo(3, 5, 9, "final", 0),
        VersionInfo(2, 7, 18, "final", 0),
        VersionInfo(3, 0, 0, "final", 0),
    ],
)
def test_import_fails_below_supported_version(monkeypatch, version):
    _clear_timemarker_modules()
    monkeypatch.setattr(sys, "version_info", version, raising=False)

    with pytest.raises(AssertionError, match="python >= 3\\.6"):
        importlib.import_module("timemarker")
