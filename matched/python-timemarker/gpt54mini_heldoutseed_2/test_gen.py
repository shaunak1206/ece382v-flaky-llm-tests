import importlib
import inspect
import io
import os
import sys
import types
from pathlib import Path

import pytest


def test_package_import_exposes_timemarker_class():
    import timemarker

    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is not None


def test_from_import_returns_same_object():
    from timemarker import TimeMarker

    import timemarker

    assert TimeMarker is timemarker.TimeMarker


def test_time_marker_is_a_class():
    from timemarker import TimeMarker

    assert inspect.isclass(TimeMarker)


def test_time_marker_module_points_to_timer_submodule():
    from timemarker import TimeMarker

    assert TimeMarker.__module__ == "timemarker.timer"


def test_time_marker_name_is_preserved():
    from timemarker import TimeMarker

    assert TimeMarker.__name__ == "TimeMarker"


def test_time_marker_qualname_is_preserved():
    from timemarker import TimeMarker

    assert TimeMarker.__qualname__.endswith("TimeMarker")


def test_time_marker_mro_contains_object():
    from timemarker import TimeMarker

    assert object in TimeMarker.mro()


def test_package_dir_contains_time_marker_attribute():
    import timemarker

    assert "TimeMarker" in dir(timemarker)


def test_package_import_is_idempotent_for_time_marker():
    import timemarker

    first = timemarker.TimeMarker
    reloaded = importlib.reload(timemarker)

    assert reloaded.TimeMarker is first


def test_timer_module_exports_same_class_as_package():
    import timemarker
    from timemarker.timer import TimeMarker as TimerTimeMarker

    assert timemarker.TimeMarker is TimerTimeMarker


def _read_init_source():
    package_dir = Path(__file__).resolve().parent.parent / "timemarker"
    init_path = package_dir / "__init__.py"
    return init_path.read_text(encoding="utf-8")


def test_init_source_has_supported_version_guard():
    source = _read_init_source()
    assert "assert version_info >= (3, 6)" in source
    assert "Currently only python >= 3.6 is supported" in source


def test_init_source_reexports_time_marker():
    source = _read_init_source()
    assert "from .timer import TimeMarker" in source


def test_version_guard_passes_for_supported_version(monkeypatch):
    source = _read_init_source().splitlines()
    guarded_prefix = "\n".join(source[:2])

    fake_sys = types.ModuleType("sys")
    fake_sys.version_info = (3, 8, 0)

    monkeypatch.setitem(sys.modules, "sys", fake_sys)
    ns = {}

    exec(guarded_prefix, ns, ns)

    assert ns["version_info"] == (3, 8, 0)


def test_version_guard_raises_for_unsupported_version(monkeypatch):
    source = _read_init_source().splitlines()
    guarded_prefix = "\n".join(source[:2])

    fake_sys = types.ModuleType("sys")
    fake_sys.version_info = (3, 5, 9)

    monkeypatch.setitem(sys.modules, "sys", fake_sys)
    ns = {}

    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        exec(guarded_prefix, ns, ns)
