import importlib
import pathlib
import sys
import types

import pytest

import timemarker


def test_package_exports_timemarker_name():
    assert hasattr(timemarker, "TimeMarker")
    assert timemarker.TimeMarker is not None


def test_time_marker_is_same_as_direct_timer_import():
    from timemarker.timer import TimeMarker as direct

    assert timemarker.TimeMarker is direct


def test_time_marker_is_a_class():
    assert isinstance(timemarker.TimeMarker, type)


def test_time_marker_module_is_timer_module():
    assert timemarker.TimeMarker.__module__ == "timemarker.timer"


def test_time_marker_has_expected_name():
    assert timemarker.TimeMarker.__name__ == "TimeMarker"


def test_time_marker_repr_mentions_class_name():
    rep = repr(timemarker.TimeMarker)
    assert "TimeMarker" in rep


def test_imported_class_is_present_in_dir():
    assert "TimeMarker" in dir(timemarker)


def test_reload_keeps_same_time_marker_object():
    before = timemarker.TimeMarker
    after = importlib.reload(timemarker).TimeMarker
    assert before is after


def test_package_source_contains_version_guard():
    init_path = pathlib.Path(timemarker.__file__).resolve()
    source = init_path.read_text(encoding="utf-8")
    assert "assert version_info >= (3, 6)" in source
    assert "Currently only python >= 3.6 is supported" in source


def _exec_init_in_fake_package(fake_version_info, fake_time_marker):
    init_path = pathlib.Path(timemarker.__file__).resolve()
    source = init_path.read_text(encoding="utf-8")

    package_name = "fake_timemarker_pkg"
    pkg = types.ModuleType(package_name)
    pkg.__path__ = []  # mark as package

    timer_mod = types.ModuleType(f"{package_name}.timer")
    timer_mod.TimeMarker = fake_time_marker

    old_pkg = sys.modules.get(package_name)
    old_timer = sys.modules.get(f"{package_name}.timer")

    try:
        sys.modules[package_name] = pkg
        sys.modules[f"{package_name}.timer"] = timer_mod

        glb = {
            "__name__": package_name,
            "__package__": package_name,
            "__file__": str(init_path),
            "__builtins__": __builtins__,
            "version_info": fake_version_info,
        }
        exec(compile(source, str(init_path), "exec"), glb, glb)
        return glb
    finally:
        if old_pkg is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = old_pkg

        if old_timer is None:
            sys.modules.pop(f"{package_name}.timer", None)
        else:
            sys.modules[f"{package_name}.timer"] = old_timer


def test_version_guard_allows_supported_version_in_isolation():
    sentinel = object()
    glb = _exec_init_in_fake_package((3, 6, 0), sentinel)
    assert glb["TimeMarker"] is sentinel


def test_version_guard_allows_newer_version_in_isolation():
    sentinel = object()
    glb = _exec_init_in_fake_package((3, 8, 0), sentinel)
    assert glb["TimeMarker"] is sentinel


def test_version_guard_rejects_older_version_in_isolation():
    with pytest.raises(AssertionError, match="Currently only python >= 3.6 is supported"):
        _exec_init_in_fake_package((3, 5, 9), object())
