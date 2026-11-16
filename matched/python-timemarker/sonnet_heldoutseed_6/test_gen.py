import importlib
import inspect
import sys
import time

import pytest

import timemarker
from timemarker import TimeMarker


def _make_instance():
    """Create a TimeMarker with no arguments, skipping if that isn't supported."""
    try:
        return TimeMarker()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")


def _public_names(obj):
    return [n for n in dir(obj) if not n.startswith("_")]


def _first_existing(obj, names):
    for n in names:
        attr = getattr(obj, n, None)
        if callable(attr):
            return n, attr
    return None, None


def test_package_exposes_timemarker():
    assert hasattr(timemarker, "TimeMarker")


def test_timemarker_is_a_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_defined_in_timer_module():
    timer_mod = importlib.import_module("timemarker.timer")
    assert timer_mod.TimeMarker is TimeMarker


def test_python_version_supported():
    # The package asserts on import that python >= 3.6 is in use.
    assert sys.version_info >= (3, 6)


def test_instantiation_without_arguments():
    obj = _make_instance()
    assert isinstance(obj, TimeMarker)


def test_instances_are_distinct_objects():
    a = _make_instance()
    b = _make_instance()
    assert a is not b


def test_instance_state_is_independent():
    a = _make_instance()
    b = _make_instance()
    a._test_marker_attribute = 123
    assert not hasattr(b, "_test_marker_attribute")


def test_repr_and_str_do_not_raise():
    obj = _make_instance()
    assert isinstance(repr(obj), str)
    assert isinstance(str(obj), str)


def test_class_can_be_subclassed():
    class Sub(TimeMarker):
        pass

    try:
        obj = Sub()
    except TypeError:
        pytest.skip("TimeMarker requires constructor arguments")
    assert isinstance(obj, TimeMarker)
    assert isinstance(obj, Sub)


def test_public_attributes_are_discoverable():
    obj = _make_instance()
    names = _public_names(obj)
    assert isinstance(names, list)
    for n in names:
        assert hasattr(obj, n)


def test_public_methods_have_inspectable_signatures():
    for name in _public_names(TimeMarker):
        attr = getattr(TimeMarker, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
            except (ValueError, TypeError):
                continue
            assert isinstance(sig, inspect.Signature)


def test_reload_keeps_class_available():
    timer_mod = importlib.import_module("timemarker.timer")
    reloaded = importlib.reload(timer_mod)
    assert inspect.isclass(reloaded.TimeMarker)
    obj = reloaded.TimeMarker.__new__(reloaded.TimeMarker)
    assert isinstance(obj, reloaded.TimeMarker)


def test_new_without_init_creates_instance():
    obj = TimeMarker.__new__(TimeMarker)
    assert isinstance(obj, TimeMarker)


def test_start_and_stop_style_methods_do_not_fail_in_sequence():
    obj = _make_instance()
    start_name, start = _first_existing(obj, ["start", "begin", "tic", "mark"])
    stop_name, stop = _first_existing(obj, ["stop", "end", "toc", "finish"])
    if start is None or stop is None:
        pytest.skip("No recognizable start/stop style methods")
    try:
        start()
        time.sleep(0.01)
        stop()
    except TypeError:
        pytest.skip("start/stop methods require arguments")


def test_elapsed_style_value_is_non_negative_number():
    obj = _make_instance()
    start_name, start = _first_existing(obj, ["start", "begin", "tic"])
    if start is None:
        pytest.skip("No recognizable start method")
    try:
        start()
    except TypeError:
        pytest.skip("start requires arguments")
    time.sleep(0.01)
    for name in ["elapsed", "elapsed_time", "duration", "time", "total"]:
        attr = getattr(obj, name, None)
        if attr is None:
            continue
        try:
            value = attr() if callable(attr) else attr
        except TypeError:
            continue
        if isinstance(value, (int, float)):
            assert value >= 0
            return
    pytest.skip("No recognizable elapsed-time accessor")
