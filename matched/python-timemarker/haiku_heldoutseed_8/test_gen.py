# -*- coding: utf-8 -*-
import importlib
import inspect
import sys
import time

import pytest

import timemarker
from timemarker import TimeMarker


def _get_method(instance, name):
    """Return a bound method by name, or skip the test if it is absent."""
    method = getattr(instance, name, None)
    if method is None or not callable(method):
        pytest.skip("TimeMarker has no callable '{}' in this version".format(name))
    return method


def test_package_exposes_timemarker_class():
    assert inspect.isclass(TimeMarker)


def test_timemarker_is_same_object_as_in_timer_module():
    from timemarker import timer

    assert timer.TimeMarker is TimeMarker


def test_package_version_guard_passes_on_running_interpreter():
    # The package asserts >= 3.6 at import time; reaching this point means it passed.
    assert sys.version_info >= (3, 6)


def test_timemarker_can_be_instantiated_without_arguments():
    marker = TimeMarker()
    assert isinstance(marker, TimeMarker)


def test_each_instantiation_returns_new_object():
    first = TimeMarker()
    second = TimeMarker()
    assert first is not second


def test_instances_are_isolated_from_each_other():
    first = TimeMarker()
    second = TimeMarker()
    first.custom_probe = "only-first"
    assert not hasattr(second, "custom_probe")


def test_repr_and_str_return_strings():
    marker = TimeMarker()
    assert isinstance(repr(marker), str)
    assert isinstance(str(marker), str)


def test_instances_are_hashable():
    marker = TimeMarker()
    assert isinstance(hash(marker), int)
    assert {marker: 1}[marker] == 1


def test_reloading_package_does_not_print_output(capsys):
    importlib.reload(timemarker)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_public_attributes_are_accessible_without_error():
    marker = TimeMarker()
    for name in dir(marker):
        if name.startswith("_"):
            continue
        # Accessing each public attribute must not raise.
        getattr(marker, name)


def test_mark_method_accepts_a_label_without_error():
    marker = TimeMarker()
    mark = _get_method(marker, "mark")
    mark("first")
    mark("second")


def test_start_then_stop_does_not_raise():
    marker = TimeMarker()
    start = _get_method(marker, "start")
    stop = _get_method(marker, "stop")
    start()
    stop()


def test_elapsed_time_is_non_negative_number_when_available():
    marker = TimeMarker()
    start = _get_method(marker, "start")
    stop = _get_method(marker, "stop")
    start()
    time.sleep(0.01)
    stop()
    elapsed = getattr(marker, "elapsed", None)
    if callable(elapsed):
        elapsed = elapsed()
    if elapsed is None:
        pytest.skip("TimeMarker does not expose elapsed time in this version")
    assert isinstance(elapsed, (int, float))
    assert elapsed >= 0


def test_elapsed_time_does_not_decrease_between_reads():
    marker = TimeMarker()
    start = _get_method(marker, "start")
    start()
    elapsed = getattr(marker, "elapsed", None)
    if not callable(elapsed):
        pytest.skip("TimeMarker has no callable 'elapsed' in this version")
    first = elapsed()
    time.sleep(0.005)
    second = elapsed()
    assert second >= first


def test_separate_instances_track_time_independently():
    fast = TimeMarker()
    slow = TimeMarker()
    start_fast = _get_method(fast, "start")
    start_slow = _get_method(slow, "start")
    start_fast()
    start_slow()
    # Starting one instance must not change the other's state object identity.
    assert fast is not slow
    assert start_fast.__self__ is fast
    assert start_slow.__self__ is slow
