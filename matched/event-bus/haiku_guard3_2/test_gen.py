"""Pytest unit tests for the public surface of the event_bus package.

Only event_bus/__init__.py was provided, so these tests cover what that file
guarantees: the re-export of EventBus, __all__, and __version__. They also
check that EventBus is the class defined in event_bus.bus, without assuming
any of its methods. Behavioral tests for EventBus methods should be added
once event_bus/bus.py is available.

These tests use no randomness, no clock, and no ordering. Module attributes
are replaced only through monkeypatch.
"""

import importlib
import inspect
import re

import pytest

import event_bus
import event_bus.bus


SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def test_package_imports_successfully():
    module = importlib.import_module("event_bus")
    assert module is event_bus


def test_eventbus_is_exposed_on_package():
    assert hasattr(event_bus, "EventBus")


def test_eventbus_is_a_class():
    assert inspect.isclass(event_bus.EventBus)


def test_exported_eventbus_is_the_bus_module_class():
    assert event_bus.EventBus is event_bus.bus.EventBus


def test_eventbus_is_defined_in_bus_module():
    assert event_bus.EventBus.__module__ == "event_bus.bus"


def test_all_is_a_list():
    assert isinstance(event_bus.__all__, list)


def test_all_contains_only_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_every_name_in_all_is_an_attribute():
    for name in event_bus.__all__:
        assert hasattr(event_bus, name)


def test_star_import_exposes_only_eventbus():
    namespace = {}
    exec("from event_bus import *", namespace)
    exported = {key for key in namespace if key != "__builtins__"}
    assert exported == {"EventBus"}
    assert namespace["EventBus"] is event_bus.EventBus


def test_version_is_a_string():
    assert isinstance(event_bus.__version__, str)


def test_version_has_semver_shape():
    assert SEMVER_PATTERN.match(event_bus.__version__) is not None


def test_version_value_is_1_0_0():
    assert event_bus.__version__ == "1.0.0"


def test_dir_lists_eventbus_and_version():
    names = dir(event_bus)
    assert "EventBus" in names
    assert "__version__" in names


def test_reexport_binding_ignores_later_changes_to_bus_module(monkeypatch):
    sentinel = object()
    original = event_bus.EventBus
    monkeypatch.setattr(event_bus.bus, "EventBus", sentinel)
    # The package bound its name at import time, so it keeps the original class.
    assert event_bus.EventBus is original
    assert event_bus.EventBus is not sentinel


def test_package_attribute_replacement_is_visible_then_restored(monkeypatch):
    original = event_bus.EventBus
    replacement = type("Replacement", (), {})
    monkeypatch.setattr(event_bus, "EventBus", replacement)
    assert event_bus.EventBus is replacement
    monkeypatch.undo()
    assert event_bus.EventBus is original


@pytest.mark.parametrize("attribute", ["__version__", "__all__", "EventBus"])
def test_public_package_attributes_are_present(attribute):
    assert hasattr(event_bus, attribute)
