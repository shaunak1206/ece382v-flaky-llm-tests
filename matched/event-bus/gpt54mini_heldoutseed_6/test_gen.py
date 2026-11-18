import importlib
import types

import event_bus
from event_bus import EventBus
from event_bus.bus import EventBus as BusEventBus


def test_version_is_exposed_and_matches_expected():
    assert event_bus.__version__ == "1.0.0"


def test_all_is_exposed_and_contains_only_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_eventbus_is_reexported_from_bus_module():
    assert event_bus.EventBus is BusEventBus


def test_from_import_eventbus_returns_same_object():
    assert EventBus is BusEventBus
    assert EventBus is event_bus.EventBus


def test_eventbus_is_present_in_module_dict():
    assert "EventBus" in event_bus.__dict__
    assert event_bus.__dict__["EventBus"] is BusEventBus


def test_version_is_present_in_module_dict():
    assert "__version__" in event_bus.__dict__
    assert event_bus.__dict__["__version__"] == "1.0.0"


def test_all_is_present_in_module_dict():
    assert "__all__" in event_bus.__dict__
    assert event_bus.__dict__["__all__"] == ["EventBus"]


def test_star_import_exports_eventbus_only():
    namespace = {}
    exec("from event_bus import *", namespace)

    assert "EventBus" in namespace
    assert namespace["EventBus"] is BusEventBus
    assert "__version__" not in namespace
    assert "__all__" not in namespace


def test_reimporting_package_keeps_same_eventbus_object():
    reloaded = importlib.import_module("event_bus")
    assert reloaded.EventBus is event_bus.EventBus
    assert reloaded.EventBus is BusEventBus


def test_reexported_eventbus_is_class_like_object():
    assert isinstance(event_bus.EventBus, type)
    assert isinstance(EventBus, type)


def test_eventbus_attribute_lookup_matches_direct_import():
    assert getattr(event_bus, "EventBus") is EventBus


def test_package_name_is_event_bus():
    assert event_bus.__name__ == "event_bus"
