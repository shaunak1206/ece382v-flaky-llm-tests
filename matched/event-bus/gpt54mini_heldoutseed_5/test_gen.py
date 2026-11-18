import importlib
import inspect
import types

import event_bus
from event_bus import EventBus


def test_package_exports_eventbus_in_all():
    assert hasattr(event_bus, "__all__")
    assert event_bus.__all__ == ["EventBus"]


def test_package_has_version_string():
    assert hasattr(event_bus, "__version__")
    assert event_bus.__version__ == "1.0.0"


def test_eventbus_is_importable_from_package():
    assert EventBus is not None


def test_eventbus_is_same_as_bus_module_class():
    bus_module = importlib.import_module("event_bus.bus")
    assert EventBus is bus_module.EventBus


def test_eventbus_name_is_correct():
    assert EventBus.__name__ == "EventBus"


def test_eventbus_qualname_ends_with_eventbus():
    assert EventBus.__qualname__.endswith("EventBus")


def test_eventbus_module_attribute_points_to_bus_module():
    assert EventBus.__module__ == "event_bus.bus"


def test_eventbus_is_a_class():
    assert inspect.isclass(EventBus)


def test_eventbus_can_be_imported_via_from_import():
    namespace = {}
    exec("from event_bus import EventBus as ImportedEventBus", namespace)
    assert namespace["ImportedEventBus"] is EventBus


def test_reimporting_package_preserves_eventbus_identity():
    reloaded = importlib.reload(event_bus)
    assert reloaded.EventBus is EventBus


def test_reimporting_bus_module_preserves_eventbus_identity():
    bus_module = importlib.import_module("event_bus.bus")
    reloaded_bus = importlib.reload(bus_module)
    assert reloaded_bus.EventBus is EventBus


def test_import_star_respects_all():
    namespace = {}
    exec("from event_bus import *", namespace)
    assert "EventBus" in namespace
    assert namespace["EventBus"] is EventBus
    assert "__version__" not in namespace


def test_package_eventbus_attribute_matches_direct_import():
    assert getattr(event_bus, "EventBus") is EventBus


def test_eventbus_is_not_a_module():
    assert not isinstance(EventBus, types.ModuleType)


def test_eventbus_can_be_referenced_from_dir_output():
    assert "EventBus" in dir(event_bus)
