import importlib
import inspect

import event_bus
from event_bus import EventBus
import event_bus.bus as bus_module


def test_package_exports_eventbus_attribute():
    assert hasattr(event_bus, "EventBus")


def test_package_eventbus_is_same_as_direct_import():
    assert event_bus.EventBus is EventBus


def test_eventbus_is_same_as_bus_module_class():
    assert EventBus is bus_module.EventBus


def test_package_all_contains_only_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_eventbus_name_is_eventbus():
    assert EventBus.__name__ == "EventBus"


def test_eventbus_qualname_endswith_eventbus():
    assert EventBus.__qualname__.endswith("EventBus")


def test_eventbus_module_is_bus_module():
    assert EventBus.__module__ == "event_bus.bus"


def test_version_is_string():
    assert isinstance(event_bus.__version__, str)


def test_version_matches_expected_value():
    assert event_bus.__version__ == "1.0.0"


def test_import_star_exports_eventbus_only_via_all():
    namespace = {}
    exec("from event_bus import *", namespace)
    exported_names = {
        name for name in namespace if not name.startswith("__")
    }
    assert exported_names == {"EventBus"}


def test_eventbus_is_class_like_object():
    assert inspect.isclass(EventBus)


def test_reimport_preserves_eventbus_identity():
    reloaded = importlib.reload(event_bus)
    assert reloaded.EventBus is EventBus
    assert reloaded.__version__ == "1.0.0"
    assert reloaded.__all__ == ["EventBus"]
