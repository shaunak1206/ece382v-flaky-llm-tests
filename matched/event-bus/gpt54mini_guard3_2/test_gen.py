import importlib
import sys

import event_bus
from event_bus import EventBus


def test_package_exports_eventbus():
    assert hasattr(event_bus, "EventBus")
    assert event_bus.EventBus is EventBus


def test_all_contains_only_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_version_value():
    assert event_bus.__version__ == "1.0.0"


def test_version_is_string():
    assert isinstance(event_bus.__version__, str)


def test_eventbus_is_class_like():
    assert isinstance(EventBus, type)


def test_eventbus_module_is_bus_module():
    assert EventBus.__module__ == "event_bus.bus"


def test_from_import_star_exposes_eventbus_only():
    namespace = {}
    exec("from event_bus import *", namespace, namespace)

    assert "EventBus" in namespace
    assert namespace["EventBus"] is EventBus
    assert "__version__" not in namespace
    assert "bus" not in namespace


def test_reimport_returns_same_module_object():
    module_before = sys.modules["event_bus"]
    module_after = importlib.import_module("event_bus")
    assert module_before is module_after


def test_reload_preserves_exports():
    reloaded = importlib.reload(event_bus)
    assert reloaded.EventBus is EventBus
    assert reloaded.__all__ == ["EventBus"]
    assert reloaded.__version__ == "1.0.0"


def test_reloaded_module_is_same_name_and_exports():
    reloaded = importlib.reload(event_bus)
    assert reloaded.__name__ == "event_bus"
    assert hasattr(reloaded, "EventBus")
    assert hasattr(reloaded, "__version__")
    assert hasattr(reloaded, "__all__")


def test_eventbus_object_same_via_module_and_from_import():
    imported_again = importlib.import_module("event_bus").EventBus
    assert imported_again is EventBus


def test_dir_contains_eventbus():
    assert "EventBus" in dir(event_bus)


def test_init_does_not_override_eventbus_reference_on_reload():
    before = event_bus.EventBus
    importlib.reload(event_bus)
    after = event_bus.EventBus
    assert before is after


def test_all_is_a_list():
    assert isinstance(event_bus.__all__, list)


def test_all_entry_is_exactly_eventbus():
    assert len(event_bus.__all__) == 1
    assert event_bus.__all__[0] == "EventBus"
