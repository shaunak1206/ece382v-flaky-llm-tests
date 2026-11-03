import importlib
import sys
import types

import event_bus


def test_version_is_defined_and_expected():
    assert hasattr(event_bus, "__version__")
    assert event_bus.__version__ == "1.0.0"


def test_all_is_defined_and_list():
    assert hasattr(event_bus, "__all__")
    assert isinstance(event_bus.__all__, list)


def test_all_contains_eventbus_only():
    assert event_bus.__all__ == ["EventBus"]


def test_eventbus_is_exported_at_package_level():
    assert hasattr(event_bus, "EventBus")
    assert event_bus.EventBus is not None


def test_eventbus_is_same_object_as_bus_module_attribute():
    bus_module = importlib.import_module("event_bus.bus")
    assert event_bus.EventBus is bus_module.EventBus


def test_from_import_eventbus_returns_same_object():
    from event_bus import EventBus as imported_eventbus

    assert imported_eventbus is event_bus.EventBus


def test_star_import_exposes_only_eventbus(monkeypatch):
    namespace = {}
    exec("from event_bus import *", namespace)

    assert "EventBus" in namespace
    assert namespace["EventBus"] is event_bus.EventBus

    public_keys = {k for k in namespace.keys() if not k.startswith("__")}
    assert public_keys == {"EventBus"}


def test_package_module_name():
    assert event_bus.__name__ == "event_bus"


def test_package_has_expected_dunder_file_attribute():
    assert hasattr(event_bus, "__file__")
    assert event_bus.__file__


def test_package_has_expected_dunder_package_attribute():
    assert getattr(event_bus, "__package__", None) == "event_bus"


def test_reimport_does_not_change_exported_eventbus():
    first = event_bus.EventBus
    reloaded = importlib.reload(event_bus)
    assert reloaded.EventBus is first


def test_importing_package_after_removing_from_sys_modules_keeps_interface():
    original_module = sys.modules.get("event_bus")
    try:
        sys.modules.pop("event_bus", None)
        reloaded = importlib.import_module("event_bus")
        assert reloaded.__version__ == "1.0.0"
        assert reloaded.__all__ == ["EventBus"]
        assert hasattr(reloaded, "EventBus")
    finally:
        if original_module is not None:
            sys.modules["event_bus"] = original_module
        importlib.reload(event_bus)


def test_eventbus_attribute_is_documented_by_all():
    assert "EventBus" in event_bus.__all__
    assert getattr(event_bus, event_bus.__all__[0]) is event_bus.EventBus
