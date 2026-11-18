import importlib
import sys

import pytest


def test_eventbus_is_reexported_from_package():
    import event_bus

    from event_bus.bus import EventBus

    assert event_bus.EventBus is EventBus
    assert event_bus.__all__ == ["EventBus"]


def test_version_is_defined_and_string():
    import event_bus

    assert isinstance(event_bus.__version__, str)
    assert event_bus.__version__ == "1.0.0"


def test_init_module_exports_only_eventbus():
    import event_bus

    assert "EventBus" in event_bus.__all__
    assert len(event_bus.__all__) == 1


def test_import_eventbus_from_package_imports_bus_module():
    import event_bus

    assert hasattr(event_bus, "EventBus")
    assert event_bus.EventBus.__module__ == "event_bus.bus"


def test_package_import_does_not_override_existing_bus_module(monkeypatch):
    import event_bus.bus as bus_module

    sentinel = object()
    monkeypatch.setattr(bus_module, "EventBus", sentinel, raising=True)

    import event_bus

    assert event_bus.EventBus is sentinel


def test_reimporting_package_preserves_export(monkeypatch):
    import event_bus
    import event_bus.bus as bus_module

    original = bus_module.EventBus
    sentinel = object()
    monkeypatch.setattr(bus_module, "EventBus", sentinel, raising=True)

    reloaded = importlib.reload(event_bus)
    assert reloaded.EventBus is sentinel
    assert reloaded.__all__ == ["EventBus"]

    monkeypatch.setattr(bus_module, "EventBus", original, raising=True)
    reloaded_again = importlib.reload(event_bus)
    assert reloaded_again.EventBus is original


def test_from_import_eventbus_matches_package_attribute():
    import event_bus
    from event_bus import EventBus

    assert EventBus is event_bus.EventBus


def test_package_module_has_expected_dunder_name():
    import event_bus

    assert event_bus.__name__ == "event_bus"


def test_package_attribute_eventbus_is_class_or_replacement():
    import event_bus

    assert event_bus.EventBus is not None


def test_eventbus_symbol_is_accessible_after_reload_of_bus(monkeypatch):
    import event_bus
    import event_bus.bus as bus_module

    original = bus_module.EventBus
    sentinel = object()
    monkeypatch.setattr(bus_module, "EventBus", sentinel, raising=True)

    bus_reloaded = importlib.reload(bus_module)
    assert hasattr(bus_reloaded, "EventBus")
    assert bus_reloaded.EventBus is not sentinel

    package_reloaded = importlib.reload(event_bus)
    assert package_reloaded.EventBus is bus_reloaded.EventBus

    monkeypatch.setattr(bus_module, "EventBus", original, raising=True)
    importlib.reload(event_bus)


def test_import_all_contains_eventbus_only_once():
    import event_bus

    assert event_bus.__all__.count("EventBus") == 1


def test_importing_package_does_not_require_extra_public_names():
    import event_bus

    public_names = [name for name in dir(event_bus) if not name.startswith("_")]
    assert "EventBus" in public_names
    assert "__version__" in dir(event_bus)


def test_eventbus_attribute_is_same_object_on_multiple_accesses():
    import event_bus

    first = event_bus.EventBus
    second = event_bus.EventBus
    assert first is second
