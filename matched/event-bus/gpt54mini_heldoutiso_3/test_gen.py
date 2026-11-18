import importlib
import sys
import types

import event_bus
import event_bus.bus


def test_package_exposes_eventbus_attribute():
    assert hasattr(event_bus, "EventBus")


def test_package_eventbus_is_class_or_type_like():
    assert isinstance(event_bus.EventBus, type)


def test_package_all_contains_only_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_package_version_string_exists():
    assert hasattr(event_bus, "__version__")
    assert isinstance(event_bus.__version__, str)


def test_package_version_matches_expected():
    assert event_bus.__version__ == "1.0.0"


def test_eventbus_export_matches_bus_module_symbol():
    assert event_bus.EventBus is event_bus.bus.EventBus


def test_from_package_import_eventbus_returns_same_symbol():
    namespace = {}
    exec("from event_bus import EventBus", namespace)
    assert namespace["EventBus"] is event_bus.EventBus


def test_importing_package_does_not_remove_bus_submodule():
    assert "event_bus.bus" in sys.modules
    assert sys.modules["event_bus.bus"] is event_bus.bus


def test_reload_package_rebinds_eventbus_from_bus_module(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(event_bus.bus, "EventBus", sentinel)
    reloaded = importlib.reload(event_bus)
    assert reloaded.EventBus is sentinel


def test_reload_package_keeps_version_constant(monkeypatch):
    original_version = event_bus.__version__
    sentinel = object()
    monkeypatch.setattr(event_bus.bus, "EventBus", sentinel)
    reloaded = importlib.reload(event_bus)
    assert reloaded.__version__ == original_version


def test_reload_package_keeps_all_constant(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(event_bus.bus, "EventBus", sentinel)
    reloaded = importlib.reload(event_bus)
    assert reloaded.__all__ == ["EventBus"]


def test_star_import_exposes_eventbus_only():
    namespace = {}
    exec("from event_bus import *", namespace)
    public_names = {k for k in namespace.keys() if not k.startswith("__")}
    assert public_names == {"EventBus"}


def test_package_has_expected_minimal_public_api():
    public_names = {name for name in dir(event_bus) if not name.startswith("_")}
    assert "EventBus" in public_names
    assert "bus" in public_names
    assert "__version__" in dir(event_bus)
