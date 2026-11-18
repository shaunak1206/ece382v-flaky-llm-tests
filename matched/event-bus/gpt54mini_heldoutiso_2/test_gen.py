import importlib
import inspect

import event_bus
from event_bus import EventBus


def test_package_exports_eventbus_attribute():
    assert hasattr(event_bus, "EventBus")
    assert event_bus.EventBus is EventBus


def test_imported_eventbus_matches_bus_module_class():
    bus_module = importlib.import_module("event_bus.bus")
    assert EventBus is bus_module.EventBus


def test_package_version_is_defined():
    assert hasattr(event_bus, "__version__")
    assert event_bus.__version__ == "1.0.0"


def test_package_all_is_defined():
    assert hasattr(event_bus, "__all__")
    assert isinstance(event_bus.__all__, list)


def test_package_all_contains_eventbus_only():
    assert event_bus.__all__ == ["EventBus"]


def test_from_package_import_eventbus_works():
    namespace = {}
    exec("from event_bus import EventBus", namespace)
    assert "EventBus" in namespace
    assert namespace["EventBus"] is EventBus


def test_star_import_exports_eventbus_only():
    namespace = {}
    exec("from event_bus import *", namespace)
    exported = {k for k in namespace.keys() if not k.startswith("__")}
    assert exported == {"EventBus"}


def test_module_dir_includes_eventbus():
    assert "EventBus" in dir(event_bus)


def test_eventbus_is_callable_type_or_class():
    # EventBus is expected to be a class/type re-exported from event_bus.bus
    assert inspect.isclass(EventBus)


def test_eventbus_repr_contains_class_name():
    assert "EventBus" in repr(EventBus)


def test_reload_preserves_eventbus_reexport():
    reloaded = importlib.reload(event_bus)
    assert reloaded.EventBus is EventBus


def test_reload_preserves_version_and_all():
    reloaded = importlib.reload(event_bus)
    assert reloaded.__version__ == "1.0.0"
    assert reloaded.__all__ == ["EventBus"]
