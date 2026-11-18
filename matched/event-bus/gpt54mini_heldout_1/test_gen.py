import importlib
import sys
import types


def _clear_event_bus_modules():
    for name in list(sys.modules):
        if name == "event_bus" or name.startswith("event_bus."):
            sys.modules.pop(name, None)


def test_package_exports_version():
    import event_bus

    assert hasattr(event_bus, "__version__")
    assert isinstance(event_bus.__version__, str)


def test_package_version_value():
    import event_bus

    assert event_bus.__version__ == "1.0.0"


def test_package_exports_all():
    import event_bus

    assert hasattr(event_bus, "__all__")
    assert isinstance(event_bus.__all__, list)


def test_package_all_contains_only_eventbus():
    import event_bus

    assert event_bus.__all__ == ["EventBus"]


def test_eventbus_is_reexported_from_package():
    import event_bus
    from event_bus.bus import EventBus as BusEventBus

    assert event_bus.EventBus is BusEventBus


def test_from_import_eventbus_matches_package_attribute():
    import event_bus
    from event_bus import EventBus

    assert EventBus is event_bus.EventBus


def test_star_import_respects_all(monkeypatch):
    _clear_event_bus_modules()

    fake_bus = types.ModuleType("event_bus.bus")

    class SentinelEventBus:
        pass

    fake_bus.EventBus = SentinelEventBus
    monkeypatch.setitem(sys.modules, "event_bus.bus", fake_bus)

    namespace = {}
    exec("from event_bus import *", namespace, namespace)

    assert "EventBus" in namespace
    assert namespace["EventBus"] is SentinelEventBus
    assert "__version__" not in namespace


def test_package_import_uses_bus_eventbus_from_sys_modules(monkeypatch):
    _clear_event_bus_modules()

    fake_bus = types.ModuleType("event_bus.bus")

    class FakeEventBus:
        pass

    fake_bus.EventBus = FakeEventBus
    monkeypatch.setitem(sys.modules, "event_bus.bus", fake_bus)

    module = importlib.import_module("event_bus")

    assert module.EventBus is FakeEventBus


def test_package_reimport_after_clear_still_exports_eventbus():
    _clear_event_bus_modules()
    module = importlib.import_module("event_bus")

    assert hasattr(module, "EventBus")
    assert module.__all__ == ["EventBus"]


def test_package_eventbus_name_is_class_like():
    import event_bus

    assert hasattr(event_bus.EventBus, "__name__")
    assert event_bus.EventBus.__name__ == "EventBus"


def test_import_eventbus_directly_from_package_namespace():
    namespace = {}
    exec("from event_bus import EventBus", namespace, namespace)

    assert "EventBus" in namespace
    assert namespace["EventBus"].__name__ == "EventBus"


def test_module_does_not_override_standard_dunder_attrs():
    import event_bus

    assert event_bus.__name__ == "event_bus"
    assert hasattr(event_bus, "__package__")
    assert hasattr(event_bus, "__file__")
