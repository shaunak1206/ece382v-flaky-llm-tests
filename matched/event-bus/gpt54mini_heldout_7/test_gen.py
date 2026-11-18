import importlib
import sys
import types


def test_package_exports_eventbus():
    import event_bus

    assert hasattr(event_bus, "EventBus")
    assert event_bus.EventBus is not None


def test_eventbus_is_reexported_from_bus_module():
    import event_bus
    from event_bus.bus import EventBus as BusEventBus

    assert event_bus.EventBus is BusEventBus


def test_version_string_present():
    import event_bus

    assert event_bus.__version__ == "1.0.0"


def test_all_contains_only_eventbus():
    import event_bus

    assert event_bus.__all__ == ["EventBus"]


def test_star_import_exports_eventbus_only():
    ns = {}
    exec("from event_bus import *", ns)

    assert "EventBus" in ns
    assert ns["EventBus"] is not None
    assert "__version__" not in ns
    assert "__all__" not in ns


def test_from_import_eventbus_works():
    from event_bus import EventBus

    assert EventBus is not None


def test_eventbus_attribute_matches_from_import():
    import event_bus
    from event_bus import EventBus

    assert event_bus.EventBus is EventBus


def test_reload_preserves_public_metadata():
    import event_bus

    reloaded = importlib.reload(event_bus)

    assert reloaded.__version__ == "1.0.0"
    assert reloaded.__all__ == ["EventBus"]
    assert reloaded.EventBus is event_bus.EventBus


def test_module_has_expected_dunder_name():
    import event_bus

    assert event_bus.__name__ == "event_bus"


def test_eventbus_is_same_object_after_multiple_imports():
    import event_bus as first_import
    second_import = importlib.import_module("event_bus")

    assert first_import.EventBus is second_import.EventBus


def test_bus_module_is_loaded_after_importing_package():
    import event_bus

    assert "event_bus.bus" in sys.modules
    assert isinstance(sys.modules["event_bus.bus"], types.ModuleType)
    assert event_bus.EventBus is getattr(sys.modules["event_bus.bus"], "EventBus")


def test_package_all_can_be_used_for_manual_exports():
    import event_bus

    exported = {name: getattr(event_bus, name) for name in event_bus.__all__}

    assert list(exported.keys()) == ["EventBus"]
    assert exported["EventBus"] is event_bus.EventBus
