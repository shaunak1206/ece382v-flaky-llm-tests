import importlib
import sys


def test_package_exports_eventbus_attribute():
    import event_bus

    assert hasattr(event_bus, "EventBus")
    assert event_bus.EventBus is not None


def test_package_version_is_expected():
    import event_bus

    assert event_bus.__version__ == "1.0.0"


def test_package_all_is_expected():
    import event_bus

    assert event_bus.__all__ == ["EventBus"]


def test_package_all_contains_only_strings():
    import event_bus

    assert all(isinstance(name, str) for name in event_bus.__all__)


def test_from_package_import_eventbus_matches_package_attribute():
    import event_bus
    from event_bus import EventBus

    assert EventBus is event_bus.EventBus


def test_import_module_attribute_matches_direct_import():
    import event_bus
    module = importlib.import_module("event_bus")

    assert module.EventBus is event_bus.EventBus


def test_star_import_exposes_eventbus_only():
    namespace = {}
    exec("from event_bus import *", namespace)

    assert "EventBus" in namespace
    assert namespace["EventBus"] is importlib.import_module("event_bus").EventBus
    assert "__version__" not in namespace


def test_eventbus_is_present_in_module_dict():
    import event_bus

    assert "EventBus" in event_bus.__dict__
    assert event_bus.__dict__["EventBus"] is event_bus.EventBus


def test_imported_package_has_version_and_all_attributes():
    import event_bus

    assert hasattr(event_bus, "__version__")
    assert hasattr(event_bus, "__all__")
    assert isinstance(event_bus.__version__, str)
    assert isinstance(event_bus.__all__, list)


def test_reloading_package_picks_up_eventbus_from_bus_module(monkeypatch):
    import event_bus

    fake_bus_module = type(sys)("event_bus.bus")

    class FakeEventBus:
        pass

    fake_bus_module.EventBus = FakeEventBus
    monkeypatch.setitem(sys.modules, "event_bus.bus", fake_bus_module)

    reloaded = importlib.reload(event_bus)
    try:
        assert reloaded.EventBus is FakeEventBus
        assert reloaded.__all__ == ["EventBus"]
        assert reloaded.__version__ == "1.0.0"
    finally:
        # Restore the real package module state for any following tests.
        monkeypatch.undo()
        importlib.reload(event_bus)
