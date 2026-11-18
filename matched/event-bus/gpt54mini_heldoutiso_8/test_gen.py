import importlib
import inspect


def test_module_import_exposes_eventbus():
    import event_bus

    assert hasattr(event_bus, "EventBus")


def test_module_import_exposes_version():
    import event_bus

    assert hasattr(event_bus, "__version__")


def test_module_import_exposes_all():
    import event_bus

    assert hasattr(event_bus, "__all__")


def test_version_is_expected_string():
    import event_bus

    assert event_bus.__version__ == "1.0.0"


def test_all_contains_eventbus_only():
    import event_bus

    assert event_bus.__all__ == ["EventBus"]


def test_eventbus_is_imported_from_bus_module():
    import event_bus
    from event_bus.bus import EventBus as BusEventBus

    assert event_bus.EventBus is BusEventBus


def test_from_import_eventbus_matches_package_attribute():
    import event_bus
    from event_bus import EventBus

    assert EventBus is event_bus.EventBus


def test_eventbus_name_is_eventbus():
    from event_bus import EventBus

    assert EventBus.__name__ == "EventBus"


def test_eventbus_module_is_bus_submodule():
    from event_bus import EventBus

    assert EventBus.__module__ == "event_bus.bus"


def test_eventbus_is_class_or_callable():
    from event_bus import EventBus

    assert inspect.isclass(EventBus) or callable(EventBus)


def test_reload_keeps_same_public_api():
    import event_bus

    reloaded = importlib.reload(event_bus)

    assert reloaded.EventBus is event_bus.EventBus
    assert reloaded.__version__ == "1.0.0"
    assert reloaded.__all__ == ["EventBus"]


def test_package_dir_contains_eventbus_attribute():
    import event_bus

    assert "EventBus" in dir(event_bus)


def test_package_dir_contains_version_attribute():
    import event_bus

    assert "__version__" in dir(event_bus)


def test_package_dir_contains_all_attribute():
    import event_bus

    assert "__all__" in dir(event_bus)


def test_star_import_would_export_eventbus():
    import event_bus

    exported_names = set(event_bus.__all__)
    assert exported_names == {"EventBus"}
