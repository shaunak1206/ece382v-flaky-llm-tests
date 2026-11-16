import inspect

import pytest

import event_bus
from event_bus import EventBus
from event_bus.bus import EventBus as BusModuleEventBus


# The subscribe/publish/unsubscribe method names are not visible in the
# provided source, so these helpers accept the common aliases.

def _subscribe(bus, event_name, handler):
    if hasattr(bus, "subscribe"):
        return bus.subscribe(event_name, handler)
    return bus.on(event_name, handler)


def _publish(bus, event_name, *args, **kwargs):
    if hasattr(bus, "publish"):
        return bus.publish(event_name, *args, **kwargs)
    return bus.emit(event_name, *args, **kwargs)


def _unsubscribe(bus, event_name, handler):
    if hasattr(bus, "unsubscribe"):
        return bus.unsubscribe(event_name, handler)
    return bus.off(event_name, handler)


class Recorder:
    """Handler that records every call it receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return None


@pytest.fixture
def bus():
    return EventBus()


# --- Package-level exports -------------------------------------------------

def test_version_is_defined():
    assert event_bus.__version__ == "1.0.0"


def test_all_exports_eventbus():
    assert "EventBus" in event_bus.__all__


def test_all_contains_only_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_eventbus_importable_from_package():
    assert inspect.isclass(EventBus)


def test_package_eventbus_is_bus_module_eventbus():
    assert EventBus is BusModuleEventBus


# --- Construction ----------------------------------------------------------

def test_instantiation_creates_instance(bus):
    assert isinstance(bus, EventBus)


def test_two_instances_are_distinct_objects():
    first = EventBus()
    second = EventBus()
    assert first is not second


def test_instances_do_not_share_subscribers():
    first = EventBus()
    second = EventBus()
    recorder = Recorder()
    _subscribe(first, "ping", recorder)
    _publish(second, "ping", 1)
    assert recorder.calls == []


# --- Subscribing and publishing --------------------------------------------

def test_published_event_invokes_subscriber(bus):
    recorder = Recorder()
    _subscribe(bus, "ping", recorder)
    _publish(bus, "ping")
    assert len(recorder.calls) == 1


def test_publish_passes_positional_arguments(bus):
    recorder = Recorder()
    _subscribe(bus, "data", recorder)
    _publish(bus, "data", 1, "two", 3.0)
    assert recorder.calls[0][0] == (1, "two", 3.0)


def test_publish_passes_keyword_arguments(bus):
    recorder = Recorder()
    _subscribe(bus, "data", recorder)
    _publish(bus, "data", key="value", count=2)
    assert recorder.calls[0][1] == {"key": "value", "count": 2}


def test_publish_invokes_every_subscriber(bus):
    first = Recorder()
    second = Recorder()
    _subscribe(bus, "news", first)
    _subscribe(bus, "news", second)
    _publish(bus, "news", "hello")
    assert len(first.calls) == 1
    assert len(second.calls) == 1


def test_subscribers_are_notified_in_subscription_order(bus):
    order = []

    def make_handler(name):
        def handler(*args, **kwargs):
            order.append(name)
        return handler

    _subscribe(bus, "tick", make_handler("a"))
    _subscribe(bus, "tick", make_handler("b"))
    _subscribe(bus, "tick", make_handler("c"))
    _publish(bus, "tick")
    assert order == ["a", "b", "c"]


def test_publish_only_notifies_subscribers_of_that_event(bus):
    ping_handler = Recorder()
    pong_handler = Recorder()
    _subscribe(bus, "ping", ping_handler)
    _subscribe(bus, "pong", pong_handler)
    _publish(bus, "ping")
    assert len(ping_handler.calls) == 1
    assert pong_handler.calls == []


def test_publish_without_subscribers_does_not_raise(bus):
    _publish(bus, "nobody_listening", 42)


def test_publish_repeated_delivers_each_time(bus):
    recorder = Recorder()
    _subscribe(bus, "tick", recorder)
    for i in range(3):
        _publish(bus, "tick", i)
    assert [args[0] for args, _ in recorder.calls] == [0, 1, 2]


# --- Unsubscribing ---------------------------------------------------------

def test_unsubscribed_handler_no_longer_receives_events(bus):
    recorder = Recorder()
    _subscribe(bus, "ping", recorder)
    _unsubscribe(bus, "ping", recorder)
    _publish(bus, "ping")
    assert recorder.calls == []


def test_unsubscribe_affects_only_target_handler(bus):
    kept = Recorder()
    removed = Recorder()
    _subscribe(bus, "ping", kept)
    _subscribe(bus, "ping", removed)
    _unsubscribe(bus, "ping", removed)
    _publish(bus, "ping")
    assert len(kept.calls) == 1
    assert removed.calls == []
