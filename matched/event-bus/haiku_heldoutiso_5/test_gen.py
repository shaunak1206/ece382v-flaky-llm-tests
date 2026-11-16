"""Unit tests for the EventBus class and the event_bus package surface.

Assumed API (event_bus/bus.py was not provided):
    EventBus()
    bus.subscribe(event_name, callback)
    bus.publish(event_name, *args, **kwargs)
    bus.unsubscribe(event_name, callback)

If the real method names differ, adjust the helper methods below.
"""
import pytest

import event_bus
from event_bus import EventBus
from event_bus.bus import EventBus as BusModuleEventBus


class Recorder:
    """Callable that records every invocation it receives."""

    def __init__(self, name="recorder", log=None):
        self.name = name
        self.log = log if log is not None else []
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.log.append(self.name)


@pytest.fixture
def bus():
    return EventBus()


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_version_is_1_0_0():
    assert event_bus.__version__ == "1.0.0"


def test_all_exports_only_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_package_eventbus_is_bus_module_eventbus():
    assert EventBus is BusModuleEventBus


def test_star_import_exposes_eventbus():
    namespace = {}
    exec("from event_bus import *", namespace)
    assert "EventBus" in namespace
    assert namespace["EventBus"] is EventBus


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_eventbus_can_be_instantiated_without_arguments():
    instance = EventBus()
    assert isinstance(instance, EventBus)


def test_two_instances_are_distinct_objects():
    assert EventBus() is not EventBus()


# ---------------------------------------------------------------------------
# Publishing and delivery
# ---------------------------------------------------------------------------

def test_publish_without_subscribers_does_not_raise(bus):
    bus.publish("nobody-listening", 1, 2, key="value")


def test_subscriber_receives_published_event(bus):
    recorder = Recorder()
    bus.subscribe("greet", recorder)

    bus.publish("greet")

    assert len(recorder.calls) == 1


def test_subscriber_receives_positional_arguments(bus):
    recorder = Recorder()
    bus.subscribe("sum", recorder)

    bus.publish("sum", 1, 2, 3)

    assert recorder.calls == [((1, 2, 3), {})]


def test_subscriber_receives_keyword_arguments(bus):
    recorder = Recorder()
    bus.subscribe("config", recorder)

    bus.publish("config", host="localhost", port=8080)

    assert recorder.calls == [((), {"host": "localhost", "port": 8080})]


def test_all_subscribers_to_event_are_called(bus):
    first = Recorder("first")
    second = Recorder("second")
    bus.subscribe("tick", first)
    bus.subscribe("tick", second)

    bus.publish("tick")

    assert len(first.calls) == 1
    assert len(second.calls) == 1


def test_subscribers_called_in_subscription_order(bus):
    log = []
    bus.subscribe("ordered", Recorder("a", log))
    bus.subscribe("ordered", Recorder("b", log))
    bus.subscribe("ordered", Recorder("c", log))

    bus.publish("ordered")

    assert log == ["a", "b", "c"]


def test_subscriber_not_called_for_other_event(bus):
    recorder = Recorder()
    bus.subscribe("alpha", recorder)

    bus.publish("beta")

    assert recorder.calls == []


def test_same_callback_can_subscribe_to_multiple_events(bus):
    recorder = Recorder()
    bus.subscribe("one", recorder)
    bus.subscribe("two", recorder)

    bus.publish("one", "x")
    bus.publish("two", "y")

    assert recorder.calls == [(("x",), {}), (("y",), {})]


def test_publish_repeatedly_delivers_each_time(bus):
    recorder = Recorder()
    bus.subscribe("repeat", recorder)

    for i in range(5):
        bus.publish("repeat", i)

    assert [args for args, _ in recorder.calls] == [(0,), (1,), (2,), (3,), (4,)]


# ---------------------------------------------------------------------------
# Unsubscribing
# ---------------------------------------------------------------------------

def test_unsubscribe_stops_delivery(bus):
    recorder = Recorder()
    bus.subscribe("event", recorder)
    bus.unsubscribe("event", recorder)

    bus.publish("event")

    assert recorder.calls == []


def test_unsubscribe_removes_only_target_callback(bus):
    kept = Recorder("kept")
    removed = Recorder("removed")
    bus.subscribe("event", kept)
    bus.subscribe("event", removed)

    bus.unsubscribe("event", removed)
    bus.publish("event")

    assert len(kept.calls) == 1
    assert removed.calls == []


def test_unsubscribe_does_not_affect_other_events(bus):
    recorder = Recorder()
    bus.subscribe("first", recorder)
    bus.subscribe("second", recorder)

    bus.unsubscribe("first", recorder)
    bus.publish("second", "still here")

    assert recorder.calls == [(("still here",), {})]


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_separate_bus_instances_do_not_share_subscribers():
    bus_a = EventBus()
    bus_b = EventBus()
    recorder = Recorder()
    bus_a.subscribe("shared-name", recorder)

    bus_b.publish("shared-name")

    assert recorder.calls == []
