"""
Unit tests for event_bus.EventBus.

Assumed API (event_bus/bus.py was not provided, so these tests use the
common pub/sub method names):
    EventBus()
    bus.subscribe(event_name, handler)
    bus.publish(event_name, *args, **kwargs)
    bus.unsubscribe(event_name, handler)
If the real method names differ, adjust the calls in the helpers below.
"""

import pytest

import event_bus
from event_bus import EventBus as ExportedEventBus
from event_bus.bus import EventBus


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def recorder():
    """Returns a handler that records every call it receives."""
    class Recorder:
        def __init__(self):
            self.calls = []

        def __call__(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    return Recorder()


# --- package export / metadata -------------------------------------------

def test_package_exports_eventbus_class():
    assert ExportedEventBus is EventBus


def test_package_all_contains_eventbus():
    assert "EventBus" in event_bus.__all__


def test_package_version_is_nonempty_string():
    assert isinstance(event_bus.__version__, str)
    assert event_bus.__version__ == "1.0.0"


# --- construction ---------------------------------------------------------

def test_new_bus_is_eventbus_instance(bus):
    assert isinstance(bus, EventBus)


def test_two_buses_are_distinct_objects():
    assert EventBus() is not EventBus()


# --- subscribe / publish --------------------------------------------------

def test_publish_invokes_subscribed_handler(bus, recorder):
    bus.subscribe("user.created", recorder)
    bus.publish("user.created")
    assert len(recorder.calls) == 1


def test_publish_passes_positional_and_keyword_arguments(bus, recorder):
    bus.subscribe("order.placed", recorder)
    bus.publish("order.placed", 42, "express", priority=True)
    assert recorder.calls == [((42, "express"), {"priority": True})]


def test_publish_notifies_all_handlers_for_event(bus):
    first, second = [], []
    bus.subscribe("tick", lambda: first.append("a"))
    bus.subscribe("tick", lambda: second.append("b"))
    bus.publish("tick")
    assert first == ["a"]
    assert second == ["b"]


def test_handlers_are_called_in_subscription_order(bus):
    order = []
    bus.subscribe("step", lambda: order.append(1))
    bus.subscribe("step", lambda: order.append(2))
    bus.subscribe("step", lambda: order.append(3))
    bus.publish("step")
    assert order == [1, 2, 3]


def test_publish_only_reaches_handlers_of_matching_event(bus, recorder):
    other = []
    bus.subscribe("alpha", recorder)
    bus.subscribe("beta", lambda: other.append("beta"))
    bus.publish("alpha")
    assert len(recorder.calls) == 1
    assert other == []


def test_publish_without_subscribers_is_harmless(bus):
    bus.publish("nobody.listening", 1, 2, key="value")


def test_repeated_publish_invokes_handler_each_time(bus, recorder):
    bus.subscribe("ping", recorder)
    for _ in range(3):
        bus.publish("ping")
    assert len(recorder.calls) == 3


def test_handler_receives_no_args_when_none_published(bus, recorder):
    bus.subscribe("empty", recorder)
    bus.publish("empty")
    assert recorder.calls == [((), {})]


# --- unsubscribe ----------------------------------------------------------

def test_unsubscribe_stops_delivery(bus, recorder):
    bus.subscribe("news", recorder)
    bus.unsubscribe("news", recorder)
    bus.publish("news")
    assert recorder.calls == []


def test_unsubscribe_removes_only_given_handler(bus, recorder):
    kept = []
    keep_handler = lambda: kept.append("kept")
    bus.subscribe("event", recorder)
    bus.subscribe("event", keep_handler)
    bus.unsubscribe("event", recorder)
    bus.publish("event")
    assert recorder.calls == []
    assert kept == ["kept"]


def test_unsubscribe_does_not_affect_other_events(bus, recorder):
    bus.subscribe("left", recorder)
    bus.subscribe("right", recorder)
    bus.unsubscribe("left", recorder)
    bus.publish("right")
    assert len(recorder.calls) == 1


# --- isolation ------------------------------------------------------------

def test_separate_bus_instances_do_not_share_subscribers(recorder):
    bus_a = EventBus()
    bus_b = EventBus()
    bus_a.subscribe("shared.name", recorder)
    bus_b.publish("shared.name")
    assert recorder.calls == []


def test_handler_can_be_added_after_publish_without_retroactive_call(bus, recorder):
    bus.publish("late")
    bus.subscribe("late", recorder)
    assert recorder.calls == []
    bus.publish("late")
    assert len(recorder.calls) == 1
