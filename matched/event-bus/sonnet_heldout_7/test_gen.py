"""Tests for event_bus.EventBus.

The source provided only showed event_bus/__init__.py, so the concrete method
names of EventBus are not known. The tests locate the subscribe / publish /
unsubscribe methods by trying common names, and skip the behavioural tests
if no matching method exists.
"""
import pytest

import event_bus
from event_bus import EventBus

SUBSCRIBE_NAMES = ("subscribe", "on", "register", "add_listener",
                   "add_handler", "listen", "connect")
PUBLISH_NAMES = ("publish", "emit", "fire", "dispatch", "trigger",
                 "post", "send")
UNSUBSCRIBE_NAMES = ("unsubscribe", "off", "unregister", "remove_listener",
                     "remove_handler", "disconnect")


def _find(bus, names, label):
    for name in names:
        method = getattr(bus, name, None)
        if callable(method):
            return method
    pytest.skip("EventBus has no recognised %s method" % label)


def _subscribe(bus):
    return _find(bus, SUBSCRIBE_NAMES, "subscribe")


def _publish(bus):
    return _find(bus, PUBLISH_NAMES, "publish")


def _unsubscribe(bus):
    return _find(bus, UNSUBSCRIBE_NAMES, "unsubscribe")


class Recorder:
    """Callable that records every call made to it."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    @property
    def count(self):
        return len(self.calls)

    def received(self, value):
        for args, kwargs in self.calls:
            if value in args or value in kwargs.values():
                return True
        return False


@pytest.fixture
def bus():
    return EventBus()


# ---------------------------------------------------------------- package

def test_eventbus_exported_from_package():
    assert hasattr(event_bus, "EventBus")
    assert event_bus.EventBus is EventBus


def test_eventbus_is_defined_in_bus_module():
    from event_bus.bus import EventBus as BusFromModule
    assert BusFromModule is EventBus


def test_version_string():
    assert event_bus.__version__ == "1.0.0"


def test_all_lists_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_eventbus_is_a_class():
    assert isinstance(EventBus, type)


# ------------------------------------------------------------- instances

def test_can_instantiate_without_arguments():
    assert isinstance(EventBus(), EventBus)


def test_instances_are_distinct():
    assert EventBus() is not EventBus()


# ------------------------------------------------------------- behaviour

def test_subscriber_called_on_publish(bus):
    handler = Recorder()
    _subscribe(bus)("greet", handler)
    _publish(bus)("greet")
    assert handler.count == 1


def test_payload_is_delivered_to_subscriber(bus):
    handler = Recorder()
    _subscribe(bus)("greet", handler)
    _publish(bus)("greet", "hello-payload")
    assert handler.count == 1
    assert handler.received("hello-payload")


def test_publish_without_subscribers_does_not_raise(bus):
    _publish(bus)("nobody-listens")


def test_subscriber_not_called_for_other_events(bus):
    handler = Recorder()
    _subscribe(bus)("event-a", handler)
    _publish(bus)("event-b")
    assert handler.count == 0


def test_multiple_subscribers_all_called(bus):
    first, second = Recorder(), Recorder()
    subscribe = _subscribe(bus)
    subscribe("evt", first)
    subscribe("evt", second)
    _publish(bus)("evt")
    assert first.count == 1
    assert second.count == 1


def test_repeated_publish_calls_handler_each_time(bus):
    handler = Recorder()
    _subscribe(bus)("tick", handler)
    publish = _publish(bus)
    publish("tick")
    publish("tick")
    publish("tick")
    assert handler.count == 3


def test_subscribers_invoked_in_registration_order(bus):
    order = []
    subscribe = _subscribe(bus)
    subscribe("evt", lambda *a, **k: order.append("first"))
    subscribe("evt", lambda *a, **k: order.append("second"))
    _publish(bus)("evt")
    assert order == ["first", "second"]


def test_unsubscribe_stops_delivery(bus):
    handler = Recorder()
    _subscribe(bus)("evt", handler)
    unsubscribe = _unsubscribe(bus)
    publish = _publish(bus)
    publish("evt")
    assert handler.count == 1
    unsubscribe("evt", handler)
    publish("evt")
    assert handler.count == 1


def test_unsubscribe_only_removes_given_handler(bus):
    keep, drop = Recorder(), Recorder()
    subscribe = _subscribe(bus)
    subscribe("evt", keep)
    subscribe("evt", drop)
    _unsubscribe(bus)("evt", drop)
    _publish(bus)("evt")
    assert keep.count == 1
    assert drop.count == 0


def test_buses_do_not_share_subscribers():
    bus_a, bus_b = EventBus(), EventBus()
    handler = Recorder()
    _subscribe(bus_a)("evt", handler)
    _publish(bus_b)("evt")
    assert handler.count == 0
    _publish(bus_a)("evt")
    assert handler.count == 1
