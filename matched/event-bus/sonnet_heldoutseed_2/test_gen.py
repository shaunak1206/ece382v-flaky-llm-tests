"""Tests for event_bus.EventBus.

Only ``event_bus/__init__.py`` was provided; the implementation in
``event_bus/bus.py`` was not visible. These tests therefore assume the usual
pub/sub shape: ``subscribe(event, handler)``, ``publish(event, payload)`` and
``unsubscribe(event, handler)``. The method names are looked up from a list of
common aliases (on/emit/off, etc.). Behavioural tests are skipped if no
matching method is found.
"""
import inspect

import pytest

import event_bus
from event_bus import EventBus

SUBSCRIBE_NAMES = ("subscribe", "on", "register", "add_listener", "add_handler", "listen")
PUBLISH_NAMES = ("publish", "emit", "fire", "dispatch", "trigger", "post", "send")
UNSUBSCRIBE_NAMES = ("unsubscribe", "off", "unregister", "remove_listener", "remove_handler")


def _find(bus, names, label):
    for name in names:
        method = getattr(bus, name, None)
        if callable(method):
            return method
    pytest.skip("EventBus has no %s method (tried %s)" % (label, ", ".join(names)))


class Recorder:
    """Callable that records every call made to it."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def received(self, payload):
        for args, kwargs in self.calls:
            candidates = list(args) + list(kwargs.values())
            for c in candidates:
                if (
                    c == payload
                    or getattr(c, "data", None) == payload
                    or getattr(c, "payload", None) == payload
                ):
                    return True
        return False


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def api(bus):
    return (
        _find(bus, SUBSCRIBE_NAMES, "subscribe"),
        _find(bus, PUBLISH_NAMES, "publish"),
    )


# ---------------------------------------------------------------- package level

def test_eventbus_exported_from_package():
    from event_bus.bus import EventBus as BusFromModule

    assert event_bus.EventBus is BusFromModule


def test_all_lists_eventbus():
    assert event_bus.__all__ == ["EventBus"]


def test_version_string():
    assert event_bus.__version__ == "1.0.0"


def test_eventbus_is_a_class():
    assert inspect.isclass(EventBus)


def test_eventbus_instantiates_without_arguments():
    assert isinstance(EventBus(), EventBus)


def test_eventbus_has_subscribe_and_publish_methods(bus):
    assert any(callable(getattr(bus, n, None)) for n in SUBSCRIBE_NAMES)
    assert any(callable(getattr(bus, n, None)) for n in PUBLISH_NAMES)


# ---------------------------------------------------------------- behaviour

def test_subscribed_handler_is_called_on_publish(api):
    subscribe, publish = api
    handler = Recorder()
    subscribe("greeting", handler)
    publish("greeting", "hello")
    assert len(handler.calls) == 1


def test_handler_receives_published_payload(api):
    subscribe, publish = api
    handler = Recorder()
    subscribe("greeting", handler)
    publish("greeting", {"name": "world"})
    assert handler.received({"name": "world"})


def test_publish_without_subscribers_does_not_raise(api):
    _, publish = api
    publish("nobody-listens", "payload")


def test_multiple_handlers_all_called(api):
    subscribe, publish = api
    first, second = Recorder(), Recorder()
    subscribe("evt", first)
    subscribe("evt", second)
    publish("evt", 1)
    assert len(first.calls) == 1
    assert len(second.calls) == 1


def test_handlers_called_in_registration_order(api):
    subscribe, publish = api
    order = []
    subscribe("evt", lambda *a, **k: order.append("a"))
    subscribe("evt", lambda *a, **k: order.append("b"))
    subscribe("evt", lambda *a, **k: order.append("c"))
    publish("evt", None)
    assert order == ["a", "b", "c"]


def test_handler_only_called_for_its_event(api):
    subscribe, publish = api
    a_handler, b_handler = Recorder(), Recorder()
    subscribe("a", a_handler)
    subscribe("b", b_handler)
    publish("a", "x")
    assert len(a_handler.calls) == 1
    assert b_handler.calls == []


def test_repeated_publish_calls_handler_each_time(api):
    subscribe, publish = api
    handler = Recorder()
    subscribe("tick", handler)
    for i in range(3):
        publish("tick", i)
    assert len(handler.calls) == 3


def test_same_handler_on_different_events(api):
    subscribe, publish = api
    handler = Recorder()
    subscribe("one", handler)
    subscribe("two", handler)
    publish("one", 1)
    publish("two", 2)
    assert len(handler.calls) == 2


def test_instances_do_not_share_subscriptions():
    bus1, bus2 = EventBus(), EventBus()
    subscribe1 = _find(bus1, SUBSCRIBE_NAMES, "subscribe")
    publish2 = _find(bus2, PUBLISH_NAMES, "publish")
    handler = Recorder()
    subscribe1("evt", handler)
    publish2("evt", "data")
    assert handler.calls == []


def test_unsubscribe_stops_delivery(bus, api):
    subscribe, publish = api
    unsubscribe = _find(bus, UNSUBSCRIBE_NAMES, "unsubscribe")
    handler = Recorder()
    subscribe("evt", handler)
    publish("evt", 1)
    unsubscribe("evt", handler)
    publish("evt", 2)
    assert len(handler.calls) == 1


def test_unsubscribe_leaves_other_handlers_active(bus, api):
    subscribe, publish = api
    unsubscribe = _find(bus, UNSUBSCRIBE_NAMES, "unsubscribe")
    removed, kept = Recorder(), Recorder()
    subscribe("evt", removed)
    subscribe("evt", kept)
    unsubscribe("evt", removed)
    publish("evt", "x")
    assert removed.calls == []
    assert len(kept.calls) == 1
