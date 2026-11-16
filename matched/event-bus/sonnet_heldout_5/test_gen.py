"""Tests for event_bus.EventBus.

NOTE: the source of event_bus/bus.py was not available when these tests were
written, so the tests resolve the bus's subscribe / publish / unsubscribe
methods by common names and skip when none is found.
"""
import pytest

import event_bus
from event_bus import EventBus

SUBSCRIBE_NAMES = ("subscribe", "on", "register", "add_listener", "add_handler", "listen", "add_subscriber")
PUBLISH_NAMES = ("publish", "emit", "fire", "dispatch", "trigger", "post", "send", "notify")
UNSUBSCRIBE_NAMES = ("unsubscribe", "off", "unregister", "remove_listener", "remove_handler", "remove", "remove_subscriber")


def _find(bus, names):
    for name in names:
        fn = getattr(bus, name, None)
        if callable(fn):
            return fn
    return None


def _require(bus, names, label):
    fn = _find(bus, names)
    if fn is None:
        pytest.skip("EventBus has no recognizable %s method" % label)
    return fn


class Recorder(object):
    """Callable that records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    @property
    def count(self):
        return len(self.calls)


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def sub(bus):
    return _require(bus, SUBSCRIBE_NAMES, "subscribe")


@pytest.fixture
def pub(bus):
    return _require(bus, PUBLISH_NAMES, "publish")


def test_package_exports():
    assert event_bus.__version__ == "1.0.0"
    assert event_bus.__all__ == ["EventBus"]
    assert event_bus.EventBus is EventBus


def test_can_instantiate_eventbus():
    assert EventBus() is not None


def test_instances_are_distinct():
    assert EventBus() is not EventBus()


def test_has_subscribe_and_publish_methods(bus):
    assert _find(bus, SUBSCRIBE_NAMES) is not None
    assert _find(bus, PUBLISH_NAMES) is not None


def test_subscribed_handler_called_on_publish(bus, sub, pub):
    handler = Recorder()
    sub("greeting", handler)
    pub("greeting")
    assert handler.count == 1


def test_handler_receives_payload(bus, sub, pub):
    handler = Recorder()
    sub("greeting", handler)
    pub("greeting", "unique-payload-123")
    assert handler.count == 1
    args, kwargs = handler.calls[0]
    assert "unique-payload-123" in repr(args) + repr(kwargs)


def test_publish_with_no_subscribers_does_not_raise(bus, pub):
    pub("nobody-listens")


def test_handler_not_called_for_other_events(bus, sub, pub):
    handler = Recorder()
    sub("event-a", handler)
    pub("event-b")
    assert handler.count == 0


def test_multiple_handlers_all_called(bus, sub, pub):
    h1, h2, h3 = Recorder(), Recorder(), Recorder()
    for h in (h1, h2, h3):
        sub("evt", h)
    pub("evt")
    assert (h1.count, h2.count, h3.count) == (1, 1, 1)


def test_handlers_called_in_registration_order(bus, sub, pub):
    order = []
    sub("evt", lambda *a, **k: order.append("first"))
    sub("evt", lambda *a, **k: order.append("second"))
    sub("evt", lambda *a, **k: order.append("third"))
    pub("evt")
    assert order == ["first", "second", "third"]


def test_handler_called_once_per_publish(bus, sub, pub):
    handler = Recorder()
    sub("evt", handler)
    pub("evt")
    pub("evt")
    pub("evt")
    assert handler.count == 3


def test_same_handler_on_different_events(bus, sub, pub):
    handler = Recorder()
    sub("one", handler)
    sub("two", handler)
    pub("one")
    pub("two")
    assert handler.count == 2


def test_handlers_for_different_events_are_independent(bus, sub, pub):
    ha, hb = Recorder(), Recorder()
    sub("a", ha)
    sub("b", hb)
    pub("a")
    assert (ha.count, hb.count) == (1, 0)
    pub("b")
    assert (ha.count, hb.count) == (1, 1)


def test_buses_are_isolated_from_each_other():
    bus1, bus2 = EventBus(), EventBus()
    sub1 = _require(bus1, SUBSCRIBE_NAMES, "subscribe")
    pub2 = _require(bus2, PUBLISH_NAMES, "publish")
    handler = Recorder()
    sub1("evt", handler)
    pub2("evt")
    assert handler.count == 0


def test_unsubscribe_stops_handler_being_called(bus, sub, pub):
    unsub = _require(bus, UNSUBSCRIBE_NAMES, "unsubscribe")
    handler = Recorder()
    sub("evt", handler)
    pub("evt")
    assert handler.count == 1
    unsub("evt", handler)
    pub("evt")
    assert handler.count == 1


def test_unsubscribe_only_removes_target_handler(bus, sub, pub):
    unsub = _require(bus, UNSUBSCRIBE_NAMES, "unsubscribe")
    keep, drop = Recorder(), Recorder()
    sub("evt", keep)
    sub("evt", drop)
    unsub("evt", drop)
    pub("evt")
    assert keep.count == 1
    assert drop.count == 0
