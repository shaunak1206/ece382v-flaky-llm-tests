"""Tests for event_bus.EventBus.

The source of event_bus/bus.py was not provided, so these tests resolve the
subscribe / publish / unsubscribe methods by their most common names and
assume the conventional call shapes:

    bus.subscribe(event_name, handler)
    bus.publish(event_name, *args, **kwargs)
    bus.unsubscribe(event_name, handler)

If a method cannot be found, the dependent tests are skipped.
"""
import pytest

import event_bus
from event_bus import EventBus

SUBSCRIBE_NAMES = ("subscribe", "on", "register", "add_listener",
                   "add_handler", "listen", "connect")
PUBLISH_NAMES = ("publish", "emit", "dispatch", "trigger", "fire",
                 "send", "post")
UNSUBSCRIBE_NAMES = ("unsubscribe", "off", "unregister", "remove_listener",
                     "remove_handler", "disconnect")


def _resolve(bus, names, label):
    for name in names:
        member = getattr(bus, name, None)
        if callable(member):
            return member
    pytest.skip("EventBus has no recognizable %s method" % label)


def subscribe(bus, event, handler):
    return _resolve(bus, SUBSCRIBE_NAMES, "subscribe")(event, handler)


def publish(bus, event, *args, **kwargs):
    return _resolve(bus, PUBLISH_NAMES, "publish")(event, *args, **kwargs)


def unsubscribe(bus, event, handler):
    func = _resolve(bus, UNSUBSCRIBE_NAMES, "unsubscribe")
    try:
        return func(event, handler)
    except TypeError:
        return func(handler)


def _contains_payload(call, payload):
    args, kwargs = call
    candidates = list(args) + list(kwargs.values())
    for item in list(candidates):
        attrs = getattr(item, "__dict__", None)
        if attrs:
            candidates.extend(attrs.values())
    return any(item is payload or item == payload for item in candidates)


class Recorder:
    """Callable handler that records every call."""

    def __init__(self, name=None, log=None):
        self.calls = []
        self.name = name
        self.log = log

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.log is not None:
            self.log.append(self.name)


@pytest.fixture
def bus():
    return EventBus()


def test_package_exports_event_bus():
    assert "EventBus" in event_bus.__all__
    assert event_bus.EventBus is EventBus
    assert event_bus.__version__ == "1.0.0"


def test_can_instantiate_event_bus():
    instance = EventBus()
    assert isinstance(instance, EventBus)


def test_instances_are_independent_objects():
    assert EventBus() is not EventBus()


def test_subscribed_handler_is_called_on_publish(bus):
    handler = Recorder()
    subscribe(bus, "user.created", handler)
    publish(bus, "user.created")
    assert len(handler.calls) == 1


def test_handler_receives_published_payload(bus):
    handler = Recorder()
    payload = {"id": 42}
    subscribe(bus, "user.created", handler)
    publish(bus, "user.created", payload)
    assert len(handler.calls) == 1
    assert _contains_payload(handler.calls[0], payload)


def test_handler_receives_keyword_payload(bus):
    handler = Recorder()
    subscribe(bus, "user.created", handler)
    publish(bus, "user.created", name="alice")
    assert len(handler.calls) == 1
    assert _contains_payload(handler.calls[0], "alice")


def test_publish_without_subscribers_does_not_raise(bus):
    publish(bus, "nobody.listens")
    publish(bus, "nobody.listens", {"x": 1})


def test_handler_not_called_for_other_events(bus):
    handler = Recorder()
    subscribe(bus, "event.a", handler)
    publish(bus, "event.b")
    assert handler.calls == []


def test_multiple_handlers_all_called(bus):
    first, second, third = Recorder(), Recorder(), Recorder()
    for h in (first, second, third):
        subscribe(bus, "evt", h)
    publish(bus, "evt")
    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert len(third.calls) == 1


def test_handlers_called_in_registration_order(bus):
    log = []
    subscribe(bus, "evt", Recorder("first", log))
    subscribe(bus, "evt", Recorder("second", log))
    subscribe(bus, "evt", Recorder("third", log))
    publish(bus, "evt")
    assert log == ["first", "second", "third"]


def test_each_publish_triggers_handler_again(bus):
    handler = Recorder()
    subscribe(bus, "evt", handler)
    publish(bus, "evt")
    publish(bus, "evt")
    publish(bus, "evt")
    assert len(handler.calls) == 3


def test_different_events_dispatch_to_own_handlers(bus):
    handler_a, handler_b = Recorder(), Recorder()
    subscribe(bus, "a", handler_a)
    subscribe(bus, "b", handler_b)
    publish(bus, "a")
    assert len(handler_a.calls) == 1
    assert handler_b.calls == []
    publish(bus, "b")
    assert len(handler_a.calls) == 1
    assert len(handler_b.calls) == 1


def test_unsubscribe_stops_handler_from_being_called(bus):
    handler = Recorder()
    subscribe(bus, "evt", handler)
    publish(bus, "evt")
    unsubscribe(bus, "evt", handler)
    publish(bus, "evt")
    assert len(handler.calls) == 1


def test_unsubscribe_only_removes_specified_handler(bus):
    keep, drop = Recorder(), Recorder()
    subscribe(bus, "evt", keep)
    subscribe(bus, "evt", drop)
    unsubscribe(bus, "evt", drop)
    publish(bus, "evt")
    assert len(keep.calls) == 1
    assert drop.calls == []


def test_separate_buses_do_not_share_subscribers():
    bus_one, bus_two = EventBus(), EventBus()
    handler = Recorder()
    subscribe(bus_one, "evt", handler)
    publish(bus_two, "evt")
    assert handler.calls == []
    publish(bus_one, "evt")
    assert len(handler.calls) == 1


def test_lambda_and_function_handlers_are_supported(bus):
    seen = []

    def func_handler(*args, **kwargs):
        seen.append("func")

    subscribe(bus, "evt", func_handler)
    subscribe(bus, "evt", lambda *a, **k: seen.append("lambda"))
    publish(bus, "evt")
    assert sorted(seen) == ["func", "lambda"]
