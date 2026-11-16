"""
Unit tests for the EventBus class exported by the event_bus package.

These tests use only the standard library and pytest. The EventBus API is
not shown in the provided source, so the helpers below accept the common
method names (subscribe/on, publish/emit, unsubscribe/off) and use whichever
one the implementation provides.
"""
import inspect

import pytest

import event_bus
from event_bus import EventBus


# ---------------------------------------------------------------------------
# Helpers that adapt to the EventBus method naming
# ---------------------------------------------------------------------------

def _first_attr(obj, names):
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return method
    raise AttributeError(
        "EventBus has none of the expected methods: {}".format(", ".join(names))
    )


def _subscribe(bus, event, handler):
    return _first_attr(bus, ("subscribe", "on", "add_listener"))(event, handler)


def _publish(bus, event, *args, **kwargs):
    return _first_attr(bus, ("publish", "emit", "trigger", "dispatch"))(
        event, *args, **kwargs
    )


def _unsubscribe(bus, event, handler):
    return _first_attr(bus, ("unsubscribe", "off", "remove_listener"))(event, handler)


class Recorder:
    """Callable handler that records every invocation."""

    def __init__(self, name=None, log=None):
        self.name = name
        self.calls = []
        self.log = log

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.log is not None:
            self.log.append(self.name)


# ---------------------------------------------------------------------------
# Package-level tests
# ---------------------------------------------------------------------------

def test_package_exports_eventbus_in_all():
    assert "EventBus" in event_bus.__all__


def test_package_version_is_defined():
    assert event_bus.__version__ == "1.0.0"


def test_eventbus_is_a_class():
    assert inspect.isclass(EventBus)


def test_eventbus_can_be_instantiated_without_arguments():
    bus = EventBus()
    assert isinstance(bus, EventBus)


# ---------------------------------------------------------------------------
# Subscribe / publish behaviour
# ---------------------------------------------------------------------------

def test_publish_invokes_subscribed_handler():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "user.created", handler)

    _publish(bus, "user.created")

    assert len(handler.calls) == 1


def test_handler_receives_positional_payload():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "order.placed", handler)

    _publish(bus, "order.placed", 42, "widget")

    assert handler.calls == [((42, "widget"), {})]


def test_handler_receives_keyword_payload():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "order.placed", handler)

    _publish(bus, "order.placed", quantity=3, sku="ABC")

    assert handler.calls == [((), {"quantity": 3, "sku": "ABC"})]


def test_publish_without_subscribers_does_not_raise():
    bus = EventBus()
    _publish(bus, "nobody.listening", 1, 2, key="value")


def test_all_subscribers_receive_the_event():
    bus = EventBus()
    first = Recorder()
    second = Recorder()
    third = Recorder()
    for handler in (first, second, third):
        _subscribe(bus, "tick", handler)

    _publish(bus, "tick")

    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert len(third.calls) == 1


def test_handlers_are_called_in_subscription_order():
    bus = EventBus()
    log = []
    _subscribe(bus, "start", Recorder("a", log))
    _subscribe(bus, "start", Recorder("b", log))
    _subscribe(bus, "start", Recorder("c", log))

    _publish(bus, "start")

    assert log == ["a", "b", "c"]


def test_each_publish_invokes_handler_once_per_publish():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "heartbeat", handler)

    for _ in range(5):
        _publish(bus, "heartbeat")

    assert len(handler.calls) == 5


def test_events_are_isolated_by_name():
    bus = EventBus()
    alpha = Recorder()
    beta = Recorder()
    _subscribe(bus, "alpha", alpha)
    _subscribe(bus, "beta", beta)

    _publish(bus, "alpha")

    assert len(alpha.calls) == 1
    assert beta.calls == []


def test_publishing_unrelated_event_does_not_call_handler():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "file.saved", handler)

    _publish(bus, "file.deleted")

    assert handler.calls == []


# ---------------------------------------------------------------------------
# Unsubscribe behaviour
# ---------------------------------------------------------------------------

def test_unsubscribed_handler_is_not_called():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "news", handler)
    _unsubscribe(bus, "news", handler)

    _publish(bus, "news")

    assert handler.calls == []


def test_unsubscribe_only_removes_the_given_handler():
    bus = EventBus()
    kept = Recorder()
    removed = Recorder()
    _subscribe(bus, "news", kept)
    _subscribe(bus, "news", removed)
    _unsubscribe(bus, "news", removed)

    _publish(bus, "news")

    assert len(kept.calls) == 1
    assert removed.calls == []


def test_unsubscribe_does_not_affect_other_events():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "weather", handler)
    _subscribe(bus, "sports", handler)
    _unsubscribe(bus, "weather", handler)

    _publish(bus, "weather")
    _publish(bus, "sports")

    assert len(handler.calls) == 1


# ---------------------------------------------------------------------------
# Instance isolation
# ---------------------------------------------------------------------------

def test_separate_bus_instances_do_not_share_subscribers():
    bus_one = EventBus()
    bus_two = EventBus()
    handler = Recorder()
    _subscribe(bus_one, "ping", handler)

    _publish(bus_two, "ping")

    assert handler.calls == []


def test_handler_exceptions_are_not_swallowed_silently_by_other_buses():
    # Publishing on a fresh bus with no subscribers must never invoke
    # handlers registered on a different bus instance.
    bus_with_handler = EventBus()
    bus_empty = EventBus()

    def boom(*_args, **_kwargs):
        raise AssertionError("should not be called")

    _subscribe(bus_with_handler, "danger", boom)

    _publish(bus_empty, "danger")  # must not raise
