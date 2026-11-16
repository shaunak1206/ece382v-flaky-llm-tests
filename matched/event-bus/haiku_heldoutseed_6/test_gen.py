"""Unit tests for the EventBus class exposed by the event_bus package.

These tests assume the public API uses the common names
``subscribe(event_name, handler)``, ``unsubscribe(event_name, handler)``
and ``publish(event_name, *args, **kwargs)``. The helpers below also accept
``on`` and ``emit`` as aliases, so either naming style works.
"""
import event_bus
from event_bus import EventBus
from event_bus.bus import EventBus as BusEventBus


def _subscribe(bus, name, handler):
    method = getattr(bus, "subscribe", None) or getattr(bus, "on")
    return method(name, handler)


def _unsubscribe(bus, name, handler):
    method = getattr(bus, "unsubscribe", None) or getattr(bus, "off")
    return method(name, handler)


def _publish(bus, name, *args, **kwargs):
    method = getattr(bus, "publish", None) or getattr(bus, "emit")
    return method(name, *args, **kwargs)


class Recorder:
    """Callable handler that records every invocation."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_package_exports_eventbus_class():
    assert EventBus is BusEventBus
    assert isinstance(EventBus, type)


def test_package_all_contains_eventbus():
    assert "EventBus" in event_bus.__all__


def test_package_version_is_string():
    assert isinstance(event_bus.__version__, str)
    assert event_bus.__version__ == "1.0.0"


def test_bus_can_be_instantiated():
    bus = EventBus()
    assert isinstance(bus, EventBus)


def test_two_instances_are_distinct_objects():
    assert EventBus() is not EventBus()


def test_publish_invokes_subscribed_handler():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "user.created", handler)

    _publish(bus, "user.created")

    assert len(handler.calls) == 1


def test_publish_passes_positional_and_keyword_arguments():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "order.placed", handler)

    _publish(bus, "order.placed", 42, "widget", quantity=3, express=True)

    assert handler.calls == [((42, "widget"), {"quantity": 3, "express": True})]


def test_publish_without_subscribers_does_not_raise():
    bus = EventBus()
    _publish(bus, "nobody.listening", 1, 2, key="value")


def test_all_handlers_for_event_are_called():
    bus = EventBus()
    first, second, third = Recorder(), Recorder(), Recorder()
    for handler in (first, second, third):
        _subscribe(bus, "tick", handler)

    _publish(bus, "tick")

    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert len(third.calls) == 1


def test_handlers_are_invoked_in_subscription_order():
    bus = EventBus()
    order = []

    def make(label):
        def handler(*_args, **_kwargs):
            order.append(label)
        return handler

    _subscribe(bus, "ordered", make("a"))
    _subscribe(bus, "ordered", make("b"))
    _subscribe(bus, "ordered", make("c"))

    _publish(bus, "ordered")

    assert order == ["a", "b", "c"]


def test_events_are_isolated_by_name():
    bus = EventBus()
    alpha, beta = Recorder(), Recorder()
    _subscribe(bus, "alpha", alpha)
    _subscribe(bus, "beta", beta)

    _publish(bus, "alpha", "x")

    assert len(alpha.calls) == 1
    assert beta.calls == []


def test_publishing_same_event_repeatedly_calls_handler_each_time():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "repeat", handler)

    for i in range(5):
        _publish(bus, "repeat", i)

    assert [args for args, _ in handler.calls] == [(0,), (1,), (2,), (3,), (4,)]


def test_subscribing_does_not_invoke_handler():
    bus = EventBus()
    handler = Recorder()

    _subscribe(bus, "quiet", handler)

    assert handler.calls == []


def test_handler_runs_synchronously_before_publish_returns():
    bus = EventBus()
    state = {"done": False}

    def handler(*_args, **_kwargs):
        state["done"] = True

    _subscribe(bus, "sync", handler)
    _publish(bus, "sync")

    assert state["done"] is True


def test_unsubscribe_stops_delivery():
    bus = EventBus()
    handler = Recorder()
    _subscribe(bus, "news", handler)
    _publish(bus, "news", "first")

    _unsubscribe(bus, "news", handler)
    _publish(bus, "news", "second")

    assert handler.calls == [(("first",), {})]


def test_unsubscribe_one_handler_keeps_others_active():
    bus = EventBus()
    kept, removed = Recorder(), Recorder()
    _subscribe(bus, "shared", kept)
    _subscribe(bus, "shared", removed)

    _unsubscribe(bus, "shared", removed)
    _publish(bus, "shared")

    assert len(kept.calls) == 1
    assert removed.calls == []


def test_separate_bus_instances_do_not_share_subscribers():
    bus_a = EventBus()
    bus_b = EventBus()
    handler_a, handler_b = Recorder(), Recorder()
    _subscribe(bus_a, "topic", handler_a)
    _subscribe(bus_b, "topic", handler_b)

    _publish(bus_a, "topic")

    assert len(handler_a.calls) == 1
    assert handler_b.calls == []


def test_handler_can_publish_another_event_reentrantly():
    bus = EventBus()
    downstream = Recorder()

    def relay(value, *_args, **_kwargs):
        _publish(bus, "downstream", value * 2)

    _subscribe(bus, "upstream", relay)
    _subscribe(bus, "downstream", downstream)

    _publish(bus, "upstream", 21)

    assert downstream.calls == [((42,), {})]
