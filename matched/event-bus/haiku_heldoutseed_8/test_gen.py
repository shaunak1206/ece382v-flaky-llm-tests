import pytest

from event_bus import EventBus, __all__ as package_all, __version__


@pytest.fixture
def bus():
    return EventBus()


class Recorder:
    """Callable handler that records every call it receives."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_package_exports_event_bus():
    assert "EventBus" in package_all
    assert EventBus.__name__ == "EventBus"


def test_version_is_non_empty_string():
    assert isinstance(__version__, str)
    assert __version__


def test_new_bus_is_event_bus_instance(bus):
    assert isinstance(bus, EventBus)


def test_publish_invokes_subscribed_handler(bus):
    handler = Recorder()
    bus.subscribe("user.created", handler)

    bus.publish("user.created", {"id": 1})

    assert len(handler.calls) == 1


def test_handler_receives_published_payload(bus):
    handler = Recorder()
    payload = {"id": 42, "name": "alice"}
    bus.subscribe("user.created", handler)

    bus.publish("user.created", payload)

    args, _ = handler.calls[0]
    assert args[0] == payload


def test_publish_without_subscribers_does_not_raise(bus):
    bus.publish("nobody.listening", "data")


def test_publish_only_reaches_handlers_of_matching_event(bus):
    matching = Recorder()
    other = Recorder()
    bus.subscribe("order.placed", matching)
    bus.subscribe("order.cancelled", other)

    bus.publish("order.placed", "x")

    assert len(matching.calls) == 1
    assert other.calls == []


def test_all_handlers_for_event_are_called(bus):
    first = Recorder()
    second = Recorder()
    third = Recorder()
    for handler in (first, second, third):
        bus.subscribe("tick", handler)

    bus.publish("tick", 1)

    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert len(third.calls) == 1


def test_handlers_are_called_in_subscription_order(bus):
    order = []

    def make_handler(name):
        def handler(*_args, **_kwargs):
            order.append(name)
        return handler

    bus.subscribe("step", make_handler("a"))
    bus.subscribe("step", make_handler("b"))
    bus.subscribe("step", make_handler("c"))

    bus.publish("step", None)

    assert order == ["a", "b", "c"]


def test_each_publish_invokes_handler_again(bus):
    handler = Recorder()
    bus.subscribe("ping", handler)

    for i in range(3):
        bus.publish("ping", i)

    assert len(handler.calls) == 3
    assert [args[0] for args, _ in handler.calls] == [0, 1, 2]


def test_unsubscribe_stops_delivery(bus):
    handler = Recorder()
    bus.subscribe("news", handler)
    bus.publish("news", "first")

    bus.unsubscribe("news", handler)
    bus.publish("news", "second")

    assert len(handler.calls) == 1
    assert handler.calls[0][0][0] == "first"


def test_unsubscribe_removes_only_given_handler(bus):
    removed = Recorder()
    kept = Recorder()
    bus.subscribe("news", removed)
    bus.subscribe("news", kept)

    bus.unsubscribe("news", removed)
    bus.publish("news", "hello")

    assert removed.calls == []
    assert len(kept.calls) == 1


def test_same_handler_can_subscribe_to_multiple_events(bus):
    handler = Recorder()
    bus.subscribe("alpha", handler)
    bus.subscribe("beta", handler)

    bus.publish("alpha", 1)
    bus.publish("beta", 2)

    assert len(handler.calls) == 2


def test_unsubscribe_from_one_event_keeps_other_event_subscription(bus):
    handler = Recorder()
    bus.subscribe("alpha", handler)
    bus.subscribe("beta", handler)

    bus.unsubscribe("alpha", handler)
    bus.publish("alpha", 1)
    bus.publish("beta", 2)

    assert len(handler.calls) == 1
    assert handler.calls[0][0][0] == 2


def test_separate_bus_instances_are_isolated():
    bus_one = EventBus()
    bus_two = EventBus()
    handler = Recorder()
    bus_one.subscribe("shared.name", handler)

    bus_two.publish("shared.name", "payload")

    assert handler.calls == []


def test_publish_with_none_payload_still_delivers(bus):
    handler = Recorder()
    bus.subscribe("empty", handler)

    bus.publish("empty", None)

    assert len(handler.calls) == 1
    assert handler.calls[0][0][0] is None
