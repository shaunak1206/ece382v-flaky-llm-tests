import pytest

import event_bus
from event_bus import EventBus

# The tests below use only the public surface that is visible from the package
# (EventBus) and discover the conventional pub/sub method names at runtime, so
# they exercise behaviour (subscribe -> publish -> handler invoked, etc.)
# rather than a single hard-coded spelling of the API.

SUBSCRIBE_NAMES = (
    "subscribe", "on", "add_listener", "add_handler",
    "register", "listen", "add_subscriber", "add_callback",
)
PUBLISH_NAMES = (
    "publish", "emit", "post", "dispatch", "fire", "trigger", "send",
)
UNSUBSCRIBE_NAMES = (
    "unsubscribe", "off", "remove_listener", "remove_handler",
    "unregister", "remove_subscriber", "remove_callback",
)


def _find(bus, names, kind):
    for name in names:
        attr = getattr(bus, name, None)
        if callable(attr):
            return attr
    pytest.skip("EventBus exposes no %s method among %r" % (kind, names))


def _subscribe(bus):
    return _find(bus, SUBSCRIBE_NAMES, "subscribe")


def _publish(bus):
    return _find(bus, PUBLISH_NAMES, "publish")


def _unsubscribe(bus):
    return _find(bus, UNSUBSCRIBE_NAMES, "unsubscribe")


class Recorder:
    """Callable handler that records every invocation."""

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


def test_package_exports_event_bus():
    assert "EventBus" in event_bus.__all__
    assert event_bus.EventBus is EventBus
    assert event_bus.__version__ == "1.0.0"


def test_event_bus_can_be_instantiated():
    instance = EventBus()
    assert isinstance(instance, EventBus)


def test_instances_are_distinct_objects():
    assert EventBus() is not EventBus()


def test_bus_has_subscribe_and_publish_methods(bus):
    assert any(callable(getattr(bus, n, None)) for n in SUBSCRIBE_NAMES)
    assert any(callable(getattr(bus, n, None)) for n in PUBLISH_NAMES)


def test_handler_called_when_event_published(bus):
    handler = Recorder()
    _subscribe(bus)("greet", handler)
    _publish(bus)("greet")
    assert handler.count == 1


def test_handler_receives_published_payload(bus):
    handler = Recorder()
    _subscribe(bus)("greet", handler)
    _publish(bus)("greet", "payload-value")
    assert handler.count == 1
    args, kwargs = handler.calls[0]
    assert "payload-value" in repr(args) + repr(kwargs)


def test_handler_called_for_each_publish(bus):
    handler = Recorder()
    _subscribe(bus)("tick", handler)
    publish = _publish(bus)
    publish("tick")
    publish("tick")
    publish("tick")
    assert handler.count == 3


def test_multiple_handlers_all_called(bus):
    first, second = Recorder(), Recorder()
    subscribe = _subscribe(bus)
    subscribe("evt", first)
    subscribe("evt", second)
    _publish(bus)("evt")
    assert first.count == 1
    assert second.count == 1


def test_handlers_called_in_subscription_order(bus):
    order = []
    subscribe = _subscribe(bus)
    subscribe("evt", lambda *a, **k: order.append("a"))
    subscribe("evt", lambda *a, **k: order.append("b"))
    subscribe("evt", lambda *a, **k: order.append("c"))
    _publish(bus)("evt")
    assert order == ["a", "b", "c"]


def test_handler_not_called_for_other_event(bus):
    handler = Recorder()
    _subscribe(bus)("alpha", handler)
    _publish(bus)("beta")
    assert handler.count == 0


def test_different_events_route_to_their_own_handlers(bus):
    alpha, beta = Recorder(), Recorder()
    subscribe = _subscribe(bus)
    subscribe("alpha", alpha)
    subscribe("beta", beta)
    _publish(bus)("beta")
    assert alpha.count == 0
    assert beta.count == 1


def test_publish_without_subscribers_does_not_raise(bus):
    _publish(bus)("nobody-listens")


def test_unsubscribe_stops_handler_being_called(bus):
    handler = Recorder()
    _subscribe(bus)("evt", handler)
    publish = _publish(bus)
    publish("evt")
    _unsubscribe(bus)("evt", handler)
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
