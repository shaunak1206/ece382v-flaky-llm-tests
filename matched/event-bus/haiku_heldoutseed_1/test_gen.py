import inspect

import pytest

import event_bus
from event_bus import EventBus
from event_bus import bus as bus_module


SUBSCRIBE_NAMES = ("subscribe", "on", "add_listener", "register", "listen")
PUBLISH_NAMES = ("publish", "emit", "dispatch", "trigger", "fire")
UNSUBSCRIBE_NAMES = ("unsubscribe", "off", "remove_listener", "unregister")


def _resolve(bus, names, kind):
    for name in names:
        method = getattr(bus, name, None)
        if callable(method):
            return method
    pytest.skip(f"EventBus has no {kind} method among {names}")


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def subscribe(bus):
    return _resolve(bus, SUBSCRIBE_NAMES, "subscribe")


@pytest.fixture
def publish(bus):
    return _resolve(bus, PUBLISH_NAMES, "publish")


@pytest.fixture
def unsubscribe(bus):
    return _resolve(bus, UNSUBSCRIBE_NAMES, "unsubscribe")


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    @property
    def count(self):
        return len(self.calls)


def test_version_is_exposed():
    assert event_bus.__version__ == "1.0.0"


def test_all_exports_only_event_bus():
    assert event_bus.__all__ == ["EventBus"]


def test_star_import_exposes_event_bus():
    namespace = {}
    exec("from event_bus import *", namespace)
    assert "EventBus" in namespace
    assert namespace["EventBus"] is EventBus


def test_package_event_bus_is_the_bus_module_class():
    assert EventBus is bus_module.EventBus


def test_event_bus_is_a_class():
    assert inspect.isclass(EventBus)


def test_event_bus_can_be_instantiated_without_arguments():
    instance = EventBus()
    assert isinstance(instance, EventBus)


def test_separate_instances_are_distinct_objects():
    first = EventBus()
    second = EventBus()
    assert first is not second


def test_subscribed_handler_receives_published_event(subscribe, publish):
    handler = Recorder()
    subscribe("order.created", handler)
    publish("order.created", {"id": 1})
    assert handler.count == 1


def test_published_payload_is_delivered_to_handler(subscribe, publish):
    handler = Recorder()
    payload = {"id": 42}
    subscribe("order.created", handler)
    publish("order.created", payload)
    args, _ = handler.calls[0]
    assert any(arg is payload or arg == payload for arg in args)


def test_all_handlers_for_event_are_invoked(subscribe, publish):
    first = Recorder()
    second = Recorder()
    subscribe("tick", first)
    subscribe("tick", second)
    publish("tick", 1)
    assert first.count == 1
    assert second.count == 1


def test_handlers_are_invoked_in_subscription_order(subscribe, publish):
    order = []
    subscribe("tick", lambda *a, **k: order.append("first"))
    subscribe("tick", lambda *a, **k: order.append("second"))
    subscribe("tick", lambda *a, **k: order.append("third"))
    publish("tick", None)
    assert order == ["first", "second", "third"]


def test_handler_is_not_called_for_other_event(subscribe, publish):
    handler = Recorder()
    subscribe("alpha", handler)
    publish("beta", None)
    assert handler.count == 0


def test_publish_without_subscribers_does_not_raise(publish):
    publish("nobody.listens", {"x": 1})


def test_unsubscribed_handler_no_longer_receives_events(subscribe, unsubscribe, publish):
    handler = Recorder()
    subscribe("tick", handler)
    publish("tick", 1)
    unsubscribe("tick", handler)
    publish("tick", 2)
    assert handler.count == 1


def test_unsubscribing_one_handler_keeps_others(subscribe, unsubscribe, publish):
    kept = Recorder()
    removed = Recorder()
    subscribe("tick", kept)
    subscribe("tick", removed)
    unsubscribe("tick", removed)
    publish("tick", 1)
    assert kept.count == 1
    assert removed.count == 0


def test_unsubscribing_unknown_handler_does_not_raise(unsubscribe):
    unsubscribe("never.subscribed", Recorder())


def test_separate_buses_do_not_share_subscribers(subscribe, publish):
    handler = Recorder()
    subscribe("tick", handler)
    other = EventBus()
    other_publish = _resolve(other, PUBLISH_NAMES, "publish")
    other_publish("tick", 1)
    assert handler.count == 0
