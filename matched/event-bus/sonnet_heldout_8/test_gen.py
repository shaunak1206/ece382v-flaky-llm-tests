import pytest

import event_bus
from event_bus import EventBus

SUBSCRIBE_NAMES = ("subscribe", "on", "register", "listen", "add_listener", "add_handler", "connect")
PUBLISH_NAMES = ("publish", "emit", "post", "dispatch", "fire", "trigger", "send")
UNSUBSCRIBE_NAMES = ("unsubscribe", "off", "unregister", "remove_listener", "remove_handler", "disconnect")


def _find(bus, names):
    for name in names:
        fn = getattr(bus, name, None)
        if callable(fn):
            return fn
    return None


def _api(bus):
    sub = _find(bus, SUBSCRIBE_NAMES)
    pub = _find(bus, PUBLISH_NAMES)
    if sub is None or pub is None:
        pytest.skip("EventBus subscribe/publish API not found")
    return sub, pub


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def received(self, payload):
        for args, kwargs in self.calls:
            if payload in args or payload in kwargs.values():
                return True
        return False


@pytest.fixture
def bus():
    return EventBus()


def test_version_string():
    assert event_bus.__version__ == "1.0.0"


def test_all_exports_event_bus():
    assert event_bus.__all__ == ["EventBus"]


def test_event_bus_is_same_as_bus_module_class():
    from event_bus.bus import EventBus as BusClass
    assert EventBus is BusClass


def test_event_bus_is_a_class():
    assert isinstance(EventBus, type)


def test_instantiation_returns_event_bus():
    assert isinstance(EventBus(), EventBus)


def test_instances_are_distinct():
    assert EventBus() is not EventBus()


def test_has_subscribe_and_publish_methods(bus):
    assert _find(bus, SUBSCRIBE_NAMES) is not None
    assert _find(bus, PUBLISH_NAMES) is not None


def test_subscribed_handler_is_called_on_publish(bus):
    sub, pub = _api(bus)
    rec = Recorder()
    sub("greet", rec)
    pub("greet")
    assert len(rec.calls) == 1


def test_handler_receives_payload(bus):
    sub, pub = _api(bus)
    rec = Recorder()
    payload = {"name": "world"}
    sub("greet", rec)
    pub("greet", payload)
    assert rec.received(payload)


def test_publish_without_subscribers_does_not_raise(bus):
    _, pub = _api(bus)
    pub("nobody-listens")


def test_multiple_handlers_all_called(bus):
    sub, pub = _api(bus)
    a, b = Recorder(), Recorder()
    sub("evt", a)
    sub("evt", b)
    pub("evt")
    assert len(a.calls) == 1
    assert len(b.calls) == 1


def test_handlers_only_called_for_their_event(bus):
    sub, pub = _api(bus)
    a, b = Recorder(), Recorder()
    sub("a", a)
    sub("b", b)
    pub("a")
    assert len(a.calls) == 1
    assert len(b.calls) == 0


def test_repeated_publish_calls_handler_each_time(bus):
    sub, pub = _api(bus)
    rec = Recorder()
    sub("tick", rec)
    pub("tick")
    pub("tick")
    pub("tick")
    assert len(rec.calls) == 3


def test_buses_are_isolated_from_each_other():
    bus1, bus2 = EventBus(), EventBus()
    sub1, _ = _api(bus1)
    _, pub2 = _api(bus2)
    rec = Recorder()
    sub1("evt", rec)
    pub2("evt")
    assert rec.calls == []


def test_unsubscribe_stops_delivery(bus):
    sub, pub = _api(bus)
    unsub = _find(bus, UNSUBSCRIBE_NAMES)
    if unsub is None:
        pytest.skip("EventBus has no unsubscribe API")
    rec = Recorder()
    sub("evt", rec)
    pub("evt")
    assert len(rec.calls) == 1
    unsub("evt", rec)
    pub("evt")
    assert len(rec.calls) == 1


def test_handlers_called_in_registration_order(bus):
    sub, pub = _api(bus)
    order = []
    sub("evt", lambda *a, **k: order.append(1))
    sub("evt", lambda *a, **k: order.append(2))
    sub("evt", lambda *a, **k: order.append(3))
    pub("evt")
    assert sorted(order) == [1, 2, 3]
    assert order == [1, 2, 3]
