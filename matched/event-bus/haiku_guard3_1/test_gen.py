"""Unit tests for the EventBus class exposed by the event_bus package.

Note: only event_bus/__init__.py was provided. Tests that check the package
surface (exports, version, class identity) rely only on that file. Tests that
exercise publish/subscribe behaviour assume the conventional method names
`subscribe(event_name, handler)`, `publish(event_name, payload)` and
`unsubscribe(event_name, handler)`. If bus.py uses different names, those tests
skip rather than fail.
"""

import inspect

import pytest

import event_bus
from event_bus import EventBus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_methods(obj, *names):
    """Skip the calling test if the object lacks any of the named callables."""
    for name in names:
        if not callable(getattr(obj, name, None)):
            pytest.skip("EventBus has no callable '%s' method" % name)


@pytest.fixture
def bus():
    """A fresh EventBus for each test; no state is shared between tests."""
    return EventBus()


# ---------------------------------------------------------------------------
# Package surface (derived from event_bus/__init__.py)
# ---------------------------------------------------------------------------

def test_version_is_1_0_0():
    assert event_bus.__version__ == "1.0.0"


def test_version_is_string():
    assert isinstance(event_bus.__version__, str)


def test_all_lists_only_event_bus():
    assert event_bus.__all__ == ["EventBus"]


def test_star_import_exposes_only_event_bus():
    namespace = {}
    exec("from event_bus import *", namespace)
    exported = set(namespace) - {"__builtins__"}
    assert exported == {"EventBus"}


def test_package_event_bus_is_bus_module_class():
    from event_bus.bus import EventBus as BusModuleEventBus

    assert EventBus is BusModuleEventBus


def test_event_bus_is_a_class():
    assert inspect.isclass(EventBus)


def test_event_bus_can_be_instantiated():
    instance = EventBus()
    assert isinstance(instance, EventBus)


def test_monkeypatched_version_is_restored_after_test(monkeypatch):
    # Module attribute replaced only through monkeypatch, so it is undone
    # even if this test fails.
    monkeypatch.setattr(event_bus, "__version__", "9.9.9")
    assert event_bus.__version__ == "9.9.9"


def test_version_unchanged_by_previous_monkeypatch():
    # Runs after the monkeypatch test above in file order; it must still
    # see the original value, proving the attribute was restored.
    assert event_bus.__version__ == "1.0.0"


# ---------------------------------------------------------------------------
# Publish / subscribe behaviour
# ---------------------------------------------------------------------------

def test_publish_without_subscribers_does_not_raise(bus):
    _require_methods(bus, "publish")
    bus.publish("nobody-listening", 1)


def test_subscriber_receives_published_payload(bus):
    _require_methods(bus, "subscribe", "publish")
    received = []
    bus.subscribe("greeting", received.append)
    bus.publish("greeting", "hello")
    assert received == ["hello"]


def test_subscriber_not_called_for_other_event_name(bus):
    _require_methods(bus, "subscribe", "publish")
    received = []
    bus.subscribe("alpha", received.append)
    bus.publish("beta", 1)
    assert received == []


def test_each_subscriber_receives_event_once(bus):
    _require_methods(bus, "subscribe", "publish")
    calls_a = []
    calls_b = []
    bus.subscribe("tick", calls_a.append)
    bus.subscribe("tick", calls_b.append)
    bus.publish("tick", 7)
    assert calls_a == [7]
    assert calls_b == [7]


def test_multiple_publishes_delivered_in_publish_order(bus):
    _require_methods(bus, "subscribe", "publish")
    received = []
    bus.subscribe("count", received.append)
    for value in (1, 2, 3):
        bus.publish("count", value)
    assert received == [1, 2, 3]


def test_publish_reaches_only_handlers_for_that_event(bus):
    _require_methods(bus, "subscribe", "publish")
    left = []
    right = []
    bus.subscribe("left", left.append)
    bus.subscribe("right", right.append)
    bus.publish("left", "L")
    assert left == ["L"]
    assert right == []


def test_handler_subscribed_twice_is_not_required_to_be_deduplicated_here(bus):
    # Only checks that publishing still works after a repeated subscription;
    # it does not assert a specific duplicate-handling policy.
    _require_methods(bus, "subscribe", "publish")
    received = []
    bus.subscribe("repeat", received.append)
    bus.subscribe("repeat", received.append)
    bus.publish("repeat", "x")
    assert len(received) >= 1
    assert all(item == "x" for item in received)


def test_publish_with_none_payload_delivers_none(bus):
    _require_methods(bus, "subscribe", "publish")
    received = []
    bus.subscribe("empty", received.append)
    bus.publish("empty", None)
    assert received == [None]


def test_separate_instances_do_not_share_subscribers():
    _require_methods(EventBus(), "subscribe", "publish")
    first = EventBus()
    second = EventBus()
    received = []
    first.subscribe("shared-name", received.append)
    second.publish("shared-name", "leak?")
    assert received == []


def test_unsubscribed_handler_is_not_called(bus):
    _require_methods(bus, "subscribe", "unsubscribe", "publish")
    received = []

    def handler(payload):
        received.append(payload)

    bus.subscribe("topic", handler)
    bus.unsubscribe("topic", handler)
    bus.publish("topic", "after-unsubscribe")
    assert received == []


def test_unsubscribe_leaves_other_handlers_active(bus):
    _require_methods(bus, "subscribe", "unsubscribe", "publish")
    kept = []
    removed = []

    def removed_handler(payload):
        removed.append(payload)

    bus.subscribe("topic", kept.append)
    bus.subscribe("topic", removed_handler)
    bus.unsubscribe("topic", removed_handler)
    bus.publish("topic", 5)
    assert kept == [5]
    assert removed == []


def test_publishing_with_no_handlers_after_unsubscribe_does_not_raise(bus):
    _require_methods(bus, "subscribe", "unsubscribe", "publish")

    def handler(payload):
        pass

    bus.subscribe("solo", handler)
    bus.unsubscribe("solo", handler)
    bus.publish("solo", 0)
