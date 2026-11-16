import pytest

import event_bus
from event_bus import EventBus


@pytest.fixture
def bus():
    return EventBus()


def test_package_exports_eventbus():
    assert "EventBus" in event_bus.__all__
    assert event_bus.EventBus is EventBus
    assert event_bus.__version__ == "1.0.0"


def test_eventbus_can_be_instantiated():
    assert isinstance(EventBus(), EventBus)


def test_on_decorator_registers_handler_and_emit_calls_it(bus):
    calls = []

    @bus.on("greet")
    def handler():
        calls.append("called")

    bus.emit("greet")
    assert calls == ["called"]


def test_emit_passes_positional_arguments(bus):
    received = []

    @bus.on("data")
    def handler(a, b):
        received.append((a, b))

    bus.emit("data", 1, 2)
    assert received == [(1, 2)]


def test_emit_passes_keyword_arguments(bus):
    received = []

    @bus.on("data")
    def handler(x=None, y=None):
        received.append((x, y))

    bus.emit("data", x="a", y="b")
    assert received == [("a", "b")]


def test_emit_passes_mixed_arguments(bus):
    received = []

    @bus.on("mixed")
    def handler(a, b, c=None):
        received.append((a, b, c))

    bus.emit("mixed", 1, 2, c=3)
    assert received == [(1, 2, 3)]


def test_multiple_handlers_for_same_event_are_all_called(bus):
    calls = []

    @bus.on("evt")
    def first():
        calls.append("first")

    @bus.on("evt")
    def second():
        calls.append("second")

    bus.emit("evt")
    assert sorted(calls) == ["first", "second"]
    assert len(calls) == 2


def test_emit_unknown_event_does_not_call_other_handlers(bus):
    calls = []

    @bus.on("known")
    def handler():
        calls.append(1)

    bus.emit("unknown")
    assert calls == []


def test_handlers_only_receive_their_own_event(bus):
    calls = []

    @bus.on("a")
    def handler_a():
        calls.append("a")

    @bus.on("b")
    def handler_b():
        calls.append("b")

    bus.emit("a")
    assert calls == ["a"]
    bus.emit("b")
    assert calls == ["a", "b"]


def test_emit_multiple_times_calls_handler_each_time(bus):
    counter = {"n": 0}

    @bus.on("tick")
    def handler():
        counter["n"] += 1

    bus.emit("tick")
    bus.emit("tick")
    bus.emit("tick")
    assert counter["n"] == 3


def test_stacked_decorators_register_handler_for_multiple_events(bus):
    calls = []

    @bus.on("one")
    @bus.on("two")
    def handler():
        calls.append(1)

    bus.emit("one")
    bus.emit("two")
    assert calls == [1, 1]


def test_decorated_function_remains_directly_callable(bus):
    calls = []

    @bus.on("evt")
    def handler(value):
        calls.append(value)

    handler("direct")
    assert calls == ["direct"]


def test_separate_bus_instances_do_not_share_handlers():
    bus1 = EventBus()
    bus2 = EventBus()
    calls = []

    @bus1.on("evt")
    def handler():
        calls.append("bus1")

    bus2.emit("evt")
    assert calls == []
    bus1.emit("evt")
    assert calls == ["bus1"]


def test_add_event_registers_plain_function(bus):
    if not hasattr(bus, "add_event"):
        pytest.skip("add_event not available")
    calls = []

    def handler(x):
        calls.append(x)

    bus.add_event(handler, "evt")
    bus.emit("evt", 42)
    assert calls == [42]
