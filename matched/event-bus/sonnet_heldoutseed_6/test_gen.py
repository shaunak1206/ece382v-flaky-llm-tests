import pytest

from event_bus import EventBus
import event_bus


def test_package_exports_event_bus():
    assert "EventBus" in event_bus.__all__
    assert event_bus.__version__ == "1.0.0"


def test_can_instantiate():
    bus = EventBus()
    assert isinstance(bus, EventBus)


def test_on_decorator_registers_and_emit_calls_handler():
    bus = EventBus()
    calls = []

    @bus.on("greet")
    def handler():
        calls.append("called")

    bus.emit("greet")
    assert calls == ["called"]


def test_emit_passes_positional_args():
    bus = EventBus()
    received = []

    @bus.on("data")
    def handler(a, b):
        received.append((a, b))

    bus.emit("data", 1, 2)
    assert received == [(1, 2)]


def test_emit_passes_keyword_args():
    bus = EventBus()
    received = {}

    @bus.on("data")
    def handler(**kwargs):
        received.update(kwargs)

    bus.emit("data", x=1, y="two")
    assert received == {"x": 1, "y": "two"}


def test_emit_passes_mixed_args_and_kwargs():
    bus = EventBus()
    received = []

    @bus.on("mixed")
    def handler(a, b=None):
        received.append((a, b))

    bus.emit("mixed", "first", b="second")
    assert received == [("first", "second")]


def test_decorated_function_remains_callable_directly():
    bus = EventBus()

    @bus.on("add")
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_multiple_handlers_for_same_event_all_called():
    bus = EventBus()
    calls = set()

    @bus.on("multi")
    def first():
        calls.add("first")

    @bus.on("multi")
    def second():
        calls.add("second")

    bus.emit("multi")
    assert calls == {"first", "second"}


def test_handlers_only_called_for_their_event():
    bus = EventBus()
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
    assert sorted(calls) == ["a", "b"]


def test_same_function_registered_for_multiple_events():
    bus = EventBus()
    calls = []

    @bus.on("one")
    @bus.on("two")
    def handler():
        calls.append(1)

    bus.emit("one")
    bus.emit("two")
    assert len(calls) == 2


def test_add_event_registers_function():
    bus = EventBus()
    calls = []

    def handler(value):
        calls.append(value)

    bus.add_event(handler, "added")
    bus.emit("added", 42)
    assert calls == [42]


def test_emit_multiple_times_calls_handler_each_time():
    bus = EventBus()
    counter = []

    @bus.on("tick")
    def handler():
        counter.append(1)

    bus.emit("tick")
    bus.emit("tick")
    bus.emit("tick")
    assert len(counter) == 3


def test_remove_event_stops_handler_being_called():
    bus = EventBus()
    calls = []

    @bus.on("removable")
    def handler():
        calls.append(1)

    bus.emit("removable")
    assert len(calls) == 1

    bus.remove_event("handler", "removable")
    bus.emit("removable")
    assert len(calls) == 1


def test_remove_event_only_removes_named_function():
    bus = EventBus()
    calls = []

    @bus.on("shared")
    def keep():
        calls.append("keep")

    @bus.on("shared")
    def drop():
        calls.append("drop")

    bus.remove_event("drop", "shared")
    bus.emit("shared")
    assert calls == ["keep"]


def test_remove_nonexistent_function_raises():
    bus = EventBus()

    @bus.on("evt")
    def handler():
        pass

    with pytest.raises(Exception):
        bus.remove_event("does_not_exist", "evt")
