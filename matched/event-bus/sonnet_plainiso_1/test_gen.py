import threading

import pytest

from event_bus import EventBus
from event_bus.exceptions import EventDoesntExist


@pytest.fixture
def bus():
    return EventBus()


def test_new_bus_has_no_events(bus):
    assert bus.event_count == 0


def test_cls_name_and_str(bus):
    assert bus.cls_name == "EventBus"
    assert str(bus) == "EventBus"


def test_repr_reports_event_count(bus):
    assert repr(bus) == "<EventBus: 0 subscribed events>"

    def f():
        pass

    bus.add_event(f, "a")
    assert repr(bus) == "<EventBus: 1 subscribed events>"


def test_add_event_and_emit_calls_function(bus):
    calls = []

    def handler():
        calls.append(1)

    bus.add_event(handler, "evt")
    bus.emit("evt")
    assert calls == [1]


def test_emit_passes_args_and_kwargs(bus):
    received = []

    def handler(a, b, c=None):
        received.append((a, b, c))

    bus.add_event(handler, "evt")
    bus.emit("evt", 1, 2, c=3)
    assert received == [(1, 2, 3)]


def test_emit_unknown_event_is_noop(bus):
    bus.emit("nothing", 1, 2, x=3)
    assert bus.event_count == 0


def test_on_decorator_registers_and_returns_working_wrapper(bus):
    calls = []

    @bus.on("evt")
    def handler(x):
        calls.append(x)
        return x * 2

    assert bus.event_count == 1
    assert handler.__name__ == "handler"
    assert handler(4) == 8
    calls.clear()
    bus.emit("evt", 5)
    assert calls == [5]


def test_multiple_functions_on_same_event_all_called(bus):
    calls = set()

    def one():
        calls.add("one")

    def two():
        calls.add("two")

    bus.add_event(one, "evt")
    bus.add_event(two, "evt")
    bus.emit("evt")
    assert calls == {"one", "two"}
    assert bus.event_count == 2


def test_adding_same_function_twice_is_deduplicated(bus):
    calls = []

    def handler():
        calls.append(1)

    bus.add_event(handler, "evt")
    bus.add_event(handler, "evt")
    assert bus.event_count == 1
    bus.emit("evt")
    assert calls == [1]


def test_event_count_sums_across_events(bus):
    def f():
        pass

    def g():
        pass

    bus.add_event(f, "a")
    bus.add_event(g, "a")
    bus.add_event(f, "b")
    assert bus.event_count == 3


def test_events_are_isolated(bus):
    calls = []

    def a_handler():
        calls.append("a")

    def b_handler():
        calls.append("b")

    bus.add_event(a_handler, "a")
    bus.add_event(b_handler, "b")
    bus.emit("a")
    assert calls == ["a"]


def test_emit_with_threads_runs_functions(bus):
    done = threading.Event()
    received = []

    def handler(x):
        received.append(x)
        done.set()

    bus.add_event(handler, "evt")
    bus.emit("evt", 42, threads=True)
    assert done.wait(timeout=5)
    assert received == [42]


def test_emit_threads_kwarg_not_forwarded_to_function(bus):
    done = threading.Event()
    received = {}

    def handler(**kwargs):
        received.update(kwargs)
        done.set()

    bus.add_event(handler, "evt")
    bus.emit("evt", a=1, threads=True)
    assert done.wait(timeout=5)
    assert received == {"a": 1}

    received.clear()
    bus.emit("evt", b=2, threads=False)
    assert received == {"b": 2}


def test_emit_only_runs_selected_function_by_name(bus):
    calls = []

    def first(x):
        calls.append(("first", x))

    def second(x):
        calls.append(("second", x))

    bus.add_event(first, "evt")
    bus.add_event(second, "evt")
    bus.emit_only("evt", "second", 7)
    assert calls == [("second", 7)]


def test_emit_only_accepts_list_of_names(bus):
    calls = set()

    def first():
        calls.add("first")

    def second():
        calls.add("second")

    def third():
        calls.add("third")

    for f in (first, second, third):
        bus.add_event(f, "evt")
    bus.emit_only("evt", ["first", "third"])
    assert calls == {"first", "third"}


def test_emit_only_no_match_calls_nothing(bus):
    calls = []

    def handler():
        calls.append(1)

    bus.add_event(handler, "evt")
    bus.emit_only("evt", "other")
    assert calls == []


def test_emit_after_runs_event_after_function_and_returns_value(bus):
    order = []

    @bus.on("done")
    def post():
        order.append("post")

    @bus.emit_after("done")
    def work(x):
        order.append("work")
        return x + 1

    assert work(1) == 2
    assert order == ["work", "post"]
    assert work.__name__ == "work"


def test_remove_event_removes_only_named_function(bus):
    calls = []

    def keep():
        calls.append("keep")

    def drop():
        calls.append("drop")

    bus.add_event(keep, "evt")
    bus.add_event(drop, "evt")
    bus.remove_event("drop", "evt")
    assert bus.event_count == 1
    bus.emit("evt")
    assert calls == ["keep"]


def test_remove_event_missing_function_raises(bus):
    def handler():
        pass

    bus.add_event(handler, "evt")
    with pytest.raises(EventDoesntExist):
        bus.remove_event("nope", "evt")
    assert bus.event_count == 1


def test_remove_event_on_unknown_event_raises(bus):
    with pytest.raises(EventDoesntExist):
        bus.remove_event("anything", "never_registered")
