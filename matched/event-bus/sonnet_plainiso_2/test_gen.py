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

    def handler():
        pass

    bus.add_event(handler, "a")
    assert repr(bus) == "<EventBus: 1 subscribed events>"


def test_add_event_and_emit_calls_function_with_args(bus):
    calls = []

    def handler(x, y=None):
        calls.append((x, y))

    bus.add_event(handler, "evt")
    bus.emit("evt", 1, y=2)
    assert calls == [(1, 2)]


def test_on_decorator_registers_and_returns_working_wrapper(bus):
    calls = []

    @bus.on("evt")
    def handler(value):
        calls.append(value)
        return value * 2

    assert bus.event_count == 1
    bus.emit("evt", 5)
    assert calls == [5]
    # The decorated name remains directly callable and returns the result.
    assert handler(3) == 6
    assert handler.__name__ == "handler"


def test_emit_unknown_event_does_nothing(bus):
    bus.emit("nobody-listens", 1, 2, key="value")
    assert bus.event_count == 0


def test_emit_only_runs_functions_of_matching_event(bus):
    calls = []

    def first():
        calls.append("first")

    def second():
        calls.append("second")

    bus.add_event(first, "one")
    bus.add_event(second, "two")
    bus.emit("one")
    assert calls == ["first"]


def test_multiple_subscribers_all_called(bus):
    calls = set()

    def a():
        calls.add("a")

    def b():
        calls.add("b")

    def c():
        calls.add("c")

    for f in (a, b, c):
        bus.add_event(f, "evt")

    bus.emit("evt")
    assert calls == {"a", "b", "c"}


def test_adding_same_function_twice_is_deduplicated(bus):
    calls = []

    def handler():
        calls.append(1)

    bus.add_event(handler, "evt")
    bus.add_event(handler, "evt")
    assert bus.event_count == 1
    bus.emit("evt")
    assert calls == [1]


def test_event_count_sums_functions_across_events(bus):
    def f():
        pass

    def g():
        pass

    bus.add_event(f, "a")
    bus.add_event(g, "a")
    bus.add_event(f, "b")
    assert bus.event_count == 3


def test_emit_with_threads_runs_handlers_in_other_threads(bus):
    done = threading.Event()
    seen = {}

    def handler(value):
        seen["value"] = value
        seen["thread"] = threading.current_thread()
        done.set()

    bus.add_event(handler, "evt")
    bus.emit("evt", 42, threads=True)

    assert done.wait(timeout=5)
    assert seen["value"] == 42
    assert seen["thread"] is not threading.main_thread()


def test_emit_threads_kwarg_not_forwarded_to_handlers(bus):
    done = threading.Event()
    received = {}

    def handler(**kwargs):
        received.update(kwargs)
        done.set()

    bus.add_event(handler, "evt")
    bus.emit("evt", threads=True, name="x")
    assert done.wait(timeout=5)
    assert received == {"name": "x"}

    received.clear()
    bus.emit("evt", threads=False, name="y")
    assert received == {"name": "y"}


def test_emit_with_threads_false_runs_synchronously(bus):
    calls = []

    def handler():
        calls.append(threading.current_thread())

    bus.add_event(handler, "evt")
    bus.emit("evt", threads=False)
    assert calls == [threading.current_thread()]


def test_emit_only_with_single_name_string(bus):
    calls = []

    def keep():
        calls.append("keep")

    def skip():
        calls.append("skip")

    bus.add_event(keep, "evt")
    bus.add_event(skip, "evt")
    bus.emit_only("evt", "keep")
    assert calls == ["keep"]


def test_emit_only_with_list_of_names_passes_args(bus):
    calls = []

    def a(x):
        calls.append(("a", x))

    def b(x):
        calls.append(("b", x))

    def c(x):
        calls.append(("c", x))

    for f in (a, b, c):
        bus.add_event(f, "evt")

    bus.emit_only("evt", ["a", "c"], 7)
    assert sorted(calls) == [("a", 7), ("c", 7)]


def test_emit_only_no_match_calls_nothing(bus):
    calls = []

    def handler():
        calls.append(1)

    bus.add_event(handler, "evt")
    bus.emit_only("evt", "missing")
    assert calls == []


def test_emit_after_triggers_event_after_function_and_returns_value(bus):
    order = []

    def listener():
        order.append("listener")

    bus.add_event(listener, "done")

    @bus.emit_after("done")
    def work(a, b):
        order.append("work")
        return a + b

    assert work(2, 3) == 5
    assert order == ["work", "listener"]
    assert work.__name__ == "work"


def test_emit_after_does_not_pass_arguments_to_listeners(bus):
    received = []

    def listener(*args, **kwargs):
        received.append((args, kwargs))

    bus.add_event(listener, "done")

    @bus.emit_after("done")
    def work(x, key=None):
        return x

    work(1, key="k")
    assert received == [((), {})]


def test_remove_event_removes_only_named_function(bus):
    calls = []

    def one():
        calls.append("one")

    def two():
        calls.append("two")

    bus.add_event(one, "evt")
    bus.add_event(two, "evt")
    bus.remove_event("one", "evt")

    assert bus.event_count == 1
    bus.emit("evt")
    assert calls == ["two"]


def test_remove_event_missing_function_raises(bus):
    def one():
        pass

    bus.add_event(one, "evt")
    with pytest.raises(EventDoesntExist):
        bus.remove_event("nope", "evt")
    assert bus.event_count == 1


def test_remove_event_on_unknown_event_raises(bus):
    with pytest.raises(EventDoesntExist):
        bus.remove_event("anything", "never-registered")


def test_private_helpers_report_function_names(bus):
    def alpha():
        pass

    def beta():
        pass

    bus.add_event(alpha, "evt")
    bus.add_event(beta, "evt")
    assert sorted(bus._event_func_names("evt")) == ["alpha", "beta"]
    assert bus._event_func_names("other") == []
