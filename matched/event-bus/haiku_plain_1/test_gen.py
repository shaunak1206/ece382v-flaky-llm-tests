import threading

import pytest

from event_bus import EventBus
from event_bus.exceptions import EventDoesntExist


def test_new_bus_has_no_subscriptions():
    bus = EventBus()
    assert bus.event_count == 0


def test_on_decorator_subscribes_and_wraps_function():
    bus = EventBus()

    @bus.on("greet")
    def greet(name):
        return "hello " + name

    assert bus.event_count == 1
    assert greet("bob") == "hello bob"


def test_on_wrapper_keeps_function_name():
    bus = EventBus()

    @bus.on("ping")
    def ping():
        pass

    assert ping.__name__ == "ping"


def test_add_event_registers_function():
    bus = EventBus()

    def handler():
        pass

    bus.add_event(handler, "evt")
    assert bus.event_count == 1


def test_same_function_twice_on_one_event_counts_once():
    bus = EventBus()

    def handler():
        pass

    bus.add_event(handler, "evt")
    bus.add_event(handler, "evt")
    assert bus.event_count == 1


def test_same_function_on_two_events_counts_twice():
    bus = EventBus()

    def handler():
        pass

    bus.add_event(handler, "first")
    bus.add_event(handler, "second")
    assert bus.event_count == 2


def test_emit_passes_args_and_kwargs_to_all_subscribers():
    bus = EventBus()
    calls = []

    def one(a, b=None):
        calls.append(("one", a, b))

    def two(a, b=None):
        calls.append(("two", a, b))

    bus.add_event(one, "evt")
    bus.add_event(two, "evt")
    bus.emit("evt", 1, b=2)

    assert sorted(calls) == [("one", 1, 2), ("two", 1, 2)]


def test_emit_unknown_event_is_noop():
    bus = EventBus()
    bus.emit("nobody_listens", 1, 2, key="value")
    assert bus.event_count == 0


def test_emit_threads_false_runs_handler_inline():
    bus = EventBus()
    seen = []

    def handler():
        seen.append(threading.get_ident())

    bus.add_event(handler, "evt")
    bus.emit("evt", threads=False)

    assert seen == [threading.get_ident()]


def test_emit_threads_true_runs_handlers_on_worker_threads():
    bus = EventBus()
    main_ident = threading.get_ident()
    seen = []
    done = threading.Event()

    def handler(value):
        seen.append((threading.get_ident(), value))
        done.set()

    bus.add_event(handler, "evt")
    bus.emit("evt", 7, threads=True)

    assert done.wait(timeout=5)
    assert len(seen) == 1
    ident, value = seen[0]
    assert ident != main_ident
    assert value == 7


def test_emit_does_not_forward_threads_kwarg_to_handlers():
    bus = EventBus()
    received = []

    def handler(**kwargs):
        received.append(kwargs)

    bus.add_event(handler, "evt")
    bus.emit("evt", threads=False, extra=3)

    assert received == [{"extra": 3}]


def test_emit_propagates_handler_exception():
    bus = EventBus()

    def broken():
        raise ValueError("boom")

    bus.add_event(broken, "evt")

    with pytest.raises(ValueError):
        bus.emit("evt")


def test_emit_only_runs_only_named_subscriber():
    bus = EventBus()
    calls = []

    def alpha(x):
        calls.append(("alpha", x))

    def beta(x):
        calls.append(("beta", x))

    bus.add_event(alpha, "evt")
    bus.add_event(beta, "evt")
    bus.emit_only("evt", "alpha", 5)

    assert calls == [("alpha", 5)]


def test_emit_only_accepts_list_of_names():
    bus = EventBus()
    calls = []

    def alpha():
        calls.append("alpha")

    def beta():
        calls.append("beta")

    def gamma():
        calls.append("gamma")

    bus.add_event(alpha, "evt")
    bus.add_event(beta, "evt")
    bus.add_event(gamma, "evt")
    bus.emit_only("evt", ["alpha", "gamma"])

    assert sorted(calls) == ["alpha", "gamma"]


def test_emit_only_forwards_arguments_and_kwargs():
    bus = EventBus()
    received = []

    def target(x, k=None):
        received.append((x, k))

    bus.add_event(target, "evt")
    bus.emit_only("evt", "target", 10, k="kw")

    assert received == [(10, "kw")]


def test_emit_after_runs_listeners_after_call_and_returns_value():
    bus = EventBus()
    order = []

    def listener():
        order.append("listener")

    bus.add_event(listener, "done")

    @bus.emit_after("done")
    def work(x, y=2):
        order.append("work")
        return x + y

    assert work(1, y=3) == 4
    assert order == ["work", "listener"]


def test_emit_after_listeners_receive_no_arguments():
    bus = EventBus()
    received = []

    def listener(*args, **kwargs):
        received.append((args, kwargs))

    bus.add_event(listener, "done")

    @bus.emit_after("done")
    def work(a, b=1):
        return a * b

    assert work(3, b=4) == 12
    assert received == [((), {})]


def test_remove_event_removes_only_named_function():
    bus = EventBus()
    calls = []

    @bus.on("evt")
    def keep():
        calls.append("keep")

    @bus.on("evt")
    def drop():
        calls.append("drop")

    bus.remove_event("drop", "evt")
    bus.emit("evt")

    assert calls == ["keep"]
    assert bus.event_count == 1


def test_remove_event_missing_function_raises():
    bus = EventBus()

    def present():
        pass

    bus.add_event(present, "evt")

    with pytest.raises(EventDoesntExist):
        bus.remove_event("absent", "evt")
    assert bus.event_count == 1


def test_remove_event_unknown_event_raises():
    bus = EventBus()

    with pytest.raises(EventDoesntExist):
        bus.remove_event("anything", "never_registered")


def test_remove_last_subscriber_brings_count_to_zero():
    bus = EventBus()

    def solo():
        pass

    bus.add_event(solo, "evt")
    assert bus.event_count == 1

    bus.remove_event("solo", "evt")
    assert bus.event_count == 0


def test_repr_and_str_report_class_and_count():
    bus = EventBus()

    def first():
        pass

    def second():
        pass

    bus.add_event(first, "a")
    bus.add_event(second, "b")

    assert repr(bus) == "<EventBus: 2 subscribed events>"
    assert str(bus) == "EventBus"


def test_instances_do_not_share_subscriptions():
    bus_one = EventBus()
    bus_two = EventBus()

    def handler():
        pass

    bus_one.add_event(handler, "evt")

    assert bus_one.event_count == 1
    assert bus_two.event_count == 0


def test_slots_block_new_attributes():
    bus = EventBus()

    with pytest.raises(AttributeError):
        bus.unexpected_attribute = 1
