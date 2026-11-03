import threading

import pytest

from event_bus import EventBus
from event_bus.exceptions import EventDoesntExist


def make_recorder(name, log):
    """Return a subscriber that appends (name, args, kwargs) to log."""

    def recorder(*args, **kwargs):
        log.append((name, args, kwargs))

    recorder.__name__ = name
    return recorder


def test_new_bus_has_no_subscriptions():
    bus = EventBus()
    assert bus.event_count == 0


def test_add_event_registers_subscriber_and_emit_calls_it():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("sub", log), "greet")

    bus.emit("greet")

    assert log == [("sub", (), {})]
    assert bus.event_count == 1


def test_emit_passes_positional_and_keyword_args():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("sub", log), "data")

    bus.emit("data", 1, "two", key="value")

    assert log == [("sub", (1, "two"), {"key": "value"})]


def test_emit_calls_every_subscriber_of_event():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("first", log), "tick")
    bus.add_event(make_recorder("second", log), "tick")

    bus.emit("tick")

    names = sorted(entry[0] for entry in log)
    assert names == ["first", "second"]


def test_emit_unknown_event_is_noop():
    bus = EventBus()
    bus.emit("nothing_subscribed", 1, 2, x=3)
    assert bus.event_count == 0


def test_same_function_subscribed_twice_is_called_once():
    bus = EventBus()
    log = []
    sub = make_recorder("sub", log)
    bus.add_event(sub, "evt")
    bus.add_event(sub, "evt")

    bus.emit("evt")

    assert len(log) == 1
    assert bus.event_count == 1


def test_subscribers_are_scoped_to_their_event():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("a_sub", log), "a")
    bus.add_event(make_recorder("b_sub", log), "b")

    bus.emit("a")

    assert [entry[0] for entry in log] == ["a_sub"]


def test_event_count_totals_subscriptions_across_events():
    bus = EventBus()
    log = []
    sub = make_recorder("sub", log)
    bus.add_event(sub, "x")
    bus.add_event(sub, "y")
    bus.add_event(make_recorder("other", log), "y")

    assert bus.event_count == 3


def test_on_decorator_subscribes_and_preserves_return_value():
    bus = EventBus()
    received = []

    @bus.on("compute")
    def handler(value):
        received.append(value)
        return value * 10

    result = handler(4)
    bus.emit("compute", 7)

    assert result == 40
    assert received == [4, 7]
    assert bus.event_count == 1


def test_remove_event_stops_delivery_to_removed_subscriber():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("gone", log), "evt")

    bus.remove_event("gone", "evt")
    bus.emit("evt")

    assert log == []
    assert bus.event_count == 0


def test_remove_event_removes_all_functions_sharing_a_name():
    bus = EventBus()

    def make_handler():
        def handler():
            return None

        return handler

    first = make_handler()
    second = make_handler()
    assert first is not second
    bus.add_event(first, "evt")
    bus.add_event(second, "evt")

    bus.remove_event("handler", "evt")

    assert bus.event_count == 0


def test_remove_event_keeps_other_subscribers():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("remove_me", log), "evt")
    bus.add_event(make_recorder("keep_me", log), "evt")

    bus.remove_event("remove_me", "evt")
    bus.emit("evt")

    assert [entry[0] for entry in log] == ["keep_me"]
    assert bus.event_count == 1


def test_remove_event_missing_function_name_raises():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("present", log), "evt")

    with pytest.raises(EventDoesntExist):
        bus.remove_event("absent", "evt")

    assert bus.event_count == 1


def test_remove_event_unknown_event_raises():
    bus = EventBus()

    with pytest.raises(EventDoesntExist):
        bus.remove_event("anything", "never_registered")


def test_emit_only_with_string_name_calls_only_that_subscriber():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("alpha", log), "evt")
    bus.add_event(make_recorder("beta", log), "evt")

    bus.emit_only("evt", "alpha", 5, flag=True)

    assert log == [("alpha", (5,), {"flag": True})]


def test_emit_only_with_list_of_names_calls_each_match():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("alpha", log), "evt")
    bus.add_event(make_recorder("beta", log), "evt")
    bus.add_event(make_recorder("gamma", log), "evt")

    bus.emit_only("evt", ["alpha", "gamma"])

    assert sorted(entry[0] for entry in log) == ["alpha", "gamma"]


def test_emit_only_unmatched_name_calls_nothing():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("alpha", log), "evt")

    bus.emit_only("evt", "not_subscribed")

    assert log == []


def test_emit_after_runs_subscribers_after_function_and_returns_result():
    bus = EventBus()
    order = []

    def on_done():
        order.append("subscriber")

    bus.add_event(on_done, "done")

    @bus.emit_after("done")
    def work(x):
        order.append("work")
        return x * 2

    result = work(3)

    assert result == 6
    assert order == ["work", "subscriber"]


def test_emit_after_subscribers_receive_no_arguments():
    bus = EventBus()
    received = []

    def on_done(*args, **kwargs):
        received.append((args, kwargs))

    bus.add_event(on_done, "done")

    @bus.emit_after("done")
    def work(a, b=1):
        return a + b

    assert work(1, b=2) == 3
    assert received == [((), {})]


def test_emit_after_skips_emit_when_function_raises():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("sub", log), "done")

    @bus.emit_after("done")
    def failing():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        failing()

    assert log == []


def test_emit_threads_runs_subscriber_in_another_thread():
    bus = EventBus()
    seen = {}
    finished = threading.Event()

    def handler(*args, **kwargs):
        seen["thread"] = threading.current_thread().ident
        seen["args"] = args
        finished.set()

    bus.add_event(handler, "evt")

    bus.emit("evt", 1, 2, threads=True)

    assert finished.wait(5)
    assert seen["thread"] != threading.current_thread().ident
    assert seen["args"] == (1, 2)


def test_threads_kwarg_is_not_forwarded_to_subscribers():
    bus = EventBus()
    log = []
    bus.add_event(make_recorder("sub", log), "evt")

    bus.emit("evt", 1, threads=False, extra=2)

    assert log == [("sub", (1,), {"extra": 2})]


def test_repr_and_str_describe_bus():
    bus = EventBus()
    bus.add_event(make_recorder("a", []), "first")
    bus.add_event(make_recorder("b", []), "first")
    bus.add_event(make_recorder("c", []), "second")

    assert repr(bus) == "<EventBus: 3 subscribed events>"
    assert str(bus) == "EventBus"


def test_slots_block_new_attributes():
    bus = EventBus()

    with pytest.raises(AttributeError):
        bus.unexpected_attribute = 1
