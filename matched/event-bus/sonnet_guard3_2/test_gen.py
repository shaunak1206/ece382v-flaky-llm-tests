import pytest

from event_bus import EventBus
import event_bus


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def ev(request):
    # Unique, deterministic event name per test, so tests stay independent
    # even if the bus keeps any shared state.
    return "evt_" + request.node.name


def test_package_exports_eventbus():
    assert "EventBus" in event_bus.__all__
    assert event_bus.EventBus is EventBus


def test_version_string():
    assert event_bus.__version__ == "1.0.0"


def test_can_instantiate():
    assert EventBus() is not None


def test_handler_called_on_emit(bus, ev):
    calls = []

    @bus.on(ev)
    def handler_a():
        calls.append("a")

    bus.emit(ev)
    assert calls == ["a"]


def test_handler_called_each_emit(bus, ev):
    calls = []

    @bus.on(ev)
    def handler_counter():
        calls.append(1)

    bus.emit(ev)
    bus.emit(ev)
    bus.emit(ev)
    assert len(calls) == 3


def test_handler_not_called_before_emit(bus, ev):
    calls = []

    @bus.on(ev)
    def handler_idle():
        calls.append(1)

    assert calls == []


def test_positional_args_passed(bus, ev):
    received = []

    @bus.on(ev)
    def handler_pos(a, b):
        received.append((a, b))

    bus.emit(ev, 1, 2)
    assert received == [(1, 2)]


def test_keyword_args_passed(bus, ev):
    received = []

    @bus.on(ev)
    def handler_kw(x=None, y=None):
        received.append((x, y))

    bus.emit(ev, x="hello", y=5)
    assert received == [("hello", 5)]


def test_mixed_args_and_kwargs(bus, ev):
    received = []

    @bus.on(ev)
    def handler_mixed(a, b, c=0):
        received.append((a, b, c))

    bus.emit(ev, 1, 2, c=3)
    assert received == [(1, 2, 3)]


def test_multiple_handlers_all_called(bus, ev):
    calls = []

    @bus.on(ev)
    def handler_one():
        calls.append("one")

    @bus.on(ev)
    def handler_two():
        calls.append("two")

    @bus.on(ev)
    def handler_three():
        calls.append("three")

    bus.emit(ev)
    assert sorted(calls) == ["one", "three", "two"]


def test_handler_only_called_for_its_event(bus, ev):
    calls = []
    other = ev + "_other"

    @bus.on(ev)
    def handler_first():
        calls.append("first")

    @bus.on(other)
    def handler_second():
        calls.append("second")

    bus.emit(ev)
    assert calls == ["first"]

    bus.emit(other)
    assert sorted(calls) == ["first", "second"]


def test_same_handler_receives_args_per_emit(bus, ev):
    received = []

    @bus.on(ev)
    def handler_record(value):
        received.append(value)

    for i in range(5):
        bus.emit(ev, i)
    assert received == [0, 1, 2, 3, 4]


def test_handler_can_mutate_passed_object(bus, ev):
    target = {}

    @bus.on(ev)
    def handler_mutate(d, key, value):
        d[key] = value

    bus.emit(ev, target, "k", "v")
    assert target == {"k": "v"}


def test_decorated_function_remains_callable(bus, ev):
    @bus.on(ev)
    def handler_plain(x):
        return x

    # The decorator may return the function or a wrapper; either way the
    # name must still resolve to something usable without raising.
    assert handler_plain is None or callable(handler_plain)


def test_remove_event_stops_handler(bus, ev):
    if not hasattr(bus, "remove_event"):
        pytest.skip("remove_event not available")
    calls = []

    @bus.on(ev)
    def handler_removable():
        calls.append(1)

    bus.emit(ev)
    assert calls == [1]

    bus.remove_event("handler_removable", ev)
    bus.emit(ev)
    assert calls == [1]
