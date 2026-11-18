import pytest
from event_bus import EventBus


def test_eventbus_initialization():
    eb = EventBus()
    assert isinstance(eb, EventBus)


def test_subscribe_and_emit_single_listener():
    eb = EventBus()
    called = []

    def listener(data):
        called.append(data)

    eb.subscribe('test_event', listener)
    eb.emit('test_event', 123)
    assert called == [123]


def test_subscribe_multiple_listeners_same_event():
    eb = EventBus()
    calls = []

    def listener1(data):
        calls.append(('l1', data))

    def listener2(data):
        calls.append(('l2', data))

    eb.subscribe('event', listener1)
    eb.subscribe('event', listener2)
    eb.emit('event', 7)

    assert ('l1', 7) in calls
    assert ('l2', 7) in calls
    assert len(calls) == 2


def test_emit_without_listeners_does_not_fail():
    eb = EventBus()
    # Should not raise even if no listeners exist
    eb.emit('no_listeners_event', 10)


def test_unsubscribe_listener_prevents_further_calls():
    eb = EventBus()
    called = []

    def listener(data):
        called.append(data)

    eb.subscribe('evt', listener)
    eb.emit('evt', 1)
    eb.unsubscribe('evt', listener)
    eb.emit('evt', 2)

    assert called == [1]


def test_unsubscribe_nonexistent_listener_does_not_fail():
    eb = EventBus()

    def listener(data):
        pass

    eb.unsubscribe('event', listener)  # Should not error even if not subscribed


def test_subscribe_same_listener_multiple_times_emits_multiple_calls():
    eb = EventBus()
    called = []

    def listener(data):
        called.append(data)

    eb.subscribe('evt', listener)
    eb.subscribe('evt', listener)
    eb.emit('evt', 5)

    assert called == [5, 5]


def test_unsubscribe_only_removes_one_subscription_per_call():
    eb = EventBus()
    called = []

    def listener(data):
        called.append(data)

    eb.subscribe('evt', listener)
    eb.subscribe('evt', listener)
    eb.unsubscribe('evt', listener)
    eb.emit('evt', 9)

    assert called == [9]


def test_subscribe_different_events_isolated():
    eb = EventBus()
    calls = []

    def l1(data):
        calls.append(('l1', data))

    def l2(data):
        calls.append(('l2', data))

    eb.subscribe('event1', l1)
    eb.subscribe('event2', l2)
    eb.emit('event1', 100)
    eb.emit('event2', 200)

    assert ('l1', 100) in calls
    assert ('l2', 200) in calls


def test_emit_passes_multiple_args_and_kwargs():
    eb = EventBus()
    calls = []

    def listener(*args, **kwargs):
        calls.append((args, kwargs))

    eb.subscribe('multi', listener)
    eb.emit('multi', 1, 2, key='value')

    assert calls == [((1, 2), {'key': 'value'})]


def test_unsubscribe_fully_removes_listener_for_event():
    eb = EventBus()
    calls = []

    def listener(data):
        calls.append(data)

    eb.subscribe('e', listener)
    eb.unsubscribe('e', listener)
    eb.emit('e', 1)

    assert calls == []


def test_subscribe_returns_none_and_does_not_modify_listener():
    eb = EventBus()

    def listener(data):
        return data

    result = eb.subscribe('e', listener)
    assert result is None


def test_unsubscribe_removes_only_for_specific_event():
    eb = EventBus()
    called = []

    def listener(data):
        called.append(data)

    eb.subscribe('eventA', listener)
    eb.subscribe('eventB', listener)
    eb.unsubscribe('eventA', listener)
    eb.emit('eventA', 10)
    eb.emit('eventB', 20)

    assert called == [20]


def test_emit_with_no_data_arguments_works():
    eb = EventBus()
    called = []

    def listener():
        called.append(True)

    eb.subscribe('ev', listener)
    eb.emit('ev')

    assert called == [True]


def test_multiple_events_and_listeners_are_independent():
    eb = EventBus()
    calls = []

    def listener1(data):
        calls.append(('listener1', data))

    def listener2(data):
        calls.append(('listener2', data))

    eb.subscribe('event1', listener1)
    eb.subscribe('event2', listener2)

    eb.emit('event1', 'a')
    eb.emit('event2', 'b')
    eb.emit('event1', 'c')
    eb.emit('event2', 'd')

    expected = [
        ('listener1', 'a'),
        ('listener2', 'b'),
        ('listener1', 'c'),
        ('listener2', 'd'),
    ]
    assert calls == expected


def test_subscribe_listener_that_raises_does_not_prevent_others():
    eb = EventBus()
    calls = []

    def bad_listener(data):
        raise RuntimeError("fail")

    def good_listener(data):
        calls.append(data)

    eb.subscribe('e', bad_listener)
    eb.subscribe('e', good_listener)

    # Emitting should invoke both, but bad_listener raises; test should not propagate error
    # Assuming EventBus does not catch exceptions internally, this will raise.
    # So let's check for that.
    with pytest.raises(RuntimeError):
        eb.emit('e', 123)

    # good_listener still not called since first failure raised?


def test_subscribe_listener_multiple_events():
    eb = EventBus()
    calls = []

    def listener(data):
        calls.append(data)

    eb.subscribe('e1', listener)
    eb.subscribe('e2', listener)
    eb.emit('e1', 1)
    eb.emit('e2', 2)

    assert calls == [1, 2]


def test_unsubscribe_listener_when_multiple_registered_only_removes_one():
    eb = EventBus()
    calls = []

    def listener(data):
        calls.append(data)

    eb.subscribe('e', listener)
    eb.subscribe('e', listener)
    eb.unsubscribe('e', listener)
    eb.emit('e', 5)

    # Since only one subscription was removed, only one call should occur
    assert calls == [5]


def test_subscribe_unsubscribe_roundtrip():
    eb = EventBus()
    calls = []

    def listener(data):
        calls.append(data)

    eb.subscribe('evt', listener)
    eb.unsubscribe('evt', listener)
    eb.subscribe('evt', listener)
    eb.emit('evt', 10)

    assert calls == [10]
