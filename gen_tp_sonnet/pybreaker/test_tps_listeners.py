from datetime import timedelta

import pybreaker
import pytest
from freezegun import freeze_time


class Boom(Exception):
    pass


def boom():
    raise Boom("boom")


def ok():
    return "ok"


class Recorder(pybreaker.CircuitBreakerListener):
    def __init__(self):
        self.events = []
        self.breakers = []

    def before_call(self, cb, func, *args, **kwargs):
        self.breakers.append(cb)
        self.events.append(("before_call", func, args, kwargs))

    def failure(self, cb, exc):
        self.breakers.append(cb)
        self.events.append(("failure", exc))

    def success(self, cb):
        self.breakers.append(cb)
        self.events.append(("success",))

    def state_change(self, cb, old_state, new_state):
        self.breakers.append(cb)
        old_name = old_state.name if old_state is not None else None
        self.events.append(("state_change", old_name, new_state.name))

    def kinds(self):
        return [e[0] for e in self.events]

    def transitions(self):
        return [(e[1], e[2]) for e in self.events if e[0] == "state_change"]


def test_no_listeners_by_default():
    assert tuple(pybreaker.CircuitBreaker().listeners) == ()


def test_listeners_from_constructor_are_registered():
    rec = Recorder()
    cb = pybreaker.CircuitBreaker(listeners=[rec])
    assert cb.listeners == (rec,)


def test_add_listener_and_add_listeners():
    a, b, c = Recorder(), Recorder(), Recorder()
    cb = pybreaker.CircuitBreaker()
    cb.add_listener(a)
    cb.add_listeners(b, c)
    assert cb.listeners == (a, b, c)


def test_remove_listener_stops_notifications():
    rec = Recorder()
    cb = pybreaker.CircuitBreaker(listeners=[rec])
    cb.remove_listener(rec)
    assert rec not in cb.listeners
    cb.call(ok)
    assert rec.events == []


def test_successful_call_notifies_before_call_then_success():
    rec = Recorder()
    cb = pybreaker.CircuitBreaker(listeners=[rec])
    cb.call(lambda a, b=0: a + b, 1, b=2)
    assert rec.kinds() == ["before_call", "success"]
    _, func, args, kwargs = rec.events[0]
    assert args == (1,)
    assert kwargs == {"b": 2}
    assert callable(func)


def test_failed_call_notifies_failure_with_exception():
    rec = Recorder()
    cb = pybreaker.CircuitBreaker(fail_max=5, listeners=[rec])
    err = Boom("specific")

    def raiser():
        raise err

    with pytest.raises(Boom):
        cb.call(raiser)
    assert rec.kinds() == ["before_call", "failure"]
    assert rec.events[1][1] is err


def test_listener_receives_the_circuit_breaker():
    rec = Recorder()
    cb = pybreaker.CircuitBreaker(listeners=[rec])
    cb.call(ok)
    assert rec.breakers
    assert all(b is cb for b in rec.breakers)


def test_state_change_notified_when_circuit_trips():
    rec = Recorder()
    cb = pybreaker.CircuitBreaker(fail_max=2, listeners=[rec])
    with pytest.raises(Boom):
        cb.call(boom)
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(boom)
    assert ("closed", "open") in rec.transitions()


def test_state_change_notified_on_manual_open_and_close():
    rec = Recorder()
    cb = pybreaker.CircuitBreaker(listeners=[rec])
    cb.open()
    assert ("closed", "open") in rec.transitions()
    cb.close()
    assert rec.transitions()[-1] == ("open", "closed")


def test_state_changes_through_full_recovery_cycle():
    with freeze_time("2024-01-01 12:00:00") as frozen:
        rec = Recorder()
        cb = pybreaker.CircuitBreaker(fail_max=1, reset_timeout=10, listeners=[rec])
        with pytest.raises(pybreaker.CircuitBreakerError):
            cb.call(boom)
        frozen.tick(timedelta(seconds=11))
        cb.call(ok)
    transitions = rec.transitions()
    assert transitions[0] == ("closed", "open")
    assert ("open", "half-open") in transitions
    assert transitions[-1] == ("half-open", "closed")


def test_multiple_listeners_all_notified():
    first, second = Recorder(), Recorder()
    cb = pybreaker.CircuitBreaker(listeners=[first, second])
    cb.call(ok)
    assert first.kinds() == ["before_call", "success"]
    assert second.kinds() == ["before_call", "success"]


def test_base_listener_methods_are_noops():
    base = pybreaker.CircuitBreakerListener()
    cb = pybreaker.CircuitBreaker(fail_max=2, listeners=[base])
    assert cb.call(ok) == "ok"
    with pytest.raises(Boom):
        cb.call(boom)
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(boom)
    assert cb.current_state == "open"
