import logging
from datetime import datetime, timedelta

import pytest
from freezegun import freeze_time

import pybreaker


class DummyListener(pybreaker.CircuitBreakerListener):
    def __init__(self):
        self.events = []

    def before_call(self, cb, func, *args, **kwargs):
        self.events.append(("before", cb.current_state, func.__name__, args, kwargs))

    def failure(self, cb, exc):
        self.events.append(("failure", cb.current_state, type(exc).__name__, str(exc)))

    def success(self, cb):
        self.events.append(("success", cb.current_state))

    def state_change(self, cb, old_state, new_state):
        self.events.append(("state_change", old_state.name if old_state else None, new_state.name))


class FakeRedisError(Exception):
    pass


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.ops = []
        self._current = None

    def get(self, key):
        self._current = self.redis.get(key)
        return self._current

    def multi(self):
        self.ops.append(("multi",))

    def set(self, key, value):
        self.ops.append(("set", key, value))
        self.redis.set(key, value)


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.calls = []
        self.raise_on = set()

    def _maybe_raise(self, method):
        self.calls.append(method)
        if method in self.raise_on:
            raise FakeRedisError(method)

    def setnx(self, key, value):
        self._maybe_raise("setnx")
        self.data.setdefault(key, value)
        return 1

    def set(self, key, value):
        self._maybe_raise("set")
        self.data[key] = value
        return True

    def get(self, key):
        self._maybe_raise("get")
        return self.data.get(key)

    def incr(self, key):
        self._maybe_raise("incr")
        self.data[key] = int(self.data.get(key, 0)) + 1
        return self.data[key]

    def transaction(self, func, key):
        self._maybe_raise("transaction")
        pipe = FakePipeline(self)
        func(pipe)
        return True


def make_breaker(**kwargs):
    return pybreaker.CircuitBreaker(**kwargs)


def test_memory_storage_defaults():
    storage = pybreaker.CircuitMemoryStorage(pybreaker.STATE_CLOSED)
    assert storage.name == "memory"
    assert storage.state == pybreaker.STATE_CLOSED
    assert storage.counter == 0
    assert storage.success_counter == 0
    assert storage.opened_at is None


def test_memory_storage_increment_and_reset():
    storage = pybreaker.CircuitMemoryStorage(pybreaker.STATE_CLOSED)
    storage.increment_counter()
    storage.increment_counter()
    storage.increment_success_counter()
    assert storage.counter == 2
    assert storage.success_counter == 1
    storage.reset_counter()
    storage.reset_success_counter()
    assert storage.counter == 0
    assert storage.success_counter == 0


def test_memory_storage_opened_at_roundtrip():
    storage = pybreaker.CircuitMemoryStorage(pybreaker.STATE_CLOSED)
    now = datetime.utcnow()
    storage.opened_at = now
    assert storage.opened_at == now


def test_breaker_initial_state_and_properties():
    cb = make_breaker(fail_max=3, reset_timeout=7, success_threshold=2, name="svc")
    assert cb.fail_max == 3
    assert cb.reset_timeout == 7
    assert cb.success_threshold == 2
    assert cb.current_state == pybreaker.STATE_CLOSED
    assert cb.name == "svc"
    assert cb.fail_counter == 0
    assert cb.success_counter == 0


def test_breaker_setters_update_values():
    cb = make_breaker()
    cb.fail_max = 9
    cb.reset_timeout = 11
    cb.success_threshold = 4
    cb.name = "newname"
    assert cb.fail_max == 9
    assert cb.reset_timeout == 11
    assert cb.success_threshold == 4
    assert cb.name == "newname"


def test_excluded_exceptions_property_is_tuple():
    cb = make_breaker(exclude=[ValueError])
    excs = cb.excluded_exceptions
    assert isinstance(excs, tuple)
    assert excs == (ValueError,)


def test_add_and_remove_excluded_exception():
    cb = make_breaker()
    cb.add_excluded_exception(ValueError)
    assert ValueError in cb.excluded_exceptions
    cb.remove_excluded_exception(ValueError)
    assert ValueError not in cb.excluded_exceptions


def test_add_excluded_exceptions_variadic():
    cb = make_breaker()
    cb.add_excluded_exceptions(ValueError, KeyError)
    assert cb.excluded_exceptions == (ValueError, KeyError)


def test_is_system_error_with_exception_type_exclusion():
    cb = make_breaker(exclude=[ValueError])
    assert cb.is_system_error(RuntimeError("x")) is True
    assert cb.is_system_error(ValueError("x")) is False
    assert cb.is_system_error(UnicodeError("x")) is True


def test_is_system_error_with_callable_exclusion():
    cb = make_breaker(exclude=[lambda exc: "ignore" in str(exc)])
    assert cb.is_system_error(RuntimeError("ignore me")) is False
    assert cb.is_system_error(RuntimeError("boom")) is True


def test_is_system_error_with_mixed_exclusions():
    cb = make_breaker(exclude=[ValueError, lambda exc: isinstance(exc, RuntimeError)])
    assert cb.is_system_error(ValueError("x")) is False
    assert cb.is_system_error(RuntimeError("x")) is False
    assert cb.is_system_error(KeyError("x")) is True


def test_call_success_returns_value_and_resets_counter():
    cb = make_breaker()
    cb._state_storage.increment_counter()
    result = cb.call(lambda x, y: x + y, 2, 3)
    assert result == 5
    assert cb.fail_counter == 0


def test_call_failure_raises_original_exception_before_threshold():
    cb = make_breaker(fail_max=10)
    with pytest.raises(RuntimeError, match="boom"):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cb.fail_counter == 1


def test_call_failure_opens_and_raises_circuitbreakererror_at_threshold():
    cb = make_breaker(fail_max=1)
    with pytest.raises(pybreaker.CircuitBreakerError, match="Failures threshold reached"):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cb.current_state == pybreaker.STATE_OPEN
    assert cb.fail_counter == 1


def test_call_failure_with_throw_new_error_on_trip_false_raises_original():
    cb = make_breaker(fail_max=1, throw_new_error_on_trip=False)
    with pytest.raises(RuntimeError, match="boom"):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cb.current_state == pybreaker.STATE_OPEN


def test_open_sets_state_and_opened_at():
    cb = make_breaker()
    with freeze_time("2020-01-01 12:00:00"):
        should_throw = cb.open()
    assert should_throw is True
    assert cb.current_state == pybreaker.STATE_OPEN
    assert cb.state.name == pybreaker.STATE_OPEN
    assert cb._state_storage.opened_at == datetime(2020, 1, 1, 12, 0, 0, tzinfo=cb._state_storage.opened_at.tzinfo)


def test_open_returns_configuration_flag():
    cb = make_breaker(throw_new_error_on_trip=False)
    assert cb.open() is False


def test_half_open_sets_state():
    cb = make_breaker()
    cb.half_open()
    assert cb.current_state == pybreaker.STATE_HALF_OPEN
    assert cb.state.name == pybreaker.STATE_HALF_OPEN


def test_close_resets_success_counter_and_state():
    cb = make_breaker()
    cb._state_storage.increment_success_counter()
    cb.close()
    assert cb.current_state == pybreaker.STATE_CLOSED
    assert cb.success_counter == 0


def test_state_property_reloads_when_storage_changes():
    cb = make_breaker()
    cb._state_storage.state = pybreaker.STATE_OPEN
    assert cb.state.name == pybreaker.STATE_OPEN


def test_state_setter_notifies_listener():
    listener = DummyListener()
    cb = make_breaker(listeners=[listener])
    cb.state = pybreaker.STATE_OPEN
    assert ("state_change", pybreaker.STATE_CLOSED, pybreaker.STATE_OPEN) in listener.events


def test_state_setter_unknown_state_raises_value_error():
    cb = make_breaker()
    with pytest.raises(ValueError, match="Unknown state"):
        cb._create_new_state("invalid")


def test_listeners_property_returns_tuple_copy():
    listener = DummyListener()
    cb = make_breaker(listeners=[listener])
    lst = cb.listeners
    assert isinstance(lst, tuple)
    assert lst == (listener,)


def test_add_and_remove_listener():
    listener = DummyListener()
    cb = make_breaker()
    cb.add_listener(listener)
    assert cb.listeners == (listener,)
    cb.remove_listener(listener)
    assert cb.listeners == ()


def test_add_listeners_variadic():
    l1, l2 = DummyListener(), DummyListener()
    cb = make_breaker()
    cb.add_listeners(l1, l2)
    assert cb.listeners == (l1, l2)


def test_call_invokes_listener_hooks_in_order():
    listener = DummyListener()
    cb = make_breaker(listeners=[listener])
    assert cb.call(lambda: "ok") == "ok"
    assert listener.events[0][0] == "before"
    assert listener.events[-1][0] == "success"


def test_call_listener_failure_hook_on_exception():
    listener = DummyListener()
    cb = make_breaker(listeners=[listener], fail_max=10)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert any(e[0] == "failure" for e in listener.events)


def test_calling_context_manager_runs_block():
    cb = make_breaker()
    with cb.calling():
        value = 123
    assert value == 123


def test_calling_context_manager_with_failure_raises():
    cb = make_breaker(fail_max=10)
    with pytest.raises(RuntimeError):
        with cb.calling():
            raise RuntimeError("boom")


def test_call_wrapper_preserves_function_metadata():
    cb = make_breaker()

    @cb
    def sample():
        """docstring"""
        return 42

    assert sample.__name__ == "sample"
    assert sample.__doc__ == "docstring"
    assert sample() == 42


def test_call_wrapper_without_parentheses_works():
    cb = make_breaker()

    @cb()
    def sample(x):
        return x + 1

    assert sample(2) == 3


def test_call_wrapper_with_args_kwargs():
    cb = make_breaker()

    @cb
    def sample(x, y=4):
        return x + y

    assert sample(3, y=5) == 8


def test_state_change_notification_on_open():
    listener = DummyListener()
    cb = make_breaker(listeners=[listener])
    cb.open()
    assert any(e[0] == "state_change" and e[2] == pybreaker.STATE_OPEN for e in listener.events)


def test_state_change_notification_on_close():
    listener = DummyListener()
    cb = make_breaker(listeners=[listener])
    cb.open()
    cb.close()
    assert any(e[0] == "state_change" and e[2] == pybreaker.STATE_CLOSED for e in listener.events)


def test_close_from_half_open_resets_success_counter():
    cb = make_breaker()
    cb.half_open()
    cb._state_storage.increment_success_counter()
    cb.close()
    assert cb.success_counter == 0
    assert cb.current_state == pybreaker.STATE_CLOSED


def test_excluded_exception_prevents_counter_increment():
    cb = make_breaker(exclude=[ValueError])
    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError("business")))
    assert cb.fail_counter == 0


def test_excluded_exception_triggers_success_listener():
    listener = DummyListener()
    cb = make_breaker(exclude=[ValueError], listeners=[listener])
    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError("business")))
    assert any(e[0] == "success" for e in listener.events)


def test_half_open_success_closes_after_threshold():
    cb = make_breaker(success_threshold=2)
    cb.half_open()
    assert cb.call(lambda: "a") == "a"
    assert cb.current_state == pybreaker.STATE_HALF_OPEN
    assert cb.call(lambda: "b") == "b"
    assert cb.current_state == pybreaker.STATE_CLOSED


def test_half_open_failure_opens_circuit():
    cb = make_breaker(throw_new_error_on_trip=False)
    cb.half_open()
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cb.current_state == pybreaker.STATE_OPEN


def test_open_state_blocks_before_timeout():
    cb = make_breaker(reset_timeout=60)
    with freeze_time("2020-01-01 12:00:00"):
        cb.open()
    with freeze_time("2020-01-01 12:00:10"):
        with pytest.raises(pybreaker.CircuitBreakerError, match="Timeout not elapsed yet"):
            cb.call(lambda: "ok")


def test_open_state_allows_call_after_timeout():
    cb = make_breaker(reset_timeout=1)
    with freeze_time("2020-01-01 12:00:00"):
        cb.open()
    with freeze_time("2020-01-01 12:00:02"):
        assert cb.call(lambda: "ok") == "ok"
    assert cb.current_state == pybreaker.STATE_CLOSED


def test_open_state_transitions_to_half_open_after_timeout():
    cb = make_breaker(reset_timeout=1)
    with freeze_time("2020-01-01 12:00:00"):
        cb.open()
    with freeze_time("2020-01-01 12:00:02"):
        cb.call(lambda: "ok")
    assert cb.current_state == pybreaker.STATE_CLOSED


def test_breaker_with_initial_nonclosed_storage_state():
    storage = pybreaker.CircuitMemoryStorage(pybreaker.STATE_OPEN)
    cb = pybreaker.CircuitBreaker(state_storage=storage)
    assert cb.current_state == pybreaker.STATE_OPEN
    assert cb.state.name == pybreaker.STATE_OPEN


def test_fail_counter_property_tracks_storage():
    cb = make_breaker()
    cb._state_storage.increment_counter()
    assert cb.fail_counter == 1


def test_success_counter_property_tracks_storage():
    cb = make_breaker()
    cb._state_storage.increment_success_counter()
    assert cb.success_counter == 1


def test_open_after_multiple_failures_threshold():
    cb = make_breaker(fail_max=3)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cb.current_state == pybreaker.STATE_CLOSED
    with pytest.raises(pybreaker.CircuitBreakerError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cb.current_state == pybreaker.STATE_OPEN


def test_listener_state_change_receives_old_and_new_state_objects():
    listener = DummyListener()
    cb = make_breaker(listeners=[listener])
    cb.open()
    change = next(e for e in listener.events if e[0] == "state_change")
    assert change[1] == pybreaker.STATE_CLOSED
    assert change[2] == pybreaker.STATE_OPEN


def test_name_property_none_by_default():
    cb = make_breaker()
    assert cb.name is None


def test_memory_storage_state_setter():
    storage = pybreaker.CircuitMemoryStorage(pybreaker.STATE_CLOSED)
    storage.state = pybreaker.STATE_OPEN
    assert storage.state == pybreaker.STATE_OPEN


def test_create_new_state_returns_state_instance():
    cb = make_breaker()
    assert cb._create_new_state(pybreaker.STATE_CLOSED).name == pybreaker.STATE_CLOSED
    assert cb._create_new_state(pybreaker.STATE_OPEN).name == pybreaker.STATE_OPEN
    assert cb._create_new_state(pybreaker.STATE_HALF_OPEN).name == pybreaker.STATE_HALF_OPEN


def test_calling_context_manager_propagates_exception():
    cb = make_breaker()
    with pytest.raises(ValueError):
        with cb.calling():
            raise ValueError("bad")


def test_state_property_syncs_after_external_storage_change():
    cb = make_breaker()
    cb._state_storage.state = pybreaker.STATE_HALF_OPEN
    assert cb.state.name == pybreaker.STATE_HALF_OPEN


def test_half_open_success_counter_incremented():
    cb = make_breaker(success_threshold=3)
    cb.half_open()
    cb.call(lambda: "ok")
    assert cb.success_counter == 1
    assert cb.current_state == pybreaker.STATE_HALF_OPEN


def test_success_counter_resets_when_opening():
    cb = make_breaker()
    cb._state_storage.increment_success_counter()
    cb.open()
    assert cb.success_counter == 0


def test_removed_excluded_exception_affects_system_error():
    cb = make_breaker(exclude=[ValueError])
    cb.remove_excluded_exception(ValueError)
    assert cb.is_system_error(ValueError("x")) is True


def test_multiple_listeners_all_receive_success():
    l1, l2 = DummyListener(), DummyListener()
    cb = make_breaker(listeners=[l1, l2])
    cb.call(lambda: "ok")
    assert any(e[0] == "success" for e in l1.events)
    assert any(e[0] == "success" for e in l2.events)


def test_opened_at_property_on_memory_storage_none_until_set():
    cb = make_breaker()
    assert cb._state_storage.opened_at is None


def test_logging_module_is_available_for_redis_class():
    assert hasattr(pybreaker.CircuitRedisStorage, "logger")
    assert isinstance(pybreaker.CircuitRedisStorage.logger, logging.Logger)


def test_redis_storage_import_error_when_dependency_missing(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", False)
    with pytest.raises(ImportError, match="CircuitRedisStorage can only be used"):
        pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, object())


def test_redis_storage_namespace_format(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, redis, namespace="ns")
    assert storage._namespace("state") == "ns:pybreaker:state"
    assert storage._namespace("fail_counter") == "ns:pybreaker:fail_counter"


def test_redis_storage_initialization_sets_keys(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_OPEN, redis, namespace="ns")
    assert redis.data["ns:pybreaker:fail_counter"] == 0
    assert redis.data["ns:pybreaker:success_counter"] == 0
    assert redis.data["ns:pybreaker:state"] == pybreaker.STATE_OPEN


def test_redis_storage_state_reads_existing_value(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, redis)
    redis.data["pybreaker:state"] = pybreaker.STATE_OPEN
    assert storage.state == pybreaker.STATE_OPEN


def test_redis_storage_state_missing_reinitializes(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, redis)
    redis.data.pop("pybreaker:state", None)
    assert storage.state == pybreaker.STATE_CLOSED
    assert redis.data["pybreaker:state"] == pybreaker.STATE_CLOSED


def test_redis_storage_state_get_error_falls_back(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    redis.raise_on.add("get")
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_OPEN, redis)
    assert storage.state == pybreaker.STATE_OPEN


def test_redis_storage_counter_and_success_counter(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, redis)
    redis.data["pybreaker:fail_counter"] = b"3"
    redis.data["pybreaker:success_counter"] = b"2"
    assert storage.counter == 3
    assert storage.success_counter == 2


def test_redis_storage_opened_at_reads_datetime(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, redis)
    redis.data["pybreaker:opened_at"] = b"0"
    dt = storage.opened_at
    assert dt.year == 1970 and dt.month == 1 and dt.day == 1


def test_redis_storage_opened_at_setter_non_cluster(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, redis, cluster_mode=False)
    with freeze_time("2020-01-01 00:00:00"):
        storage.opened_at = datetime(2020, 1, 1)
    assert redis.data["pybreaker:opened_at"] == 1577836800


def test_redis_storage_opened_at_setter_cluster(monkeypatch):
    monkeypatch.setattr(pybreaker, "HAS_REDIS_SUPPORT", True)
    monkeypatch.setattr(pybreaker, "RedisError", FakeRedisError, raising=False)
    redis = FakeRedis()
    storage = pybreaker.CircuitRedisStorage(pybreaker.STATE_CLOSED, redis, cluster_mode=True)
    with freeze_time("2020-01-01 00:00:00"):
        storage.opened_at = datetime(2020, 1, 1)
    assert redis.data["pybreaker:opened_at"] == 1577836800
