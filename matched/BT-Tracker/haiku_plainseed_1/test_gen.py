import pytest

from Tracker import event as event_module
from Tracker.event import Event, status


@pytest.fixture(autouse=True)
def restore_finished_property(monkeypatch):
    # Event.check assigns to the class attribute ``finished``, which replaces the
    # property on the class. Record the original property so it is restored
    # after every test and tests stay independent of each other.
    monkeypatch.setattr(Event, "finished", vars(Event)["finished"])


@pytest.fixture
def sleeps(monkeypatch):
    # Record time.sleep calls instead of actually sleeping.
    calls = []
    monkeypatch.setattr(event_module.time, "sleep", lambda seconds: calls.append(seconds))
    return calls


# --- Event initial state and finished property -----------------------------

def test_new_event_is_not_finished():
    assert Event().finished is False


def test_new_event_stores_false_in_private_flag():
    assert Event()._finish is False


def test_setting_finished_true_marks_event_finished():
    e = Event()
    e.finished = True
    assert e.finished is True


def test_setting_finished_true_updates_private_flag():
    e = Event()
    e.finished = True
    assert e._finish is True


def test_setting_finished_false_is_ignored_after_true():
    e = Event()
    e.finished = True
    e.finished = False
    assert e.finished is True


def test_setting_finished_false_on_new_event_keeps_it_unfinished():
    e = Event()
    e.finished = False
    assert e.finished is False


def test_setting_truthy_non_bool_does_not_mark_finished():
    e = Event()
    e.finished = 1
    assert e.finished is False


def test_setting_none_does_not_mark_finished():
    e = Event()
    e.finished = None
    assert e.finished is False


def test_setting_string_does_not_mark_finished():
    e = Event()
    e.finished = "yes"
    assert e.finished is False


def test_setting_finished_true_twice_stays_true():
    e = Event()
    e.finished = True
    e.finished = True
    assert e.finished is True


def test_instances_have_independent_finished_state():
    first = Event()
    second = Event()
    first.finished = True
    assert first.finished is True
    assert second.finished is False


# --- Module-level singleton ------------------------------------------------

def test_module_status_is_event_instance():
    assert isinstance(status, Event)


def test_reimporting_status_returns_same_singleton():
    from Tracker.event import status as again
    assert again is status


# --- Event.check decorator -------------------------------------------------

def test_check_returns_wrapped_function_result(sleeps):
    @Event.check
    def double(x):
        return x * 2

    assert double(3) == 6


def test_check_passes_positional_and_keyword_arguments(sleeps):
    @Event.check
    def combine(a, b, sep="-"):
        return f"{a}{sep}{b}"

    assert combine("x", "y") == "x-y"
    assert combine("x", "y", sep="+") == "x+y"


def test_check_returns_none_when_wrapped_function_returns_none(sleeps):
    @Event.check
    def nothing():
        return None

    assert nothing() is None


def test_check_preserves_function_name(sleeps):
    @Event.check
    def my_task():
        return 1

    assert my_task.__name__ == "my_task"


def test_check_preserves_function_docstring(sleeps):
    @Event.check
    def documented():
        """Some docs."""
        return 1

    assert documented.__doc__ == "Some docs."


def test_check_wrapper_is_callable(sleeps):
    @Event.check
    def f():
        return 1

    assert callable(f)


def test_check_calls_wrapped_function_exactly_once_per_call(sleeps):
    counter = {"n": 0}

    @Event.check
    def tick():
        counter["n"] += 1
        return counter["n"]

    assert tick() == 1
    assert tick() == 2
    assert counter["n"] == 2


def test_check_sleeps_once_for_0_1_seconds_after_call(sleeps):
    @Event.check
    def f():
        return "ok"

    f()
    assert sleeps == [0.1]


def test_check_does_not_sleep_when_function_raises(sleeps):
    @Event.check
    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        boom()
    assert sleeps == []


def test_check_propagates_exceptions_from_wrapped_function(sleeps):
    @Event.check
    def boom():
        raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        boom()


def test_check_reports_finished_true_during_sleep(monkeypatch):
    probe = Event()
    observed = []

    def fake_sleep(seconds):
        observed.append(probe.finished)

    monkeypatch.setattr(event_module.time, "sleep", fake_sleep)

    @Event.check
    def job():
        return "done"

    assert job() == "done"
    assert observed == [True]


def test_check_resets_finished_to_false_after_call(sleeps):
    probe = Event()

    @Event.check
    def job():
        return "done"

    job()
    assert probe.finished is False


def test_check_decorator_can_wrap_multiple_functions(sleeps):
    @Event.check
    def add(a, b):
        return a + b

    @Event.check
    def mul(a, b):
        return a * b

    assert add(2, 3) == 5
    assert mul(2, 3) == 6
    assert sleeps == [0.1, 0.1]
