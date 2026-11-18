import time
import pytest
from timemarker import TimeMarker

def test_timemarker_initial_state():
    tm = TimeMarker()
    assert isinstance(tm, TimeMarker)

def test_timemarker_start_and_stop():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.stop()
    elapsed = tm.elapsed
    assert elapsed > 0

def test_elapsed_property_without_start_stop():
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        _ = tm.elapsed

def test_start_twice_raises():
    tm = TimeMarker()
    tm.start()
    with pytest.raises(RuntimeError):
        tm.start()

def test_stop_without_start_raises():
    tm = TimeMarker()
    with pytest.raises(RuntimeError):
        tm.stop()

def test_reset_resets_elapsed():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.stop()
    assert tm.elapsed > 0
    tm.reset()
    with pytest.raises(RuntimeError):
        _ = tm.elapsed

def test_context_manager_usage():
    with TimeMarker() as tm:
        time.sleep(0.01)
    assert tm.elapsed > 0

def test_context_manager_explicit_stop():
    tm = TimeMarker()
    with tm:
        time.sleep(0.01)
    elapsed1 = tm.elapsed
    # Subsequent calls to stop should raise since it already stopped
    with pytest.raises(RuntimeError):
        tm.stop()
    assert elapsed1 > 0

def test_multiple_start_stop_cycles():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.005)
    tm.stop()
    e1 = tm.elapsed
    tm.reset()
    tm.start()
    time.sleep(0.01)
    tm.stop()
    e2 = tm.elapsed
    assert e2 > e1

def test_repr_contains_elapsed_and_states():
    tm = TimeMarker()
    repr_before = repr(tm)
    tm.start()
    time.sleep(0.01)
    tm.stop()
    repr_after = repr(tm)
    assert "elapsed" in repr_after
    assert repr_before != repr_after

def test_elapsed_access_after_stop():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.stop()
    e1 = tm.elapsed
    e2 = tm.elapsed
    assert e1 == e2

def test_elapsed_access_during_running_timer():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    # Accessing elapsed while timer is running should raise
    with pytest.raises(RuntimeError):
        _ = tm.elapsed
    tm.stop()

def test_reset_while_running():
    tm = TimeMarker()
    tm.start()
    with pytest.raises(RuntimeError):
        tm.reset()
    tm.stop()

def test_context_manager_raises_propagation():
    class CustomError(Exception):
        pass
    try:
        with TimeMarker() as tm:
            raise CustomError()
    except CustomError:
        pass
    else:
        pytest.fail("CustomError was not propagated")

def test_stop_twice_raises():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.stop()
    with pytest.raises(RuntimeError):
        tm.stop()

def test_start_after_stop_raises():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.01)
    tm.stop()
    with pytest.raises(RuntimeError):
        tm.start()

def test_elapsed_is_float():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.02)
    tm.stop()
    assert isinstance(tm.elapsed, float)

def test_reset_after_stop_allows_restart():
    tm = TimeMarker()
    tm.start()
    time.sleep(0.005)
    tm.stop()
    tm.reset()
    tm.start()
    time.sleep(0.005)
    tm.stop()
    assert tm.elapsed > 0
