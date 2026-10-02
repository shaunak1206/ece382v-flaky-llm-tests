import time
from itertools import count

import pytest

from more_itertools import time_limited


def test_time_limited_negative_raises():
    with pytest.raises(ValueError):
        time_limited(-1, [1, 2, 3])


def test_time_limited_zero_yields_nothing():
    it = time_limited(0, [1, 2, 3])
    assert list(it) == []
    assert it.timed_out is True


def test_time_limited_finishes_before_limit():
    it = time_limited(5, [1, 2, 3])
    assert list(it) == [1, 2, 3]
    assert it.timed_out is False


def test_time_limited_times_out_on_slow_generator():
    def gen():
        yield 1
        yield 2
        time.sleep(0.3)
        yield 3

    it = time_limited(0.1, gen())
    assert list(it) == [1, 2]
    assert it.timed_out is True


def test_time_limited_infinite_iterator_stops():
    it = time_limited(0.05, count())
    start = time.monotonic()
    result = list(it)
    elapsed = time.monotonic() - start
    assert it.timed_out is True
    assert len(result) > 0
    assert elapsed < 5


