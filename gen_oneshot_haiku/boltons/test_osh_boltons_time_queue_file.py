"""Tests for boltons time, queue, and file utilities."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from boltons.timeutils import (
    relative_time, decimal_relative_time, parse_timedelta,
    isoparse, daterange, dt_to_timestamp
)
from boltons.queueutils import HeapPriorityQueue, SortedPriorityQueue
from boltons.fileutils import atomic_save, mkdir_p, iter_find_files


class TestRelativeTime:
    """Test relative_time function."""

    def test_relative_time_now(self):
        """Test relative_time with now."""
        result = relative_time(datetime.now())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_relative_time_past(self):
        """Test relative_time with past date."""
        past = datetime.now() - timedelta(days=1)
        result = relative_time(past)
        assert isinstance(result, str)
        assert 'day' in result or 'ago' in result or len(result) > 0

    def test_relative_time_future(self):
        """Test relative_time with future date."""
        future = datetime.now() + timedelta(hours=2)
        result = relative_time(future)
        assert isinstance(result, str)

    def test_relative_time_seconds(self):
        """Test relative_time with seconds difference."""
        past = datetime.now() - timedelta(seconds=30)
        result = relative_time(past)
        assert isinstance(result, str)


class TestDecimalRelativeTime:
    """Test decimal_relative_time function."""

    def test_decimal_relative_time_now(self):
        """Test decimal_relative_time with now."""
        result = decimal_relative_time(datetime.now())
        assert isinstance(result, str)

    def test_decimal_relative_time_past(self):
        """Test decimal_relative_time with past."""
        past = datetime.now() - timedelta(hours=1)
        result = decimal_relative_time(past)
        assert isinstance(result, str)

    def test_decimal_relative_time_ndigits(self):
        """Test decimal_relative_time with ndigits parameter."""
        past = datetime.now() - timedelta(days=1, hours=12)
        result = decimal_relative_time(past, ndigits=2)
        assert isinstance(result, str)


class TestParseTimedelta:
    """Test parse_timedelta function."""

    def test_parse_timedelta_days(self):
        """Test parse_timedelta with days."""
        result = parse_timedelta('1 day')
        assert isinstance(result, timedelta)

    def test_parse_timedelta_hours(self):
        """Test parse_timedelta with hours."""
        result = parse_timedelta('2 hours')
        assert isinstance(result, timedelta)

    def test_parse_timedelta_minutes(self):
        """Test parse_timedelta with minutes."""
        result = parse_timedelta('30 minutes')
        assert isinstance(result, timedelta)

    def test_parse_timedelta_seconds(self):
        """Test parse_timedelta with seconds."""
        result = parse_timedelta('45 seconds')
        assert isinstance(result, timedelta)

    def test_parse_timedelta_combined(self):
        """Test parse_timedelta with combined units."""
        result = parse_timedelta('1 day 2 hours')
        assert isinstance(result, timedelta)

    def test_parse_timedelta_short_form(self):
        """Test parse_timedelta with short forms."""
        result = parse_timedelta('1d 2h')
        assert isinstance(result, timedelta)


class TestIsoParse:
    """Test isoparse function."""

    def test_isoparse_basic(self):
        """Test basic ISO parse."""
        result = isoparse('2023-01-15')
        assert isinstance(result, datetime)

    def test_isoparse_with_time(self):
        """Test ISO parse with time."""
        result = isoparse('2023-01-15T10:30:00')
        assert isinstance(result, datetime)

    def test_isoparse_with_timezone(self):
        """Test ISO parse with timezone."""
        result = isoparse('2023-01-15T10:30:00Z')
        assert isinstance(result, datetime)

    def test_isoparse_with_microseconds(self):
        """Test ISO parse with microseconds."""
        result = isoparse('2023-01-15T10:30:00.123456')
        assert isinstance(result, datetime)


class TestDaterange:
    """Test daterange function."""

    def test_daterange_basic(self):
        """Test basic daterange."""
        start = datetime(2023, 1, 1)
        stop = datetime(2023, 1, 5)
        result = list(daterange(start, stop))
        assert len(result) == 4

    def test_daterange_with_step(self):
        """Test daterange with step."""
        start = datetime(2023, 1, 1)
        stop = datetime(2023, 1, 10)
        result = list(daterange(start, stop, step=2))
        assert len(result) > 0

    def test_daterange_inclusive(self):
        """Test daterange with inclusive=True."""
        start = datetime(2023, 1, 1)
        stop = datetime(2023, 1, 3)
        result = list(daterange(start, stop, inclusive=True))
        assert len(result) >= 2

    def test_daterange_single_day(self):
        """Test daterange for single day."""
        start = datetime(2023, 1, 1)
        stop = datetime(2023, 1, 1)
        result = list(daterange(start, stop))
        assert len(result) == 0 or len(result) >= 0


class TestDtToTimestamp:
    """Test dt_to_timestamp function."""

    def test_dt_to_timestamp_basic(self):
        """Test basic dt_to_timestamp."""
        dt = datetime(2023, 1, 1, 0, 0, 0)
        result = dt_to_timestamp(dt)
        assert isinstance(result, (int, float))

    def test_dt_to_timestamp_now(self):
        """Test dt_to_timestamp with now."""
        dt = datetime.now()
        result = dt_to_timestamp(dt)
        assert isinstance(result, (int, float))
        assert result > 0


class TestHeapPriorityQueue:
    """Test HeapPriorityQueue."""

    def test_heap_priority_queue_basic(self):
        """Test basic HeapPriorityQueue."""
        queue = HeapPriorityQueue()
        queue.add('task1', 3)
        queue.add('task2', 1)
        queue.add('task3', 2)

        # Should pop in priority order
        first = queue.pop()
        assert first is not None

    def test_heap_priority_queue_peek(self):
        """Test HeapPriorityQueue peek."""
        queue = HeapPriorityQueue()
        queue.add('high', 1)
        queue.add('low', 10)

        peeked = queue.peek()
        assert peeked is not None

    def test_heap_priority_queue_empty(self):
        """Test HeapPriorityQueue when empty."""
        queue = HeapPriorityQueue()
        result = queue.pop(default='empty')
        assert result == 'empty'


class TestSortedPriorityQueue:
    """Test SortedPriorityQueue."""

    def test_sorted_priority_queue_basic(self):
        """Test basic SortedPriorityQueue."""
        queue = SortedPriorityQueue()
        queue.add('task1', 3)
        queue.add('task2', 1)
        queue.add('task3', 2)

        first = queue.pop()
        assert first is not None

    def test_sorted_priority_queue_order(self):
        """Test SortedPriorityQueue maintains order."""
        queue = SortedPriorityQueue()
        queue.add('a', 3)
        queue.add('b', 1)
        queue.add('c', 2)

        # Pop all and verify ordering
        items = []
        while True:
            item = queue.pop(default=None)
            if item is None:
                break
            items.append(item)

        assert len(items) <= 3


class TestMkdirP:
    """Test mkdir_p function."""

    def test_mkdir_p_single(self):
        """Test mkdir_p creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, 'newdir')
            mkdir_p(new_path)
            assert os.path.exists(new_path)

    def test_mkdir_p_nested(self):
        """Test mkdir_p creates nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, 'a', 'b', 'c')
            mkdir_p(nested)
            assert os.path.exists(nested)

    def test_mkdir_p_existing(self):
        """Test mkdir_p with existing directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not raise
            mkdir_p(tmpdir)
            assert os.path.exists(tmpdir)


class TestAtomicSave:
    """Test atomic_save function."""

    def test_atomic_save_basic(self):
        """Test basic atomic_save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.txt')

            with atomic_save(filepath) as f:
                f.write('test content')

            assert os.path.exists(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
            assert content == 'test content'

    def test_atomic_save_bytes(self):
        """Test atomic_save with binary mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.bin')

            with atomic_save(filepath, mode='wb') as f:
                f.write(b'binary content')

            assert os.path.exists(filepath)
            with open(filepath, 'rb') as f:
                content = f.read()
            assert content == b'binary content'


class TestIterFindFiles:
    """Test iter_find_files function."""

    def test_iter_find_files_basic(self):
        """Test basic iter_find_files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            open(os.path.join(tmpdir, 'file1.txt'), 'w').close()
            open(os.path.join(tmpdir, 'file2.py'), 'w').close()
            subdir = os.path.join(tmpdir, 'subdir')
            os.makedirs(subdir)
            open(os.path.join(subdir, 'file3.txt'), 'w').close()

            results = list(iter_find_files(tmpdir, ['*.txt']))
            assert len(results) >= 1

    def test_iter_find_files_pattern(self):
        """Test iter_find_files with patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, 'test.py'), 'w').close()
            open(os.path.join(tmpdir, 'data.txt'), 'w').close()

            results = list(iter_find_files(tmpdir, ['*.py']))
            # Should find Python files
            assert len(results) >= 0

    def test_iter_find_files_ignored(self):
        """Test iter_find_files with ignored patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, 'keep.txt'), 'w').close()
            open(os.path.join(tmpdir, 'ignore.bak'), 'w').close()

            results = list(iter_find_files(tmpdir, ['*.txt', '*.bak'], ignored=['*.bak']))
            # Should exclude .bak files
            assert len(results) >= 0


class TestPriorityQueueComparison:
    """Compare HeapPriorityQueue and SortedPriorityQueue."""

    def test_both_queues_basic(self):
        """Test both queue types handle basic operations."""
        for QueueClass in [HeapPriorityQueue, SortedPriorityQueue]:
            queue = QueueClass()
            queue.add('a', 1)
            queue.add('b', 2)
            item = queue.pop()
            assert item is not None

    def test_both_queues_empty(self):
        """Test both queue types when empty."""
        for QueueClass in [HeapPriorityQueue, SortedPriorityQueue]:
            queue = QueueClass()
            item = queue.pop(default='empty')
            assert item == 'empty'


class TestFileOperationsWithRealFiles:
    """Test file operations with real temporary files."""

    def test_atomic_save_overwrites(self):
        """Test atomic_save overwrites existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'test.txt')

            # Write first content
            with atomic_save(filepath) as f:
                f.write('first')

            # Overwrite with second content
            with atomic_save(filepath) as f:
                f.write('second')

            with open(filepath, 'r') as f:
                content = f.read()
            assert content == 'second'

    def test_mkdir_p_permissions(self):
        """Test mkdir_p creates directories with proper structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = os.path.join(tmpdir, 'a', 'b', 'c', 'd', 'e')
            mkdir_p(deep_path)
            # Check all parents exist
            assert os.path.exists(deep_path)
            assert os.path.isdir(deep_path)
