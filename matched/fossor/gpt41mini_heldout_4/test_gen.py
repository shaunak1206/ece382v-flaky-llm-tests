import builtins
import io
import os
import tempfile
from collections import OrderedDict
import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


@pytest.fixture
def similar_log_errors():
    return SimilarLogErrors()


def test_remove_blacklisted_logs_removes_blacklisted(similar_log_errors):
    logs = [
        "error.log",
        "server_configuration.log",
        "app.out",
        "gc.log",
        "normal.log",
        "some_oor.log",
        "jetty_thread_pool_stats.log",
        "engine_access_log.log",
        "custom.log",
    ]
    filtered = similar_log_errors._remove_blacklisted_logs(logs)
    expected = ["error.log", "normal.log", "custom.log"]
    assert sorted(filtered) == sorted(expected)


def test_remove_blacklisted_logs_no_blacklisted(similar_log_errors):
    logs = ["file1.log", "file2.txt"]
    filtered = similar_log_errors._remove_blacklisted_logs(logs)
    assert filtered == logs


def test__process_line_skips_whitespace_start(similar_log_errors):
    line = "  continuation line"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test__process_line_skips_alpha_not_exception(similar_log_errors):
    line = "Info: something happened"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test__process_line_skips_caused_by(similar_log_errors):
    line = "Caused by java.lang.Exception"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test__process_line_non_error_level(similar_log_errors):
    line = "2023-06-01 12:00:00 INFO Application started"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test__process_line_handles_error_line(similar_log_errors):
    line = "2023-06-01 12:00:00 ERROR Something bad happened"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line.startswith("ERROR Something bad happened")
    assert date == "2023-06-01 12:00:00"


def test__process_line_truncates_long_line(similar_log_errors):
    long_text = "a" * 500
    line = f"2023-06-01 12:00:00 ERROR {long_text}"
    processed_line, _ = similar_log_errors._process_line(line)
    # The line should be truncated to ~200 characters after 'ERROR '
    # Because the slice occurs after join from 2: -> "ERROR " + "a"*500 truncated to 200 chars
    assert len(processed_line) <= 205  # 'ERROR ' plus about 200 chars


def test__process_line_handles_index_error(monkeypatch, similar_log_errors):
    # simulate a line with fewer than 3 tokens to cause IndexError
    line = "2023-06-01 12:00:00"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test__group_similar_log_lines_adds_new_line(similar_log_errors):
    common_lines = OrderedDict()
    line = "ERROR Something happened"
    date = "2023-06-01 12:00:00"
    common_lines = similar_log_errors._group_similar_log_lines(line, date, common_lines)
    assert line in common_lines
    assert common_lines[line]["count"] == 1
    assert common_lines[line]["first_seen"] == date


def test__group_similar_log_lines_increments_similar_line(similar_log_errors):
    existing_line = "ERROR Disk full"
    common_lines = OrderedDict({existing_line: {"count": 2, "first_seen": "2023-01-01 00:00:00"}})
    line = "ERROR Disk almost full"
    date = "2023-06-01 12:00:00"

    # Make similarity high by patching SequenceMatcher.quick_ratio
    original_quick_ratio = similar_log_errors._group_similar_log_lines.__globals__['SequenceMatcher'].quick_ratio

    def fake_quick_ratio(self):
        return 0.9  # Always above SIMILARITY_RATIO

    similar_log_errors._group_similar_log_lines.__globals__['SequenceMatcher'].quick_ratio = fake_quick_ratio

    updated = similar_log_errors._group_similar_log_lines(line, date, common_lines)
    # Reverted patched method to avoid side effects
    similar_log_errors._group_similar_log_lines.__globals__['SequenceMatcher'].quick_ratio = original_quick_ratio

    assert existing_line in updated
    assert updated[existing_line]["count"] == 3
    assert updated[existing_line]["first_seen"] == "2023-01-01 00:00:00"
    # The new line is not added since it's similar


def test__group_similar_log_lines_too_many_unique_lines(similar_log_errors):
    # Populate common_lines with MAX_COMMON_LINES unique lines
    common_lines = OrderedDict()
    for i in range(similar_log_errors.MAX_COMMON_LINES):
        key = f"ERROR line {i}"
        common_lines[key] = {"count": 1, "first_seen": f"2023-06-01 12:00:{i:02d}"}

    new_line = "ERROR this is a new unique line"
    date = "2023-06-01 12:15:00"

    updated = similar_log_errors._group_similar_log_lines(new_line, date, common_lines)
    # The new line should be counted under 'Other Error Lines'
    assert "Other Error Lines" in updated
    assert updated["Other Error Lines"]["count"] == 1
    assert updated["Other Error Lines"]["first_seen"] == date


def test_get_error_pattern_counts_reads_file(monkeypatch, similar_log_errors):
    # Create a fake file with some error and non-error lines
    content = (
        "  continuation line\n"
        "Info message\n"
        "Caused by java.lang.Exception\n"
        "2023-06-01 12:00:00 INFO Application started\n"
        "2023-06-01 12:00:00 ERROR Something failed\n"
        "2023-06-01 12:00:01 ERROR Something failed badly\n"
    )

    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write(content)
        tf_filename = tf.name

    try:
        filename, common_lines = similar_log_errors.get_error_pattern_counts(tf_filename)
        # Should process only the two ERROR lines
        assert filename == tf_filename
        # common_lines keys are processed lines starting with "ERROR"
        keys = list(common_lines.keys())
        assert any(key.startswith("ERROR Something failed") for key in keys)
        # Counts should add up to 2
        total_counts = sum(v['count'] for v in common_lines.values())
        assert total_counts == 2
    finally:
        os.unlink(tf_filename)


def test_run_returns_none_with_no_logfiles(similar_log_errors):
    # variables without 'LogFiles' key or empty LogFiles
    assert similar_log_errors.run({}) is None
    assert similar_log_errors.run({'LogFiles': []}) is None


def test_run_filters_blacklisted_and_returns(monkeypatch, similar_log_errors):
    logs = ["normal.log", "foo.out", "error.log"]

    # Patch _remove_blacklisted_logs to control output
    monkeypatch.setattr(similar_log_errors, "_remove_blacklisted_logs", lambda x: ["normal.log"])

    # Patch get_error_pattern_counts to return fake data
    monkeypatch.setattr(similar_log_errors, "get_error_pattern_counts", lambda filename: (filename, OrderedDict({
        "ERROR 1": {"count": 3, "first_seen": "2023-01-01 00:00:00"},
        "ERROR 2": {"count": 1}
    })))

    # Patch multiprocessing.Pool to avoid actual multiprocessing in tests
    class DummyPool:
        def __init__(self, x):
            pass

        def map(self, func, iterable):
            return [func(f) for f in iterable]

        def close(self):
            pass

        def join(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(similar_log_errors, "cpu_count", lambda: 1)
    monkeypatch.setattr(similar_log_errors, "log", type("Log", (), {"debug": lambda *a, **k: None}))

    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", DummyPool)

    result = similar_log_errors.run({"LogFiles": logs})
    # Result should include filename and counts
    assert "filename: normal.log" in result
    assert "Count: 3" in result or "Count: 1" in result


def test__group_similar_log_lines_returns_unique_with_low_similarity(similar_log_errors):
    # Provide a common_lines dict with a key that has low similarity with line
    common_line = "ERROR totally different error"
    common_lines = OrderedDict({common_line: {"count": 1}})
    line = "ERROR something unrelated"
    date = "2023-06-01 12:00:00"

    # Patch SequenceMatcher.quick_ratio to return low similarity
    original_quick_ratio = similar_log_errors._group_similar_log_lines.__globals__['SequenceMatcher'].quick_ratio

    def low_quick_ratio(self):
        return 0.4  # below SIMILARITY_RATIO (0.7)

    similar_log_errors._group_similar_log_lines.__globals__['SequenceMatcher'].quick_ratio = low_quick_ratio

    updated = similar_log_errors._group_similar_log_lines(line, date, common_lines)

    similar_log_errors._group_similar_log_lines.__globals__['SequenceMatcher'].quick_ratio = original_quick_ratio

    # The new line should be added separate as unique (not merged)
    assert line in updated
    assert updated[line]["count"] == 1
    # The old line count unchanged
    assert updated[common_line]["count"] == 1


def test__group_similar_log_lines_adds_first_seen_only_once(similar_log_errors):
    line = "ERROR unique error"
    date1 = "2023-06-01 12:00:00"
    date2 = "2023-06-01 13:00:00"
    common_lines = OrderedDict()

    updated = similar_log_errors._group_similar_log_lines(line, date1, common_lines)
    # Increase count, second call with a different date
    updated = similar_log_errors._group_similar_log_lines(line, date2, updated)

    # first_seen should be the first date only
    assert updated[line]["first_seen"] == date1
    assert updated[line]["count"] == 2


def test__process_line_empty_string(similar_log_errors):
    line = ""
    with pytest.raises(IndexError):
        # Accessing line[0] when line empty raises IndexError, but no except catches
        similar_log_errors._process_line(line)


def test_run_returns_none_when_no_results(monkeypatch, similar_log_errors):
    # Patch _remove_blacklisted_logs to return something
    monkeypatch.setattr(similar_log_errors, "_remove_blacklisted_logs", lambda x: ["file.log"])

    # Patch get_error_pattern_counts to return empty common_lines
    monkeypatch.setattr(similar_log_errors, "get_error_pattern_counts", lambda filename: (filename, OrderedDict()))

    # Patch Pool to dummy to avoid multiproc
    class DummyPool:
        def __init__(self, x):
            pass
        def map(self, func, iterable):
            return [func(f) for f in iterable]
    monkeypatch.setattr(similar_log_errors, "cpu_count", lambda: 1)
    monkeypatch.setattr(similar_log_errors, "log", type("Log", (), {"debug": lambda *a, **k: None}))
    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", DummyPool)

    result = similar_log_errors.run({"LogFiles": ["file.log"]})
    # Nothing to return because common_lines empty
    assert result is None
