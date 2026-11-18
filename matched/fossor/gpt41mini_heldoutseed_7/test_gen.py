import os
import tempfile
import pytest
from fossor.checks.similar_log_errors import SimilarLogErrors
from collections import OrderedDict


class DummyLog:
    def __init__(self):
        self.debug_calls = []

    def debug(self, msg):
        self.debug_calls.append(msg)


@pytest.fixture
def checker():
    c = SimilarLogErrors()
    c.log = DummyLog()
    return c


def test_remove_blacklisted_logs_removes_blacklisted(checker):
    input_logs = [
        "app_configuration.log",
        "server.out",
        "normal.log",
        "my_gc.log",
        "other.log",
        "some_oor.log",
        "jetty_thread_pool_stats.log",
        "engine_access_log.log",
        "valid_log.log"
    ]
    expected = ["normal.log", "other.log", "valid_log.log"]

    filtered = checker._remove_blacklisted_logs(input_logs)
    assert sorted(filtered) == sorted(expected)


def test_remove_blacklisted_logs_empty_list(checker):
    filtered = checker._remove_blacklisted_logs([])
    assert filtered == []


def test_process_line_returns_none_for_blank_line(checker):
    line, date = checker._process_line("    at some.stack.trace")
    assert line is None and date is None


def test_process_line_returns_none_for_line_starting_alpha_without_exception(checker):
    line, date = checker._process_line("Info: This is a log line")
    assert line is None and date is None


def test_process_line_skips_caused_by(checker):
    line, date = checker._process_line("Caused by some exception")
    assert line is None and date is None


def test_process_line_returns_line_and_date_for_valid_error_line(checker):
    log_line = "2024-01-01 12:34:56 ERROR Something bad happened"
    line, date = checker._process_line(log_line)
    assert line is not None and date is not None
    assert line.startswith("ERROR Something bad happened")
    assert date == "2024-01-01 12:34:56"


def test_process_line_truncates_long_lines(checker):
    long_msg = "A" * 500
    log_line = f"2024-01-01 12:00:00 ERROR {long_msg}"
    line, date = checker._process_line(log_line)
    assert len(line) <= 200 + len("ERROR ")  # truncated


def test_process_line_returns_none_if_index_error_logged(checker):
    # simulate short line with less than 3 split parts to cause IndexError
    short_line = "2024-01-01 something"
    line, date = checker._process_line(short_line)
    assert line is None and date is None
    # Confirm debug message includes 'IndexError'
    found = any("IndexError" in msg for msg in checker.log.debug_calls)
    assert found


def test_group_similar_log_lines_creates_new_entry(checker):
    common_lines = OrderedDict()
    new_line = "ERROR something failed"
    updated = checker._group_similar_log_lines(new_line, "2024-01-01 12:00:00", common_lines)
    assert new_line in updated
    assert updated[new_line]['count'] == 1
    assert updated[new_line]['first_seen'] == "2024-01-01 12:00:00"


def test_group_similar_log_lines_increments_similar_line(checker):
    common_lines = OrderedDict()
    base_line = "ERROR something failed"
    common_lines[base_line] = {'count': 2, 'first_seen': '2023-01-01 00:00:00'}
    similar_line = "ERROR something failed!"
    updated = checker._group_similar_log_lines(similar_line, "2024-01-01 12:00:00", common_lines)
    # Same key should be used because similarity ratio is high
    assert base_line in updated
    assert updated[base_line]['count'] == 3
    # first_seen should remain the original
    assert updated[base_line]['first_seen'] == '2023-01-01 00:00:00'
    # No new keys added
    assert len(updated) == 1


def test_group_similar_log_lines_switches_to_other_error_lines(checker):
    common_lines = OrderedDict()
    # fill with MAX_COMMON_LINES keys
    for i in range(checker.MAX_COMMON_LINES):
        common_lines[f"ERROR line {i}"] = {'count': 1}

    # A line that is not similar to any existing
    new_line = "ERROR brand new error line"
    updated = checker._group_similar_log_lines(new_line, None, common_lines)
    # The line added should be the catch-all category
    assert 'Other Error Lines' in updated
    assert updated['Other Error Lines']['count'] == 1


def test_group_similar_log_lines_similar_ratio_below_threshold_adds_unique_line(checker):
    common_lines = OrderedDict()
    # Add one line that is dissimilar
    common_lines["ERROR first line"] = {'count': 1}
    # Add a dissimilar line that won't match similarity ratio
    dissimilar_line = "ERROR completely different error"
    updated = checker._group_similar_log_lines(dissimilar_line, "2024-01-01", common_lines)
    # dissimilar_line should be a new key
    assert dissimilar_line in updated
    assert updated[dissimilar_line]['count'] == 1


def test_get_error_pattern_counts_reads_file_and_group_lines(checker):
    content = """2024-01-01 12:00:00 ERROR First error happened
Info: Just an info line
2024-01-01 12:00:01 ERROR Second error happened
Caused by some exception
  at stacktrace
"""
    with tempfile.NamedTemporaryFile(mode='wt', delete=False) as tf:
        tf.write(content)
        tmp_file = tf.name

    try:
        filename, common_lines = checker.get_error_pattern_counts(tmp_file)
        # Only two lines should be processed and counted
        assert filename == tmp_file
        assert isinstance(common_lines, OrderedDict)
        found_first = any("First error happened" in k for k in common_lines.keys())
        found_second = any("Second error happened" in k for k in common_lines.keys())
        assert found_first and found_second
        all_counts = sum(v['count'] for v in common_lines.values())
        assert all_counts == 2
    finally:
        os.remove(tmp_file)


def test_run_returns_none_without_log_files(checker):
    result = checker.run({})
    assert result is None


def test_run_removes_blacklisted_and_processes_files(monkeypatch, checker):
    # Patch get_error_pattern_counts to avoid actual file IO and multiprocessing
    processed = []

    def fake_get_error_pattern_counts(filename):
        processed.append(filename)
        # Return fake results of common lines
        return filename, OrderedDict({
            "ERROR line one": {'count': 1, 'first_seen': "2024-01-01 00:00:01"},
            "ERROR line two": {'count': 2},
        })

    monkeypatch.setattr(checker, "get_error_pattern_counts", fake_get_error_pattern_counts)
    monkeypatch.setattr(checker, "_remove_blacklisted_logs", lambda logs: logs)

    logs = ["app.log", "app_configuration.log"]  # blacklist would remove second but we bypassed

    result = checker.run({'LogFiles': logs})
    # Should include both log filenames
    for log_file in logs:
        assert f"filename: {log_file}" in result
        assert "Count: 1" in result
        assert "Count: 2" in result


def test_run_returns_none_when_no_common_lines(monkeypatch, checker):
    monkeypatch.setattr(checker, "_remove_blacklisted_logs", lambda x: x)
    monkeypatch.setattr(checker, "get_error_pattern_counts", lambda filename: (filename, OrderedDict()))
    logs = ["app.log"]
    result = checker.run({'LogFiles': logs})
    assert result is None


def test_run_uses_pool_map(monkeypatch, checker):
    called = []

    class DummyPool:
        def __init__(self, threads):
            self.threads = threads

        def map(self, func, iterable):
            called.append((func, list(iterable)))
            # Return fake result
            return [(filename, OrderedDict()) for filename in iterable]

        def close(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(checker, "_remove_blacklisted_logs", lambda x: x)
    monkeypatch.setattr('fossor.checks.similar_log_errors.Pool', DummyPool)

    logs = ["a.log", "b.log"]
    result = checker.run({'LogFiles': logs})
    # Our dummy pool should have been called with the get_error_pattern_counts func and logs
    assert called
    func_called, files_passed = called[0]
    assert callable(func_called)
    assert files_passed == logs
    # Since get_error_pattern_counts returns empty dicts, result is None
    assert result is None


def test_process_line_line_with_no_level_field_logs_index_error(checker):
    line = "2024-01-01 12:00"  # Only two parts, missing level field
    line_out, date = checker._process_line(line)
    assert line_out is None and date is None
    assert any("IndexError" in msg for msg in checker.log.debug_calls)


def test_process_line_line_with_non_error_level_returns_none(checker):
    line = "2024-01-01 12:00:00 INFO This is info, not error"
    line_out, date = checker._process_line(line)
    assert line_out is None and date is None


def test_group_similar_log_lines_increments_first_seen_only_once(checker):
    common_lines = OrderedDict()
    line = "ERROR foo happened"
    # First add line with first_seen=some date
    updated = checker._group_similar_log_lines(line, "2021-01-01", common_lines)
    assert updated[line]['count'] == 1
    assert updated[line]['first_seen'] == "2021-01-01"
    # Add again with new date, first_seen should remain
    updated = checker._group_similar_log_lines(line, "2022-02-02", common_lines)
    assert updated[line]['count'] == 2
    assert updated[line]['first_seen'] == "2021-01-01"

