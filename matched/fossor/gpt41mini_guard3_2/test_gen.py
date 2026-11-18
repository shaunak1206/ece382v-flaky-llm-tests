import io
import random
import pytest
import numpy as np

import fossor.checks.similar_log_errors as sle


class DummyLog:
    def __init__(self):
        self.debug_msgs = []
    def debug(self, msg):
        self.debug_msgs.append(msg)


@pytest.fixture
def similar_errors_instance():
    inst = sle.SimilarLogErrors()
    inst.log = DummyLog()
    return inst


def test_remove_blacklisted_logs_removes_blacklisted(similar_errors_instance):
    logs = [
        "app.log",
        "error.gclog",
        "user.out",
        "run_gc.log",
        "test_configuration.log",
        "good.log",
        "jetty_thread_pool_stats.log",
    ]
    filtered = similar_errors_instance._remove_blacklisted_logs(logs)
    # None of the blacklisted strings should be in any string in filtered
    for bl in similar_errors_instance.log_name_blacklist:
        assert all(bl not in x for x in filtered)
    # Non blacklisted files remain
    assert "app.log" in filtered
    assert "good.log" in filtered
    # Blacklisted files are removed
    assert all(not any(bl in x for bl in similar_errors_instance.log_name_blacklist) for x in filtered)


def test_process_line_skips_initial_whitespace(similar_errors_instance):
    random.seed(123)
    np.random.seed(123)

    line = "     some stack trace line"
    processed_line, date = similar_errors_instance._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_skips_alpha_starting_without_exception(similar_errors_instance):
    random.seed(456)
    np.random.seed(456)

    line = "Started service"
    processed_line, date = similar_errors_instance._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_skips_caused_by_line(similar_errors_instance):
    random.seed(789)
    np.random.seed(789)

    line = "Caused by java.lang.NullPointerException"
    processed_line, date = similar_errors_instance._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_returns_error_line_and_date(similar_errors_instance):
    random.seed(101112)
    np.random.seed(101112)

    line = "2024-06-01 12:34:56 ERROR This is an error message"
    processed_line, date = similar_errors_instance._process_line(line)
    assert date == "2024-06-01 12:34:56"
    # processed_line starts with ERROR
    assert processed_line.startswith("ERROR ")
    # processed line is truncated to max 200 chars (all here shorter)
    assert len(processed_line) <= 200
    # original error message included (at least partially)
    assert "This is an error message" in processed_line


def test_process_line_non_error_level_line_returns_none(similar_errors_instance):
    random.seed(131415)
    np.random.seed(131415)

    line = "2024-06-01 12:34:56 INFO Not an error"
    processed_line, date = similar_errors_instance._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_indexerror_handling(similar_errors_instance, monkeypatch):
    random.seed(161718)
    np.random.seed(161718)

    # Disable debug logs for clean test; patch log.debug to record messages
    logs = []
    def fake_debug(msg):
        logs.append(msg)
    monkeypatch.setattr(similar_errors_instance.log, "debug", fake_debug)

    # Line that causes IndexError: e.g. only two tokens; line_split[2] triggers IndexError
    bad_line = "2024-06-01 12:34"
    processed_line, date = similar_errors_instance._process_line(bad_line)
    assert processed_line is None
    assert date is None
    assert any("IndexError parsing line" in m for m in logs)


def test_group_similar_log_lines_counts_and_similarity(similar_errors_instance):
    random.seed(192021)
    np.random.seed(192021)

    common_lines = {}
    line1 = "ERROR Something bad happened"
    common_lines = similar_errors_instance._group_similar_log_lines(line1, "2024-06-01 12:00", common_lines)
    assert line1 in common_lines
    assert common_lines[line1]['count'] == 1
    assert common_lines[line1]['first_seen'] == "2024-06-01 12:00"

    # Add very similar line, should increment count of existing line
    similar_line = line1 + " extra detail"
    common_lines = similar_errors_instance._group_similar_log_lines(similar_line, "2024-06-01 12:01", common_lines)
    # The key should still be the original line (line1)
    assert line1 in common_lines
    assert common_lines[line1]['count'] == 2
    # first_seen should not change (ignore later date)
    assert common_lines[line1]['first_seen'] == "2024-06-01 12:00"

    # Add different line that is not similar (> SIMILARITY_RATIO)
    different_line = "ERROR Completely different error message"
    common_lines = similar_errors_instance._group_similar_log_lines(different_line, "2024-06-01 12:02", common_lines)
    assert different_line in common_lines
    assert common_lines[different_line]['count'] == 1
    assert common_lines[different_line]['first_seen'] == "2024-06-01 12:02"


def test_group_similar_log_lines_max_common_lines_buckets(similar_errors_instance):
    random.seed(222324)
    np.random.seed(222324)

    common_lines = {}
    # Add MAX_COMMON_LINES unique lines
    for i in range(similar_errors_instance.MAX_COMMON_LINES):
        line = f"ERROR unique error line {i}"
        common_lines = similar_errors_instance._group_similar_log_lines(line, f"2024-06-01 12:{i:02d}", common_lines)
    assert len(common_lines) == similar_errors_instance.MAX_COMMON_LINES

    # Now add line not similar to any existing lines
    new_line = "ERROR another unique error line that is different"
    common_lines = similar_errors_instance._group_similar_log_lines(new_line, "2024-06-01 12:30", common_lines)
    # The new line key should be 'Other Error Lines'
    assert 'Other Error Lines' in common_lines
    assert common_lines['Other Error Lines']['count'] == 1
    # The total keys count might have increased by one or not if "Other Error Lines" existed before
    assert len(common_lines) >= similar_errors_instance.MAX_COMMON_LINES


def test_get_error_pattern_counts_reads_file_and_groups(monkeypatch, similar_errors_instance):
    random.seed(252627)
    np.random.seed(252627)

    # Prepare content with multiple lines, some to be ignored, some error lines
    file_content = (
        "2024-06-01 12:00:00 ERROR First error line\n"
        "2024-06-01 12:05:00 INFO Not an error\n"
        "Started service\n"
        " Caused by something\n"
        "2024-06-01 12:10:00 ERROR Second error line\n"
        "   stacktrace line\n"
    )

    filename = "fake_log.log"

    # Patch open to provide our content
    def fake_open(file, mode='rt', *args, **kwargs):
        if file == filename:
            return io.StringIO(file_content)
        return open(file, mode, *args, **kwargs)

    monkeypatch.setattr(sle, 'open', fake_open)

    # Patch log.debug to not print anything (or collect logs later if needed)
    monkeypatch.setattr(similar_errors_instance.log, 'debug', lambda x: None)

    fname, common_lines = similar_errors_instance.get_error_pattern_counts(filename)

    assert fname == filename
    # Expect common_lines to contain exactly the two error lines processed (or grouped)
    found_lines = list(common_lines.keys())
    # The truncated error lines start with "ERROR First error line" and "ERROR Second error line"
    found_error_lines = [l for l in found_lines if l.startswith("ERROR")]
    assert len(found_error_lines) == 2
    counts = [common_lines[l]['count'] for l in found_error_lines]
    assert all(isinstance(c, int) and c >= 1 for c in counts)


def test_run_returns_none_for_no_logs(similar_errors_instance):
    random.seed(282930)
    np.random.seed(282930)

    res = similar_errors_instance.run({})
    assert res is None

    res = similar_errors_instance.run({'LogFiles': []})
    assert res is None


def test_run_returns_results(monkeypatch, similar_errors_instance):
    random.seed(313233)
    np.random.seed(313233)

    # Provide list of filenames, patch internal functions to isolate test from filesystem and multiprocessing

    log_files = ["log1", "log2", "log3.out", "log4_gc.log"]  # last two blacklisted; should be filtered out

    # Patch _remove_blacklisted_logs to original implementation so filtering occurs
    # No monkeypatch needed because we want actual filtering

    # Patch get_error_pattern_counts to return controlled dummy data
    def dummy_get_error_pattern_counts(filename):
        return (filename, {"ERROR dummy line": {'count': 2, 'first_seen': '2024-06-01 12:00'}})

    monkeypatch.setattr(similar_errors_instance, "get_error_pattern_counts", dummy_get_error_pattern_counts)

    # Patch cpu_count to 1 so Pool size is small
    monkeypatch.setattr(sle, "cpu_count", lambda : 1)

    # Patch Pool to a dummy pool that maps function normally without real multiprocessing
    class DummyPool:
        def __init__(self, n): pass
        def map(self, func, iterable):
            # Use list comprehension directly
            return [func(x) for x in iterable]
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def close(self): pass
        def join(self): pass

    monkeypatch.setattr(sle, "Pool", DummyPool)

    # Run
    result = similar_errors_instance.run({'LogFiles': log_files})
    # The blacklisted logs (log3.out, log4_gc.log) filtered out; result contains only log1, log2
    assert result is not None
    # The result should include "filename: log1" and "filename: log2"
    assert "filename: log1" in result
    assert "filename: log2" in result
    assert "filename: log3.out" not in result
    assert "filename: log4_gc.log" not in result
    # Each error line string should contain count and first seen info
    assert "Count: 2" in result
    assert "First seen: 2024-06-01 12:00" in result


def test_run_returns_none_when_no_errors(monkeypatch, similar_errors_instance):
    random.seed(343536)
    np.random.seed(343536)

    log_files = ["log1"]

    def dummy_get_error_pattern_counts(filename):
        return (filename, {})

    monkeypatch.setattr(similar_errors_instance, "get_error_pattern_counts", dummy_get_error_pattern_counts)
    monkeypatch.setattr(sle, "cpu_count", lambda: 1)
    class DummyPool:
        def __init__(self, n): pass
        def map(self, func, iterable):
            return [func(x) for x in iterable]
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def close(self): pass
        def join(self): pass
    monkeypatch.setattr(sle, "Pool", DummyPool)

    result = similar_errors_instance.run({'LogFiles': log_files})
    assert result is None


def test_group_similar_log_lines_first_seen_only_sets_once(similar_errors_instance):
    random.seed(373839)
    np.random.seed(373839)

    common_lines = {}
    line = "ERROR unique error"
    # Add line with first_seen date1
    common_lines = similar_errors_instance._group_similar_log_lines(line, "2024-06-01 10:00", common_lines)
    # Add similar line with first_seen date2 later, should not overwrite first_seen
    common_lines = similar_errors_instance._group_similar_log_lines(line, "2024-06-02 12:00", common_lines)
    assert common_lines[line]['count'] == 2
    assert common_lines[line]['first_seen'] == "2024-06-01 10:00"


def test_group_similar_log_lines_no_date_preserves_count(similar_errors_instance):
    random.seed(404142)
    np.random.seed(404142)

    common_lines = {}
    line = "ERROR something"
    # Add line with date
    common_lines = similar_errors_instance._group_similar_log_lines(line, "2024-06-01 10:00", common_lines)
    # Add line without date (None)
    common_lines = similar_errors_instance._group_similar_log_lines(line, None, common_lines)
    assert common_lines[line]['count'] == 2
    # first_seen remains the original date
    assert common_lines[line]['first_seen'] == "2024-06-01 10:00"


def test_group_similar_log_lines_similar_line_updates_correctly(similar_errors_instance):
    random.seed(434445)
    np.random.seed(434445)

    common_lines = {}
    base_line = "ERROR Base error line"
    similar_line = "ERROR Base error linE"  # minor difference, should be similar > 0.7 ratio

    common_lines = similar_errors_instance._group_similar_log_lines(base_line, "2024-06-01 09:00", common_lines)
    common_lines = similar_errors_instance._group_similar_log_lines(similar_line, "2024-06-01 10:00", common_lines)

    # Because similarity > threshold, count should be 2 for base_line key only
    assert base_line in common_lines
    # The similar_line should not be a separate key
    assert similar_line not in common_lines.keys()
    assert common_lines[base_line]['count'] == 2
    # first_seen should be original date, not updated
    assert common_lines[base_line]['first_seen'] == "2024-06-01 09:00"
