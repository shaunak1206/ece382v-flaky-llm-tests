import builtins
import io
import os
import tempfile
from collections import OrderedDict

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def __init__(self):
        self.debug_msgs = []

    def debug(self, msg):
        self.debug_msgs.append(msg)


@pytest.fixture
def check():
    c = SimilarLogErrors()
    c.log = DummyLog()
    return c


def test_remove_blacklisted_logs_removes_blacklisted(check):
    logs = [
        "app_configuration.log",
        "service.out",
        "gc.log",
        "firewall.log",
        "access.log",
        "my_oor.log",
        "normal.log",
    ]
    filtered = check._remove_blacklisted_logs(logs)
    # Blacklisted should be removed, only 'firewall.log', 'access.log', 'normal.log' remain
    assert "gc.log" not in filtered
    assert "service.out" not in filtered
    assert "app_configuration.log" not in filtered
    assert "my_oor.log" not in filtered
    assert all(x in filtered for x in ("firewall.log", "access.log", "normal.log"))


def test_process_line_returns_none_for_leading_whitespace(check):
    line = "  at some.stacktrace.line"
    result, date = check._process_line(line)
    assert result is None


def test_process_line_returns_none_for_alpha_line_without_exception(check):
    line = "Random informational message here"
    result, date = check._process_line(line)
    assert result is None


def test_process_line_returns_none_for_caused_by_line(check):
    line = "Caused by: java.lang.Exception"
    result, date = check._process_line(line)
    assert result is None


def test_process_line_filters_non_error_level(check):
    # non-ERROR level but with digit start
    line = "2024-01-01 12:01:01 WARNING This is a warning line"
    result, date = check._process_line(line)
    assert result is None


def test_process_line_cuts_off_date_and_truncates(check):
    line = "2024-01-01 12:01:02 ERROR This is an error message with details"
    result, date = check._process_line(line)
    expected_start = "ERROR This is an error message"
    assert result.startswith("ERROR")
    assert result.endswith("details") or len(result) <= 200
    assert date == "2024-01-01 12:01:02"


def test_process_line_handles_short_line_and_index_error(check):
    # Line too short to have line_level at index 2
    line = "2024-01-01 12"
    result, date = check._process_line(line)
    # Should trigger debug log and return None line
    assert result is None
    # Check debug message includes IndexError
    assert any("IndexError" in m for m in check.log.debug_msgs)


def test_group_similar_log_lines_increments_existing_similar_line(check):
    common_lines = OrderedDict()
    common_lines["ERROR something bad happened"] = {'count': 1}

    # Provide a line very similar to the existing one
    line = "ERROR something bad happene"  # missing last 'd'
    date = "2024-01-01 12:01:01"

    result = check._group_similar_log_lines(line, date, common_lines)
    # The key should be 'ERROR something bad happened' (original) with count incremented
    assert "ERROR something bad happened" in result
    assert result["ERROR something bad happened"]['count'] == 2
    assert result["ERROR something bad happened"]['first_seen'] == date


def test_group_similar_log_lines_adds_new_unique_line(check):
    common_lines = OrderedDict()
    line = "ERROR a unique error message"
    date = "2024-01-01 12:01:01"

    result = check._group_similar_log_lines(line, date, common_lines)
    # It should add the new line directly
    assert line in result
    assert result[line]['count'] == 1
    assert result[line]['first_seen'] == date


def test_group_similar_log_lines_threshold_behavior(check):
    # Set similarity ratio to a fixed value so we can test borderline case
    check.SIMILARITY_RATIO = 0.7
    common_lines = OrderedDict()
    base_line = "ERROR base line message"
    common_lines[base_line] = {'count': 1}

    # Create a line similar but below threshold
    line = "ERROR very different message"
    date = "2024-01-01 12:01:01"

    # The similarity ratio between base_line and line should be below 0.7
    result = check._group_similar_log_lines(line, date, common_lines)
    # Since dissimilar and not too many lines, it should add a new unique line
    assert line in result
    assert result[line]['count'] == 1


def test_group_similar_log_lines_max_common_lines_triggers_other_category(check):
    check.MAX_COMMON_LINES = 2
    common_lines = OrderedDict()
    common_lines["ERROR A"] = {'count': 1}
    common_lines["ERROR B"] = {'count': 1}

    # Now add a completely unique line - should be grouped under 'Other Error Lines'
    line = "ERROR C completely unique"
    date = "2024-01-01 12:01:01"
    result = check._group_similar_log_lines(line, date, common_lines)

    assert 'Other Error Lines' in result
    assert result['Other Error Lines']['count'] == 1
    # The new line should NOT be directly in result
    assert line not in result


def test_get_error_pattern_counts_reads_file_and_groups_lines(check):
    # Create a temp file and write log lines in expected format
    with tempfile.NamedTemporaryFile('w+t', delete=False) as tmp:
        tmp.write("2024-01-01 12:01:01 ERROR First error message\n")
        tmp.write("2024-01-01 12:01:02 ERROR First error message\n")
        tmp.write("2024-01-01 12:01:03 INFO This is info\n")
        tmp.write("  at stacktrace continuation\n")
        tmp.write("Caused by: an exception\n")
        tmp.flush()
        file_name = tmp.name

    try:
        filename, common_lines = check.get_error_pattern_counts(file_name)
        # Expect only one line counted, with count 2
        expected_key = "ERROR First error message"
        assert filename == file_name
        assert expected_key in common_lines
        assert common_lines[expected_key]['count'] == 2

        # Lines that should be skipped (INFO, stacktrace, caused by) are not counted
        assert len(common_lines) == 1
    finally:
        os.remove(file_name)


def test_run_returns_none_if_no_LogFiles_variable(check):
    variables = {}
    output = check.run(variables)
    assert output is None


def test_run_removes_blacklisted_and_processes_files(monkeypatch, check):
    # Prepare variables with some log files, one blacklisted
    variables = {
        'LogFiles': ['good.log', 'gc.log']
    }

    # Blacklisted by name 'gc.log' should be removed
    # Monkeypatch _remove_blacklisted_logs to confirm only one file remains
    def fake_remove_blacklisted(log_files):
        return [f for f in log_files if f != 'gc.log']

    check._remove_blacklisted_logs = fake_remove_blacklisted

    # Monkeypatch get_error_pattern_counts to return deterministic results
    def fake_get_error_pattern_counts(filename):
        if filename == 'good.log':
            return filename, OrderedDict({
                "ERROR something bad": {'count': 5, 'first_seen': '2024-01-01 00:00:01'}
            })
        else:
            return filename, OrderedDict()

    check.get_error_pattern_counts = fake_get_error_pattern_counts

    # Monkeypatch Pool to avoid actual multiprocessing
    class DummyPool:
        def __init__(self, _):
            pass

        def map(self, func, iterable):
            return [check.get_error_pattern_counts(f) for f in iterable]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", DummyPool)
    monkeypatch.setattr("fossor.checks.similar_log_errors.cpu_count", lambda: 1)

    output = check.run(variables)
    assert output is not None
    assert "good.log" in output
    assert "Count: 5" in output
    assert "First seen: 2024-01-01 00:00:01" in output
    assert "ERROR something bad" in output
    assert "gc.log" not in output


def test_run_returns_none_if_no_results(monkeypatch, check):
    # variables with log file that returns empty common_lines
    variables = {'LogFiles': ['empty.log']}

    check._remove_blacklisted_logs = lambda x: x
    check.get_error_pattern_counts = lambda filename: (filename, OrderedDict())

    class DummyPool:
        def __init__(self, _):
            pass

        def map(self, func, iterable):
            return [check.get_error_pattern_counts(f) for f in iterable]

    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", DummyPool)
    monkeypatch.setattr("fossor.checks.similar_log_errors.cpu_count", lambda: 1)

    result = check.run(variables)
    assert result is None


def test_debug_logging_in_group_similar_log_lines_on_increment_and_unique(check):
    common_lines = OrderedDict()
    common_lines["ERROR line one"] = {'count': 1}
    # Line similar enough to existing
    line_similar = "ERROR line one modified"
    check._group_similar_log_lines(line_similar, "date", common_lines)
    # Line dissimilar (unique)
    line_unique = "ERROR totally unique line"
    check._group_similar_log_lines(line_unique, "date", common_lines)

    # Check debug messages captured
    debug_msgs = check.log.debug_msgs
    found_increment = False
    found_unique = False
    for msg in debug_msgs:
        if "incrementing count" in msg:
            found_increment = True
        if "unique" in msg or "under the threshold" in msg:
            found_unique = True
    assert found_increment
    assert found_unique


def test_run_integration_with_real_temp_log_files(monkeypatch, check):
    # Create two temp files with error lines, one blacklisted
    with tempfile.NamedTemporaryFile('w+t', delete=False) as f1, tempfile.NamedTemporaryFile('w+t', delete=False) as f2:
        f1.write("2024-01-01 12:00:00 ERROR First error line\n")
        f1.write("2024-01-01 12:00:01 ERROR First error line\n")
        f1.flush()
        f2.write("2024-01-01 12:00:00 ERROR Another error\n")
        f2.flush()

        blacklisted = f2.name + "_configuration.log"
        # Add a blacklisted file name by suffix to check removal
        variables = {'LogFiles': [f1.name, blacklisted]}

        monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", lambda _: type("DummyPool", (), {
            "map": lambda self, func, iterable: [func(filename) for filename in iterable]
        })())
        monkeypatch.setattr("fossor.checks.similar_log_errors.cpu_count", lambda: 1)

        # Use actual _remove_blacklisted_logs
        result = check.run(variables)

        # f1.name should appear, blacklisted file excluded
        assert f1.name in result
        assert blacklisted not in result

    os.remove(f1.name)
    os.remove(f2.name)


def test_group_similar_log_lines_preserves_first_seen_only_once(check):
    common_lines = OrderedDict()
    # Add a line with first_seen set
    common_lines["ERROR preserved"] = {'count': 1, 'first_seen': "2023-12-31 10:00:00"}

    # Similar line arrives with different date
    line = "ERROR preserved error"
    date = "2024-01-01 12:00:00"

    result = check._group_similar_log_lines(line, date, common_lines)

    # first_seen should remain original, not overwritten
    assert result["ERROR preserved"]['first_seen'] == "2023-12-31 10:00:00"
    # Count should be incremented
    assert result["ERROR preserved"]['count'] == 2


def test_process_line_truncates_long_lines(check):
    long_message = "X" * 300
    line = f"2024-01-01 12:00:00 ERROR {long_message}"
    processed_line, date = check._process_line(line)
    # Length should be at most 200
    assert len(processed_line) <= 200
    assert processed_line.startswith("ERROR")
    assert date == "2024-01-01 12:00:00"


def test__process_line_with_non_digit_first_char_but_contains_exception(check):
    line = "Exception occurred while processing"
    result, date = check._process_line(line)
    # It should not filter out lines that contain 'Exception' even if starting alpha
    assert result == line
    assert date is None


def test__process_line_with_line_starting_with_digit_but_invalid_format(check):
    # A line starting with a digit but missing parts for error level
    line = "20240101"
    result, date = check._process_line(line)
    assert result is None
    # Should have logged IndexError
    assert any("IndexError" in msg for msg in check.log.debug_msgs)
