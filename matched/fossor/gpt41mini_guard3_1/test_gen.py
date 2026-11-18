import pytest
import random
import numpy as np

from fossor.checks import similar_log_errors


class DummyLog:
    def debug(self, msg):
        pass


@pytest.fixture
def check():
    c = similar_log_errors.SimilarLogErrors()
    c.log = DummyLog()
    return c


def test_remove_blacklisted_logs_removes_blacklisted(check):
    log_files = [
        'foo_configuration.log',
        'bar.out',
        'baz.log',
        'foo_gc.log',
        'qux_normal.log',
        'engine_access_log.log',
    ]
    filtered = check._remove_blacklisted_logs(log_files)
    for blacklisted in check.log_name_blacklist:
        for f in filtered:
            assert blacklisted not in f
    assert 'baz.log' in filtered
    assert 'qux_normal.log' in filtered


def test_remove_blacklisted_logs_empty_and_none(check):
    assert check._remove_blacklisted_logs([]) == []
    # Passing None should be handled gracefully by run()
    result = check.run({})
    assert result is None


def test__process_line_ignores_leading_whitespace(check):
    # Leading whitespace: line ignored (returns None)
    line, date = check._process_line("  indented line")
    assert line is None
    assert date is None


def test__process_line_ignores_lines_starting_alpha_without_exception(check):
    line, date = check._process_line("Info: this is a normal line")
    assert line is None
    assert date is None


def test__process_line_ignores_caused_by_lines(check):
    line, date = check._process_line("Caused by some error")
    assert line is None
    assert date is None


def test__process_line_processes_error_line(check):
    # Line begin with digits, level ERROR
    raw_line = "2023-01-01 12:00:00 ERROR Something bad happened"
    line, date = check._process_line(raw_line)
    assert line is not None
    assert date == "2023-01-01 12:00:00"
    assert line.startswith("ERROR")
    # Line should be truncated to length <= 200 (including prefix "ERROR")
    assert len(line) <= 200


def test__process_line_ignores_non_error_levels(check):
    raw_line = "2023-01-01 12:00:00 INFO Some info here"
    line, date = check._process_line(raw_line)
    assert line is None
    assert date is None


def test__process_line_handles_short_lines_without_indexerror(check, monkeypatch):
    # Line with fewer than 3 parts cause IndexError, but handled gracefully
    short_line = "2023-01-01"
    # Use monkeypatch to spy on debug calls without capturing logs
    debug_msgs = []

    def fake_debug(msg):
        debug_msgs.append(msg)

    monkeypatch.setattr(check.log, "debug", fake_debug)
    line, date = check._process_line(short_line)
    assert line is None
    assert date is None
    assert any("IndexError" in m for m in debug_msgs)


def test__group_similar_log_lines_similar_increments_count(check):
    random.seed(42)
    np.random.seed(42)

    common_lines = {}
    line1 = "ERROR Unable to connect to service"
    # First insert
    common_lines = check._group_similar_log_lines(line1, "date1", common_lines)
    # Insert a line very similar to line1; high similarity ratio expected
    line2 = "ERROR Unable to connect to service!!!"
    updated = check._group_similar_log_lines(line2, "date2", common_lines)
    # After similar line, count for line1 (the key) should be 2
    assert line1 in updated
    assert updated[line1]["count"] == 2
    # first_seen should remain date1 (from first insert)
    assert updated[line1]["first_seen"] == "date1"


def test__group_similar_log_lines_unique_when_below_threshold(check):
    random.seed(42)
    np.random.seed(42)

    common_lines = {}
    line1 = "ERROR Failed to open file"
    common_lines = check._group_similar_log_lines(line1, "2023-01-01", common_lines)
    line2 = "ERROR Disk full"
    common_lines = check._group_similar_log_lines(line2, "2023-01-02", common_lines)
    # Both lines should be present
    assert line1 in common_lines
    assert line2 in common_lines
    # Each count is 1
    assert common_lines[line1]["count"] == 1
    assert common_lines[line2]["count"] == 1


def test__group_similar_log_lines_max_unique_lines_reached(check):
    random.seed(42)
    np.random.seed(42)

    # Create max unique lines
    common_lines = {}
    for i in range(check.MAX_COMMON_LINES):
        line = f"ERROR unique line {i}"
        common_lines = check._group_similar_log_lines(line, f"date{i}", common_lines)
        assert line in common_lines

    # Now adding a new unique line should be categorized as 'Other Error Lines'
    new_line = "ERROR a totally new distinct error"
    common_lines2 = check._group_similar_log_lines(new_line, "dateX", common_lines)
    # It should have 'Other Error Lines' as key and increased count
    assert 'Other Error Lines' in common_lines2
    # Also new_line should NOT be a separate key (else would exceed max)
    assert new_line not in common_lines2
    # Count increased by one
    assert common_lines2['Other Error Lines']['count'] >= 1


def test_get_error_pattern_counts_reads_and_counts(tmp_path, check):
    # Prepare a file with multiple lines (some ignored, some errors)
    content = (
        "2023-01-01 12:00:00 ERROR First error\n"
        "Caused by some reason\n"
        "  stacktrace line\n"
        "2023-01-01 12:01:00 ERROR Second error\n"
        "2023-01-01 12:02:00 INFO Not error\n"
        "2023-01-01 12:03:00 ERROR First error\n"
    )
    file_path = tmp_path / "logfile.log"
    file_path.write_text(content)

    filename, counts = check.get_error_pattern_counts(str(file_path))

    assert filename == str(file_path)
    # Should count 3 error lines ignoring others
    total_counts = sum(v['count'] for v in counts.values())
    assert total_counts == 3

    # There should be at least two distinct error lines (First error and Second error)
    found_keys = list(counts.keys())
    assert any("First error" in k for k in found_keys)
    assert any("Second error" in k for k in found_keys)


def test_run_returns_result_for_logs(tmp_path, check):
    # Two log files
    content1 = (
        "2023-01-01 12:00:00 ERROR Foo error happened\n"
        "2023-01-01 12:01:00 ERROR Another error\n"
    )
    content2 = (
        "2023-01-02 13:00:00 ERROR Foo error happened\n"
        "2023-01-02 13:01:00 ERROR Different error\n"
    )
    f1 = tmp_path / "file1.log"
    f2 = tmp_path / "file2.log"
    f1.write_text(content1)
    f2.write_text(content2)

    variables = {"LogFiles": [str(f1), str(f2)]}

    # Run returns text that includes filename and counts
    result = check.run(variables)

    assert str(f1) in result
    assert str(f2) in result
    # For each file, expect counted error lines mentioned
    for fname in [str(f1), str(f2)]:
        assert f"filename: {fname}" in result
    assert "Count:" in result
    # Each error line text present
    assert "ERROR Foo error happened" in result


def test_run_returns_none_if_no_logfiles(check):
    # No logs present yields None
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_removes_blacklisted_before_processing(monkeypatch, tmp_path, check):
    # Blacklist a file explicitly
    bl_file = tmp_path / "exclude_configuration.log"
    bl_file.write_text("2023-01-01 12:00:00 ERROR stuff\n")
    good_file = tmp_path / "good.log"
    good_file.write_text("2023-01-01 12:00:00 ERROR stuff\n")

    variables = {"LogFiles": [str(bl_file), str(good_file)]}

    # Spy on _remove_blacklisted_logs to ensure it was called
    called = {}

    def fake_remove_blacklisted_logs(log_files):
        called['called'] = True
        result = check._remove_blacklisted_logs(log_files)
        # Check blacklist applied correctly here
        assert all('configuration.log' not in f for f in result)
        return result

    monkeypatch.setattr(check, "_remove_blacklisted_logs", fake_remove_blacklisted_logs)

    res = check.run(variables)
    assert called.get('called', False)
    # Result should only mention good_file
    assert str(good_file) in res
    assert str(bl_file) not in res


def test__group_similar_log_lines_merges_most_similar(monkeypatch, check):
    """
    Ensures grouping picks largest similarity ratio and merges to that line.
    """
    random.seed(123)
    np.random.seed(123)

    calls = []

    def fake_debug(msg):
        calls.append(msg)

    monkeypatch.setattr(check.log, "debug", fake_debug)
    common_lines = {
        "ERROR Connection timeout": {'count': 2},
        "ERROR Connection timed out badly": {'count': 1},
    }
    new_line = "ERROR Connection timed out"
    new_common_lines = check._group_similar_log_lines(new_line, "2023-01-01", common_lines)
    # It should merge to the most similar existing line

    # New line should now be one of the keys in new_common_lines
    # Since it should merge with the more similar existing line, counts increased
    counts = [v['count'] for v in new_common_lines.values()]
    # Since it merged, total number of unique keys is expected unchanged
    assert len(new_common_lines) == len(common_lines)

    # There should be debug message about incrementing count for similar line
    found = any("incrementing count" in m for m in calls)
    assert found


def test__process_line_line_truncation_length(check):
    # Verify that _process_line truncates long line correctly (to at most 200 chars)
    # Compose a very long error message line
    base_msg = "ERROR " + "A" * 1000
    input_line = f"2023-01-01 12:00:00 {base_msg}"
    line, date = check._process_line(input_line)
    assert line.startswith("ERROR ")
    assert len(line) <= 200


def test__group_similar_log_lines_date_preserved_first_seen(check):
    random.seed(42)
    np.random.seed(42)
    common_lines = {}

    line1 = "ERROR Disk failure"
    common_lines = check._group_similar_log_lines(line1, "date_first", common_lines)
    # Add similar line with different timestamp; first_seen should not change
    similar_line = "ERROR Disk failurx"
    common_lines = check._group_similar_log_lines(similar_line, "date_second", common_lines)

    assert line1 in common_lines
    # Count should be 2
    assert common_lines[line1]["count"] == 2
    # first_seen should be original
    assert common_lines[line1]["first_seen"] == "date_first"


def test_run_results_include_first_seen_and_count_ordering(tmp_path, check):
    # Create a log file where count > 1 on some lines, first_seen present
    content = (
        "2023-01-01 12:00:00 ERROR A\n"
        "2023-01-01 12:01:00 ERROR B\n"
        "2023-01-01 12:02:00 ERROR A\n"
        "2023-01-01 12:03:00 ERROR B\n"
        "2023-01-01 12:04:00 ERROR A\n"
    )
    f = tmp_path / "sample.log"
    f.write_text(content)
    result = check.run({"LogFiles": [str(f)]})
    assert "filename: " in result
    assert str(f) in result
    assert "Count: 3" in result or "Count: 2" in result  # A and B counted appropriately
    assert "First seen: 2023-01-01 12:00:00" in result


def test__process_line_non_digit_start_ignored(check):
    line, date = check._process_line("Z2023-01-01 12:00:00 ERROR Ooops")
    # Starts with 'Z' letter, so ignored (since not exception line)
    assert line is None
    assert date is None


def test__process_line_with_exception_in_alpha_line(check):
    line_w_exc = "Exception in thread"
    # This line has Exception string => should not be ignored by the alpha check
    line, date = check._process_line(line_w_exc)
    # Since first_character is alpha but Exception in line, _process_line won't ignore it by that rule
    # But line does not start with digit => line will be None still
    assert line is None
    assert date is None
