import os
import tempfile
import pytest
from collections import OrderedDict
from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLogger:
    def __init__(self):
        self.debug_messages = []

    def debug(self, msg):
        self.debug_messages.append(msg)


@pytest.fixture
def similar_log_errors():
    obj = SimilarLogErrors()
    obj.log = DummyLogger()
    return obj


def test_remove_blacklisted_logs_removes_blacklisted(similar_log_errors):
    logs = [
        "app.log",
        "error_gc.log",
        "access.engine_access_log.log",
        "my_oor.logfile",
        "normal.log",
        "foo.out",
        "bar_configuration.log",
        "valid.log",
    ]
    filtered = similar_log_errors._remove_blacklisted_logs(logs)
    for blacklisted in similar_log_errors.log_name_blacklist:
        for f in filtered:
            assert blacklisted not in f
    # Check that valid logs still present
    assert "app.log" in filtered
    assert "normal.log" in filtered
    assert "valid.log" in filtered
    # Check that logs with blacklisted substrings are removed
    assert "error_gc.log" not in filtered
    assert "foo.out" not in filtered


@pytest.mark.parametrize(
    "line,expected_line,expected_date",
    [
        ('  at foo.bar (file.java:23)', (None, None)),  # Leading space => None
        ('Exception in thread main', (None, None)),  # Starts with alpha, no Exception keyword => None
        ('ExceptionOccurred at foo', ('ExceptionOccurred at foo', None)),  # Has 'Exception'
        ('Caused by some reason', (None, None)),  # Starts with 'Caused by' => None
        ('2023-06-01 12:00:00 ERROR something bad happened',
         ('ERROR something bad happened', '2023-06-01 12:00:00')),
        ('2023-06-01 12:00:00 INFO something', (None, None)),  # Not ERROR level
        ('2023-06-01 12:00:00 ERROR a'*250,
         ('ERROR ' + 'a'*193, '2023-06-01 12:00:00')),  # Truncation to 200 chars
        ('2023-06-01 12:00:00', (None, None)),  # Less than 3 tokens
    ],
)
def test_process_line_various_cases(similar_log_errors, line, expected_line):
    out_line, out_date = similar_log_errors._process_line(line)
    expected_line_val = expected_line[0]
    expected_date_val = expected_line[1]
    assert out_line == expected_line_val
    assert out_date == expected_date_val


def test_process_line_debug_on_indexerror(similar_log_errors):
    # Line with too few tokens triggers IndexError
    line = "2023-06-01"
    out_line, out_date = similar_log_errors._process_line(line)
    assert out_line is None
    assert any("IndexError" in msg for msg in similar_log_errors.log.debug_messages)


def test_group_similar_log_lines_counts_and_groups(similar_log_errors):
    common_lines = OrderedDict()
    # Add a line manually
    common_lines["ERROR something happened"] = {"count": 2, "first_seen": "2023-01-01 00:00:00"}
    common_lines = similar_log_errors._group_similar_log_lines(
        "ERROR something happened", "2023-06-01 12:00:00", common_lines
    )
    # The count for that line should increase since it's identical
    assert common_lines["ERROR something happened"]["count"] == 3
    # First seen date should remain unchanged
    assert common_lines["ERROR something happened"]["first_seen"] == "2023-01-01 00:00:00"


def test_group_similar_log_lines_merges_similar(similar_log_errors):
    common_lines = OrderedDict()
    common_lines["ERROR failed to connect to DB"] = {"count": 1, "first_seen": "2023-01-01 00:00:00"}
    # A very similar line
    similar_line = "ERROR failed to connect to DB at host"
    common_lines = similar_log_errors._group_similar_log_lines(similar_line, "2023-06-01 12:00:00", common_lines)
    # The similar_line should map to original line, so count increases for original line
    assert len(common_lines) == 1
    key = next(iter(common_lines))
    assert key == "ERROR failed to connect to DB"
    assert common_lines[key]["count"] == 2
    # first_seen remains the first date
    assert common_lines[key]["first_seen"] == "2023-01-01 00:00:00"


def test_group_similar_log_lines_adds_new_line_when_unique(similar_log_errors):
    common_lines = OrderedDict()
    common_lines["ERROR something happened"] = {"count": 1}
    new_line = "ERROR a totally new error message"
    common_lines = similar_log_errors._group_similar_log_lines(new_line, "2023-06-01 12:00:00", common_lines)
    assert len(common_lines) == 2
    assert new_line in common_lines
    assert common_lines[new_line]["count"] == 1
    assert common_lines[new_line]["first_seen"] == "2023-06-01 12:00:00"


def test_group_similar_log_lines_above_max_unique_lines(similar_log_errors):
    common_lines = OrderedDict()
    for i in range(similar_log_errors.MAX_COMMON_LINES):
        common_lines[f"ERROR unique error {i}"] = {"count": 1}
    new_line = "ERROR this error should be categorized as Other Error Lines"
    common_lines = similar_log_errors._group_similar_log_lines(new_line, "2023-06-01 12:00:00", common_lines)
    # The new line should be added as 'Other Error Lines'
    assert "Other Error Lines" in common_lines
    values = common_lines.get("Other Error Lines", {})
    # Count should be 1 first time
    assert values["count"] >= 1


def test_get_error_pattern_counts_reads_file_and_groups(tmp_path, similar_log_errors):
    # Create a temp log file
    log_lines = [
        "2023-06-01 12:00:00 ERROR first error occurred\n",
        "2023-06-01 12:01:00 ERROR first error occurred again\n",
        "2023-06-01 12:02:00 ERROR a different error\n",
        "Caused by some reason\n",
        "   indented stacktrace\n",
        "Not an error line\n",
    ]
    logfile = tmp_path / "test.log"
    logfile.write_text(''.join(log_lines))

    filename, common_lines = similar_log_errors.get_error_pattern_counts(str(logfile))
    assert filename == str(logfile)
    # The common_lines keys should at least have "ERROR first error occurred" and "ERROR a different error"
    keys = list(common_lines.keys())
    assert any("ERROR first error occurred" in k for k in keys)
    assert any("ERROR a different error" in k for k in keys)
    # Lines ignored should not appear
    for line in keys:
        assert not line.startswith("Caused by")
        assert not line.startswith(" ")
    # The count of the first error line should be 2 since it occurred twice
    found = False
    for key, value in common_lines.items():
        if "first error" in key:
            assert value["count"] == 2
            found = True
    assert found


def test_run_returns_none_when_no_logfiles(similar_log_errors):
    result = similar_log_errors.run({})
    assert result is None


def test_run_removes_blacklisted_logs_and_processes(tmp_path, similar_log_errors, monkeypatch):
    # Create 3 files: 1 blacklisted and 2 valid
    valid1 = tmp_path / "log1.log"
    valid2 = tmp_path / "log2.log"
    blacklisted = tmp_path / "gc.log"

    valid1.write_text("2023-06-01 12:00:00 ERROR error1\n")
    valid2.write_text("2023-06-01 12:00:00 ERROR error2\n")
    blacklisted.write_text("2023-06-01 12:00:00 ERROR should be ignored\n")

    variables = {"LogFiles": [str(valid1), str(valid2), str(blacklisted)]}

    # Patch get_error_pattern_counts for quick predictable output
    def fake_get_error_pattern_counts(filename):
        return filename, OrderedDict({f"ERROR fake for {os.path.basename(filename)}": {"count": 1, "first_seen": "2023-06-01"}})

    monkeypatch.setattr(similar_log_errors, "get_error_pattern_counts", fake_get_error_pattern_counts)
    # Patch Pool to avoid multiprocessing in test and use single-threaded map
    class DummyPool:
        def __init__(self, _):
            pass
        def map(self, func, items):
            return [func(i) for i in items]
        def close(self):
            pass
        def join(self):
            pass

    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", DummyPool)

    result = similar_log_errors.run(variables)
    # Result should contain only the two valid files, not blacklisted one
    assert str(valid1) in result
    assert str(valid2) in result
    assert str(blacklisted) not in result
    # Should have count and first seen text for each line
    assert "Count: 1" in result
    assert "First seen: 2023-06-01" in result


def test_run_returns_none_when_no_result(monkeypatch, similar_log_errors):
    # Patch _remove_blacklisted_logs to return empty
    monkeypatch.setattr(similar_log_errors, "_remove_blacklisted_logs", lambda x: [])
    result = similar_log_errors.run({"LogFiles": ["file1.log"]})
    assert result is None


def test_run_handles_no_logfiles_key(similar_log_errors):
    result = similar_log_errors.run({})
    assert result is None


def test_group_similar_log_lines_prefers_highest_similarity(similar_log_errors):
    common_lines = OrderedDict()
    common_lines["ERROR connection timeout"] = {"count": 1}
    common_lines["ERROR connection refused"] = {"count": 1}

    # A line more similar to "connection refused"
    test_line = "ERROR connection refused by server"
    common_lines = similar_log_errors._group_similar_log_lines(test_line, "2023", common_lines)

    # The count of "connection refused" should be 2 now
    assert common_lines["ERROR connection refused"]["count"] == 2
    # The count of "connection timeout" should remain 1
    assert common_lines["ERROR connection timeout"]["count"] == 1


def test_process_line_with_exact_200_chars_truncation(similar_log_errors):
    # Create a line with exactly 200 characters after skipping date + space
    message = "E" * 193  # To make total 200 including 'ERROR '
    line = f"2023-01-01 00:00:00 ERROR {message}"
    processed_line, date = similar_log_errors._process_line(line)
    # Should have length 200 (ERROR + 193 chars)
    assert len(processed_line) == 200
    assert processed_line.startswith("ERROR ")
    assert date == "2023-01-01 00:00:00"


def test_group_similar_log_lines_handles_no_date(similar_log_errors):
    common_lines = OrderedDict()
    new_line = "ERROR unexpected failure"
    # Pass None as date
    common_lines = similar_log_errors._group_similar_log_lines(new_line, None, common_lines)
    # Date should not appear in dict keys
    assert new_line in common_lines
    assert "first_seen" not in common_lines[new_line]
    assert common_lines[new_line]["count"] == 1


def test_group_similar_log_lines_increments_existing_line_first_seen_preserved(similar_log_errors):
    common_lines = OrderedDict()
    common_lines["ERROR foo"] = {"count": 1, "first_seen": "2020-01-01"}
    common_lines = similar_log_errors._group_similar_log_lines("ERROR foo", "2022-01-01", common_lines)
    assert common_lines["ERROR foo"]["count"] == 2
    # first_seen date unchanged
    assert common_lines["ERROR foo"]["first_seen"] == "2020-01-01"


def test_run_integration(tmp_path, similar_log_errors):
    # Create logs with multiple lines, mixed errors
    log1 = tmp_path / "log1.log"
    content1 = (
        "2023-01-01 10:00:00 ERROR failure 1\n"
        "2023-01-01 10:01:00 ERROR failure 1 extended\n"
        "2023-01-01 10:02:00 ERROR failure 2\n"
        "Caused by some reason\n"
        " Not an error line\n"
    )
    log1.write_text(content1)

    variables = {"LogFiles": [str(log1)]}

    result = similar_log_errors.run(variables)
    # Should contain filename
    assert str(log1) in result
    # Should contain "Count: " lines for each error category
    assert "Count: " in result
    # Should contain first seen dates
    assert "First seen: 2023-01-01 10:00:00" in result


def test_get_error_pattern_counts_reads_empty_file(tmp_path, similar_log_errors):
    empty_file = tmp_path / "empty.log"
    empty_file.write_text("")
    filename, common_lines = similar_log_errors.get_error_pattern_counts(str(empty_file))
    assert filename == str(empty_file)
    # common_lines should be empty OrderedDict
    assert isinstance(common_lines, OrderedDict)
    assert len(common_lines) == 0


def test_group_similar_log_lines_exact_threshold(similar_log_errors):
    # Create two lines that have similarity exactly .7
    line1 = "ERROR abcdefghijklmnop"
    line2 = "ERROR abcdefgrijklmnop"  # One char changed to get ~0.7 similarity
    common_lines = OrderedDict()
    common_lines[line1] = {"count": 1}
    # Use SequenceMatcher directly to get exact ratio and check
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(a=line1, b=line2).quick_ratio()
    assert abs(ratio - 0.7) <= 0.05  # Close to threshold

    # Provide line2, should NOT merge since threshold > 0.7 is required
    updated = similar_log_errors._group_similar_log_lines(line2, "date", common_lines)
    # Should add new line instead of merging
    assert len(updated) == 2
    assert line2 in updated
