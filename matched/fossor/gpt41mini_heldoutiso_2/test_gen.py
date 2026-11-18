import builtins
import io
import os
import tempfile
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


def test_remove_blacklisted_logs_removes_blacklisted():
    sle = SimilarLogErrors()
    logs = [
        'app.log',
        'error.out',
        'gc.log',
        'access_public_access.log',
        'foo_configuration.log',
        'normal.log',
        'engine_access_log.log',
    ]
    filtered = sle._remove_blacklisted_logs(logs)
    # Only logs without any blacklist substrings remain
    assert 'app.log' in filtered
    assert 'normal.log' in filtered
    # All logs with any blacklisted substring are removed
    for b in sle.log_name_blacklist:
        for log in filtered:
            assert b not in log


def test_process_line_skips_leading_whitespace(monkeypatch):
    sle = SimilarLogErrors()
    # line starting with space -> ignored
    line = " some log line"
    out_line, date = sle._process_line(line)
    assert out_line is None and date is None


def test_process_line_skips_alpha_without_exception(monkeypatch):
    sle = SimilarLogErrors()
    line = "abc just a log line"
    out_line, date = sle._process_line(line)
    assert out_line is None and date is None


def test_process_line_skips_caused_by_line():
    sle = SimilarLogErrors()
    line = "Caused by some error"
    out_line, date = sle._process_line(line)
    assert out_line is None and date is None


def test_process_line_keeps_error_line_and_extracts_date(monkeypatch):
    sle = SimilarLogErrors()
    # Normal error line, simulate line with date + level ERROR
    line = "2024-04-27 12:00:00 ERROR Something bad happened in module"
    processed_line, date = sle._process_line(line)
    assert processed_line.startswith("ERROR ")
    assert "Something bad happened" in processed_line
    assert date == "2024-04-27 12:00:00"


def test_process_line_skips_non_error_log_levels():
    sle = SimilarLogErrors()
    for lvl in ['INFO', 'WARN', 'DEBUG']:
        line = f"2024-04-27 12:00:00 {lvl} This is not error"
        processed_line, date = sle._process_line(line)
        assert processed_line is None and date is None


def test_process_line_handles_index_error(monkeypatch):
    sle = SimilarLogErrors()
    # The line is too short/split does not have index 2
    line = "2024-04-27"
    # Patch log.debug to catch call
    sle.log = MagicMock()
    out_line, date = sle._process_line(line)
    assert out_line is None and date is None
    sle.log.debug.assert_called_once()


def test_group_similar_log_lines_adds_new_line(monkeypatch):
    sle = SimilarLogErrors()
    sle.SIMILARITY_RATIO = 0.7
    sle.MAX_COMMON_LINES = 20
    sle.log = MagicMock()

    common_lines = OrderedDict()
    line = "ERROR something bad happened here"
    date = "2024-04-27 12:00:00"

    # Start with empty common_lines
    common_lines = sle._group_similar_log_lines(line, date, common_lines)
    assert line in common_lines
    values = common_lines[line]
    assert values['count'] == 1
    assert values['first_seen'] == date


def test_group_similar_log_lines_counts_similar(monkeypatch):
    sle = SimilarLogErrors()
    sle.SIMILARITY_RATIO = 0.7
    sle.MAX_COMMON_LINES = 20
    sle.log = MagicMock()

    existing_line = "ERROR something bad happened here"
    common_lines = OrderedDict({existing_line: {'count': 1, 'first_seen': '2024-04-27 11:00:00'}})

    # A line similar enough to existing_line
    new_line = "ERROR something bad happened here!"
    updated = sle._group_similar_log_lines(new_line, "2024-04-27 12:00:00", common_lines)
    # The key should remain as existing_line since it's similar enough
    assert existing_line in updated
    assert updated[existing_line]['count'] == 2
    # first_seen should be preserved, not overwritten
    assert updated[existing_line]['first_seen'] == '2024-04-27 11:00:00'
    sle.log.debug.assert_any_call(
        f"Line had a ratio of {mocked_ratio(updated, new_line, existing_line)}, incrementing count. New line: {new_line}, Old line: {existing_line}"
    ) if hasattr(sle.log.debug, 'assert_any_call') else None


def mocked_ratio(common_lines, new_line, existing_line):
    from difflib import SequenceMatcher
    return SequenceMatcher(a=existing_line, b=new_line).quick_ratio()


def test_group_similar_log_lines_limits_unique_lines(monkeypatch):
    sle = SimilarLogErrors()
    sle.SIMILARITY_RATIO = 0.7
    sle.MAX_COMMON_LINES = 3
    sle.log = MagicMock()

    # Fill common_lines to max
    common_lines = OrderedDict()
    for i in range(sle.MAX_COMMON_LINES):
        common_lines[f"ERROR line {i}"] = {'count': 1, 'first_seen': '2024-04-27 11:00:00'}

    new_line = "ERROR totally different message"
    updated = sle._group_similar_log_lines(new_line, "2024-04-27 12:00:00", common_lines)
    # The added line key should be 'Other Error Lines'
    assert 'Other Error Lines' in updated
    assert updated['Other Error Lines']['count'] == 1
    sle.log.debug.assert_any_call(
        f"Too many unique lines, {sle.MAX_COMMON_LINES} of {sle.MAX_COMMON_LINES}, adding this to category: Other Error Lines"
    )


def test_get_error_pattern_counts_reads_file_and_groups(tmp_path):
    sle = SimilarLogErrors()
    sle.log = MagicMock()

    # Create a test log file with lines
    file_path = tmp_path / "test.log"
    lines = [
        "2024-04-27 12:00:00 ERROR Failure in system component\n",
        "2024-04-27 12:01:00 ERROR Failure in system component\n",
        "Caused by something\n",
        "  stack trace line\n",
        "2024-04-27 12:02:00 INFO this is info\n",
    ]
    file_path.write_text(''.join(lines))

    filename, common_lines = sle.get_error_pattern_counts(str(file_path))
    assert filename == str(file_path)
    # Should group the two ERROR lines as same
    assert any("Failure in system component" in k for k in common_lines)
    # count for that line should be 2
    counts = list(common_lines.values())[0]['count']
    assert counts == 2
    # Lines that are stack traces, caused by or info should not appear
    for k in common_lines:
        assert not k.startswith('Caused by')
        assert not k.isspace()
        assert k is not None


def test_run_returns_none_if_no_logfiles():
    sle = SimilarLogErrors()
    results = sle.run({})
    assert results is None


def test_run_removes_blacklisted_and_returns_result(tmp_path):
    sle = SimilarLogErrors()
    sle.log = MagicMock()

    # Create two files, one blacklisted and one not
    allowed_file = tmp_path / "allowed.log"
    blacklisted_file = tmp_path / "test.out"
    allowed_file.write_text("2024-04-27 12:00:00 ERROR Error here\n")
    blacklisted_file.write_text("2024-04-27 12:00:00 ERROR Should be removed\n")

    variables = {
        "LogFiles": [str(allowed_file), str(blacklisted_file)]
    }

    result = sle.run(variables)
    # The blacklisted file content should NOT appear
    assert str(blacklisted_file) not in result
    # Allowed file should be processed and appear
    assert str(allowed_file) in result
    assert "Count:" in result
    assert "filename:" in result


def test_run_returns_none_if_no_errors(tmp_path):
    sle = SimilarLogErrors()
    sle.log = MagicMock()

    # File with no ERROR lines
    file = tmp_path / "empty.log"
    file.write_text("2024-04-27 12:00:00 INFO Just info\n")

    variables = {
        "LogFiles": [str(file)]
    }

    result = sle.run(variables)
    assert result is None


def test_run_aggregates_multiple_files(tmp_path):
    sle = SimilarLogErrors()
    sle.log = MagicMock()

    file1 = tmp_path / "file1.log"
    file2 = tmp_path / "file2.log"
    file1.write_text("2024-04-27 12:00:00 ERROR error 1\n")
    file2.write_text("2024-04-27 13:00:00 ERROR error 2\n")

    variables = {"LogFiles": [str(file1), str(file2)]}

    res = sle.run(variables)
    assert str(file1) in res
    assert str(file2) in res
    assert "Count:" in res


def test__process_line_truncates_long_line():
    sle = SimilarLogErrors()
    # Create a very long error line > 200 chars
    base_line = "E" * 300
    line = f"2024-04-27 12:00:00 ERROR {base_line}"
    out_line, date = sle._process_line(line)
    assert len(out_line) <= 200 + len("ERROR ")  # prefix plus 200 chars truncated the rest


@pytest.mark.parametrize("line,result", [
    ("2024-04-27 12:00:00 ERROR Something happened", True),
    ("  indented line", False),
    ("Caused by cause", False),
    ("INFO this is info", False),
    ("2024-04-27 12:00:00 INFO Not error line", False),
])
def test_process_line_various_cases(line, result):
    sle = SimilarLogErrors()
    out_line, date = sle._process_line(line)
    if result:
        assert out_line is not None
    else:
        assert out_line is None


def test_group_similar_log_lines_similar_scores(monkeypatch):
    sle = SimilarLogErrors()
    sle.log = MagicMock()
    sle.SIMILARITY_RATIO = 0.7
    sle.MAX_COMMON_LINES = 5

    common_lines = OrderedDict()
    base_line = "ERROR database connection failed"
    common_lines[base_line] = {"count": 1, "first_seen": "2024-01-01 00:00:00"}

    # Similar line, slightly different text
    new_line_similar = "ERROR database connection failed due to timeout"
    updated = sle._group_similar_log_lines(new_line_similar, "2024-06-01 12:00:00", common_lines)
    # Should increment count of base_line
    assert base_line in updated
    assert updated[base_line]['count'] == 2

    # Very different line - below threshold
    new_line_unique = "ERROR unexpected shutdown failure"
    updated = sle._group_similar_log_lines(new_line_unique, "2024-06-02 12:00:00", updated)
    assert new_line_unique in updated or 'Other Error Lines' in updated


def test_group_similar_log_lines_first_seen_set_once(monkeypatch):
    sle = SimilarLogErrors()
    sle.log = MagicMock()
    common_lines = OrderedDict()

    line = "ERROR something"
    date1 = "2024-01-01 01:01:01"
    date2 = "2024-01-01 02:02:02"

    common_lines = sle._group_similar_log_lines(line, date1, common_lines)
    # Add same line, with different date, first_seen should remain the first date
    common_lines = sle._group_similar_log_lines(line, date2, common_lines)

    assert common_lines[line]['first_seen'] == date1
    assert common_lines[line]['count'] == 2


def test_get_error_pattern_counts_empty_file(tmp_path):
    sle = SimilarLogErrors()
    sle.log = MagicMock()
    file = tmp_path / "empty.log"
    file.write_text("")

    filename, common_lines = sle.get_error_pattern_counts(str(file))
    assert filename == str(file)
    assert isinstance(common_lines, OrderedDict)
    assert len(common_lines) == 0


def test_get_error_pattern_counts_uses_debug(monkeypatch, tmp_path):
    sle = SimilarLogErrors()
    log_mock = MagicMock()
    sle.log = log_mock
    file = tmp_path / "file.log"
    file.write_text("2024-04-27 12:00:00 ERROR error\n")
    sle.get_error_pattern_counts(str(file))
    # log.debug should be called at least twice
    assert log_mock.debug.call_count >= 2
    log_mock.debug.assert_any_call(f"Processing filename: {str(file)}")
    log_mock.debug.assert_any_call(f"Finished processing filename: {str(file)}")


def test_run_parallel_map(monkeypatch, tmp_path):
    sle = SimilarLogErrors()
    sle.log = MagicMock()

    # Create two log files
    file1 = tmp_path / "file1.log"
    file2 = tmp_path / "file2.log"
    file1.write_text("2024-04-27 12:00:00 ERROR error 1\n")
    file2.write_text("2024-04-27 12:00:00 ERROR error 2\n")

    # Patch Pool.map to call function directly (simulate parallel call)
    def fake_map(func, lst):
        return [func(f) for f in lst]

    with patch('fossor.checks.similar_log_errors.Pool') as pool_mock:
        pool_inst = pool_mock.return_value
        pool_inst.map.side_effect = fake_map
        result = sle.run({"LogFiles": [str(file1), str(file2)]})

    assert str(file1) in result
    assert str(file2) in result
    assert "Count:" in result
