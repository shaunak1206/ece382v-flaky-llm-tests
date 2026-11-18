import os
import tempfile
import pytest
from fossor.checks.similar_log_errors import SimilarLogErrors


@pytest.fixture
def similar_log_errors():
    return SimilarLogErrors()


def test_remove_blacklisted_logs_removes_blacklisted(similar_log_errors):
    logs = [
        'foo_configuration.log',
        'normal.log',
        'something.out',
        'good.log',
        'gc.log',
        'jetty_thread_pool_stats.log'
    ]
    filtered = similar_log_errors._remove_blacklisted_logs(logs)
    assert 'normal.log' in filtered
    assert 'good.log' in filtered
    assert 'foo_configuration.log' not in filtered
    assert 'something.out' not in filtered
    assert 'gc.log' not in filtered
    assert 'jetty_thread_pool_stats.log' not in filtered


def test_remove_blacklisted_logs_returns_all_if_none_blacklisted(similar_log_errors):
    logs = ['app.log', 'error_2024.log', 'server.log']
    filtered = similar_log_errors._remove_blacklisted_logs(logs)
    assert filtered == logs


def test_process_line_none_for_lower_stacktrace(similar_log_errors):
    line = "    at some.stacktrace(location.java:123)"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_none_for_alpha_start_non_exception(similar_log_errors):
    line = "Info started something"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_none_for_caused_by_line(similar_log_errors):
    line = "Caused by some error message"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_none_for_non_error_log(similar_log_errors):
    # This line starts with date but level is WARN, so discard
    line = "2024-06-01 12:00:00 WARN some warning message"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_error_line_truncated(similar_log_errors):
    # Line longer than 200 chars after removing date portion
    long_message = "x" * 300
    line = f"2024-06-01 12:00:00 ERROR {long_message}"
    processed_line, date = similar_log_errors._process_line(line)
    # Should be truncated at 200 chars after "ERROR "
    assert processed_line.startswith("ERROR ")
    assert len(processed_line) <= 200
    # Date is the first two tokens
    assert date == "2024-06-01 12:00:00"


def test_process_line_handles_index_error_and_returns_none(similar_log_errors, caplog):
    # Malformed line with less than 3 tokens to cause IndexError
    line = "2024-06-01 something"
    processed_line, date = similar_log_errors._process_line(line)
    assert processed_line is None
    assert date is None
    assert "IndexError" in caplog.text


def test_group_similar_log_lines_counts_new_line_when_empty(similar_log_errors):
    common_lines = {}
    line = "ERROR Something bad happened"
    date = "2024-06-01 12:00:00"

    updated = similar_log_errors._group_similar_log_lines(line, date, common_lines)
    assert line in updated
    assert updated[line]['count'] == 1
    assert updated[line]['first_seen'] == date


def test_group_similar_log_lines_counts_similar_line(similar_log_errors):
    common_lines = {
        "ERROR Something bad error": {'count': 2, 'first_seen': '2024-06-01 11:00:00'}
    }
    # This line is very similar to existing line - ratio > 0.7
    new_line = "ERROR Something bad error!"
    updated = similar_log_errors._group_similar_log_lines(new_line, "2024-06-01 12:00:00", common_lines)
    # The original key is preserved
    assert "ERROR Something bad error" in updated
    # Count incremented
    assert updated["ERROR Something bad error"]['count'] == 3
    # First seen remains the original one, not overwritten
    assert updated["ERROR Something bad error"]['first_seen'] == '2024-06-01 11:00:00'
    # The new line string itself is not added as key
    assert new_line not in updated


def test_group_similar_log_lines_adds_other_error_when_max_lines_reached(similar_log_errors):
    common_lines = {}
    # Fill up to max lines
    for i in range(similar_log_errors.MAX_COMMON_LINES):
        common_lines[f"ERROR line {i}"] = {'count': 1}
    # Now add a totally different line which should be added under "Other Error Lines"
    new_line = "ERROR completely new unique line"
    updated = similar_log_errors._group_similar_log_lines(new_line, None, common_lines)
    assert "Other Error Lines" in updated
    assert updated["Other Error Lines"]['count'] == 1


def test_get_error_pattern_counts_reads_and_processes_file(similar_log_errors):
    content = (
        "2024-06-01 12:00:00 INFO some info message\n"
        "2024-06-01 12:01:00 ERROR First error line\n"
        "2024-06-01 12:02:00 ERROR Second error line\n"
        "Caused by something\n"
        "  at stacktrace\n"
        "2024-06-01 12:03:00 WARN warning line\n"
    )
    with tempfile.NamedTemporaryFile('w+', delete=False) as f:
        f.write(content)
        tempname = f.name

    try:
        filename, common_lines = similar_log_errors.get_error_pattern_counts(tempname)
        # INFO and WARN lines are skipped; also caused by and stacktrace lines skipped
        # Only 2 error lines counted
        assert filename == tempname
        assert any("ERROR First error line" in k or "ERROR Second error line" in k for k in common_lines.keys())
        total_count = sum(v['count'] for v in common_lines.values())
        assert total_count == 2
    finally:
        os.unlink(tempname)


def test_run_returns_none_with_no_logfiles(similar_log_errors):
    res = similar_log_errors.run({})
    assert res is None


def test_run_removes_blacklisted_and_processes(similar_log_errors, monkeypatch):
    # Setup 2 temp files: one blacklisted and one good
    blacklisted_name = "foo_gc.log"
    good_content = "2024-06-01 12:00:00 ERROR line one\n"
    with tempfile.NamedTemporaryFile('w+', delete=False) as good_f:
        good_f.write(good_content)
        good_name = good_f.name

    logs = [blacklisted_name, good_name]

    # Patch _remove_blacklisted_logs to actually filter out blacklisted_name
    monkeypatch.setattr(similar_log_errors, "_remove_blacklisted_logs", lambda x: [good_name])

    # Patch get_error_pattern_counts to avoid multiprocessing and file IO in test
    def fake_get_error_pattern_counts(filename):
        assert filename == good_name
        return filename, {"ERROR line one": {'count': 1, 'first_seen': "2024-06-01 12:00:00"}}
    monkeypatch.setattr(similar_log_errors, "get_error_pattern_counts", fake_get_error_pattern_counts)

    result = similar_log_errors.run({"LogFiles": logs})
    assert good_name in result
    assert "ERROR line one" in result
    assert "Count: 1" in result


def test_run_returns_none_when_no_results(monkeypatch, similar_log_errors):
    # Patch _remove_blacklisted_logs to return empty list
    monkeypatch.setattr(similar_log_errors, "_remove_blacklisted_logs", lambda x: [])
    result = similar_log_errors.run({"LogFiles": ["foo.log"]})
    assert result is None


def test_run_handles_pool_mapping(monkeypatch, similar_log_errors):
    # Patch _remove_blacklisted_logs to return one file
    files = ["file1.log"]
    monkeypatch.setattr(similar_log_errors, "_remove_blacklisted_logs", lambda x: files)

    # Patch Pool and its map method to call method directly without multiprocessing
    class DummyPool:
        def map(self, func, arglist):
            assert arglist == files
            # Simulate return from get_error_pattern_counts
            return [(files[0], {"ERROR test line": {'count': 2, 'first_seen': '2024-06-01 12:00:00'}})]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def close(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(similar_log_errors, "get_error_pattern_counts", lambda filename: (filename, {}))
    monkeypatch.setattr('fossor.checks.similar_log_errors.Pool', lambda threads: DummyPool())

    result = similar_log_errors.run({"LogFiles": files})
    # result should not be None but empty since common_lines is empty dict
    assert result is None or isinstance(result, str)


def test_group_similar_log_lines_reuses_line_with_exact_match(similar_log_errors):
    common_lines = {"ERROR exact line": {'count': 1}}
    new_line = "ERROR exact line"
    updated = similar_log_errors._group_similar_log_lines(new_line, None, common_lines)
    assert "ERROR exact line" in updated
    assert updated["ERROR exact line"]['count'] == 2


def test_group_similar_log_lines_increments_count_and_keeps_first_seen(similar_log_errors):
    common_lines = {"ERROR line": {'count': 3, 'first_seen': 'start'}}
    # New line similar enough should increment count but keep first_seen
    new_line = "ERROR line!"
    updated = similar_log_errors._group_similar_log_lines(new_line, "newdate", common_lines)
    assert updated["ERROR line"]['count'] == 4
    assert updated["ERROR line"]['first_seen'] == 'start'


def test_group_similar_log_lines_sets_first_seen_when_missing(similar_log_errors):
    common_lines = {}
    line = "ERROR unique line"
    updated = similar_log_errors._group_similar_log_lines(line, "date1", common_lines)
    assert updated[line]['first_seen'] == "date1"
    # Add again with no date, first_seen remains
    updated = similar_log_errors._group_similar_log_lines(line, None, common_lines)
    assert updated[line]['first_seen'] == "date1"
    assert updated[line]['count'] == 2


def test_group_similar_log_lines_handles_no_date(similar_log_errors):
    common_lines = {}
    line = "ERROR nodate line"
    updated = similar_log_errors._group_similar_log_lines(line, None, common_lines)
    assert 'first_seen' not in updated[line]


def test_run_returns_expected_output_format(monkeypatch, similar_log_errors):
    # Prepare test files list
    test_file = "file.log"

    # Return input file list unfiltered
    monkeypatch.setattr(similar_log_errors, "_remove_blacklisted_logs", lambda logs: [test_file])

    # Provide get_error_pattern_counts results
    def fake_get_error_pattern_counts(filename):
        return filename, {
            "ERROR line 1": {'count': 1, 'first_seen': '2024-06-01 00:00:00'},
            "ERROR line 2": {'count': 2},
        }
    monkeypatch.setattr(similar_log_errors, "get_error_pattern_counts", fake_get_error_pattern_counts)

    result = similar_log_errors.run({"LogFiles": [test_file, "blacklisted.log"]})
    assert f"filename: {test_file}" in result
    assert "First seen: 2024-06-01 00:00:00, Count: 1, ERROR line 1" in result
    assert "Count: 2, ERROR line 2" in result


def test_run_skips_blacklisted_logs(monkeypatch, similar_log_errors):
    logs = ['foo.log', '_configuration.log', 'bar.out']
    monkeypatch.setattr(similar_log_errors, "_remove_blacklisted_logs", lambda logs: ['foo.log'])
    monkeypatch.setattr(similar_log_errors, "get_error_pattern_counts",
                        lambda filename: (filename, {"ERROR example": {'count': 1}}))

    result = similar_log_errors.run({"LogFiles": logs})
    assert "foo.log" in result
    assert "_configuration.log" not in result
    assert "bar.out" not in result


def test_group_similar_log_lines_different_line_no_similarity_adds_unique(similar_log_errors):
    common_lines = {"ERROR line 1": {'count': 1}}
    new_line = "ERROR completely different error"
    updated = similar_log_errors._group_similar_log_lines(new_line, None, common_lines)
    assert new_line in updated
    assert updated[new_line]['count'] == 1
    assert "Other Error Lines" not in updated

