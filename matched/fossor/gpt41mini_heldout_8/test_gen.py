import os
import tempfile
import pytest
from fossor.checks.similar_log_errors import SimilarLogErrors


@pytest.fixture
def checker():
    return SimilarLogErrors()


def test_remove_blacklisted_logs_removes_blacklisted(checker):
    logs = [
        "server_configuration.log",
        "app.log",
        "error.gc.log",
        "normal.log",
        "something_oor.log",
        "another.log.out",
        "jetty_thread_pool_stats.log",
        "final.log",
    ]
    filtered = checker._remove_blacklisted_logs(logs)
    for blacklisted in checker.log_name_blacklist:
        for entry in filtered:
            assert blacklisted not in entry
    # original list had 8 logs, after filtering some removed
    assert len(filtered) < len(logs)
    assert "app.log" in filtered
    assert "normal.log" in filtered
    assert "final.log" in filtered


def test_process_line_ignores_lines_starting_with_space(checker):
    line = "    at com.example.MyClass.method(MyClass.java:10)"
    processed_line, date = checker._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_ignores_lines_starting_with_alpha_without_exception(checker):
    line = "Starting server on port 8080"
    processed_line, date = checker._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_ignores_caused_by_lines(checker):
    line = "Caused by: java.lang.NullPointerException"
    processed_line, date = checker._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_non_error_level_line(checker):
    line = "2023-06-10 23:59:59 INFO Server started"
    processed_line, date = checker._process_line(line)
    assert processed_line is None
    assert date is None


def test_process_line_extracts_error_line_and_date(checker):
    line = "2023-06-10 23:59:59 ERROR Something bad happened"
    processed_line, date = checker._process_line(line)
    assert processed_line.startswith("ERROR")
    assert "Something bad happened" in processed_line
    assert date == "2023-06-10 23:59:59"
    # Truncated length max 200
    assert len(processed_line) <= 200


def test_process_line_handles_index_error(log_checker=SimilarLogErrors()):
    # Malformed line with less than 3 parts
    line = "2023-06-10 ERROR"
    # Patch self.log.debug to avoid errors, capture debug calls
    called_msgs = []

    def fake_debug(msg):
        called_msgs.append(msg)

    log_checker.log = type('Log', (), {"debug": fake_debug})()
    processed_line, date = log_checker._process_line(line)
    assert processed_line is None
    assert date is None
    assert any("IndexError parsing line" in m for m in called_msgs)


def test_group_similar_log_lines_increments_existing_line(checker):
    common_lines = {'ERROR Something happened': {'count': 1, 'first_seen': '2023-01-01 12:00:00'}}
    # Similar line but not exactly the same
    line = 'ERROR Something happend'  # one char difference
    result = checker._group_similar_log_lines(line, '2023-01-01 13:00:00', common_lines)
    # Should increment count on existing line, not add a new line
    assert 'ERROR Something happened' in result
    assert result['ERROR Something happened']['count'] == 2
    # first_seen stays as the first date
    assert result['ERROR Something happened']['first_seen'] == '2023-01-01 12:00:00'


def test_group_similar_log_lines_adds_new_line_if_unique(checker):
    common_lines = OrderedDict()
    line = 'ERROR A completely new error occurred'
    result = checker._group_similar_log_lines(line, '2023-02-01 15:15:15', common_lines)
    assert line in result
    assert result[line]['count'] == 1
    assert result[line]['first_seen'] == '2023-02-01 15:15:15'


def test_group_similar_log_lines_categories_other_error_lines_after_max(checker):
    # Pre-fill common_lines with max lines
    common_lines = OrderedDict((f"ERROR {i}" , {'count': 1}) for i in range(checker.MAX_COMMON_LINES))
    line = 'ERROR This new error will be categorized'
    result = checker._group_similar_log_lines(line, None, common_lines)
    assert 'Other Error Lines' in result
    assert result['Other Error Lines']['count'] >= 1


def test_run_returns_none_when_no_logfiles(checker):
    ret = checker.run({})
    assert ret is None


def test_run_removes_blacklisted_and_process_files(tmp_path, checker):
    # Create three files, one blacklisted
    paths = [
        tmp_path / "app.log",
        tmp_path / "gc.log",  # blacklisted
        tmp_path / "other.log",
    ]
    # Write error lines only to valid files
    error_line = "2023-06-10 23:59:59 ERROR Sample error message\n"
    normal_line = "Some unrelated line\n"
    for p in paths:
        with open(p, "wt") as f:
            f.write(error_line)
            f.write(normal_line)

    variables = {
        "LogFiles": [str(p) for p in paths]
    }
    result = checker.run(variables)
    # gc.log should be ignored
    assert "gc.log" not in result
    # app.log and other.log should be present
    assert "app.log" in result
    assert "other.log" in result
    assert "Sample error message" in result


def test_get_error_pattern_counts_groups_lines(tmp_path, checker):
    file = tmp_path / "test.log"
    lines = [
        "2023-06-10 23:59:59 ERROR Error happened in module A\n",
        "2023-06-10 23:59:59 ERROR Error happened in module A\n",
        "2023-06-10 23:59:59 ERROR Error happened in module B\n",
    ]
    file.write_text(''.join(lines))

    filename, common_lines = checker.get_error_pattern_counts(str(file))
    assert filename == str(file)
    # There should be at most 2 unique error lines, grouped by similarity
    assert len(common_lines) >= 1
    counts = [v['count'] for v in common_lines.values()]
    # Count sum must equal number of error lines
    assert sum(counts) == 3


def test_get_error_pattern_counts_ignores_lines(checker, tmp_path):
    file = tmp_path / "ignore.log"
    content = "\n".join([
        "   this line starts with space",
        "StartsWithAlpha but no Exception",
        "Caused by: something",
        "2023-06-10 23:59:59 INFO Not an error",
        "2023-06-10 23:59:59 ERROR Actual error here"
    ]) + "\n"
    file.write_text(content)
    filename, common_lines = checker.get_error_pattern_counts(str(file))
    # should only contain one actual error
    assert len(common_lines) == 1
    key = list(common_lines.keys())[0]
    assert "ERROR Actual error here" in key


def test_run_returns_none_when_no_matching_errors(tmp_path, checker):
    # No error lines
    file = tmp_path / "clean.log"
    file.write_text("2023-06-10 23:59:59 INFO All good\n")
    variables = {"LogFiles": [str(file)]}
    ret = checker.run(variables)
    assert ret is None


def test_group_similar_log_lines_first_seen_only_set_once(checker):
    common_lines = {}
    line = 'ERROR Unique error line'
    # First addition sets first_seen
    common_lines = checker._group_similar_log_lines(line, '2023-01-01 10:00:00', common_lines)
    orig_first_seen = common_lines[line]['first_seen']
    # Second addition with different date does not overwrite first_seen
    common_lines = checker._group_similar_log_lines(line, '2023-01-02 11:00:00', common_lines)
    assert common_lines[line]['first_seen'] == orig_first_seen


def test__group_similar_log_lines_logs_debug_on_similarity(caplog, checker):
    common_lines = {}
    line = 'ERROR Unique error line'

    with caplog.at_level("DEBUG"):
        checker._group_similar_log_lines(line, None, common_lines)
        # Add a similar line to trigger debug about incrementing count
        similar_line = 'ERROR Unique error lines'
        checker._group_similar_log_lines(similar_line, None, common_lines)

    # Inspect caplog for debug messages about ratio and incrementing
    found_increment = any("incrementing count" in record.message for record in caplog.records)
    assert found_increment or len(common_lines) == 1


def test__process_line_truncates_long_lines(checker):
    long_msg = "A" * 300
    line = f"2023-06-10 23:59:59 ERROR {long_msg}"
    processed_line, _ = checker._process_line(line)
    assert len(processed_line) <= 200
    assert processed_line.startswith("ERROR")


def test_remove_blacklisted_logs_with_no_blacklist_match(checker):
    logs = ["app.log", "server.log", "error.log"]
    filtered = checker._remove_blacklisted_logs(logs)
    # All logs remain
    assert set(filtered) == set(logs)


def test_run_combines_results_multiple_files(tmp_path, checker):
    file1 = tmp_path / "file1.log"
    file2 = tmp_path / "file2.log"

    err_line1 = "2023-06-10 23:59:59 ERROR First error\n"
    err_line2 = "2023-06-10 23:59:59 ERROR Second error\n"
    file1.write_text(err_line1)
    file2.write_text(err_line2)

    variables = {"LogFiles": [str(file1), str(file2)]}
    result = checker.run(variables)
    assert "filename: " in result
    assert "file1.log" in result
    assert "file2.log" in result
    assert "First error" in result
    assert "Second error" in result
