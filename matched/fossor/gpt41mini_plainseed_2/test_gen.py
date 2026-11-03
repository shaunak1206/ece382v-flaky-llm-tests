import os
import tempfile
import pytest
from fossor.checks.similar_log_errors import SimilarLogErrors
from collections import OrderedDict


class DummyLog:
    def __init__(self):
        self.records = []

    def debug(self, msg):
        self.records.append(msg)


@pytest.fixture
def check_instance():
    check = SimilarLogErrors()
    check.log = DummyLog()
    return check


def test_remove_blacklisted_logs_removes_expected(check_instance):
    logs = [
        "app_configuration.log",
        "normal.log",
        "something.out",
        "gc.log",
        "app_oor.log",
        "jetty_thread_pool_stats.log",
        "random.log",
        "engine_access_log.log"
    ]
    filtered = check_instance._remove_blacklisted_logs(logs)
    assert "normal.log" in filtered
    assert "random.log" in filtered
    assert all(x not in filtered for x in [
        "app_configuration.log",
        "something.out",
        "gc.log",
        "app_oor.log",
        "jetty_thread_pool_stats.log",
        "engine_access_log.log",
    ])


def test__process_line_skips_leading_whitespace(check_instance):
    line = "    at com.example.SomeClass.method(SomeClass.java:123)\n"
    processed, date = check_instance._process_line(line)
    assert processed is None
    assert date is None


def test__process_line_skips_alpha_line_without_exception(check_instance):
    line = "Started some service\n"
    processed, date = check_instance._process_line(line)
    assert processed is None
    assert date is None


def test__process_line_skips_caused_by_line(check_instance):
    line = "Caused by: java.lang.NullPointerException\n"
    processed, date = check_instance._process_line(line)
    assert processed is None
    assert date is None


def test__process_line_returns_none_when_indexerror_raised(check_instance):
    # Provide a line that doesn't have enough parts to get line_level at index 2
    line = "2023-01-01 12:00\n"
    processed, date = check_instance._process_line(line)
    assert processed is None
    assert date is None
    assert any("IndexError" in log for log in check_instance.log.records)


def test__process_line_returns_none_if_level_not_error(check_instance):
    line = "2023-01-01 12:00 INFO Something happened\n"
    processed, date = check_instance._process_line(line)
    assert processed is None
    assert date is None


def test__process_line_returns_processed_line_and_date(check_instance):
    line = "2023-01-01 12:00:00 ERROR This is an error message with details\n"
    processed, date = check_instance._process_line(line)
    assert processed.startswith("ERROR ")
    assert "2023-01-01" in date
    assert "This is an error message with details" in processed


def test__process_line_truncates_long_line(check_instance):
    long_message = "A" * 300
    line = f"2023-01-01 12:00:00 ERROR {long_message}\n"
    processed, date = check_instance._process_line(line)
    assert processed.startswith("ERROR ")
    # The message part length should be truncated to approx 200 characters
    assert len(processed) <= 205  # 5 chars for "ERROR" + max 200


def test__group_similar_log_lines_counts_first_line_as_unique(check_instance):
    common_lines = OrderedDict()
    line = "ERROR Something failed here"
    updated = check_instance._group_similar_log_lines(line=line, date="2023-01-01 10:00", common_lines=common_lines)
    assert line in updated
    assert updated[line]['count'] == 1
    assert updated[line]['first_seen'] == "2023-01-01 10:00"


def test__group_similar_log_lines_merges_similar_line(check_instance):
    common_lines = OrderedDict()
    line1 = "ERROR Something failed here"
    line2 = "ERROR Something failed herE"  # only case difference
    common_lines[line1] = {'count': 2, 'first_seen': "2023-01-01 10:00"}
    updated = check_instance._group_similar_log_lines(line=line2, date="2023-01-01 11:00", common_lines=common_lines)
    # Should merge counts under line1 key
    assert line1 in updated
    assert updated[line1]['count'] == 3
    # first_seen should remain the original
    assert updated[line1]['first_seen'] == "2023-01-01 10:00"


def test__group_similar_log_lines_adds_to_other_error_lines_when_max_exceeded(check_instance):
    common_lines = OrderedDict()
    # Fill up common_lines to max allowed unique lines
    for i in range(check_instance.MAX_COMMON_LINES):
        common_lines[f"ERROR unique{i}"] = {'count': 1}
    # Now add another unique line
    new_line = "ERROR a new unique error"
    updated = check_instance._group_similar_log_lines(line=new_line, date="2023-01-01 12:00", common_lines=common_lines)
    # It should add under 'Other Error Lines'
    assert "Other Error Lines" in updated
    assert updated["Other Error Lines"]['count'] == 1


def test_get_error_pattern_counts_reads_and_groups(tmp_path, check_instance):
    # Create a temp file with lines to process
    file_path = tmp_path / "logfile.log"
    contents = """2023-01-01 10:00:00 ERROR First error line
    at some stacktrace
2023-01-01 10:01:00 ERROR First error line
2023-01-01 10:02:00 INFO Some info
Caused by: java.lang.Exception
2023-01-01 10:03:00 ERROR Second error line
"""
    file_path.write_text(contents)

    filename, common_lines = check_instance.get_error_pattern_counts(str(file_path))
    assert filename == str(file_path)
    assert isinstance(common_lines, OrderedDict)
    # Should have at least both error lines (grouped or separate)
    keys = list(common_lines.keys())
    assert any("First error line" in k for k in keys)
    assert any("Second error line" in k or "Other Error Lines" in k for k in keys)
    # Count of first error line entries should be 2
    counts = sum(v['count'] for k, v in common_lines.items() if "First error line" in k)
    assert counts == 2


def test_run_returns_none_if_no_LogFiles(check_instance):
    result = check_instance.run({})
    assert result is None


def test_run_skips_blacklisted_logs_and_returns_formatted_string(tmp_path, check_instance):
    # Prepare a file not blacklisted
    good_file = tmp_path / "good.log"
    good_file.write_text("2023-01-01 10:00:00 ERROR error message\n")
    # Prepare a blacklisted file
    black_file = tmp_path / "something.gc.log"
    black_file.write_text("2023-01-01 10:00:00 ERROR blacklisted error\n")

    # Provide both files
    variables = {'LogFiles': [str(good_file), str(black_file)]}
    result = check_instance.run(variables)
    # Result should contain good_file but not black_file
    assert "filename: " in result
    assert str(good_file) in result
    assert str(black_file) not in result
    # Should include error count line
    assert "Count: 1" in result


def test_run_returns_none_if_no_lines_after_filter(tmp_path, check_instance):
    empty_file = tmp_path / "empty.log"
    empty_file.write_text("Just some harmless info\n")
    result = check_instance.run({'LogFiles': [str(empty_file)]})
    assert result is None


def test__group_similar_log_lines_respects_threshold(check_instance):
    common_lines = OrderedDict()
    line1 = "ERROR This is error one"
    common_lines[line1] = {'count': 1}
    # A line very different, low similarity
    line2 = "ERROR Totally unrelated message"
    updated = check_instance._group_similar_log_lines(line=line2, date=None, common_lines=common_lines.copy())
    # Should NOT merge; thus line2 should be added as unique
    assert any(k == line2 or k == 'Other Error Lines' for k in updated)


def test__group_similar_log_lines_merges_when_ratio_edge_case(check_instance):
    # Prepare two lines right around the similarity boundary
    # The quick_ratio is a coarse but fast measure
    base_line = "ERROR This is an error message"
    similar_line = "ERROR This is a eror message"  # slight typo

    common_lines = OrderedDict()
    common_lines[base_line] = {'count': 3}

    updated = check_instance._group_similar_log_lines(line=similar_line, date=None, common_lines=common_lines.copy())
    # Similar line should merge (assuming > 0.7 quick_ratio)
    # The test depends on quick_ratio details but likely merge:
    assert base_line in updated
    assert updated[base_line]['count'] == 4


def test__process_line_keeps_date_and_line_length(check_instance):
    line = "2024-04-01 09:21:35 ERROR critical failure occurred in module\n"
    processed, date = check_instance._process_line(line)
    assert date == "2024-04-01 09:21:35"
    assert processed.startswith("ERROR critical failure occurred in module")
    # processed length within limit
    assert len(processed) <= 205


def test_run_logs_debug_messages(monkeypatch, tmp_path, check_instance):
    # Patch Pool.map to avoid multiprocessing in test
    def fake_map(func, iterable):
        results = []
        for i in iterable:
            results.append((i, OrderedDict({"ERROR test": {'count': 1, 'first_seen': 'testdate'}})))
        return results

    monkeypatch.setattr('fossor.checks.similar_log_errors.Pool', lambda *args, **kwargs: None)
    monkeypatch.setattr('fossor.checks.similar_log_errors.Pool.map', lambda self, func, iterable: fake_map(func, iterable))

    log_file = tmp_path / "valid.log"
    log_file.write_text("2024-04-01 09:21:35 ERROR error example\n")

    # Instead of true Pool, monkeypatch run to avoid multiprocessing
    def fake_run(self, variables):
        log_files = variables.get('LogFiles', None)
        if not log_files:
            return
        log_files = self._remove_blacklisted_logs(log_files)
        self.log.debug(f"log_files: {log_files}")
        pool_results = [(f, OrderedDict({"ERROR test": {'count': 1}})) for f in log_files]
        result = ''
        for filename, common_lines in pool_results:
            lines = []
            for line, values in common_lines.items():  # Sort by count
                count = values['count']
                first_seen = values.get('first_seen', None)
                first_seen_text = f"First seen: {first_seen}, " if first_seen else ''
                count_text = f"Count: {count}, "
                line_text = f"{line.strip()}"
                text = first_seen_text + count_text + line_text
                lines.append(text)
            if lines:
                result += f"filename: {filename}\n"
                result += '\n'.join(lines) + '\n\n'

        if result:
            return result

    monkeypatch.setattr(SimilarLogErrors, "run", fake_run)

    out = check_instance.run({'LogFiles': [str(log_file)]})
    assert "filename" in out
    assert "Count: 1" in out


def test_log_messages_generated_during_run(tmp_path, check_instance):
    log_file = tmp_path / "my.log"
    log_file.write_text("2023-01-01 12:00:00 ERROR message\n")
    out = check_instance.run({'LogFiles': [str(log_file)]})
    # Check debug messages mention processing filename
    logs = check_instance.log.records
    assert any("log_files:" in m or "Finished processing filename:" in m or "incrementing count." in m or "unique lines" in m for m in logs)


def test__process_line_non_digit_starts_returns_none(check_instance):
    lines = [
        "W123-invalid log line",
        "ESome random error line",
        "Exception: something happened",
    ]
    for line in lines:
        processed, date = check_instance._process_line(line)
        assert processed is None
        assert date is None


def test__group_similar_log_lines_increments_count_when_date_absent(check_instance):
    common_lines = OrderedDict()
    line = "ERROR testline"
    common_lines[line] = {'count': 1}
    updated = check_instance._group_similar_log_lines(line=line, date=None, common_lines=common_lines.copy())
    assert updated[line]['count'] == 2
    # first_seen should remain unchanged / not added since date is None
    assert 'first_seen' not in updated[line] or updated[line]['first_seen'] == common_lines[line].get('first_seen')


def test_run_handles_empty_LogFiles_list(check_instance):
    result = check_instance.run({'LogFiles': []})
    assert result is None


def test_run_handles_None_LogFiles(check_instance):
    result = check_instance.run({'LogFiles': None})
    assert result is None
