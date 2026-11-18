import os
import tempfile
import pytest
from collections import OrderedDict
from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def debug(self, msg):
        pass


@pytest.fixture
def instance():
    chk = SimilarLogErrors()
    chk.log = DummyLog()
    return chk


def test_remove_blacklisted_logs_removes_all_blacklisted(instance):
    logs = [
        "app_configuration.log",
        "normal.log",
        "something.out",
        "gc.log",
        "error_oor.log",
        "nolonger_public_access.log",
        "jetty_thread_pool_stats.log",
        "engine_access_log.log",
        "useful.log",
        "other_databus.log",
    ]
    filtered = instance._remove_blacklisted_logs(logs)
    # All logs containing any blacklisted string removed
    assert "normal.log" in filtered
    assert "useful.log" in filtered
    for bl in instance.log_name_blacklist:
        for f in filtered:
            assert bl not in f


def test_remove_blacklisted_logs_empty_list(instance):
    assert instance._remove_blacklisted_logs([]) == []


def test_process_line_non_error_lines(instance):
    # Strings that should cause line to be None
    cases = [
        (" stacktrace line with leading space", None),
        ("This is a random info message", None),
        ("SomeException: error occurred", ("SomeException: error occurred", None)),
        ("Caused by: something", None),
        ("2020-10-10 12:10 WARN something happened", None),
        ("2022-01-01 00:00 ERROR actual error here", ("ERROR actual error here", "2022-01-01 00:00")),
    ]
    for line, expected in cases:
        result = instance._process_line(line)
        if expected is None:
            assert result == (None, None)
        else:
            assert result == expected


def test_process_line_truncation(instance):
    # Create a very long error line to test truncation to max 200 chars
    date = "2023-01-01 12:00"
    level = "ERROR"
    long_msg = "A" * 250
    line = f"{date} {level} {long_msg}"
    processed_line, date_out = instance._process_line(line)
    assert len(processed_line) <= 200
    assert processed_line.startswith("ERROR " + "A" * (200 - 6))  # "ERROR " is 6 chars
    assert date_out == date


def test_group_similar_log_lines_counts_and_groups(instance):
    common_lines = OrderedDict()
    # Add a very different line
    common_lines["Line one error"] = {"count": 3}

    new_line_same = "Line one error"
    new_line_similar = "Line one error!"
    new_line_unique = "Completely different error"

    # Same line increments count
    cl = instance._group_similar_log_lines(new_line_same, "2020-01-01 00:00", common_lines)
    assert cl["Line one error"]["count"] == 4
    assert cl["Line one error"]["first_seen"] == "2020-01-01 00:00"

    # Similar line (above ratio threshold) increments the same line count
    count_before = cl["Line one error"]["count"]
    cl = instance._group_similar_log_lines(new_line_similar, "2020-01-02 00:00", common_lines)
    assert cl["Line one error"]["count"] == count_before + 1
    # first_seen does not change after first set
    assert cl["Line one error"]["first_seen"] == "2020-01-01 00:00"

    # Add up to max common lines
    for i in range(instance.MAX_COMMON_LINES - len(cl)):
        cl[f"Unique line {i}"] = {"count": 1}

    # Now add another unique line, should be grouped as 'Other Error Lines'
    cl2 = instance._group_similar_log_lines(new_line_unique, "2020-01-03 00:00", cl)
    assert "Other Error Lines" in cl2
    assert cl2["Other Error Lines"]["count"] >= 1


def test_get_error_pattern_counts_groups_lines(tmp_path, instance):
    # Create a temp log file with lines: some ERROR lines, some ignored
    log_content = (
        "2023-01-01 12:00 ERROR Error number one\n"
        "2023-01-01 12:01 ERROR Error number one\n"
        "2023-01-01 12:02 INFO This is info\n"
        "Caused by: exception trace\n"
        "2023-01-01 12:03 ERROR Error number two\n"
    )
    log_file = tmp_path / "test.log"
    log_file.write_text(log_content)

    filename, common_lines = instance.get_error_pattern_counts(str(log_file))
    assert filename == str(log_file)
    # Should contain lines for both error 1 and error 2
    keys = list(common_lines.keys())
    assert any("Error number one" in k for k in keys)
    assert any("Error number two" in k for k in keys)
    # Count of error number one lines is 2
    val = next(v['count'] for k, v in common_lines.items() if "Error number one" in k)
    assert val == 2
    # Count of error number two lines is 1
    val2 = next(v['count'] for k, v in common_lines.items() if "Error number two" in k)
    assert val2 == 1


def test_run_returns_none_if_no_log_files(instance):
    res = instance.run({})
    assert res is None
    res = instance.run({"LogFiles": []})
    assert res is None
    res = instance.run({"LogFiles": None})
    assert res is None


def test_run_filters_blacklisted_and_returns_result(tmp_path, monkeypatch, instance):
    # Prepare two log files, one blacklisted, one not
    allowed_log = tmp_path / "allowed.log"
    blacklisted_log = tmp_path / "gc.log"
    allowed_log.write_text("2023-01-01 12:00 ERROR Allowed error\n")
    blacklisted_log.write_text("2023-01-01 12:00 ERROR Blacklisted error\n")

    input_files = [str(allowed_log), str(blacklisted_log)]

    # Override Pool to map method to just call get_error_pattern_counts directly to avoid multiprocessing in test
    class DummyPool:
        def __init__(self, n): pass

        def map(self, fn, items):
            return [fn(item) for item in items]

        def close(self): pass

        def join(self): pass

    monkeypatch.setattr(instance, "MAX_COMMON_LINES", 5)
    monkeypatch.setattr(instance, "SIMILARITY_RATIO", 0.7)
    monkeypatch.setattr(instance, "log_name_blacklist", ['gc.log'])

    # monkeypatch Pool in the module
    import fossor.checks.similar_log_errors as similar_mod
    monkeypatch.setattr(similar_mod, "Pool", DummyPool)

    result = instance.run({"LogFiles": input_files})
    # Should not contain blacklisted_log filename
    assert "gc.log" not in result
    # Should contain allowed.log filename and error details
    assert "allowed.log" in result
    assert "Allowed error" in result


def test_process_line_handles_index_error(monkeypatch, instance):
    # Line that causes IndexError in _process_line (less than 3 tokens)
    line = "2023-01-01 12:00"  # only 2 tokens, no level at index 2

    # Patch log.debug to record debug messages
    debug_calls = []

    class LogWithDebugCapture:
        def debug(self, msg):
            debug_calls.append(msg)

    instance.log = LogWithDebugCapture()
    result = instance._process_line(line)
    assert result == (None, None)
    # Check debug message recorded includes IndexError parsing line
    assert any("IndexError parsing line" in m for m in debug_calls)


def test_group_similar_log_lines_similar_and_different(instance):
    common_lines = OrderedDict()
    # first line
    common_lines["ERR1 occurred"] = {"count": 1}

    # A very similar line (above threshold)
    similar_line = "ERR1 occurre"
    cl = instance._group_similar_log_lines(similar_line, None, common_lines)
    # Should increment count of original key, not add a new key
    assert len(cl) == 1
    assert cl["ERR1 occurred"]["count"] == 2

    # A different line under threshold
    diff_line = "Different error"
    cl2 = instance._group_similar_log_lines(diff_line, "2023-01-01", cl)
    # A new key added for different line
    assert len(cl2) == 2
    assert "Different error" in cl2
    # first_seen date set for new line
    assert cl2["Different error"]["first_seen"] == "2023-01-01"

    # Add lines to max limit, then add another, triggers 'Other Error Lines'
    for i in range(instance.MAX_COMMON_LINES - len(cl2)):
        cl2[f"Error {i}"] = {"count": 1}
    cl3 = instance._group_similar_log_lines("New unique error", None, cl2)
    assert "Other Error Lines" in cl3


def test_run_returns_none_if_result_empty(monkeypatch, instance):
    # If all logs removed by blacklist, run should return None
    monkeypatch.setattr(instance, "log_name_blacklist", ["allowed.log"])
    result = instance.run({"LogFiles": ["allowed.log"]})
    assert result is None


def test_run_output_format(tmp_path, monkeypatch, instance):
    log_file = tmp_path / "file.log"
    log_file.write_text(
        "2023-01-01 00:00 ERROR This is an error line\n"
        "2023-01-01 00:01 ERROR This is another error\n"
        "2023-01-01 00:02 ERROR This is an error line\n"
    )

    # Monkeypatch Pool to avoid multiprocessing
    class DummyPool:
        def __init__(self, n): pass

        def map(self, fn, items):
            return [fn(item) for item in items]

    import fossor.checks.similar_log_errors as similar_mod
    monkeypatch.setattr(similar_mod, "Pool", DummyPool)

    res = instance.run({"LogFiles": [str(log_file)]})
    assert res.startswith("filename:")
    assert str(log_file) in res
    assert "Count:" in res
    # Lines in output are trimmed lines, e.g. "ERROR This is an error line" without date


def test_get_error_pattern_counts_empty_file(tmp_path, instance):
    log_file = tmp_path / "empty.log"
    log_file.write_text("")
    filename, common_lines = instance.get_error_pattern_counts(str(log_file))
    assert filename == str(log_file)
    assert isinstance(common_lines, dict)
    assert len(common_lines) == 0


def test_process_line_line_starting_with_alpha_and_contains_exception(instance):
    line = "MyException: something went wrong"
    result = instance._process_line(line)
    assert result == (line, None)


def test_process_line_line_starting_with_alpha_and_no_exception(instance):
    line = "Startup complete"
    result = instance._process_line(line)
    assert result == (None, None)


def test_group_similar_log_lines_first_seen_set_once(instance):
    common_lines = OrderedDict()
    common_lines["Error line"] = {}
    # First time add count with date
    cl = instance._group_similar_log_lines("Error line", "2022-06-01 12:00", common_lines)
    # first_seen should be set
    assert cl["Error line"]["first_seen"] == "2022-06-01 12:00"
    # Add again with a different date, first_seen should not change
    cl2 = instance._group_similar_log_lines("Error line", "2023-07-01 12:00", cl)
    assert cl2["Error line"]["first_seen"] == "2022-06-01 12:00"
    # Count incremented
    assert cl2["Error line"]["count"] == 2


def test_run_with_multiple_log_files(tmp_path, monkeypatch, instance):
    file1 = tmp_path / "log1.log"
    file1.write_text("2023-01-01 00:00 ERROR Something failed\n")
    file2 = tmp_path / "log2.log"
    file2.write_text("2023-01-02 00:00 ERROR Another failure\n")

    files = [str(file1), str(file2)]

    class DummyPool:
        def __init__(self, n): pass

        def map(self, fn, items):
            return [fn(item) for item in items]

    import fossor.checks.similar_log_errors as similar_mod
    monkeypatch.setattr(similar_mod, "Pool", DummyPool)

    res = instance.run({"LogFiles": files})
    assert "log1.log" in res
    assert "log2.log" in res
    assert "Something failed" in res or "Another failure" in res


def test_remove_blacklisted_logs_preserves_order(instance):
    logs = [
        "a_normal.log",
        "b.gc.log",
        "c_something.log",
        "d_oor.log",
        "e_good.log",
    ]
    filtered = instance._remove_blacklisted_logs(logs)
    # Should preserve order for non-blacklisted only
    expected = ["a_normal.log", "c_something.log", "e_good.log"]
    assert filtered == expected


def test_group_similar_log_lines_empty_common_lines_first_entry(instance):
    common_lines = OrderedDict()
    cl = instance._group_similar_log_lines("Unique error line", "2021-05-05 00:00", common_lines)
    assert "Unique error line" in cl
    assert cl["Unique error line"]["count"] == 1
    assert cl["Unique error line"]["first_seen"] == "2021-05-05 00:00"
