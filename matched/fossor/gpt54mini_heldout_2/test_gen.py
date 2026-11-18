import builtins
from collections import OrderedDict
from pathlib import Path

import pytest

import fossor.checks.similar_log_errors as similar_log_errors_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def __init__(self):
        self.messages = []

    def debug(self, msg):
        self.messages.append(msg)


def make_check():
    check = SimilarLogErrors()
    check.log = DummyLog()
    return check


class DummyPool:
    def __init__(self, threads):
        self.threads = threads
        self.mapped = None

    def map(self, func, iterable):
        self.mapped = (func, list(iterable))
        return [func(item) for item in iterable]


def test_remove_blacklisted_logs_removes_matching_names():
    check = make_check()
    logs = [
        "/var/log/app.log",
        "/var/log/app_configuration.log",
        "/var/log/app.out",
        "/var/log/gc.log",
        "/var/log/keep_this.log",
    ]
    assert check._remove_blacklisted_logs(logs) == ["/var/log/app.log", "/var/log/keep_this.log"]


def test_remove_blacklisted_logs_keeps_non_matching_names():
    check = make_check()
    logs = ["a.log", "b.txt", "c.err"]
    assert check._remove_blacklisted_logs(logs) == logs


def test_process_line_ignores_indented_stacktrace_line():
    check = make_check()
    line, date = check._process_line("    at com.example.SomeClass.method(SomeClass.java:10)\n")
    assert line is None
    assert date is None


def test_process_line_ignores_alpha_line_without_exception():
    check = make_check()
    line, date = check._process_line("INFO Something happened\n")
    assert line is None
    assert date is None


def test_process_line_ignores_caused_by_line():
    check = make_check()
    line, date = check._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_keeps_error_line_and_strips_date():
    check = make_check()
    line, date = check._process_line("2020-01-01 12:00:00 ERROR Something bad happened\n")
    assert date == "2020-01-01 12:00:00"
    assert line == "ERROR Something bad happened\n"


def test_process_line_truncates_long_error_message_to_200_chars():
    check = make_check()
    payload = "x" * 250
    line, date = check._process_line(f"2020-01-01 12:00:00 ERROR {payload}\n")
    assert date == "2020-01-01 12:00:00"
    assert line.startswith("ERROR ")
    assert len(line) == 200


def test_process_line_ignores_non_error_dated_line():
    check = make_check()
    line, date = check._process_line("2020-01-01 12:00:00 INFO not an error\n")
    assert line is None
    assert date is None


def test_process_line_ignores_short_dated_line_with_index_error():
    check = make_check()
    line, date = check._process_line("2020-01-01\n")
    assert line is None
    assert date is None


def test_group_similar_log_lines_groups_above_similarity_threshold():
    check = make_check()
    common = OrderedDict()
    common = check._group_similar_log_lines(
        line="ERROR database connection failed", date="2020-01-01 00:00:00", common_lines=common
    )
    common = check._group_similar_log_lines(
        line="ERROR database connection failure", date="2020-01-01 00:01:00", common_lines=common
    )

    assert list(common.keys()) == ["ERROR database connection failed"]
    assert common["ERROR database connection failed"]["count"] == 2
    assert common["ERROR database connection failed"]["first_seen"] == "2020-01-01 00:00:00"


def test_group_similar_log_lines_keeps_unique_lines_below_threshold():
    check = make_check()
    common = OrderedDict()
    common = check._group_similar_log_lines("ERROR first issue", "2020-01-01", common)
    common = check._group_similar_log_lines("ERROR very different issue", "2020-01-02", common)

    assert list(common.keys()) == ["ERROR first issue", "ERROR very different issue"]
    assert common["ERROR first issue"]["count"] == 1
    assert common["ERROR very different issue"]["count"] == 1


def test_group_similar_log_lines_uses_other_error_lines_when_max_common_lines_exceeded():
    check = make_check()
    common = OrderedDict((f"ERROR line {i}", {"count": 1}) for i in range(check.MAX_COMMON_LINES))
    common = check._group_similar_log_lines("ERROR brand new unique line", None, common)

    assert "Other Error Lines" in common
    assert common["Other Error Lines"]["count"] == 1


def test_get_error_pattern_counts_groups_lines_from_file(tmp_path):
    check = make_check()
    log_file = tmp_path / "app.log"
    log_file.write_text(
        "2020-01-01 12:00:00 ERROR database connection failed\n"
        "2020-01-01 12:01:00 ERROR database connection failure\n"
        "2020-01-01 12:02:00 INFO ignore me\n"
        "Caused by: java.lang.RuntimeException: boom\n"
    )

    filename, common_lines = check.get_error_pattern_counts(str(log_file))
    assert filename == str(log_file)
    assert list(common_lines.keys()) == ["ERROR database connection failed"]
    assert common_lines["ERROR database connection failed"]["count"] == 2
    assert common_lines["ERROR database connection failed"]["first_seen"] == "2020-01-01 12:00:00"


def test_run_returns_none_when_no_log_files():
    check = make_check()
    assert check.run({}) is None
    assert check.run({"LogFiles": None}) is None
    assert check.run({"LogFiles": []}) is None


def test_run_filters_blacklisted_files_and_formats_output(monkeypatch, tmp_path):
    check = make_check()

    log1 = tmp_path / "keep1.log"
    log2 = tmp_path / "keep2.log"
    blacklisted = tmp_path / "skip_configuration.log"
    log1.write_text("2020-01-01 12:00:00 ERROR first issue\n")
    log2.write_text("2020-01-01 12:01:00 ERROR second issue\n")
    blacklisted.write_text("2020-01-01 12:02:00 ERROR should not be read\n")

    created_pools = []

    class RecordingPool(DummyPool):
        def __init__(self, threads):
            super().__init__(threads)
            created_pools.append(self)

    monkeypatch.setattr(similar_log_errors_module, "Pool", RecordingPool)
    monkeypatch.setattr(similar_log_errors_module, "cpu_count", lambda: 1)

    result = check.run({"LogFiles": [str(log1), str(blacklisted), str(log2)]})

    assert created_pools and created_pools[0].threads == 2
    assert str(blacklisted) not in result
    assert f"filename: {log1}" in result
    assert f"filename: {log2}" in result
    assert "Count: 1, ERROR first issue" in result
    assert "Count: 1, ERROR second issue" in result
