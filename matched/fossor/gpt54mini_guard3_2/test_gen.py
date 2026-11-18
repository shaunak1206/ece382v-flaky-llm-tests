import os
from collections import OrderedDict

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def debug(self, *args, **kwargs):
        pass


class DummyPool:
    def __init__(self, threads):
        self.threads = threads
        self.mapped = None

    def map(self, func, iterable):
        self.mapped = list(iterable)
        return [func(item) for item in iterable]


def make_check():
    check = SimilarLogErrors()
    check.log = DummyLog()
    return check


def test_remove_blacklisted_logs_filters_known_suffixes():
    check = make_check()
    logs = [
        "/tmp/app.log",
        "/tmp/app_configuration.log",
        "/tmp/output.out",
        "/tmp/gc.log",
        "/tmp/normal.txt",
    ]
    assert check._remove_blacklisted_logs(logs) == ["/tmp/app.log", "/tmp/normal.txt"]


def test_remove_blacklisted_logs_returns_empty_when_all_blacklisted():
    check = make_check()
    logs = [
        "a_configuration.log",
        "b.out",
        "cgc.log",
        "door.log",
        "x_databus.log",
        "y_public_access.log",
        "jetty_thread_pool_stats.log",
        "engine_access_log.log",
    ]
    assert check._remove_blacklisted_logs(logs) == []


def test_process_line_ignores_indented_stacktrace_lines():
    check = make_check()
    line, date = check._process_line("    at some.package.Class.method(Class.java:1)\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_error_alpha_lines_without_exception():
    check = make_check()
    line, date = check._process_line("INFO service started successfully\n")
    assert line is None
    assert date is None


def test_process_line_ignores_caused_by_lines():
    check = make_check()
    line, date = check._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_keeps_error_lines_and_extracts_date():
    check = make_check()
    line, date = check._process_line(
        "2024-01-02 03:04:05 ERROR Something bad happened here\n"
    )
    assert date == "2024-01-02 03:04:05"
    assert line == "ERROR Something bad happened here"


def test_process_line_ignores_non_error_timestamped_lines():
    check = make_check()
    line, date = check._process_line(
        "2024-01-02 03:04:05 WARN Something bad happened here\n"
    )
    assert line is None
    assert date is None


def test_process_line_handles_short_timestamped_line_without_indexerror():
    check = make_check()
    line, date = check._process_line("2024-01-02\n")
    assert line is None
    assert date is None


def test_group_similar_log_lines_merges_above_threshold():
    check = make_check()
    common_lines = OrderedDict()
    first = "ERROR unable to connect to database"
    second = "ERROR unable to connect to databaze"
    common_lines = check._group_similar_log_lines(first, "2024-01-01 00:00:01", common_lines)
    common_lines = check._group_similar_log_lines(second, "2024-01-01 00:00:02", common_lines)

    assert list(common_lines.keys()) == [first]
    assert common_lines[first]["count"] == 2
    assert common_lines[first]["first_seen"] == "2024-01-01 00:00:01"


def test_group_similar_log_lines_keeps_distinct_below_threshold():
    check = make_check()
    common_lines = OrderedDict()
    first = "ERROR alpha alpha alpha alpha"
    second = "ERROR zulu zulu zulu zulu"
    common_lines = check._group_similar_log_lines(first, None, common_lines)
    common_lines = check._group_similar_log_lines(second, None, common_lines)

    assert list(common_lines.keys()) == [first, second]
    assert common_lines[first]["count"] == 1
    assert common_lines[second]["count"] == 1


def test_group_similar_log_lines_uses_other_error_lines_after_max_unique():
    check = make_check()
    common_lines = OrderedDict()
    for i in range(check.MAX_COMMON_LINES):
        common_lines = check._group_similar_log_lines(f"ERROR unique {i}", None, common_lines)

    common_lines = check._group_similar_log_lines("ERROR another unique line", None, common_lines)

    assert "Other Error Lines" in common_lines
    assert common_lines["Other Error Lines"]["count"] == 1
    assert len(common_lines) == check.MAX_COMMON_LINES + 1


def test_get_error_pattern_counts_groups_lines_from_file(tmp_path):
    check = make_check()
    log_file = tmp_path / "example.log"
    log_file.write_text(
        "\n".join(
            [
                "2024-01-02 03:04:05 ERROR first problem",
                "2024-01-02 03:04:06 ERROR first problem",
                "INFO ignored line",
                "    at ignored stack line",
            ]
        )
        + "\n"
    )

    filename, common_lines = check.get_error_pattern_counts(str(log_file))

    assert filename == str(log_file)
    assert list(common_lines.keys()) == ["ERROR first problem"]
    assert common_lines["ERROR first problem"]["count"] == 2
    assert common_lines["ERROR first problem"]["first_seen"] == "2024-01-02 03:04:05"


def test_run_returns_none_when_variables_missing_logfiles():
    check = make_check()
    assert check.run({}) is None
    assert check.run({"LogFiles": []}) is None
    assert check.run({"LogFiles": None}) is None


def test_run_filters_blacklisted_logs_before_processing(monkeypatch):
    import fossor.checks.similar_log_errors as module

    check = make_check()
    seen = []

    def fake_get_error_pattern_counts(filename):
        seen.append(filename)
        return filename, OrderedDict()

    monkeypatch.setattr(module, "cpu_count", lambda: 1)
    monkeypatch.setattr(module, "Pool", DummyPool)
    monkeypatch.setattr(check, "get_error_pattern_counts", fake_get_error_pattern_counts)

    logs = [
        "keep_one.log",
        "remove_configuration.log",
        "keep_two.log",
        "remove.out",
    ]
    result = check.run({"LogFiles": logs})

    assert result is None
    assert seen == ["keep_one.log", "keep_two.log"]


def test_run_formats_output_for_multiple_files(monkeypatch):
    import fossor.checks.similar_log_errors as module

    check = make_check()

    def fake_get_error_pattern_counts(filename):
        if filename == "a.log":
            return filename, OrderedDict(
                [
                    ("ERROR alpha", {"count": 2, "first_seen": "2024-01-01 00:00:01"}),
                    ("ERROR beta", {"count": 1}),
                ]
            )
        return filename, OrderedDict(
            [("ERROR gamma", {"count": 3, "first_seen": "2024-01-02 00:00:01"})]
        )

    monkeypatch.setattr(module, "cpu_count", lambda: 1)
    monkeypatch.setattr(module, "Pool", DummyPool)
    monkeypatch.setattr(check, "get_error_pattern_counts", fake_get_error_pattern_counts)

    result = check.run({"LogFiles": ["a.log", "b.log"]})

    assert result == (
        "filename: a.log\n"
        "First seen: 2024-01-01 00:00:01, Count: 2, ERROR alpha\n"
        "Count: 1, ERROR beta\n\n"
        "filename: b.log\n"
        "First seen: 2024-01-02 00:00:01, Count: 3, ERROR gamma\n\n"
    )


def test_run_returns_none_when_all_files_produce_no_common_lines(monkeypatch):
    import fossor.checks.similar_log_errors as module

    check = make_check()

    def fake_get_error_pattern_counts(filename):
        return filename, OrderedDict()

    monkeypatch.setattr(module, "cpu_count", lambda: 1)
    monkeypatch.setattr(module, "Pool", DummyPool)
    monkeypatch.setattr(check, "get_error_pattern_counts", fake_get_error_pattern_counts)

    result = check.run({"LogFiles": ["a.log", "b.log"]})

    assert result is None
