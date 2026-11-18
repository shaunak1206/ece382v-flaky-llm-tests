import os
from collections import OrderedDict

import pytest

import fossor.checks.similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def __init__(self):
        self.messages = []

    def debug(self, msg):
        self.messages.append(msg)


class DummyPool:
    last_created = None

    def __init__(self, size):
        self.size = size
        self.mapped = None
        DummyPool.last_created = self

    def map(self, func, iterable):
        self.mapped = (func, list(iterable))
        return [func(item) for item in iterable]


@pytest.fixture
def checker():
    obj = SimilarLogErrors()
    obj.log = DummyLog()
    return obj


def test_remove_blacklisted_logs_removes_matching_filenames(checker):
    logs = [
        "/tmp/app.log",
        "/tmp/app_configuration.log",
        "/tmp/server.out",
        "/tmp/engine_access_log.log",
        "/tmp/error.log",
    ]
    filtered = checker._remove_blacklisted_logs(logs)
    assert filtered == ["/tmp/app.log", "/tmp/error.log"]


def test_remove_blacklisted_logs_returns_empty_when_all_blacklisted(checker):
    logs = ["a.out", "b_gc.log", "c_jetty_thread_pool_stats.log"]
    assert checker._remove_blacklisted_logs(logs) == []


def test_process_line_ignores_indented_stacktrace_line(checker):
    line, date = checker._process_line("    at package.Class.method(Class.java:1)\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_date_alpha_line_without_exception(checker):
    line, date = checker._process_line("INFO service started\n")
    assert line is None
    assert date is None


def test_process_line_ignores_caused_by_line(checker):
    line, date = checker._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_error_level_line(checker):
    line, date = checker._process_line("2024-01-01 00:00:00 WARN Something happened\n")
    assert line is None
    assert date is None


def test_process_line_parses_error_line_and_extracts_date(checker):
    line, date = checker._process_line("2024-01-01 00:00:00 ERROR Something happened\n")
    assert date == "2024-01-01 00:00:00"
    assert line == "ERROR Something happened"


def test_process_line_truncates_long_error_message(checker):
    payload = "x" * 500
    line, date = checker._process_line(f"2024-01-01 00:00:00 ERROR {payload}\n")
    assert date == "2024-01-01 00:00:00"
    assert line.startswith("ERROR ")
    assert len(line) == 200


def test_process_line_handles_too_short_error_line(checker):
    line, date = checker._process_line("2024-01-01 00:00:00 ERROR\n")
    assert date == "2024-01-01 00:00:00"
    assert line == "ERROR"


def test_group_similar_log_lines_merges_similar_lines_and_keeps_first_seen(checker):
    common = OrderedDict()
    common = checker._group_similar_log_lines(
        line="ERROR Database connection failed",
        date="2024-01-01 00:00:00",
        common_lines=common,
    )
    common = checker._group_similar_log_lines(
        line="ERROR Database connection failed!!!",
        date="2024-01-01 00:01:00",
        common_lines=common,
    )

    assert len(common) == 1
    key = next(iter(common))
    assert key == "ERROR Database connection failed"
    assert common[key]["count"] == 2
    assert common[key]["first_seen"] == "2024-01-01 00:00:00"


def test_group_similar_log_lines_stores_unique_lines_separately(checker):
    common = OrderedDict()
    common = checker._group_similar_log_lines("ERROR alpha", "d1", common)
    common = checker._group_similar_log_lines("ERROR beta", "d2", common)

    assert list(common.keys()) == ["ERROR alpha", "ERROR beta"]
    assert common["ERROR alpha"]["count"] == 1
    assert common["ERROR beta"]["count"] == 1
    assert common["ERROR alpha"]["first_seen"] == "d1"
    assert common["ERROR beta"]["first_seen"] == "d2"


def test_group_similar_log_lines_uses_other_error_lines_when_limit_reached(checker):
    common = OrderedDict()
    for i in range(checker.MAX_COMMON_LINES):
        common = checker._group_similar_log_lines(f"ERROR unique {i}", f"d{i}", common)

    common = checker._group_similar_log_lines("ERROR brand new", "d-last", common)

    assert "Other Error Lines" in common
    assert common["Other Error Lines"]["count"] == 1
    assert common["Other Error Lines"]["first_seen"] == "d-last"


def test_get_error_pattern_counts_groups_lines_from_file(tmp_path, checker):
    p = tmp_path / "app.log"
    p.write_text(
        "\n".join(
            [
                "2024-01-01 00:00:00 ERROR Database connection failed",
                "2024-01-01 00:01:00 ERROR Database connection failed!!!",
                "2024-01-01 00:02:00 WARN Not counted",
                "INFO noise",
                "    stacktrace line",
                "Caused by: java.lang.RuntimeException: boom",
            ]
        )
        + "\n"
    )

    filename, common = checker.get_error_pattern_counts(str(p))

    assert filename == str(p)
    assert list(common.keys()) == ["ERROR Database connection failed"]
    assert common["ERROR Database connection failed"]["count"] == 2
    assert common["ERROR Database connection failed"]["first_seen"] == "2024-01-01 00:00:00"


def test_run_returns_none_when_no_log_files(checker):
    assert checker.run({}) is None
    assert checker.run({"LogFiles": None}) is None
    assert checker.run({"LogFiles": []}) is None


def test_run_filters_blacklisted_files_and_formats_output(tmp_path, monkeypatch, checker):
    p1 = tmp_path / "service.log"
    p2 = tmp_path / "service.out"
    p1.write_text(
        "2024-01-01 00:00:00 ERROR Something bad happened\n"
        "2024-01-01 00:01:00 ERROR Something bad happened!!!\n"
    )
    p2.write_text("2024-01-01 00:00:00 ERROR should be ignored\n")

    monkeypatch.setattr(sle_module, "Pool", DummyPool)
    monkeypatch.setattr(sle_module, "cpu_count", lambda: 1)

    result = checker.run({"LogFiles": [str(p1), str(p2)]})

    assert DummyPool.last_created is not None
    assert DummyPool.last_created.size == 2
    assert DummyPool.last_created.mapped[1] == [str(p1)]
    assert "service.out" not in (result or "")
    assert "filename: " + str(p1) in result
    assert "Count: 2, ERROR Something bad happened" in result


def test_run_returns_none_when_all_files_produce_no_matching_errors(tmp_path, monkeypatch, checker):
    p = tmp_path / "empty.log"
    p.write_text("INFO nothing useful\n2024-01-01 00:00:00 WARN ignored\n")

    monkeypatch.setattr(sle_module, "Pool", DummyPool)
    monkeypatch.setattr(sle_module, "cpu_count", lambda: 1)

    result = checker.run({"LogFiles": [str(p)]})
    assert result is None
