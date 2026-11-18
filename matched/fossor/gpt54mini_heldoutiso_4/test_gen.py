import builtins
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def __init__(self):
        self.messages = []

    def debug(self, msg):
        self.messages.append(msg)


class DummyPool:
    def __init__(self, threads):
        self.threads = threads
        self.mapped = None

    def map(self, func, iterable):
        self.mapped = list(iterable)
        return [func(item) for item in iterable]


@pytest.fixture
def checker():
    c = object.__new__(SimilarLogErrors)
    c.log = DummyLog()
    return c


def write_file(tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content)
    return str(path)


def test_remove_blacklisted_logs_filters_matching_names(checker):
    files = [
        "/var/log/app.log",
        "/var/log/app.out",
        "/var/log/gc.log",
        "/var/log/service.log",
        "/var/log/engine_access_log.log",
    ]
    assert checker._remove_blacklisted_logs(files) == ["/var/log/app.log", "/var/log/service.log"]


def test_remove_blacklisted_logs_keeps_non_matching_names(checker):
    files = ["/tmp/a.log", "/tmp/b.txt", "/tmp/c.err"]
    assert checker._remove_blacklisted_logs(files) == files


def test_process_line_ignores_indented_stacktrace_line(checker):
    line, date = checker._process_line("    at example.Class.method(Class.java:1)\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_exception_alpha_line(checker):
    line, date = checker._process_line("INFO Something happened\n")
    assert line is None
    assert date is None


def test_process_line_ignores_caused_by_line(checker):
    line, date = checker._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_error_level(checker):
    line, date = checker._process_line("2024-01-01 12:00:00 WARN Something happened\n")
    assert line is None
    assert date is None


def test_process_line_extracts_error_and_date(checker):
    line, date = checker._process_line("2024-01-01 12:00:00 ERROR Something bad happened\n")
    assert date == "2024-01-01 12:00:00"
    assert line == "ERROR Something bad happened"


def test_process_line_truncates_long_error_line(checker):
    message = "x" * 300
    line, date = checker._process_line(f"2024-01-01 12:00:00 ERROR {message}\n")
    assert date == "2024-01-01 12:00:00"
    assert line.startswith("ERROR ")
    assert len(line) == 200


def test_process_line_handles_short_malformed_numeric_line(checker):
    line, date = checker._process_line("2024-01-01\n")
    assert line is None
    assert date is None
    assert any("IndexError parsing line" in msg for msg in checker.log.messages)


def test_group_similar_log_lines_groups_above_threshold(checker):
    common = OrderedDict()
    common["ERROR unable to connect to database"] = {"count": 1}
    result = checker._group_similar_log_lines(
        line="ERROR unable to connect to database #2",
        date="2024-01-01 12:00:00",
        common_lines=common,
    )
    assert list(result.keys()) == ["ERROR unable to connect to database"]
    assert result["ERROR unable to connect to database"]["count"] == 2
    assert result["ERROR unable to connect to database"]["first_seen"] == "2024-01-01 12:00:00"


def test_group_similar_log_lines_adds_unique_line_with_date(checker):
    common = OrderedDict()
    result = checker._group_similar_log_lines(
        line="ERROR first unique problem",
        date="2024-01-01 12:00:00",
        common_lines=common,
    )
    assert "ERROR first unique problem" in result
    assert result["ERROR first unique problem"]["count"] == 1
    assert result["ERROR first unique problem"]["first_seen"] == "2024-01-01 12:00:00"


def test_group_similar_log_lines_uses_other_bucket_when_max_unique_exceeded(checker):
    common = OrderedDict((f"ERROR line {i}", {"count": 1}) for i in range(checker.MAX_COMMON_LINES))
    result = checker._group_similar_log_lines(
        line="ERROR a brand new unrelated line",
        date=None,
        common_lines=common,
    )
    assert "Other Error Lines" in result
    assert result["Other Error Lines"]["count"] == 1


def test_get_error_pattern_counts_reads_and_groups_lines(tmp_path, checker):
    log_path = write_file(
        tmp_path,
        "app.log",
        "\n".join(
            [
                "2024-01-01 12:00:00 ERROR unable to connect to db",
                "2024-01-01 12:01:00 ERROR unable to connect to db again",
                "INFO ignored line",
                "    at stack trace line",
            ]
        )
        + "\n",
    )
    filename, common = checker.get_error_pattern_counts(log_path)
    assert filename == log_path
    assert len(common) == 1
    key = next(iter(common.keys()))
    assert key.startswith("ERROR unable to connect to db")
    assert common[key]["count"] == 2
    assert common[key]["first_seen"] == "2024-01-01 12:00:00"


def test_run_returns_none_when_logfiles_missing(checker):
    assert checker.run({}) is None
    assert checker.run({"LogFiles": []}) is None


def test_run_filters_blacklisted_files_and_formats_output(tmp_path, checker):
    good1 = write_file(
        tmp_path,
        "service.log",
        "2024-01-01 12:00:00 ERROR first problem\n"
        "2024-01-01 12:01:00 ERROR first problem again\n",
    )
    good2 = write_file(
        tmp_path,
        "api.log",
        "2024-01-01 12:00:00 ERROR second problem\n",
    )
    blacklisted = write_file(
        tmp_path,
        "service.out",
        "2024-01-01 12:00:00 ERROR should be ignored\n",
    )

    with patch("fossor.checks.similar_log_errors.cpu_count", return_value=1), patch(
        "fossor.checks.similar_log_errors.Pool", DummyPool
    ):
        result = checker.run({"LogFiles": [good1, good2, blacklisted]})

    assert "service.out" not in result
    assert f"filename: {good1}" in result
    assert f"filename: {good2}" in result
    assert "Count: 2, First seen: 2024-01-01 12:00:00, ERROR first problem" in result
    assert "Count: 1, First seen: 2024-01-01 12:00:00, ERROR second problem" in result
