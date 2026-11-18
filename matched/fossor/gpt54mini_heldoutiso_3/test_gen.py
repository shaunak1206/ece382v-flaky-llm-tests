import io
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyPool:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.mapped = None

    def map(self, func, iterable):
        self.mapped = (func, list(iterable))
        return [func(item) for item in iterable]


@pytest.fixture
def checker():
    c = SimilarLogErrors()
    c.log = MagicMock()
    return c


def test_remove_blacklisted_logs_removes_matching_paths(checker):
    logs = [
        "/var/log/app.log",
        "/var/log/app.out",
        "/var/log/gc.log",
        "/var/log/keep_this.log",
        "/var/log/service_configuration.log",
    ]
    assert checker._remove_blacklisted_logs(logs) == [
        "/var/log/app.log",
        "/var/log/keep_this.log",
    ]


def test_remove_blacklisted_logs_returns_empty_when_all_blacklisted(checker):
    logs = [
        "/tmp/engine_access_log.log",
        "/tmp/jetty_thread_pool_stats.log",
        "/tmp/other.out",
    ]
    assert checker._remove_blacklisted_logs(logs) == []


def test_process_line_ignores_indented_stacktrace_lines(checker):
    line, date = checker._process_line("    at com.example.Foo.bar(Foo.java:10)\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_exception_alpha_lines(checker):
    line, date = checker._process_line("INFO started service successfully\n")
    assert line is None
    assert date is None


def test_process_line_ignores_caused_by_lines(checker):
    line, date = checker._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_error_log_levels(checker):
    line, date = checker._process_line("2024-01-01 12:00:00 WARN something happened\n")
    assert line is None
    assert date is None


def test_process_line_parses_error_and_removes_timestamp(checker):
    line, date = checker._process_line("2024-01-01 12:00:00 ERROR Something bad happened\n")
    assert date == "2024-01-01 12:00:00"
    assert line == "ERROR Something bad happened"


def test_process_line_truncates_long_error_message(checker):
    payload = "x" * 300
    line, date = checker._process_line(f"2024-01-01 12:00:00 ERROR {payload}\n")
    assert date == "2024-01-01 12:00:00"
    assert line.startswith("ERROR ")
    assert len(line) == 200


def test_process_line_handles_short_malformed_timestamp_line(checker):
    line, date = checker._process_line("2024-01-01\n")
    assert line is None
    assert date is None


def test_group_similar_log_lines_counts_exact_match(checker):
    common = OrderedDict()
    common = checker._group_similar_log_lines(
        line="ERROR Something bad happened",
        date="2024-01-01 12:00:00",
        common_lines=common,
    )
    common = checker._group_similar_log_lines(
        line="ERROR Something bad happened",
        date=None,
        common_lines=common,
    )
    assert list(common.keys()) == ["ERROR Something bad happened"]
    assert common["ERROR Something bad happened"]["count"] == 2
    assert common["ERROR Something bad happened"]["first_seen"] == "2024-01-01 12:00:00"


def test_group_similar_log_lines_groups_similar_lines_under_threshold(checker):
    common = OrderedDict()
    common = checker._group_similar_log_lines(
        line="ERROR Database connection timeout while reading from replica",
        date="2024-01-01 12:00:00",
        common_lines=common,
    )
    common = checker._group_similar_log_lines(
        line="ERROR Database connection timeout while reading from primary",
        date=None,
        common_lines=common,
    )
    assert len(common) == 1
    key = next(iter(common))
    assert "ERROR Database connection timeout while reading from" in key
    assert common[key]["count"] == 2


def test_group_similar_log_lines_creates_other_error_lines_after_limit(checker):
    common = OrderedDict(
        (f"ERROR unique {i}", {"count": 1}) for i in range(checker.MAX_COMMON_LINES)
    )
    common = checker._group_similar_log_lines(
        line="ERROR a brand new unmatched message",
        date="2024-01-01 12:00:00",
        common_lines=common,
    )
    assert "Other Error Lines" in common
    assert common["Other Error Lines"]["count"] == 1
    assert common["Other Error Lines"]["first_seen"] == "2024-01-01 12:00:00"


def test_group_similar_log_lines_prefers_existing_similar_key_not_new_key(checker):
    common = OrderedDict()
    base = "ERROR Failed to connect to database host db01"
    similar = "ERROR Failed to connect to database host db02"
    common = checker._group_similar_log_lines(base, None, common)
    common = checker._group_similar_log_lines(similar, None, common)
    assert list(common.keys()) == [base]
    assert common[base]["count"] == 2


def test_get_error_pattern_counts_reads_file_and_groups(monkeypatch, checker, tmp_path):
    p = tmp_path / "app.log"
    p.write_text(
        "2024-01-01 12:00:00 ERROR Connection refused to db1\n"
        "2024-01-01 12:00:01 ERROR Connection refused to db2\n"
        "INFO this should be ignored\n"
        "    at com.example.Foo.bar(Foo.java:10)\n"
    )

    filename, common = checker.get_error_pattern_counts(str(p))
    assert filename == str(p)
    assert len(common) == 1
    key = next(iter(common))
    assert "ERROR Connection refused to db" in key
    assert common[key]["count"] == 2
    assert common[key]["first_seen"] == "2024-01-01 12:00:00"


def test_run_returns_none_when_no_log_files(checker):
    assert checker.run({}) is None
    assert checker.run({"LogFiles": []}) is None


def test_run_filters_blacklisted_logs_and_uses_pool(monkeypatch, checker, tmp_path):
    file1 = tmp_path / "a.log"
    file2 = tmp_path / "b.out"
    file3 = tmp_path / "c.log"
    file1.write_text("2024-01-01 12:00:00 ERROR First issue\n")
    file2.write_text("2024-01-01 12:00:00 ERROR Should be filtered out\n")
    file3.write_text("2024-01-01 12:00:01 ERROR Second issue\n")

    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", DummyPool)
    monkeypatch.setattr("fossor.checks.similar_log_errors.cpu_count", lambda: 1)

    result = checker.run({"LogFiles": [str(file1), str(file2), str(file3)]})
    assert "filename: " in result
    assert str(file2) not in result
    assert str(file1) in result
    assert str(file3) in result


def test_run_includes_first_seen_and_count_in_output(monkeypatch, checker, tmp_path):
    p = tmp_path / "errors.log"
    p.write_text(
        "2024-01-01 12:00:00 ERROR Disk full on node1\n"
        "2024-01-01 12:01:00 ERROR Disk full on node2\n"
    )

    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", DummyPool)
    monkeypatch.setattr("fossor.checks.similar_log_errors.cpu_count", lambda: 1)

    result = checker.run({"LogFiles": [str(p)]})
    assert "filename: " + str(p) in result
    assert "First seen: 2024-01-01 12:00:00, Count: 2," in result
    assert "ERROR Disk full on node" in result


def test_run_returns_none_when_all_logs_blacklisted(monkeypatch, checker, tmp_path):
    p = tmp_path / "engine_access_log.log"
    p.write_text("2024-01-01 12:00:00 ERROR ignored\n")

    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", DummyPool)
    monkeypatch.setattr("fossor.checks.similar_log_errors.cpu_count", lambda: 1)

    assert checker.run({"LogFiles": [str(p)]}) is None
