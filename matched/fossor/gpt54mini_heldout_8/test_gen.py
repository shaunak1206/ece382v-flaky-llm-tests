import builtins
from pathlib import Path

import pytest

import fossor.checks.similar_log_errors as slo


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
    c = slo.SimilarLogErrors()
    c.log = DummyLog()
    return c


def test_remove_blacklisted_logs_removes_matching_paths(checker):
    logs = [
        "/var/log/app.log",
        "/var/log/app_configuration.log",
        "/var/log/foo.out",
        "/var/log/gc.log",
        "/var/log/keep.me",
    ]
    result = checker._remove_blacklisted_logs(logs)
    assert result == ["/var/log/app.log", "/var/log/keep.me"]


def test_remove_blacklisted_logs_keeps_non_matching_paths(checker):
    logs = ["/tmp/a.log", "/tmp/b.txt", "/tmp/c.err"]
    assert checker._remove_blacklisted_logs(logs) == logs


def test_process_line_ignores_indented_stacktrace_line(checker):
    line, date = checker._process_line("    at some.package.Class.method(Class.java:12)\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_exception_alpha_line(checker):
    line, date = checker._process_line("INFO starting up service\n")
    assert line is None
    assert date is None


def test_process_line_ignores_caused_by_line(checker):
    line, date = checker._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_error_level(checker):
    line, date = checker._process_line("2020-01-01 12:00:00 WARN something happened\n")
    assert line is None
    assert date is None


def test_process_line_extracts_date_and_truncates_error_line(checker):
    long_msg = "x" * 500
    raw = f"2020-01-01 12:00:00 ERROR {long_msg}\n"
    line, date = checker._process_line(raw)
    assert date == "2020-01-01 12:00:00"
    assert line.startswith("ERROR ")
    assert len(line) <= 200
    assert line == f"ERROR {long_msg}"[:200]


def test_process_line_handles_short_error_line(checker):
    raw = "2020-01-01 12:00:00 ERROR short message\n"
    line, date = checker._process_line(raw)
    assert date == "2020-01-01 12:00:00"
    assert line == "ERROR short message"


def test_process_line_logs_index_error_for_malformed_line(checker):
    line, date = checker._process_line("2020-01-01\n")
    assert line is None
    assert date is None
    assert any("IndexError parsing line" in msg for msg in checker.log.messages)


def test_group_similar_log_lines_counts_exact_duplicates(checker):
    common_lines = {}
    common_lines = checker._group_similar_log_lines(
        line="ERROR database unavailable",
        date="2020-01-01 12:00:00",
        common_lines=common_lines,
    )
    common_lines = checker._group_similar_log_lines(
        line="ERROR database unavailable",
        date="2020-01-01 12:01:00",
        common_lines=common_lines,
    )
    assert list(common_lines.keys()) == ["ERROR database unavailable"]
    assert common_lines["ERROR database unavailable"]["count"] == 2
    assert common_lines["ERROR database unavailable"]["first_seen"] == "2020-01-01 12:00:00"


def test_group_similar_log_lines_groups_similar_lines(checker):
    common_lines = {}
    line1 = "ERROR failed to connect to database host=db1 port=5432"
    line2 = "ERROR failed to connect to database host=db2 port=5432"
    common_lines = checker._group_similar_log_lines(line1, None, common_lines)
    common_lines = checker._group_similar_log_lines(line2, None, common_lines)
    assert len(common_lines) == 1
    key = next(iter(common_lines))
    assert common_lines[key]["count"] == 2
    assert "failed to connect to database" in key


def test_group_similar_log_lines_creates_other_error_lines_bucket_when_limit_reached(checker):
    checker.MAX_COMMON_LINES = 1
    common_lines = {}
    common_lines = checker._group_similar_log_lines("ERROR first unique", None, common_lines)
    common_lines = checker._group_similar_log_lines("ERROR second unique", None, common_lines)
    assert "Other Error Lines" in common_lines
    assert common_lines["Other Error Lines"]["count"] == 1
    assert common_lines["ERROR first unique"]["count"] == 1


def test_get_error_pattern_counts_reads_and_groups_lines(tmp_path, checker):
    log_file = tmp_path / "app.log"
    log_file.write_text(
        "\n".join(
            [
                "2020-01-01 12:00:00 ERROR failed to connect to database host=db1",
                "2020-01-01 12:01:00 ERROR failed to connect to database host=db2",
                "2020-01-01 12:02:00 WARN ignored warning",
                "INFO ignored info",
                "    at stack.trace.Line",
            ]
        )
        + "\n"
    )

    filename, common_lines = checker.get_error_pattern_counts(str(log_file))
    assert filename == str(log_file)
    assert len(common_lines) == 1
    key = next(iter(common_lines))
    assert common_lines[key]["count"] == 2
    assert common_lines[key]["first_seen"] == "2020-01-01 12:00:00"


def test_run_returns_none_when_no_log_files(checker):
    assert checker.run({}) is None
    assert checker.run({"LogFiles": []}) is None


def test_run_formats_results_and_uses_blacklist_and_pool(monkeypatch, tmp_path, checker):
    monkeypatch.setattr(slo, "cpu_count", lambda: 1)
    monkeypatch.setattr(slo, "Pool", DummyPool)

    good_log = tmp_path / "good.log"
    bad_log = tmp_path / "app.out"
    good_log.write_text(
        "2020-01-01 12:00:00 ERROR one problem\n"
        "2020-01-01 12:01:00 ERROR one problem\n"
    )
    bad_log.write_text("2020-01-01 12:00:00 ERROR should be ignored\n")

    result = checker.run({"LogFiles": [str(good_log), str(bad_log)]})
    assert "good.log" in result
    assert "app.out" not in result
    assert "Count: 2" in result
    assert "filename: " in result


def test_run_returns_none_when_all_logs_are_blacklisted(monkeypatch, tmp_path, checker):
    monkeypatch.setattr(slo, "cpu_count", lambda: 1)
    monkeypatch.setattr(slo, "Pool", DummyPool)

    bad_log1 = tmp_path / "x_configuration.log"
    bad_log2 = tmp_path / "y.out"
    bad_log1.write_text("2020-01-01 12:00:00 ERROR ignored\n")
    bad_log2.write_text("2020-01-01 12:00:00 ERROR ignored\n")

    assert checker.run({"LogFiles": [str(bad_log1), str(bad_log2)]}) is None
