import io
from collections import OrderedDict

import pytest

import fossor.checks.similar_log_errors as sle


class DummyLog:
    def __init__(self):
        self.messages = []

    def debug(self, msg):
        self.messages.append(msg)


class DummyPool:
    def __init__(self, processes):
        self.processes = processes
        self.mapped = None

    def map(self, func, iterable):
        self.mapped = list(iterable)
        return [func(item) for item in self.mapped]


@pytest.fixture
def checker():
    c = sle.SimilarLogErrors()
    c.log = DummyLog()
    return c


def test_remove_blacklisted_logs_filters_known_patterns(checker):
    logs = [
        "/var/log/app.log",
        "/var/log/app_configuration.log",
        "/var/log/something.out",
        "/var/log/gc.log",
        "/var/log/clean.log",
    ]
    assert checker._remove_blacklisted_logs(logs) == ["/var/log/app.log", "/var/log/clean.log"]


def test_remove_blacklisted_logs_returns_empty_when_all_blacklisted(checker):
    logs = [
        "a_configuration.log",
        "x.out",
        "gc.log",
        "oor.log",
        "_databus.log",
        "_public_access.log",
        "jetty_thread_pool_stats.log",
        "engine_access_log.log",
    ]
    assert checker._remove_blacklisted_logs(logs) == []


@pytest.mark.parametrize(
    "line, expected_line, expected_date",
    [
        ("2020-01-01 10:00:00 ERROR Something bad happened\n",
         "ERROR Something bad happened",
         "2020-01-01 10:00:00"),
        ("2020-01-01 10:00:00 INFO Something bad happened\n",
         None,
         None),
        ("This is not a dated line\n", None, None),
        (" Caused by: inner exception\n", None, None),
    ],
)
def test_process_line_various_cases(checker, line, expected_line, expected_date):
    got_line, got_date = checker._process_line(line)
    assert got_line == expected_line
    assert got_date == expected_date


def test_process_line_ignores_lower_stacktrace_line(checker):
    line, date = checker._process_line("    at com.example.Class.method(Class.java:10)\n")
    assert line is None
    assert date is None


def test_process_line_truncates_long_error_lines(checker):
    long_message = "x" * 500
    line, date = checker._process_line(f"2020-01-01 10:00:00 ERROR {long_message}\n")
    assert date == "2020-01-01 10:00:00"
    assert line.startswith("ERROR ")
    assert len(line) == 200


def test_process_line_handles_short_invalid_numeric_line(checker):
    line, date = checker._process_line("2020-01-01\n")
    assert line is None
    assert date is None
    assert any("IndexError parsing line" in msg for msg in checker.log.messages)


def test_group_similar_log_lines_counts_unique_line(checker):
    common = OrderedDict()
    result = checker._group_similar_log_lines("ERROR alpha", "2020-01-01 10:00:00", common)
    assert list(result.keys()) == ["ERROR alpha"]
    assert result["ERROR alpha"]["count"] == 1
    assert result["ERROR alpha"]["first_seen"] == "2020-01-01 10:00:00"


def test_group_similar_log_lines_reuses_similar_line(checker):
    common = OrderedDict()
    common["ERROR connection failed"] = {"count": 1, "first_seen": "2020-01-01 10:00:00"}
    result = checker._group_similar_log_lines(
        "ERROR connection faild", "2020-01-01 10:01:00", common
    )
    assert list(result.keys()) == ["ERROR connection failed"]
    assert result["ERROR connection failed"]["count"] == 2
    assert result["ERROR connection failed"]["first_seen"] == "2020-01-01 10:00:00"


def test_group_similar_log_lines_groups_to_other_when_too_many_unique(checker):
    common = OrderedDict()
    for i in range(checker.MAX_COMMON_LINES):
        common[f"ERROR line {i}"] = {"count": 1}
    result = checker._group_similar_log_lines("ERROR brand new line", None, common)
    assert "Other Error Lines" in result
    assert result["Other Error Lines"]["count"] == 1


def test_group_similar_log_lines_logs_similarity_debug_message(checker):
    common = OrderedDict()
    common["ERROR connection failed"] = {"count": 1}
    checker._group_similar_log_lines("ERROR connection faild", None, common)
    assert any("incrementing count" in msg for msg in checker.log.messages)


def test_get_error_pattern_counts_reads_and_groups_lines(tmp_path, checker):
    p = tmp_path / "sample.log"
    p.write_text(
        "2020-01-01 10:00:00 ERROR First failure\n"
        "2020-01-01 10:00:01 ERROR First failure\n"
        "2020-01-01 10:00:02 ERROR Second failure\n"
        "INFO should be ignored\n"
        "    stacktrace line ignored\n"
    )
    filename, common = checker.get_error_pattern_counts(str(p))
    assert filename == str(p)
    assert common["ERROR First failure"]["count"] == 2
    assert common["ERROR First failure"]["first_seen"] == "2020-01-01 10:00:00"
    assert common["ERROR Second failure"]["count"] == 1
    assert "INFO should be ignored" not in common


def test_get_error_pattern_counts_uses_open_builtin_with_text_mode(tmp_path, monkeypatch, checker):
    p = tmp_path / "one.log"
    p.write_text("2020-01-01 10:00:00 ERROR hello\n")

    opened = []

    real_open = open

    def tracking_open(path, mode="r", *args, **kwargs):
        opened.append((path, mode))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(sle, "open", tracking_open)
    _, common = checker.get_error_pattern_counts(str(p))
    assert opened == [(str(p), "rt")]
    assert common["ERROR hello"]["count"] == 1


def test_run_returns_none_when_no_log_files(checker):
    assert checker.run({}) is None
    assert checker.run({"LogFiles": []}) is None


def test_run_filters_blacklisted_logs_and_uses_pool(monkeypatch, tmp_path, checker):
    good = tmp_path / "good.log"
    bad = tmp_path / "bad_configuration.log"
    good.write_text("2020-01-01 10:00:00 ERROR hello\n")
    bad.write_text("2020-01-01 10:00:00 ERROR should not be read\n")

    called = {}

    def fake_cpu_count():
        return 2

    class RecordingPool(DummyPool):
        def __init__(self, processes):
            super().__init__(processes)
            called["processes"] = processes

    monkeypatch.setattr(sle, "cpu_count", fake_cpu_count)
    monkeypatch.setattr(sle, "Pool", RecordingPool)

    result = checker.run({"LogFiles": [str(good), str(bad)]})
    assert called["processes"] == 4
    assert "bad_configuration.log" not in result
    assert "filename: " + str(good) in result
    assert "Count: 1" in result


def test_run_returns_formatted_output_with_first_seen_and_count(monkeypatch, tmp_path, checker):
    p = tmp_path / "log.log"
    p.write_text(
        "2020-01-01 10:00:00 ERROR alpha\n"
        "2020-01-01 10:00:01 ERROR alpha\n"
    )

    monkeypatch.setattr(sle, "cpu_count", lambda: 1)
    monkeypatch.setattr(sle, "Pool", DummyPool)

    result = checker.run({"LogFiles": [str(p)]})
    assert "filename: " + str(p) in result
    assert "First seen: 2020-01-01 10:00:00, Count: 2, ERROR alpha" in result


def test_run_returns_none_when_all_lines_ignored(monkeypatch, tmp_path, checker):
    p = tmp_path / "ignored.log"
    p.write_text(
        "INFO nothing here\n"
        "    stacktrace\n"
        "Caused by: nested issue\n"
    )

    monkeypatch.setattr(sle, "cpu_count", lambda: 1)
    monkeypatch.setattr(sle, "Pool", DummyPool)

    assert checker.run({"LogFiles": [str(p)]}) is None
