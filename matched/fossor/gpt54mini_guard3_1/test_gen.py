import os
from collections import OrderedDict

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def __init__(self):
        self.messages = []

    def debug(self, msg):
        self.messages.append(msg)


class FakePool:
    created_with = None
    map_called_with = None

    def __init__(self, size):
        FakePool.created_with = size

    def map(self, func, iterable):
        FakePool.map_called_with = (func, list(iterable))
        return [func(item) for item in iterable]


def make_check():
    chk = SimilarLogErrors()
    chk.log = DummyLog()
    return chk


def write_log(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("".join(lines), encoding="utf-8")
    return str(path)


def test_remove_blacklisted_logs_filters_all_blacklisted_entries():
    chk = make_check()
    logs = [
        "/var/log/app.log",
        "/var/log/app_configuration.log",
        "/var/log/app.out",
        "/var/log/gc.log",
        "/var/log/engine_access_log.log",
        "/var/log/keep.me",
    ]
    assert chk._remove_blacklisted_logs(logs) == ["/var/log/app.log", "/var/log/keep.me"]


def test_remove_blacklisted_logs_keeps_logs_without_blacklisted_substrings():
    chk = make_check()
    logs = ["/var/log/error.log", "/var/log/access.log", "/tmp/other.txt"]
    assert chk._remove_blacklisted_logs(logs) == logs


def test_process_line_ignores_indented_stacktrace_lines():
    chk = make_check()
    line, date = chk._process_line("    at com.example.Foo.bar(Foo.java:1)\n")
    assert line is None
    assert date is None


def test_process_line_ignores_alpha_lines_without_exception():
    chk = make_check()
    line, date = chk._process_line("INFO something happened\n")
    assert line is None
    assert date is None


def test_process_line_ignores_caused_by_lines():
    chk = make_check()
    line, date = chk._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_error_level_lines():
    chk = make_check()
    line, date = chk._process_line("2020-01-01 12:00:00 WARN something happened\n")
    assert line is None
    assert date is None


def test_process_line_handles_malformed_date_line_without_crashing():
    chk = make_check()
    line, date = chk._process_line("2020-01-01\n")
    assert line is None
    assert date is None


def test_process_line_parses_error_line_and_returns_timestamp():
    chk = make_check()
    line, date = chk._process_line(
        "2020-01-01 12:00:00 ERROR Something bad happened here\n"
    )
    assert date == "2020-01-01 12:00:00"
    assert line == "ERROR Something bad happened here"


def test_process_line_truncates_long_error_lines():
    chk = make_check()
    payload = "x" * 300
    line, date = chk._process_line(f"2020-01-01 12:00:00 ERROR {payload}\n")
    assert date == "2020-01-01 12:00:00"
    assert len(line) == 200
    assert line.startswith("ERROR ")
    assert line.endswith("x")


def test_group_similar_log_lines_merges_similar_lines_and_keeps_first_seen():
    chk = make_check()
    common_lines = OrderedDict()
    common_lines = chk._group_similar_log_lines(
        line="ERROR connection timeout to host A",
        date="2020-01-01 12:00:00",
        common_lines=common_lines,
    )
    common_lines = chk._group_similar_log_lines(
        line="ERROR connection timeout to host B",
        date="2020-01-01 12:01:00",
        common_lines=common_lines,
    )

    assert list(common_lines.keys()) == ["ERROR connection timeout to host A"]
    values = common_lines["ERROR connection timeout to host A"]
    assert values["count"] == 2
    assert values["first_seen"] == "2020-01-01 12:00:00"


def test_group_similar_log_lines_adds_unique_line_when_below_threshold():
    chk = make_check()
    common_lines = OrderedDict()
    common_lines = chk._group_similar_log_lines(
        line="ERROR first unique line",
        date="2020-01-01 12:00:00",
        common_lines=common_lines,
    )
    common_lines = chk._group_similar_log_lines(
        line="ERROR second unique line",
        date="2020-01-01 12:01:00",
        common_lines=common_lines,
    )

    assert len(common_lines) == 2
    assert common_lines["ERROR first unique line"]["count"] == 1
    assert common_lines["ERROR second unique line"]["count"] == 1
    assert common_lines["ERROR first unique line"]["first_seen"] == "2020-01-01 12:00:00"
    assert common_lines["ERROR second unique line"]["first_seen"] == "2020-01-01 12:01:00"


def test_group_similar_log_lines_routes_extra_unique_lines_to_other_error_lines():
    chk = make_check()
    common_lines = OrderedDict()

    for idx in range(chk.MAX_COMMON_LINES):
        common_lines = chk._group_similar_log_lines(
            line=f"ERROR unique line {idx}",
            date=None,
            common_lines=common_lines,
        )

    common_lines = chk._group_similar_log_lines(
        line="ERROR one more unique line",
        date=None,
        common_lines=common_lines,
    )

    assert "Other Error Lines" in common_lines
    assert common_lines["Other Error Lines"]["count"] == 1
    assert len(common_lines) == chk.MAX_COMMON_LINES + 1


def test_get_error_pattern_counts_groups_lines_from_file(tmp_path):
    chk = make_check()
    log_path = write_log(
        tmp_path,
        "app.log",
        [
            "2020-01-01 12:00:00 ERROR connection timeout to host A\n",
            "2020-01-01 12:01:00 ERROR connection timeout to host B\n",
            "2020-01-01 12:02:00 ERROR database unavailable\n",
            "INFO ignored line\n",
        ],
    )

    filename, common_lines = chk.get_error_pattern_counts(log_path)

    assert filename == log_path
    assert list(common_lines.keys()) == [
        "ERROR connection timeout to host A",
        "ERROR database unavailable",
    ]
    assert common_lines["ERROR connection timeout to host A"]["count"] == 2
    assert common_lines["ERROR connection timeout to host A"]["first_seen"] == "2020-01-01 12:00:00"
    assert common_lines["ERROR database unavailable"]["count"] == 1


def test_run_returns_none_when_logfiles_variable_missing():
    chk = make_check()
    assert chk.run({}) is None
    assert chk.run({"LogFiles": None}) is None
    assert chk.run({"LogFiles": []}) is None


def test_run_uses_pool_and_formats_results(monkeypatch, tmp_path):
    from fossor.checks import similar_log_errors as module

    chk = make_check()

    log1 = write_log(
        tmp_path,
        "service1.log",
        [
            "2020-01-01 12:00:00 ERROR alpha failure\n",
            "2020-01-01 12:01:00 ERROR alpha failure again\n",
        ],
    )
    log2 = write_log(
        tmp_path,
        "service2.log",
        [
            "2020-01-02 12:00:00 ERROR beta failure\n",
        ],
    )

    monkeypatch.setattr(module, "cpu_count", lambda: 1)
    monkeypatch.setattr(module, "Pool", FakePool)

    result = chk.run({"LogFiles": [log1, log2]})

    assert FakePool.created_with == 2
    assert FakePool.map_called_with[1] == [log1, log2]
    assert "filename: " + log1 in result
    assert "filename: " + log2 in result
    assert "Count: 2, ERROR alpha failure" in result
    assert "First seen: 2020-01-01 12:00:00, Count: 2, ERROR alpha failure" in result
    assert "Count: 1, ERROR beta failure" in result


def test_run_filters_blacklisted_logs_before_processing(monkeypatch, tmp_path):
    from fossor.checks import similar_log_errors as module

    chk = make_check()

    good_log = write_log(
        tmp_path,
        "good.log",
        ["2020-01-01 12:00:00 ERROR good failure\n"],
    )
    bad_log = write_log(
        tmp_path,
        "bad_configuration.log",
        ["2020-01-01 12:00:00 ERROR should not be processed\n"],
    )

    monkeypatch.setattr(module, "cpu_count", lambda: 1)
    monkeypatch.setattr(module, "Pool", FakePool)

    result = chk.run({"LogFiles": [good_log, bad_log]})

    assert FakePool.map_called_with[1] == [good_log]
    assert "filename: " + good_log in result
    assert bad_log not in result
