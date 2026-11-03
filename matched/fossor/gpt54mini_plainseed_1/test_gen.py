import builtins
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import fossor.checks.similar_log_errors as similar_log_errors_mod
from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyPool:
    def __init__(self, results):
        self._results = results
        self.mapped = None

    def map(self, func, iterable):
        self.mapped = list(iterable)
        return self._results


def make_check():
    check = SimilarLogErrors()
    check.log = Mock()
    return check


def test_remove_blacklisted_logs_filters_matching_names():
    check = make_check()
    log_files = [
        "app.log",
        "/var/log/service_configuration.log",
        "stdout.out",
        "gc.log",
        "keep_this.log",
    ]

    result = check._remove_blacklisted_logs(log_files)

    assert result == ["app.log", "keep_this.log"]


def test_remove_blacklisted_logs_leaves_non_matching_names_unchanged():
    check = make_check()
    log_files = ["a.log", "b.txt", "c.ERROR"]

    result = check._remove_blacklisted_logs(log_files)

    assert result == log_files


@pytest.mark.parametrize(
    "line, expected_line, expected_date",
    [
        ("   stacktrace line\n", None, None),
        ("INFO not an error\n", None, None),
        ("Caused by: nope\n", None, None),
    ],
)
def test_process_line_ignores_non_error_and_stacktrace_lines(line, expected_line, expected_date):
    check = make_check()

    processed_line, date = check._process_line(line)

    assert processed_line == expected_line
    assert date == expected_date


def test_process_line_ignores_digit_starting_non_error_line():
    check = make_check()

    line, date = check._process_line("2024-01-01 12:00:00 INFO not an error\n")

    assert line is None
    assert date is None


def test_process_line_returns_error_line_and_date():
    check = make_check()

    line, date = check._process_line("2024-01-01 12:00:00 ERROR Something bad happened\n")

    assert date == "2024-01-01 12:00:00"
    assert line == "ERROR Something bad happened"


def test_process_line_truncates_long_error_line():
    check = make_check()
    long_message = "x" * 500

    line, date = check._process_line(f"2024-01-01 12:00:00 ERROR {long_message}\n")

    assert date == "2024-01-01 12:00:00"
    assert line.startswith("ERROR ")
    assert len(line) == 200


def test_process_line_handles_index_error_and_returns_none():
    check = make_check()

    line, date = check._process_line("2024-01-01\n")

    assert line is None
    assert date is None


def test_process_line_allows_alpha_lines_containing_exception():
    check = make_check()

    line, date = check._process_line("MyException happened here\n")

    assert line == "MyException happened here\n"
    assert date is None


def test_group_similar_log_lines_increments_similar_existing_line_and_sets_first_seen():
    check = make_check()
    common_lines = {"ERROR same problem": {"count": 2, "first_seen": "2024-01-01 00:00:00"}}

    result = check._group_similar_log_lines(
        line="ERROR same problem again",
        date="2024-01-02 00:00:00",
        common_lines=common_lines,
    )

    assert result["ERROR same problem"]["count"] == 3
    assert result["ERROR same problem"]["first_seen"] == "2024-01-01 00:00:00"


def test_group_similar_log_lines_adds_new_unique_line_with_date():
    check = make_check()
    common_lines = {}

    result = check._group_similar_log_lines(
        line="ERROR unique issue",
        date="2024-01-03 00:00:00",
        common_lines=common_lines,
    )

    assert list(result.keys()) == ["ERROR unique issue"]
    assert result["ERROR unique issue"]["count"] == 1
    assert result["ERROR unique issue"]["first_seen"] == "2024-01-03 00:00:00"


def test_group_similar_log_lines_uses_other_error_lines_when_max_unique_exceeded():
    check = make_check()
    common_lines = {
        f"ERROR unique {i}": {"count": 1} for i in range(check.MAX_COMMON_LINES)
    }

    result = check._group_similar_log_lines(
        line="ERROR brand new unique issue",
        date=None,
        common_lines=common_lines,
    )

    assert "Other Error Lines" in result
    assert result["Other Error Lines"]["count"] == 1


def test_get_error_pattern_counts_groups_lines_from_file(monkeypatch):
    check = make_check()
    file_content = (
        "2024-01-01 00:00:00 ERROR same problem A\n"
        "2024-01-01 00:01:00 ERROR same problem B\n"
        "2024-01-01 00:02:00 INFO ignore me\n"
        "   stacktrace line\n"
        "2024-01-01 00:03:00 ERROR different problem\n"
    )

    def fake_open(filename, mode="rt"):
        assert filename == "dummy.log"
        assert mode == "rt"
        return SimpleNamespace(
            readlines=lambda: file_content.splitlines(True),
            __enter__=lambda self: self,
            __exit__=lambda self, exc_type, exc, tb: False,
        )

    monkeypatch.setattr(builtins, "open", fake_open)

    filename, common_lines = check.get_error_pattern_counts("dummy.log")

    assert filename == "dummy.log"
    assert common_lines["ERROR same problem A"]["count"] == 1
    assert common_lines["ERROR same problem A"]["first_seen"] == "2024-01-01 00:00:00"
    assert common_lines["ERROR same problem B"]["count"] == 1
    assert common_lines["ERROR different problem"]["count"] == 1
    assert "INFO ignore me" not in common_lines


def test_run_returns_none_when_no_logfiles():
    check = make_check()

    assert check.run({}) is None
    assert check.run({"LogFiles": []}) is None
    assert check.run({"LogFiles": None}) is None


def test_run_filters_blacklisted_logs_and_formats_output(monkeypatch):
    check = make_check()

    monkeypatch.setattr(similar_log_errors_mod, "cpu_count", lambda: 1)
    monkeypatch.setattr(
        similar_log_errors_mod,
        "Pool",
        lambda threads: DummyPool(
            [
                (
                    "keep.log",
                    {
                        "ERROR one": {"count": 2, "first_seen": "2024-01-01 00:00:00"},
                        "ERROR two": {"count": 1},
                    },
                )
            ]
        ),
    )

    result = check.run(
        {
            "LogFiles": [
                "keep.log",
                "skip.out",
                "service_configuration.log",
            ]
        }
    )

    assert "filename: keep.log" in result
    assert "First seen: 2024-01-01 00:00:00, Count: 2, ERROR one" in result
    assert "Count: 1, ERROR two" in result
    assert "skip.out" not in result
    assert "service_configuration.log" not in result


def test_run_returns_none_when_pool_results_are_empty(monkeypatch):
    check = make_check()

    monkeypatch.setattr(similar_log_errors_mod, "cpu_count", lambda: 1)
    monkeypatch.setattr(similar_log_errors_mod, "Pool", lambda threads: DummyPool([]))

    result = check.run({"LogFiles": ["keep.log"]})

    assert result is None
