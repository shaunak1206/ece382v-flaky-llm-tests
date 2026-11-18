import builtins
from collections import OrderedDict

import pytest

import fossor.checks.similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def __init__(self):
        self.debug_messages = []

    def debug(self, msg):
        self.debug_messages.append(msg)


class DummyPool:
    def __init__(self, processes):
        self.processes = processes
        self.mapped = None

    def map(self, func, iterable):
        self.mapped = (func, list(iterable))
        return [func(item) for item in iterable]


@pytest.fixture
def checker():
    c = SimilarLogErrors()
    c.log = DummyLog()
    return c


def test_remove_blacklisted_logs_filters_blacklisted_entries(checker):
    log_files = [
        "/var/log/app.log",
        "/var/log/app_configuration.log",
        "/var/log/app.out",
        "/var/log/gc.log",
        "/var/log/kept.log",
    ]
    result = checker._remove_blacklisted_logs(log_files)
    assert result == ["/var/log/app.log", "/var/log/kept.log"]


def test_remove_blacklisted_logs_returns_same_list_when_no_blacklist_matches(checker):
    log_files = ["/var/log/a.log", "/var/log/b.txt"]
    assert checker._remove_blacklisted_logs(log_files) == log_files


@pytest.mark.parametrize(
    "line, expected_line, expected_date",
    [
        ("2024-01-01 12:00:00 ERROR something bad happened\n", "ERROR something bad happened", "2024-01-01 12:00:00"),
        ("2024-01-01 12:00:00 INFO just info\n", None, None),
        ("    stacktrace line\n", None, None),
        ("Caused by: boom\n", None, None),
        ("Nov 1 12:00:00 ERROR not parsed as date\n", None, None),
    ],
)
def test_process_line_handles_common_cases(checker, line, expected_line, expected_date):
    actual_line, actual_date = checker._process_line(line)
    assert actual_line == expected_line
    assert actual_date == expected_date


def test_process_line_truncates_long_error_messages(checker):
    payload = "x" * 500
    line = f"2024-01-01 12:00:00 ERROR {payload}\n"
    processed, date = checker._process_line(line)
    assert date == "2024-01-01 12:00:00"
    assert processed.startswith("ERROR ")
    assert len(processed) <= 200


def test_process_line_returns_none_for_too_short_numeric_line(checker):
    line = "2024-01-01\n"
    processed, date = checker._process_line(line)
    assert processed is None
    assert date is None


def test_group_similar_log_lines_counts_identical_line_as_same_entry(checker):
    common_lines = OrderedDict()
    common_lines = checker._group_similar_log_lines(
        "ERROR something bad happened",
        "2024-01-01 12:00:00",
        common_lines,
    )
    common_lines = checker._group_similar_log_lines(
        "ERROR something bad happened",
        "2024-01-01 12:01:00",
        common_lines,
    )
    assert list(common_lines.keys()) == ["ERROR something bad happened"]
    assert common_lines["ERROR something bad happened"]["count"] == 2
    assert common_lines["ERROR something bad happened"]["first_seen"] == "2024-01-01 12:00:00"


def test_group_similar_log_lines_merges_similar_lines_above_threshold(checker):
    common_lines = OrderedDict()
    first = "ERROR failed to connect to host alpha"
    second = "ERROR failed to connect to host beta"
    common_lines = checker._group_similar_log_lines(first, None, common_lines)
    common_lines = checker._group_similar_log_lines(second, None, common_lines)
    assert len(common_lines) == 1
    key = next(iter(common_lines))
    assert common_lines[key]["count"] == 2


def test_group_similar_log_lines_uses_other_error_lines_after_max_unique(checker):
    checker.MAX_COMMON_LINES = 1
    common_lines = OrderedDict()
    common_lines = checker._group_similar_log_lines("ERROR first unique", None, common_lines)
    common_lines = checker._group_similar_log_lines("ERROR second unique", None, common_lines)
    assert "Other Error Lines" in common_lines
    assert common_lines["Other Error Lines"]["count"] == 1
    assert common_lines["ERROR first unique"]["count"] == 1


def test_get_error_pattern_counts_reads_file_and_groups_lines(tmp_path, checker):
    path = tmp_path / "app.log"
    path.write_text(
        "\n".join(
            [
                "2024-01-01 12:00:00 ERROR connection failed for user A",
                "2024-01-01 12:01:00 ERROR connection failed for user B",
                "2024-01-01 12:02:00 INFO not included",
                "   stacktrace line",
            ]
        )
        + "\n"
    )
    filename, common_lines = checker.get_error_pattern_counts(str(path))
    assert filename == str(path)
    assert len(common_lines) == 1
    key = next(iter(common_lines))
    assert key.startswith("ERROR connection failed for user")
    assert common_lines[key]["count"] == 2


def test_run_returns_none_when_logfiles_missing(checker):
    assert checker.run({}) is None
    assert checker.run({"LogFiles": None}) is None
    assert checker.run({"LogFiles": []}) is None


def test_run_removes_blacklisted_files_and_uses_pool(monkeypatch, tmp_path, checker):
    file1 = tmp_path / "a.log"
    file2 = tmp_path / "b_configuration.log"
    file3 = tmp_path / "c.log"
    file1.write_text("2024-01-01 12:00:00 ERROR alpha\n")
    file2.write_text("2024-01-01 12:00:00 ERROR should be filtered\n")
    file3.write_text("2024-01-01 12:00:00 ERROR beta\n")

    monkeypatch.setattr(sle_module, "cpu_count", lambda: 1)
    dummy_pool = DummyPool
    monkeypatch.setattr(sle_module, "Pool", dummy_pool)

    result = checker.run({"LogFiles": [str(file1), str(file2), str(file3)]})
    assert "b_configuration.log" not in result
    assert "filename: " in result
    assert str(file1) in result
    assert str(file3) in result


def test_run_formats_first_seen_and_count(monkeypatch, tmp_path, checker):
    file1 = tmp_path / "app.log"
    file1.write_text(
        "\n".join(
            [
                "2024-01-01 12:00:00 ERROR alpha issue",
                "2024-01-01 12:05:00 ERROR alpha issue",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(sle_module, "cpu_count", lambda: 1)
    monkeypatch.setattr(sle_module, "Pool", DummyPool)

    result = checker.run({"LogFiles": [str(file1)]})
    assert "filename: " in result
    assert "Count: 2, " in result
    assert "First seen: 2024-01-01 12:00:00, " in result
    assert "ERROR alpha issue" in result


def test_run_returns_none_when_no_common_lines_are_found(monkeypatch, tmp_path, checker):
    file1 = tmp_path / "app.log"
    file1.write_text(
        "\n".join(
            [
                "2024-01-01 12:00:00 INFO only info",
                "caused by something",
                "   stacktrace line",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(sle_module, "cpu_count", lambda: 1)
    monkeypatch.setattr(sle_module, "Pool", DummyPool)

    assert checker.run({"LogFiles": [str(file1)]}) is None
