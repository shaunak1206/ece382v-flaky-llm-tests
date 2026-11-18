import builtins
from pathlib import Path

import pytest

import fossor.checks.similar_log_errors as sle_mod
from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def debug(self, *args, **kwargs):
        pass


class DummyPool:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.mapped = None

    def map(self, func, iterable):
        self.mapped = list(iterable)
        return [func(x) for x in iterable]


@pytest.fixture
def check():
    obj = SimilarLogErrors()
    obj.log = DummyLog()
    return obj


def test_remove_blacklisted_logs_filters_matching_names(check):
    logs = [
        "/var/log/app.log",
        "/var/log/app_configuration.log",
        "/var/log/access.out",
        "/var/log/worker.log",
    ]
    result = check._remove_blacklisted_logs(logs)
    assert result == ["/var/log/app.log", "/var/log/worker.log"]


def test_remove_blacklisted_logs_returns_same_list_when_no_matches(check):
    logs = ["/var/log/app.log", "/var/log/worker.log"]
    assert check._remove_blacklisted_logs(logs) == logs


def test_process_line_ignores_indented_stacktrace_line(check):
    line, date = check._process_line("    at com.example.Service.method(Service.java:12)\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_exception_alpha_line(check):
    line, date = check._process_line("INFO something happened\n")
    assert line is None
    assert date is None


def test_process_line_ignores_caused_by_line(check):
    line, date = check._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_ignores_non_error_level_line(check):
    line, date = check._process_line("2020-01-01 12:00:00 WARN something\n")
    assert line is None
    assert date is None


def test_process_line_parses_error_line_and_date(check):
    line, date = check._process_line("2020-01-01 12:00:00 ERROR Something bad happened\n")
    assert date == "2020-01-01 12:00:00"
    assert line == "ERROR Something bad happened"


def test_process_line_truncates_very_long_error_line(check):
    msg = "x" * 300
    line, date = check._process_line(f"2020-01-01 12:00:00 ERROR {msg}\n")
    assert date == "2020-01-01 12:00:00"
    assert line.startswith("ERROR ")
    assert len(line) == 200


def test_process_line_handles_index_error_and_returns_none(check):
    line, date = check._process_line("2020-01-01\n")
    assert line is None
    assert date is None


def test_group_similar_log_lines_counts_similar_lines_together(check):
    common_lines = sle_mod.OrderedDict()
    common_lines = check._group_similar_log_lines(
        line="ERROR Database connection failed",
        date="2020-01-01 12:00:00",
        common_lines=common_lines,
    )
    common_lines = check._group_similar_log_lines(
        line="ERROR Database connection failed again",
        date="2020-01-01 12:01:00",
        common_lines=common_lines,
    )

    assert len(common_lines) == 1
    key = next(iter(common_lines))
    assert key == "ERROR Database connection failed"
    assert common_lines[key]["count"] == 2
    assert common_lines[key]["first_seen"] == "2020-01-01 12:00:00"


def test_group_similar_log_lines_keeps_distinct_lines_under_threshold(check):
    common_lines = sle_mod.OrderedDict()
    common_lines = check._group_similar_log_lines(
        line="ERROR A completely different error message",
        date=None,
        common_lines=common_lines,
    )
    common_lines = check._group_similar_log_lines(
        line="ERROR Another unrelated error message",
        date=None,
        common_lines=common_lines,
    )

    assert len(common_lines) == 2
    assert common_lines["ERROR A completely different error message"]["count"] == 1
    assert common_lines["ERROR Another unrelated error message"]["count"] == 1


def test_group_similar_log_lines_groups_as_other_error_lines_after_limit(check):
    common_lines = sle_mod.OrderedDict()
    for i in range(check.MAX_COMMON_LINES):
        common_lines = check._group_similar_log_lines(
            line=f"ERROR unique message {i}",
            date=None,
            common_lines=common_lines,
        )

    common_lines = check._group_similar_log_lines(
        line="ERROR one more unique message",
        date=None,
        common_lines=common_lines,
    )

    assert "Other Error Lines" in common_lines
    assert common_lines["Other Error Lines"]["count"] == 1


def test_get_error_pattern_counts_reads_file_and_groups_lines(tmp_path, check):
    log_file = tmp_path / "sample.log"
    log_file.write_text(
        "\n".join(
            [
                "2020-01-01 12:00:00 ERROR Database connection failed",
                "2020-01-01 12:01:00 ERROR Database connection failed again",
                "2020-01-01 12:02:00 WARN ignore this",
                "INFO not an error",
            ]
        )
        + "\n"
    )

    filename, common_lines = check.get_error_pattern_counts(str(log_file))
    assert filename == str(log_file)
    assert len(common_lines) == 1
    key = next(iter(common_lines))
    assert key == "ERROR Database connection failed"
    assert common_lines[key]["count"] == 2
    assert common_lines[key]["first_seen"] == "2020-01-01 12:00:00"


def test_run_returns_none_when_no_log_files(check):
    assert check.run({}) is None
    assert check.run({"LogFiles": None}) is None
    assert check.run({"LogFiles": []}) is None


def test_run_formats_results_and_uses_pool(monkeypatch, tmp_path, check):
    log1 = tmp_path / "app1.log"
    log2 = tmp_path / "app2.log"
    log1.write_text("2020-01-01 12:00:00 ERROR First error\n")
    log2.write_text("2020-01-01 12:00:00 ERROR Second error\n")

    def fake_cpu_count():
        return 1

    created_pools = []

    def fake_pool(n):
        p = DummyPool(n)
        created_pools.append(p)
        return p

    monkeypatch.setattr(sle_mod, "cpu_count", fake_cpu_count)
    monkeypatch.setattr(sle_mod, "Pool", fake_pool)

    result = check.run({"LogFiles": [str(log1), str(log2)]})
    assert created_pools and created_pools[0].args == (2,)
    assert "filename: " + str(log1) in result
    assert "filename: " + str(log2) in result
    assert "Count: 1, ERROR First error" in result
    assert "Count: 1, ERROR Second error" in result


def test_run_excludes_blacklisted_files(monkeypatch, tmp_path, check):
    allowed = tmp_path / "good.log"
    blacklisted = tmp_path / "bad.out"
    allowed.write_text("2020-01-01 12:00:00 ERROR Kept\n")
    blacklisted.write_text("2020-01-01 12:00:00 ERROR Removed\n")

    seen = []

    class TrackingPool(DummyPool):
        def map(self, func, iterable):
            seen.extend(list(iterable))
            return super().map(func, iterable)

    monkeypatch.setattr(sle_mod, "cpu_count", lambda: 1)
    monkeypatch.setattr(sle_mod, "Pool", lambda n: TrackingPool(n))

    result = check.run({"LogFiles": [str(allowed), str(blacklisted)]})
    assert str(allowed) in seen
    assert str(blacklisted) not in seen
    assert str(allowed) in result
    assert str(blacklisted) not in result
