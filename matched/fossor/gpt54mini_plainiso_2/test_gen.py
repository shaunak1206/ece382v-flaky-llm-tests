import os
from collections import OrderedDict

import pytest

import fossor.checks.similar_log_errors as similar_module
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
    c = SimilarLogErrors()
    c.log = DummyLog()
    return c


def test_run_returns_none_when_no_logfiles(checker, monkeypatch):
    monkeypatch.setattr(similar_module, "Pool", DummyPool)
    monkeypatch.setattr(similar_module, "cpu_count", lambda: 1)
    assert checker.run({}) is None
    assert checker.run({"LogFiles": None}) is None
    assert checker.run({"LogFiles": []}) is None


def test_remove_blacklisted_logs_filters_expected_names(checker):
    log_files = [
        "/var/log/app.log",
        "/var/log/app.out",
        "/var/log/gc.log",
        "/var/log/service_error.log",
        "/var/log/engine_access_log.log",
    ]
    filtered = checker._remove_blacklisted_logs(log_files)
    assert filtered == ["/var/log/app.log", "/var/log/service_error.log"]


@pytest.mark.parametrize(
    "line, expected_line, expected_date",
    [
        ("2024-01-01 12:00:00 ERROR Something bad happened\n", "ERROR Something bad happened\n", "2024-01-01 12:00:00"),
        ("2024-01-01 12:00:00 INFO Not an error\n", None, None),
        ("   at com.example.Class.method(Class.java:1)\n", None, None),
        ("Caused by: java.lang.RuntimeException: boom\n", None, None),
        ("Hello world\n", None, None),
        ("2024-01-01 12:00:00 ERROR " + ("x" * 500) + "\n", "ERROR " + ("x" * 193), "2024-01-01 12:00:00"),
    ],
)
def test_process_line_variants(checker, line, expected_line, expected_date):
    processed_line, date = checker._process_line(line)
    assert processed_line == expected_line
    assert date == expected_date


def test_process_line_handles_short_numeric_line(checker):
    processed_line, date = checker._process_line("2024-01-01\n")
    assert processed_line is None
    assert date is None


def test_group_similar_log_lines_increments_similar_line(checker):
    common_lines = OrderedDict()
    common_lines["ERROR Disk full on node 1"] = {"count": 2}
    updated = checker._group_similar_log_lines(
        line="ERROR Disk full on node 2",
        date="2024-01-01 12:00:00",
        common_lines=common_lines,
    )
    assert updated["ERROR Disk full on node 1"]["count"] == 3
    assert updated["ERROR Disk full on node 1"]["first_seen"] == "2024-01-01 12:00:00"


def test_group_similar_log_lines_adds_new_unique_line(checker):
    common_lines = OrderedDict()
    updated = checker._group_similar_log_lines(
        line="ERROR Something completely different",
        date="2024-01-01 12:00:00",
        common_lines=common_lines,
    )
    assert list(updated.keys()) == ["ERROR Something completely different"]
    assert updated["ERROR Something completely different"]["count"] == 1
    assert updated["ERROR Something completely different"]["first_seen"] == "2024-01-01 12:00:00"


def test_group_similar_log_lines_uses_other_error_lines_after_limit(checker):
    checker.MAX_COMMON_LINES = 1
    common_lines = OrderedDict()
    common_lines["ERROR first line"] = {"count": 1}
    updated = checker._group_similar_log_lines(
        line="ERROR second unique line",
        date=None,
        common_lines=common_lines,
    )
    assert "Other Error Lines" in updated
    assert updated["Other Error Lines"]["count"] == 1
    assert updated["ERROR first line"]["count"] == 1


def test_get_error_pattern_counts_groups_lines_from_file(tmp_path, checker):
    p = tmp_path / "app.log"
    p.write_text(
        "\n".join(
            [
                "2024-01-01 12:00:00 ERROR Disk failed on node 1",
                "2024-01-01 12:00:01 ERROR Disk failed on node 2",
                "2024-01-01 12:00:02 INFO Not included",
                "Caused by: ignored",
                "   stacktrace line",
                "2024-01-01 12:00:03 ERROR Something else",
            ]
        )
        + "\n"
    )

    filename, common_lines = checker.get_error_pattern_counts(str(p))
    assert filename == str(p)
    assert len(common_lines) == 2
    assert any("Disk failed" in key for key in common_lines.keys())
    assert any("Something else" in key for key in common_lines.keys())


def test_get_error_pattern_counts_sets_first_seen_for_grouped_lines(tmp_path, checker):
    p = tmp_path / "app.log"
    p.write_text(
        "\n".join(
            [
                "2024-01-01 12:00:00 ERROR Timeout talking to service A",
                "2024-01-01 12:00:01 ERROR Timeout talking to service B",
            ]
        )
        + "\n"
    )

    _, common_lines = checker.get_error_pattern_counts(str(p))
    assert len(common_lines) == 1
    key = next(iter(common_lines.keys()))
    assert common_lines[key]["count"] == 2
    assert common_lines[key]["first_seen"] == "2024-01-01 12:00:00"


def test_run_formats_single_file_result(tmp_path, checker, monkeypatch):
    p = tmp_path / "app.log"
    p.write_text(
        "\n".join(
            [
                "2024-01-01 12:00:00 ERROR Disk failed on node 1",
                "2024-01-01 12:00:01 ERROR Disk failed on node 2",
            ]
        )
        + "\n"
    )

    monkeypatch.setattr(similar_module, "cpu_count", lambda: 1)
    monkeypatch.setattr(similar_module, "Pool", DummyPool)

    result = checker.run({"LogFiles": [str(p)]})
    assert "filename: " + str(p) in result
    assert "Count: 2, " in result
    assert "First seen: 2024-01-01 12:00:00, " in result
    assert "ERROR Disk failed on node" in result


def test_run_filters_blacklisted_files_before_processing(tmp_path, checker, monkeypatch):
    good = tmp_path / "good.log"
    blacklisted = tmp_path / "bad.out"
    good.write_text("2024-01-01 12:00:00 ERROR good error\n")
    blacklisted.write_text("2024-01-01 12:00:00 ERROR should not be read\n")

    seen = []

    def fake_get_error_pattern_counts(filename):
        seen.append(os.path.basename(filename))
        return filename, OrderedDict({"ERROR good error": {"count": 1}})

    monkeypatch.setattr(similar_module, "cpu_count", lambda: 1)
    monkeypatch.setattr(similar_module, "Pool", DummyPool)
    monkeypatch.setattr(checker, "get_error_pattern_counts", fake_get_error_pattern_counts)

    result = checker.run({"LogFiles": [str(good), str(blacklisted)]})
    assert seen == ["good.log"]
    assert "good.log" in result
    assert "bad.out" not in result


def test_run_returns_none_when_all_files_blacklisted(checker, monkeypatch):
    monkeypatch.setattr(similar_module, "cpu_count", lambda: 1)
    monkeypatch.setattr(similar_module, "Pool", DummyPool)
    result = checker.run({"LogFiles": ["/tmp/app.out", "/tmp/gc.log"]})
    assert result is None


def test_run_uses_cpu_count_to_size_pool(checker, monkeypatch):
    created = {}

    class RecordingPool(DummyPool):
        def __init__(self, threads):
            super().__init__(threads)
            created["threads"] = threads

    monkeypatch.setattr(similar_module, "cpu_count", lambda: 4)
    monkeypatch.setattr(similar_module, "Pool", RecordingPool)

    def fake_get_error_pattern_counts(filename):
        return filename, OrderedDict()

    monkeypatch.setattr(checker, "get_error_pattern_counts", fake_get_error_pattern_counts)

    checker.run({"LogFiles": ["a.log"]})
    assert created["threads"] == 8
