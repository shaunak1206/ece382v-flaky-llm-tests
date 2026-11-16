import logging
from collections import OrderedDict

import pytest

from fossor.checks import similar_log_errors as sle
from fossor.checks.similar_log_errors import SimilarLogErrors


def make_check():
    """Build a SimilarLogErrors instance with a usable logger."""
    check = SimilarLogErrors.__new__(SimilarLogErrors)
    check.log = logging.getLogger("test_similar_log_errors")
    return check


def write_log(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("".join(line + "\n" for line in lines))
    return str(path)


class SerialPool:
    """Stand-in for multiprocessing.Pool that runs work in-process."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def close(self):
        pass

    def join(self):
        pass


@pytest.fixture
def serial_pool(monkeypatch):
    monkeypatch.setattr(sle, "Pool", SerialPool)


@pytest.fixture
def recording_pool(monkeypatch):
    received = []

    class RecordingPool:
        def __init__(self, processes=None):
            pass

        def map(self, func, iterable):
            received.append(list(iterable))
            return []

        def close(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(sle, "Pool", RecordingPool)
    return received


# ---------- _process_line ----------

def test_process_line_skips_indented_stack_trace_lines():
    check = make_check()
    line, date = check._process_line("\tat com.example.Foo.bar(Foo.java:10)\n")
    assert line is None
    assert date is None


def test_process_line_skips_alpha_line_without_exception():
    check = make_check()
    line, date = check._process_line("Starting service now\n")
    assert line is None
    assert date is None


def test_process_line_keeps_alpha_line_containing_exception():
    check = make_check()
    raw = "java.lang.NullPointerException: boom\n"
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


def test_process_line_skips_caused_by_lines():
    check = make_check()
    line, date = check._process_line("Caused by: java.lang.Exception: root\n")
    assert line is None
    assert date is None


def test_process_line_parses_error_line_into_message_and_date():
    check = make_check()
    line, date = check._process_line("2017-01-01 10:00:00 ERROR Disk is full\n")
    assert line == "ERROR Disk is full"
    assert date == "2017-01-01 10:00:00"


def test_process_line_skips_non_error_level():
    check = make_check()
    line, date = check._process_line("2017-01-01 10:00:00 WARN Disk is getting full\n")
    assert line is None
    assert date is None


def test_process_line_returns_none_for_short_dated_line():
    check = make_check()
    line, date = check._process_line("2017-01-01\n")
    assert line is None
    assert date is None


def test_process_line_truncates_long_error_message():
    check = make_check()
    raw = "2017-01-01 10:00:00 ERROR " + "x" * 300 + "\n"
    line, date = check._process_line(raw)
    assert line is not None
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 10:00:00"


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs_filters_matching_names():
    check = make_check()
    files = [
        "/logs/app.log",
        "/logs/app_configuration.log",
        "/logs/gc.log",
        "/logs/server.out",
        "/logs/app_databus.log",
        "/logs/app_public_access.log",
        "/logs/jetty_thread_pool_stats.log",
        "/logs/engine_access_log.log",
        "/logs/service.log",
    ]
    assert check._remove_blacklisted_logs(files) == ["/logs/app.log", "/logs/service.log"]


def test_remove_blacklisted_logs_keeps_everything_when_nothing_matches():
    check = make_check()
    files = ["/logs/a.log", "/logs/b.log"]
    assert check._remove_blacklisted_logs(files) == files


def test_remove_blacklisted_logs_handles_empty_input():
    check = make_check()
    assert check._remove_blacklisted_logs([]) == []


# ---------- _group_similar_log_lines ----------

def test_group_first_line_creates_entry_with_count_and_first_seen():
    check = make_check()
    common = check._group_similar_log_lines(
        line="ERROR Disk is full", date="2017-01-01 10:00:00", common_lines=OrderedDict()
    )
    assert list(common.keys()) == ["ERROR Disk is full"]
    assert common["ERROR Disk is full"]["count"] == 1
    assert common["ERROR Disk is full"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_repeated_line_increments_count_and_keeps_first_seen():
    check = make_check()
    common = OrderedDict()
    common = check._group_similar_log_lines("ERROR Same", "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines("ERROR Same", "2017-01-01 10:05:00", common)
    assert len(common) == 1
    assert common["ERROR Same"]["count"] == 2
    assert common["ERROR Same"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_similar_line_merges_into_existing_key():
    check = make_check()
    common = OrderedDict()
    common = check._group_similar_log_lines(
        "ERROR connection failed to host A", "2017-01-01 10:00:00", common
    )
    common = check._group_similar_log_lines(
        "ERROR connection failed to host B", "2017-01-01 10:00:01", common
    )
    assert len(common) == 1
    assert "ERROR connection failed to host A" in common
    assert common["ERROR connection failed to host A"]["count"] == 2


def test_group_dissimilar_line_creates_new_key():
    check = make_check()
    common = OrderedDict()
    common = check._group_similar_log_lines("ERROR aaaaaaaaaa", "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines("ERROR bbbbbbbbbb", "2017-01-01 10:00:01", common)
    assert len(common) == 2
    assert common["ERROR aaaaaaaaaa"]["count"] == 1
    assert common["ERROR bbbbbbbbbb"]["count"] == 1


def test_group_overflow_lines_go_to_other_bucket():
    check = make_check()
    check.MAX_COMMON_LINES = 2
    common = OrderedDict()
    common = check._group_similar_log_lines("a" * 8, "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines("b" * 8, "2017-01-01 10:00:01", common)
    common = check._group_similar_log_lines("c" * 8, "2017-01-01 10:00:02", common)
    common = check._group_similar_log_lines("d" * 8, "2017-01-01 10:00:03", common)
    assert "Other Error Lines" in common
    assert common["Other Error Lines"]["count"] == 2
    assert "c" * 8 not in common
    assert "d" * 8 not in common


def test_group_line_without_date_has_no_first_seen():
    check = make_check()
    common = check._group_similar_log_lines("ERROR no date", None, OrderedDict())
    assert common["ERROR no date"]["count"] == 1
    assert "first_seen" not in common["ERROR no date"]


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_reads_file_and_groups_lines(tmp_path):
    check = make_check()
    path = write_log(tmp_path, "app.log", [
        "2017-01-01 10:00:00 ERROR Connection refused to host alpha",
        "2017-01-01 10:00:01 ERROR Connection refused to host beta",
        "2017-01-01 10:00:02 INFO ignored line",
        "  at some.stack.frame",
        "2017-01-01 10:00:03 ERROR Disk full",
    ])
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert len(common) == 2
    first = common["ERROR Connection refused to host alpha"]
    assert first["count"] == 2
    assert first["first_seen"] == "2017-01-01 10:00:00"
    assert common["ERROR Disk full"]["count"] == 1


def test_get_error_pattern_counts_empty_file_returns_no_lines(tmp_path):
    check = make_check()
    path = tmp_path / "empty.log"
    path.write_text("")
    filename, common = check.get_error_pattern_counts(str(path))
    assert filename == str(path)
    assert len(common) == 0


# ---------- run ----------

def test_run_returns_none_when_no_log_files_variable():
    check = make_check()
    assert check.run({}) is None


def test_run_returns_none_for_empty_log_file_list():
    check = make_check()
    assert check.run({"LogFiles": []}) is None


def test_run_formats_grouped_errors_with_first_seen_and_count(tmp_path, serial_pool):
    check = make_check()
    path = write_log(tmp_path, "app.log", [
        "2017-01-01 10:00:00 ERROR Disk full",
        "2017-01-01 10:00:05 ERROR Disk full",
    ])
    result = check.run({"LogFiles": [path]})
    assert f"filename: {path}\n" in result
    assert "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Disk full" in result


def test_run_returns_none_when_no_error_lines_found(tmp_path, serial_pool):
    check = make_check()
    path = write_log(tmp_path, "info.log", [
        "2017-01-01 10:00:00 INFO all good",
        "2017-01-01 10:00:01 DEBUG still good",
    ])
    assert check.run({"LogFiles": [path]}) is None


def test_run_excludes_blacklisted_files_before_mapping(recording_pool):
    check = make_check()
    check.run({"LogFiles": [
        "/logs/app.log",
        "/logs/app_configuration.log",
        "/logs/gc.log",
    ]})
    assert recording_pool == [["/logs/app.log"]]
