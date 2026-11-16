import logging
from collections import OrderedDict

import pytest

from fossor.checks import similar_log_errors
from fossor.checks.similar_log_errors import SimilarLogErrors


@pytest.fixture
def checker():
    # Bypass Check.__init__ so tests do not depend on its constructor signature.
    # Only the attributes used by the methods under test are set.
    obj = SimilarLogErrors.__new__(SimilarLogErrors)
    obj.log = logging.getLogger("test_similar_log_errors")
    return obj


class FakePool:
    """Synchronous stand-in for multiprocessing.Pool."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def close(self):
        pass

    def join(self):
        pass


# ---------------------------------------------------------------------------
# _remove_blacklisted_logs
# ---------------------------------------------------------------------------

def test_remove_blacklisted_logs_drops_blacklisted_names(checker):
    files = [
        "/var/log/app/server.log",
        "/var/log/app/server_configuration.log",
        "/var/log/app/stdout.out",
        "/var/log/app/gc.log",
        "/var/log/app/db_databus.log",
    ]
    assert checker._remove_blacklisted_logs(files) == ["/var/log/app/server.log"]


def test_remove_blacklisted_logs_keeps_all_when_none_match(checker):
    files = ["/var/log/a.log", "/var/log/b.log"]
    assert checker._remove_blacklisted_logs(files) == files


def test_remove_blacklisted_logs_empty_input(checker):
    assert checker._remove_blacklisted_logs([]) == []


# ---------------------------------------------------------------------------
# _process_line
# ---------------------------------------------------------------------------

def test_process_line_skips_leading_whitespace(checker):
    assert checker._process_line("   at com.example.Foo\n") == (None, None)


def test_process_line_skips_alpha_line_without_exception(checker):
    assert checker._process_line("Something happened\n") == (None, None)


def test_process_line_keeps_alpha_line_containing_exception(checker):
    line = "IllegalStateException: boom\n"
    assert checker._process_line(line) == (line, None)


def test_process_line_skips_caused_by_lines(checker):
    assert checker._process_line("Caused by: java.lang.Exception: x\n") == (None, None)


def test_process_line_parses_error_line(checker):
    line = "2017-01-01 10:00:00 ERROR Disk full\n"
    assert checker._process_line(line) == ("ERROR Disk full", "2017-01-01 10:00:00")


def test_process_line_skips_non_error_level(checker):
    assert checker._process_line("2017-01-01 10:00:00 INFO all good\n") == (None, None)


def test_process_line_skips_digit_line_with_too_few_tokens(checker):
    assert checker._process_line("2017-01-01\n") == (None, None)


def test_process_line_truncates_long_lines_to_200_chars(checker):
    line = "2017-01-01 12:00:00 ERROR " + "x" * 300 + "\n"
    result, date = checker._process_line(line)
    assert len(result) == 200
    assert result.startswith("ERROR ")
    assert date == "2017-01-01 12:00:00"


def test_process_line_minimal_error_line_has_only_level(checker):
    assert checker._process_line("2017-01-01 10:00:00 ERROR\n") == ("ERROR", "2017-01-01 10:00:00")


# ---------------------------------------------------------------------------
# _group_similar_log_lines
# ---------------------------------------------------------------------------

def test_group_creates_new_entry_with_first_seen(checker):
    common = checker._group_similar_log_lines(
        line="ERROR a", date="2017-01-01 10:00:00", common_lines=OrderedDict()
    )
    assert common["ERROR a"] == {"count": 1, "first_seen": "2017-01-01 10:00:00"}


def test_group_identical_line_increments_count_and_keeps_first_seen(checker):
    common = OrderedDict()
    common = checker._group_similar_log_lines("ERROR a", "2017-01-01 10:00:00", common)
    common = checker._group_similar_log_lines("ERROR a", "2017-01-01 11:00:00", common)
    assert len(common) == 1
    assert common["ERROR a"]["count"] == 2
    assert common["ERROR a"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_similar_line_joins_existing_key(checker):
    common = OrderedDict()
    common = checker._group_similar_log_lines(
        "Connection refused to host alpha port 8080", "2017-01-01 10:00:00", common
    )
    common = checker._group_similar_log_lines(
        "Connection refused to host alpha port 8081", "2017-01-01 10:01:00", common
    )
    assert list(common.keys()) == ["Connection refused to host alpha port 8080"]
    assert common["Connection refused to host alpha port 8080"]["count"] == 2


def test_group_dissimilar_line_creates_new_key(checker):
    common = OrderedDict()
    common = checker._group_similar_log_lines("Disk full", "2017-01-01 10:00:00", common)
    common = checker._group_similar_log_lines(
        "Totally different message here", "2017-01-01 10:01:00", common
    )
    assert list(common.keys()) == ["Disk full", "Totally different message here"]


def test_group_without_date_has_no_first_seen(checker):
    common = checker._group_similar_log_lines("ERROR b", None, OrderedDict())
    assert common["ERROR b"] == {"count": 1}


def test_group_caps_unique_lines_into_other_bucket(checker):
    checker.MAX_COMMON_LINES = 2
    common = OrderedDict()
    common = checker._group_similar_log_lines("alpha alpha", None, common)
    common = checker._group_similar_log_lines("zzzz qqqq xxxx", None, common)
    common = checker._group_similar_log_lines("1234 5678 9012", None, common)
    common = checker._group_similar_log_lines("%%%% ####", None, common)
    assert list(common.keys()) == ["alpha alpha", "zzzz qqqq xxxx", "Other Error Lines"]
    assert common["Other Error Lines"]["count"] == 2


def test_group_similar_line_still_grouped_when_at_cap(checker):
    checker.MAX_COMMON_LINES = 1
    common = OrderedDict()
    common = checker._group_similar_log_lines("Disk full on volume", None, common)
    common = checker._group_similar_log_lines("Disk full on volumes", None, common)
    assert list(common.keys()) == ["Disk full on volume"]
    assert common["Disk full on volume"]["count"] == 2


# ---------------------------------------------------------------------------
# get_error_pattern_counts
# ---------------------------------------------------------------------------

def test_get_error_pattern_counts_end_to_end(checker, tmp_path):
    log_file = tmp_path / "server.log"
    log_file.write_text(
        "2017-01-01 10:00:00 ERROR Database timeout on node 1\n"
        "    at some.stack.frame(File.java:10)\n"
        "2017-01-01 10:01:00 INFO heartbeat ok\n"
        "2017-01-01 10:02:00 ERROR Database timeout on node 2\n"
        "2017-01-01 10:03:00 ERROR Disk full\n"
    )
    filename, common = checker.get_error_pattern_counts(str(log_file))

    assert filename == str(log_file)
    assert list(common.keys()) == ["ERROR Database timeout on node 1", "ERROR Disk full"]
    assert common["ERROR Database timeout on node 1"]["count"] == 2
    assert common["ERROR Database timeout on node 1"]["first_seen"] == "2017-01-01 10:00:00"
    assert common["ERROR Disk full"]["count"] == 1


def test_get_error_pattern_counts_returns_filename_unchanged(checker, tmp_path):
    log_file = tmp_path / "empty.log"
    log_file.write_text("")
    filename, common = checker.get_error_pattern_counts(str(log_file))
    assert filename == str(log_file)
    assert len(common) == 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_returns_none_without_log_files(checker):
    assert checker.run({}) is None


def test_run_returns_none_for_empty_log_files(checker):
    assert checker.run({"LogFiles": []}) is None


def test_run_returns_none_when_no_error_lines(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(similar_log_errors, "Pool", FakePool)
    log_file = tmp_path / "info.log"
    log_file.write_text("2017-01-01 10:00:00 INFO all good\n")
    assert checker.run({"LogFiles": [str(log_file)]}) is None


def test_run_formats_output_and_skips_blacklisted(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(similar_log_errors, "Pool", FakePool)

    app_log = tmp_path / "app.log"
    app_log.write_text(
        "2017-01-01 10:00:00 ERROR Disk full\n"
        "2017-01-01 10:05:00 ERROR Disk full\n"
    )
    config_log = tmp_path / "app_configuration.log"
    config_log.write_text("2017-01-01 10:00:00 ERROR Config broken\n")

    result = checker.run({"LogFiles": [str(app_log), str(config_log)]})

    assert result is not None
    assert f"filename: {app_log}\n" in result
    assert "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Disk full" in result
    assert str(config_log) not in result
    assert "Config broken" not in result
