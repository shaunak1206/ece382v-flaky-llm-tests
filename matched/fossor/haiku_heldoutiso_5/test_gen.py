import logging
from collections import OrderedDict
import string

import pytest

import fossor.checks.similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Runs Pool.map serially so tests do not depend on multiprocessing."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]


@pytest.fixture
def check():
    # Build the instance without relying on Check.__init__ details;
    # only self.log is needed by the methods under test.
    obj = SimilarLogErrors.__new__(SimilarLogErrors)
    obj.log = logging.getLogger("test_similar_log_errors")
    return obj


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(sle_module, "Pool", FakePool)
    return FakePool


# ---------------------------------------------------------------------------
# _process_line
# ---------------------------------------------------------------------------

def test_process_line_whitespace_start_is_skipped(check):
    assert check._process_line("    at com.example.Foo\n") == (None, None)


def test_process_line_alpha_without_exception_is_skipped(check):
    assert check._process_line("Starting service now\n") == (None, None)


def test_process_line_alpha_with_exception_is_kept_without_date(check):
    line = "NullPointerException happened here\n"
    assert check._process_line(line) == (line, None)


def test_process_line_caused_by_is_skipped(check):
    assert check._process_line("Caused by: java.lang.Exception: boom\n") == (None, None)


def test_process_line_dated_error_splits_date_and_message(check):
    line, date = check._process_line("2017-01-01 10:00:00 ERROR disk failure\n")
    assert line == "ERROR disk failure"
    assert date == "2017-01-01 10:00:00"


def test_process_line_dated_non_error_level_is_skipped(check):
    assert check._process_line("2017-01-01 10:00:00 INFO all good\n") == (None, None)


def test_process_line_dated_line_too_short_returns_none(check):
    assert check._process_line("2017-01-01\n") == (None, None)


def test_process_line_truncates_long_messages_to_200_chars(check):
    long_message = "x" * 500
    line, date = check._process_line(f"2017-01-01 10:00:00 ERROR {long_message}\n")
    assert date == "2017-01-01 10:00:00"
    assert len(line) == 200
    assert line.startswith("ERROR ")


def test_process_line_other_leading_char_passes_through_unchanged(check):
    raw = "[INFO] something\n"
    assert check._process_line(raw) == (raw, None)


# ---------------------------------------------------------------------------
# _group_similar_log_lines
# ---------------------------------------------------------------------------

def test_group_adds_first_line_with_count_one(check):
    result = check._group_similar_log_lines(
        line="ERROR first", date="2017-01-01 00:00:00", common_lines=OrderedDict()
    )
    assert list(result.keys()) == ["ERROR first"]
    assert result["ERROR first"]["count"] == 1


def test_group_identical_line_increments_count(check):
    common = OrderedDict()
    common = check._group_similar_log_lines("ERROR same", None, common)
    common = check._group_similar_log_lines("ERROR same", None, common)
    assert len(common) == 1
    assert common["ERROR same"]["count"] == 2


def test_group_dissimilar_line_creates_new_entry(check):
    common = OrderedDict()
    common = check._group_similar_log_lines("aaaaaaaaaa", None, common)
    common = check._group_similar_log_lines("bbbbbbbbbb", None, common)
    assert set(common.keys()) == {"aaaaaaaaaa", "bbbbbbbbbb"}
    assert common["aaaaaaaaaa"]["count"] == 1
    assert common["bbbbbbbbbb"]["count"] == 1


def test_group_similar_line_joins_most_similar_existing_key(check):
    common = OrderedDict([
        ("xxxxxxxxxx", {"count": 1}),
        ("abcdefghij", {"count": 1}),
    ])
    common = check._group_similar_log_lines("abcdefghiz", None, common)
    assert len(common) == 2
    assert "abcdefghiz" not in common
    assert common["abcdefghij"]["count"] == 2
    assert common["xxxxxxxxxx"]["count"] == 1


def test_group_first_seen_keeps_earliest_date(check):
    common = OrderedDict()
    common = check._group_similar_log_lines("ERROR repeat", "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines("ERROR repeat", "2017-01-02 10:00:00", common)
    assert common["ERROR repeat"]["first_seen"] == "2017-01-01 10:00:00"
    assert common["ERROR repeat"]["count"] == 2


def test_group_none_date_does_not_set_first_seen(check):
    common = check._group_similar_log_lines("ERROR nodate", None, OrderedDict())
    assert "first_seen" not in common["ERROR nodate"]


def test_group_caps_unique_lines_into_other_bucket(check):
    common = OrderedDict()
    # 20 mutually dissimilar lines fill up MAX_COMMON_LINES.
    for letter in string.ascii_lowercase[:20]:
        common = check._group_similar_log_lines(letter * 10, None, common)
    assert len(common) == 20

    common = check._group_similar_log_lines("z" * 10, None, common)
    assert "Other Error Lines" in common
    assert common["Other Error Lines"]["count"] == 1
    assert "z" * 10 not in common
    assert len(common) == 21


# ---------------------------------------------------------------------------
# _remove_blacklisted_logs
# ---------------------------------------------------------------------------

def test_remove_blacklisted_logs_filters_known_names(check):
    files = [
        "/logs/app.log",
        "/logs/server.out",
        "/logs/gc.log",
        "/logs/service.log",
        "/logs/foo_databus.log",
    ]
    assert check._remove_blacklisted_logs(files) == [
        "/logs/app.log",
        "/logs/service.log",
    ]


# ---------------------------------------------------------------------------
# get_error_pattern_counts
# ---------------------------------------------------------------------------

def test_get_error_pattern_counts_counts_error_lines(check, tmp_path):
    log = tmp_path / "app.log"
    log.write_text(
        "2017-01-01 10:00:00 ERROR disk failure\n"
        "    at some.stack.frame\n"
        "2017-01-01 10:05:00 ERROR disk failure\n"
        "2017-01-01 10:06:00 INFO fine\n"
    )
    filename, common_lines = check.get_error_pattern_counts(str(log))
    assert filename == str(log)
    assert list(common_lines.keys()) == ["ERROR disk failure"]
    assert common_lines["ERROR disk failure"]["count"] == 2
    assert common_lines["ERROR disk failure"]["first_seen"] == "2017-01-01 10:00:00"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_returns_none_without_log_files(check):
    assert check.run({}) is None
    assert check.run({"LogFiles": []}) is None


def test_run_returns_none_when_no_error_lines(check, fake_pool, tmp_path):
    log = tmp_path / "app.log"
    log.write_text("2017-01-01 10:06:00 INFO fine\n")
    assert check.run({"LogFiles": [str(log)]}) is None


def test_run_formats_output(check, fake_pool, tmp_path):
    log = tmp_path / "app.log"
    log.write_text(
        "2017-01-01 10:00:00 ERROR disk failure\n"
        "2017-01-01 10:05:00 ERROR disk failure\n"
    )
    result = check.run({"LogFiles": [str(log)]})
    expected = (
        f"filename: {log}\n"
        "First seen: 2017-01-01 10:00:00, Count: 2, ERROR disk failure\n\n"
    )
    assert result == expected


def test_run_excludes_blacklisted_files(check, fake_pool, tmp_path):
    app = tmp_path / "app.log"
    app.write_text("2017-01-01 10:00:00 ERROR real problem\n")
    result = check.run({"LogFiles": [str(app), "/nonexistent/gc.log"]})
    assert result is not None
    assert str(app) in result
    assert "gc.log" not in result
