import logging
import os
from collections import OrderedDict

import pytest

from fossor.checks import similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


ERROR_LINE = "2017-01-01 10:00:00 ERROR Database connection failed for user alice\n"


class FakePool:
    """Serial stand-in for multiprocessing.Pool so run() can be tested deterministically."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]


@pytest.fixture
def check():
    instance = SimilarLogErrors()
    if not hasattr(instance, 'log'):
        instance.log = logging.getLogger('test_similar_log_errors')
    return instance


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(sle_module, 'Pool', FakePool)
    return FakePool


def write_log(directory, name, content):
    path = os.path.join(str(directory), name)
    with open(path, 'w') as handle:
        handle.write(content)
    return path


# ---------- run() ----------

def test_run_returns_none_when_no_log_files_key(check):
    assert check.run({}) is None


def test_run_returns_none_when_log_files_empty(check):
    assert check.run({'LogFiles': []}) is None


def test_run_returns_none_when_no_error_lines(check, tmp_path, fake_pool):
    path = write_log(tmp_path, 'app.log', "2017-01-01 10:00:00 INFO all good\n")
    assert check.run({'LogFiles': [path]}) is None


def test_run_formats_output_with_fake_pool(check, tmp_path, fake_pool):
    path_a = write_log(tmp_path, 'a.log', ERROR_LINE + ERROR_LINE)
    path_b = write_log(tmp_path, 'b.log', ERROR_LINE)

    result = check.run({'LogFiles': [path_a, path_b]})

    assert isinstance(result, str)
    assert f"filename: {path_a}\n" in result
    assert f"filename: {path_b}\n" in result
    assert "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Database connection failed for user alice" in result
    assert "Count: 1, ERROR Database connection failed for user alice" in result


def test_run_skips_blacklisted_files(check, tmp_path, fake_pool):
    gc_path = write_log(tmp_path, 'gc.log', ERROR_LINE)
    app_path = write_log(tmp_path, 'app.log', ERROR_LINE)

    result = check.run({'LogFiles': [gc_path, app_path]})

    assert result is not None
    assert f"filename: {app_path}\n" in result
    assert gc_path not in result


# ---------- _remove_blacklisted_logs() ----------

def test_remove_blacklisted_logs_drops_each_blacklisted_pattern(check):
    files = [f"/logs/host{pattern}" for pattern in check.log_name_blacklist]
    assert check._remove_blacklisted_logs(files) == []


def test_remove_blacklisted_logs_keeps_regular_files(check):
    files = ['/logs/app.log', '/logs/server/errors.log']
    assert check._remove_blacklisted_logs(files) == files


def test_remove_blacklisted_logs_preserves_order(check):
    files = ['/logs/b.log', '/logs/gc.log', '/logs/a.log', '/logs/oor.log', '/logs/c.log']
    assert check._remove_blacklisted_logs(files) == ['/logs/b.log', '/logs/a.log', '/logs/c.log']


# ---------- _process_line() ----------

def test_process_line_whitespace_start_is_ignored(check):
    assert check._process_line("    at com.example.Foo.bar(Foo.java:10)\n") == (None, None)


def test_process_line_alpha_start_without_exception_is_ignored(check):
    assert check._process_line("INFO something happened\n") == (None, None)


def test_process_line_exception_line_is_returned_unchanged(check):
    line = "SomeException: something broke\n"
    assert check._process_line(line) == (line, None)


def test_process_line_caused_by_is_ignored(check):
    assert check._process_line("Caused by: java.io.IOException\n") == (None, None)


def test_process_line_error_entry_splits_date_and_message(check):
    line, date = check._process_line("2017-01-01 10:00:00 ERROR disk is full\n")
    assert line == "ERROR disk is full"
    assert date == "2017-01-01 10:00:00"


def test_process_line_non_error_level_is_ignored(check):
    assert check._process_line("2017-01-01 10:00:00 WARN disk is nearly full\n") == (None, None)


def test_process_line_short_digit_line_is_ignored(check):
    assert check._process_line("2017-01-01 10:00\n") == (None, None)


def test_process_line_truncates_long_messages(check):
    long_message = "x" * 500
    line, date = check._process_line(f"2017-01-01 10:00:00 ERROR {long_message}\n")
    assert len(line) == 200
    assert line.startswith("ERROR ")
    assert date == "2017-01-01 10:00:00"


# ---------- _group_similar_log_lines() ----------

def test_group_first_line_creates_entry_with_first_seen(check):
    common = check._group_similar_log_lines(
        line="ERROR disk is full", date="2017-01-01 10:00:00", common_lines=OrderedDict())
    assert list(common.keys()) == ["ERROR disk is full"]
    assert common["ERROR disk is full"]["count"] == 1
    assert common["ERROR disk is full"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_identical_line_increments_count_and_keeps_first_seen(check):
    common = OrderedDict()
    common = check._group_similar_log_lines("ERROR disk is full", "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines("ERROR disk is full", "2017-01-02 11:00:00", common)
    assert len(common) == 1
    assert common["ERROR disk is full"]["count"] == 2
    assert common["ERROR disk is full"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_similar_line_merges_into_existing_entry(check):
    first = "ERROR connection refused to host alpha on port 8080"
    second = "ERROR connection refused to host alpha on port 8081"
    common = OrderedDict()
    common = check._group_similar_log_lines(first, "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines(second, "2017-01-01 10:05:00", common)
    assert len(common) == 1
    assert list(common.keys()) == [first]
    assert common[first]["count"] == 2


def test_group_dissimilar_line_creates_new_entry(check):
    common = OrderedDict()
    common = check._group_similar_log_lines(
        "ERROR connection refused to host alpha on port 8080", "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines("ERROR disk is full", "2017-01-01 10:01:00", common)
    assert len(common) == 2
    assert common["ERROR disk is full"]["count"] == 1


def test_group_without_date_has_no_first_seen(check):
    common = check._group_similar_log_lines("ERROR disk is full", None, OrderedDict())
    assert common["ERROR disk is full"]["count"] == 1
    assert 'first_seen' not in common["ERROR disk is full"]


def test_group_caps_unique_lines_into_other_bucket(check):
    # Keys contain only 'q' and digits, so a line of "ERROR" shares no characters
    # with any of them and cannot be merged; the cap then forces the 'Other' bucket.
    common = OrderedDict(
        (f"qq{i:02d}", {'count': 1}) for i in range(check.MAX_COMMON_LINES)
    )
    common = check._group_similar_log_lines("ERROR", None, common)
    assert len(common) == check.MAX_COMMON_LINES + 1
    assert common['Other Error Lines']['count'] == 1


# ---------- get_error_pattern_counts() ----------

def test_get_error_pattern_counts_aggregates_file(check, tmp_path):
    content = (
        ERROR_LINE
        + "    at com.example.Foo.bar(Foo.java:10)\n"
        + "2017-01-01 10:00:01 INFO something fine\n"
        + "2017-01-01 10:00:02 ERROR Database connection failed for user alice\n"
    )
    path = write_log(tmp_path, 'app.log', content)

    filename, common = check.get_error_pattern_counts(path)

    assert filename == path
    assert isinstance(common, OrderedDict)
    key = "ERROR Database connection failed for user alice"
    assert list(common.keys()) == [key]
    assert common[key]['count'] == 2
    assert common[key]['first_seen'] == "2017-01-01 10:00:00"
