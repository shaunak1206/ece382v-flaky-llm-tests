import logging
from collections import OrderedDict

import pytest

from fossor.checks import similar_log_errors
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Synchronous stand-in for multiprocessing.Pool."""

    def __init__(self, *args, **kwargs):
        pass

    def map(self, func, iterable):
        return [func(x) for x in iterable]


@pytest.fixture
def check():
    c = SimilarLogErrors()
    if not hasattr(c, 'log'):
        c.log = logging.getLogger('test_similar_log_errors')
    return c


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(similar_log_errors, 'Pool', FakePool)


def write_log(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text(''.join(line + '\n' for line in lines))
    return str(p)


# ---------- _process_line ----------

def test_process_line_error_line_with_date(check):
    line, date = check._process_line("2017-01-01 10:00:00 ERROR Something bad happened\n")
    assert line == "ERROR Something bad happened"
    assert date == "2017-01-01 10:00:00"


def test_process_line_non_error_level_ignored(check):
    line, date = check._process_line("2017-01-01 10:00:00 INFO All is well\n")
    assert line is None
    assert date is None


def test_process_line_too_few_fields_ignored(check):
    line, date = check._process_line("2017-01-01 10:00:00\n")
    assert line is None
    assert date is None


def test_process_line_whitespace_start_ignored(check):
    line, date = check._process_line("    at com.example.Foo.bar(Foo.java:10)\n")
    assert line is None
    assert date is None


def test_process_line_blank_line_ignored(check):
    line, date = check._process_line("\n")
    assert line is None
    assert date is None


def test_process_line_alpha_without_exception_ignored(check):
    line, date = check._process_line("Some random text\n")
    assert line is None
    assert date is None


def test_process_line_alpha_with_exception_kept(check):
    raw = "java.lang.NullPointerException: oops\n"
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


def test_process_line_caused_by_ignored(check):
    line, date = check._process_line("Caused by: java.lang.RuntimeException: boom\n")
    assert line is None
    assert date is None


def test_process_line_truncates_to_200_chars(check):
    long_msg = 'x' * 500
    line, date = check._process_line(f"2017-01-01 10:00:00 ERROR {long_msg}\n")
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 10:00:00"


# ---------- _group_similar_log_lines ----------

def test_group_adds_new_line_with_first_seen(check):
    result = check._group_similar_log_lines(line="ERROR foo", date="2017-01-01 10:00:00",
                                            common_lines=OrderedDict())
    assert result["ERROR foo"]["count"] == 1
    assert result["ERROR foo"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_similar_lines_are_merged_and_first_seen_kept(check):
    common = OrderedDict()
    common = check._group_similar_log_lines("ERROR Connection to db1 failed", "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines("ERROR Connection to db2 failed", "2017-01-02 11:00:00", common)
    assert list(common.keys()) == ["ERROR Connection to db1 failed"]
    assert common["ERROR Connection to db1 failed"]["count"] == 2
    assert common["ERROR Connection to db1 failed"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_dissimilar_lines_kept_separate(check):
    common = OrderedDict()
    common = check._group_similar_log_lines("aaaaaaaaaa", None, common)
    common = check._group_similar_log_lines("zzzzzzzzzz", None, common)
    assert len(common) == 2
    assert common["aaaaaaaaaa"]["count"] == 1
    assert common["zzzzzzzzzz"]["count"] == 1
    assert 'first_seen' not in common["aaaaaaaaaa"]


def test_group_overflow_goes_to_other_error_lines(check):
    check.MAX_COMMON_LINES = 2
    common = OrderedDict()
    common = check._group_similar_log_lines("aaaaaaaaaa", None, common)
    common = check._group_similar_log_lines("zzzzzzzzzz", None, common)
    common = check._group_similar_log_lines("qqqqqqqqqq", "2017-01-01 10:00:00", common)
    assert "qqqqqqqqqq" not in common
    assert common["Other Error Lines"]["count"] == 1
    assert len(common) == 3


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/app.out',
             '/var/log/x_configuration.log', '/var/log/other.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log', '/var/log/other.log']


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_reads_file(check, tmp_path):
    path = write_log(tmp_path, 'app.log', [
        "2017-01-01 10:00:00 INFO started",
        "2017-01-01 10:00:01 ERROR Connection to db1 failed",
        "    at com.example.Foo.bar(Foo.java:10)",
        "2017-01-01 10:00:02 ERROR Connection to db2 failed",
        "plain text line",
    ])
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert len(common) == 1
    values = common["ERROR Connection to db1 failed"]
    assert values["count"] == 2
    assert values["first_seen"] == "2017-01-01 10:00:01"


# ---------- run ----------

def test_run_without_logfiles_returns_none(check):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, tmp_path, fake_pool):
    path = write_log(tmp_path, 'app.log', [
        "2017-01-01 10:00:01 ERROR Connection to db1 failed",
        "2017-01-01 10:00:02 ERROR Connection to db2 failed",
    ])
    result = check.run({'LogFiles': [path]})
    assert f"filename: {path}\n" in result
    assert ("First seen: 2017-01-01 10:00:01, Count: 2, "
            "ERROR Connection to db1 failed") in result
    assert result.endswith('\n\n')


def test_run_no_errors_returns_none(check, tmp_path, fake_pool):
    path = write_log(tmp_path, 'app.log', [
        "2017-01-01 10:00:01 INFO fine",
        "2017-01-01 10:00:02 DEBUG also fine",
    ])
    assert check.run({'LogFiles': [path]}) is None


def test_run_skips_blacklisted_logs(check, tmp_path, fake_pool):
    bad = write_log(tmp_path, 'gc.log', ["2017-01-01 10:00:01 ERROR gc problem"])
    good = write_log(tmp_path, 'app.log', ["2017-01-01 10:00:01 ERROR app problem"])
    result = check.run({'LogFiles': [bad, good]})
    assert "app problem" in result
    assert "gc problem" not in result
    assert f"filename: {bad}" not in result


def test_run_all_blacklisted_returns_none(check, tmp_path, fake_pool):
    bad = write_log(tmp_path, 'gc.log', ["2017-01-01 10:00:01 ERROR gc problem"])
    assert check.run({'LogFiles': [bad]}) is None
