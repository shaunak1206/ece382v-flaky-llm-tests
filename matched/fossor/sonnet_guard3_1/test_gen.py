import collections

import pytest

import fossor.checks.similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Serial stand-in for multiprocessing.Pool."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(x) for x in iterable]


@pytest.fixture
def check():
    return SimilarLogErrors()


@pytest.fixture
def serial_pool(monkeypatch):
    monkeypatch.setattr(sle_module, 'Pool', FakePool)
    monkeypatch.setattr(sle_module, 'cpu_count', lambda: 1)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs_filters_matching(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/x.out',
             '/var/log/my_configuration.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log']


def test_remove_blacklisted_logs_keeps_all_when_none_match(check):
    files = ['/a/one.log', '/a/two.log']
    assert check._remove_blacklisted_logs(files) == files


def test_remove_blacklisted_logs_empty(check):
    assert check._remove_blacklisted_logs([]) == []


# ---------- _process_line ----------

def test_process_line_indented_line_skipped(check):
    assert check._process_line("    at com.foo.Bar(Bar.java:1)\n") == (None, None)


def test_process_line_alpha_without_exception_skipped(check):
    assert check._process_line("Something happened\n") == (None, None)


def test_process_line_alpha_with_exception_kept(check):
    line = "java.lang.NullPointerException: boom\n"
    assert check._process_line(line) == (line, None)


def test_process_line_caused_by_skipped(check):
    line = "Caused by: java.lang.RuntimeException: boom\n"
    assert check._process_line(line) == (None, None)


def test_process_line_error_with_date(check):
    line, date = check._process_line("2017-01-01 10:00:00 ERROR something bad\n")
    assert line == "ERROR something bad"
    assert date == "2017-01-01 10:00:00"


def test_process_line_non_error_level_skipped(check):
    assert check._process_line("2017-01-01 10:00:00 INFO all good\n") == (None, None)


def test_process_line_too_few_fields_skipped(check):
    assert check._process_line("2017-01-01 10:00\n") == (None, None)


def test_process_line_truncates_to_200_chars(check):
    raw = "2017-01-01 10:00:00 ERROR " + "x" * 500 + "\n"
    line, date = check._process_line(raw)
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 10:00:00"


def test_process_line_other_first_char_returned_unchanged(check):
    line = "[main] something\n"
    assert check._process_line(line) == (line, None)


# ---------- _group_similar_log_lines ----------

def test_group_first_line_added_with_date(check):
    result = check._group_similar_log_lines(
        line="ERROR foo", date="2017-01-01 10:00:00", common_lines=collections.OrderedDict())
    assert list(result.keys()) == ["ERROR foo"]
    assert result["ERROR foo"] == {'count': 1, 'first_seen': "2017-01-01 10:00:00"}


def test_group_similar_lines_are_merged_and_keep_first_date(check):
    common = collections.OrderedDict()
    check._group_similar_log_lines("ERROR connection failed to host 1", "d1", common)
    check._group_similar_log_lines("ERROR connection failed to host 2", "d2", common)
    assert list(common.keys()) == ["ERROR connection failed to host 1"]
    values = common["ERROR connection failed to host 1"]
    assert values['count'] == 2
    assert values['first_seen'] == "d1"


def test_group_dissimilar_lines_are_separate(check):
    common = collections.OrderedDict()
    check._group_similar_log_lines("aaaaaaaaaa", None, common)
    check._group_similar_log_lines("bbbbbbbbbb", None, common)
    assert list(common.keys()) == ["aaaaaaaaaa", "bbbbbbbbbb"]
    assert common["aaaaaaaaaa"] == {'count': 1}
    assert 'first_seen' not in common["bbbbbbbbbb"]


def test_group_overflow_goes_to_other_error_lines(check, monkeypatch):
    monkeypatch.setattr(SimilarLogErrors, 'MAX_COMMON_LINES', 2)
    common = collections.OrderedDict()
    check._group_similar_log_lines("aaaaaaaaaa", None, common)
    check._group_similar_log_lines("bbbbbbbbbb", None, common)
    check._group_similar_log_lines("cccccccccc", None, common)
    check._group_similar_log_lines("dddddddddd", None, common)
    assert list(common.keys()) == ["aaaaaaaaaa", "bbbbbbbbbb", "Other Error Lines"]
    assert common["Other Error Lines"]['count'] == 2


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_parses_file(check, tmp_path):
    content = (
        "2017-01-01 10:00:00 INFO started\n"
        "2017-01-01 10:00:01 ERROR Something failed\n"
        "    at some.stack.Frame\n"
        "\n"
        "2017-01-01 10:00:05 ERROR Something failed\n"
        "Caused by: java.lang.Exception\n"
    )
    path = _write(tmp_path, "app.log", content)
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert list(common.keys()) == ["ERROR Something failed"]
    assert common["ERROR Something failed"] == {'count': 2, 'first_seen': "2017-01-01 10:00:01"}


# ---------- run ----------

def test_run_without_logfiles_returns_none(check):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, tmp_path, serial_pool):
    content = (
        "2017-01-01 10:00:00 ERROR Something failed\n"
        "2017-01-01 10:00:05 ERROR Something failed\n"
    )
    path = _write(tmp_path, "app.log", content)
    result = check.run({'LogFiles': [path]})
    expected = (f"filename: {path}\n"
                "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Something failed\n\n")
    assert result == expected


def test_run_returns_none_when_no_errors(check, tmp_path, serial_pool):
    path = _write(tmp_path, "app.log", "2017-01-01 10:00:00 INFO fine\n")
    assert check.run({'LogFiles': [path]}) is None


def test_run_skips_blacklisted_logs(check, tmp_path, serial_pool):
    content = "2017-01-01 10:00:00 ERROR Something failed\n"
    good = _write(tmp_path, "app.log", content)
    bad = _write(tmp_path, "gc.log", content)
    result = check.run({'LogFiles': [good, bad]})
    assert f"filename: {good}\n" in result
    assert bad not in result
