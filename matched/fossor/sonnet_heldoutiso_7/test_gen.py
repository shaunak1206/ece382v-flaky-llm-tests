import logging
from collections import OrderedDict

import pytest

import fossor.checks.similar_log_errors as module
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Synchronous stand-in for multiprocessing.Pool."""
    last_items = None

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, items):
        items = list(items)
        FakePool.last_items = items
        return [func(i) for i in items]


@pytest.fixture
def check():
    c = SimilarLogErrors()
    if not hasattr(c, 'log'):
        c.log = logging.getLogger('test_similar_log_errors')
    return c


@pytest.fixture
def fake_pool(monkeypatch):
    FakePool.last_items = None
    monkeypatch.setattr(module, 'Pool', FakePool)
    return FakePool


# ---------- _process_line ----------

def test_process_line_skips_indented_stacktrace_line(check):
    assert check._process_line("    at com.foo.Bar(Bar.java:10)\n") == (None, None)


def test_process_line_skips_tab_indented_line(check):
    assert check._process_line("\tat com.foo.Bar\n") == (None, None)


def test_process_line_skips_alpha_line_without_exception(check):
    assert check._process_line("Starting server now\n") == (None, None)


def test_process_line_keeps_alpha_line_with_exception(check):
    line = "java.lang.NullPointerException: boom\n"
    assert check._process_line(line) == (line, None)


def test_process_line_skips_caused_by(check):
    assert check._process_line("Caused by: java.lang.RuntimeException: x\n") == (None, None)


def test_process_line_error_level_splits_date_and_message(check):
    line, date = check._process_line("2017-01-01 10:00:00 ERROR Something bad\n")
    assert line == "ERROR Something bad"
    assert date == "2017-01-01 10:00:00"


def test_process_line_non_error_level_is_skipped(check):
    assert check._process_line("2017-01-01 10:00:00 INFO All good\n") == (None, None)


def test_process_line_too_short_digit_line_is_skipped(check):
    assert check._process_line("2017-01-01 10:00\n") == (None, None)


def test_process_line_truncates_long_lines_to_200_chars(check):
    long_line = "2017-01-01 10:00:00 ERROR " + "x" * 500 + "\n"
    line, date = check._process_line(long_line)
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 10:00:00"


def test_process_line_other_first_character_passes_through(check):
    line = "[main] something odd\n"
    assert check._process_line(line) == (line, None)


# ---------- _group_similar_log_lines ----------

def test_group_new_line_creates_entry_with_first_seen(check):
    result = check._group_similar_log_lines("ERROR a problem", "2017-01-01 10:00:00", OrderedDict())
    assert result["ERROR a problem"] == {'count': 1, 'first_seen': "2017-01-01 10:00:00"}


def test_group_without_date_has_no_first_seen(check):
    result = check._group_similar_log_lines("ERROR a problem", None, OrderedDict())
    assert result["ERROR a problem"] == {'count': 1}


def test_group_similar_lines_increment_count_and_keep_first_seen(check):
    common = OrderedDict()
    check._group_similar_log_lines("ERROR Connection failed to host 1", "2017-01-01 10:00:00", common)
    check._group_similar_log_lines("ERROR Connection failed to host 2", "2017-01-02 11:00:00", common)
    assert list(common.keys()) == ["ERROR Connection failed to host 1"]
    values = common["ERROR Connection failed to host 1"]
    assert values['count'] == 2
    assert values['first_seen'] == "2017-01-01 10:00:00"


def test_group_dissimilar_lines_are_tracked_separately(check):
    common = OrderedDict()
    check._group_similar_log_lines("a" * 30, "d1", common)
    check._group_similar_log_lines("z" * 30, "d2", common)
    assert len(common) == 2
    assert common["a" * 30]['count'] == 1
    assert common["z" * 30]['count'] == 1


def test_group_overflow_goes_to_other_error_lines(check):
    common = OrderedDict()
    for i in range(check.MAX_COMMON_LINES):
        common[chr(97 + i) * 30] = {'count': 1}
    check._group_similar_log_lines("1" * 30, "d", common)
    check._group_similar_log_lines("2" * 30, "d", common)
    assert 'Other Error Lines' in common
    assert common['Other Error Lines']['count'] == 2
    assert "1" * 30 not in common


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs_filters_matching_names(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/stdout.out',
             '/var/log/x_configuration.log', '/var/log/other.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log', '/var/log/other.log']


def test_remove_blacklisted_logs_empty_list(check):
    assert check._remove_blacklisted_logs([]) == []


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_parses_file(check, tmp_path):
    p = tmp_path / "app.log"
    p.write_text(
        "2017-01-01 10:00:00 INFO started\n"
        "2017-01-01 10:00:01 ERROR Connection failed to host 1\n"
        "\n"
        "    at com.foo.Bar(Bar.java:1)\n"
        "2017-01-01 10:00:02 ERROR Connection failed to host 2\n"
        "Caused by: something\n"
        "plain text line\n"
    )
    filename, common = check.get_error_pattern_counts(str(p))
    assert filename == str(p)
    assert len(common) == 1
    values = list(common.values())[0]
    assert values['count'] == 2
    assert values['first_seen'] == "2017-01-01 10:00:01"


def test_get_error_pattern_counts_empty_file(check, tmp_path):
    p = tmp_path / "empty.log"
    p.write_text("")
    filename, common = check.get_error_pattern_counts(str(p))
    assert filename == str(p)
    assert len(common) == 0


# ---------- run ----------

def test_run_returns_none_without_log_files(check, fake_pool):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, fake_pool, tmp_path):
    p = tmp_path / "app.log"
    p.write_text(
        "2017-01-01 10:00:01 ERROR Connection failed to host 1\n"
        "2017-01-01 10:00:02 ERROR Connection failed to host 2\n"
    )
    result = check.run({'LogFiles': [str(p)]})
    assert result.startswith(f"filename: {p}\n")
    assert "First seen: 2017-01-01 10:00:01, Count: 2, ERROR Connection failed to host 1" in result
    assert result.endswith("\n\n")


def test_run_returns_none_when_no_errors(check, fake_pool, tmp_path):
    p = tmp_path / "app.log"
    p.write_text("2017-01-01 10:00:01 INFO fine\n")
    assert check.run({'LogFiles': [str(p)]}) is None


def test_run_excludes_blacklisted_files(check, fake_pool, tmp_path):
    good = tmp_path / "app.log"
    bad = tmp_path / "gc.log"
    line = "2017-01-01 10:00:01 ERROR bad thing\n"
    good.write_text(line)
    bad.write_text(line)
    result = check.run({'LogFiles': [str(good), str(bad)]})
    assert FakePool.last_items == [str(good)]
    assert str(good) in result
    assert str(bad) not in result
