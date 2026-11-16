import logging
from collections import OrderedDict

import pytest

from fossor.checks import similar_log_errors
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Sequential stand-in for multiprocessing.Pool."""

    def map(self, func, iterable):
        return [func(x) for x in iterable]


@pytest.fixture
def check():
    return SimilarLogErrors()


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(similar_log_errors, 'Pool', lambda *args, **kwargs: FakePool())


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs_filters_matching_names(check):
    logs = ['/var/log/app.log', '/var/log/gc.log', '/var/log/x.out',
            '/var/log/a_configuration.log', '/var/log/b_databus.log']
    assert check._remove_blacklisted_logs(logs) == ['/var/log/app.log']


def test_remove_blacklisted_logs_keeps_all_when_none_match(check):
    logs = ['/var/log/a.log', '/var/log/b.log']
    assert check._remove_blacklisted_logs(logs) == logs


def test_remove_blacklisted_logs_empty_list(check):
    assert check._remove_blacklisted_logs([]) == []


# ---------- _process_line ----------

def test_process_line_leading_whitespace_is_skipped(check):
    assert check._process_line('    at com.foo.Bar(Bar.java:10)\n') == (None, None)


def test_process_line_newline_only_is_skipped(check):
    assert check._process_line('\n') == (None, None)


def test_process_line_alpha_without_exception_is_skipped(check):
    assert check._process_line('Some random text\n') == (None, None)


def test_process_line_alpha_with_exception_is_kept(check):
    line, date = check._process_line('java.lang.RuntimeException: boom\n')
    assert line == 'java.lang.RuntimeException: boom\n'
    assert date is None


def test_process_line_caused_by_is_skipped(check):
    assert check._process_line('Caused by: java.lang.Exception: boom\n') == (None, None)


def test_process_line_error_line_extracts_date_and_strips_it(check):
    line, date = check._process_line('2017-01-01 12:00:00 ERROR Something bad\n')
    assert line == 'ERROR Something bad'
    assert date == '2017-01-01 12:00:00'


def test_process_line_non_error_level_is_skipped(check):
    assert check._process_line('2017-01-01 12:00:00 INFO All good\n') == (None, None)


def test_process_line_too_few_fields_is_skipped(check):
    assert check._process_line('2017-01-01 12:00:00\n') == (None, None)


def test_process_line_truncates_to_200_chars(check):
    message = 'x' * 500
    line, date = check._process_line('2017-01-01 12:00:00 ERROR ' + message + '\n')
    assert len(line) == 200
    assert line.startswith('ERROR xxx')
    assert date == '2017-01-01 12:00:00'


def test_process_line_other_start_character_is_returned_unchanged(check):
    line, date = check._process_line('[main] something\n')
    assert line == '[main] something\n'
    assert date is None


# ---------- _group_similar_log_lines ----------

def test_group_new_line_creates_entry_with_first_seen(check):
    result = check._group_similar_log_lines(line='ERROR a', date='2017-01-01 00:00:00',
                                            common_lines=OrderedDict())
    assert result['ERROR a'] == {'count': 1, 'first_seen': '2017-01-01 00:00:00'}


def test_group_without_date_has_no_first_seen(check):
    result = check._group_similar_log_lines(line='Exception x', date=None,
                                            common_lines=OrderedDict())
    assert result['Exception x'] == {'count': 1}


def test_group_similar_lines_increment_count_and_keep_first_seen(check):
    common = OrderedDict()
    common = check._group_similar_log_lines('ERROR connection failed to host 1', 'd1', common)
    common = check._group_similar_log_lines('ERROR connection failed to host 2', 'd2', common)
    assert list(common.keys()) == ['ERROR connection failed to host 1']
    assert common['ERROR connection failed to host 1']['count'] == 2
    assert common['ERROR connection failed to host 1']['first_seen'] == 'd1'


def test_group_dissimilar_lines_are_tracked_separately(check):
    common = OrderedDict()
    common = check._group_similar_log_lines('a' * 10, None, common)
    common = check._group_similar_log_lines('b' * 10, None, common)
    assert len(common) == 2
    assert all(v['count'] == 1 for v in common.values())


def test_group_overflow_goes_to_other_error_lines(check):
    check.MAX_COMMON_LINES = 2
    common = OrderedDict()
    for ch in ('a', 'b', 'c', 'd'):
        common = check._group_similar_log_lines(ch * 10, None, common)
    assert len(common) == 3
    assert common['Other Error Lines']['count'] == 2


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_reads_file(check, tmp_path):
    path = tmp_path / 'app.log'
    path.write_text(
        '2017-01-01 10:00:00 ERROR Something failed\n'
        '2017-01-01 10:00:05 INFO fine\n'
        '    at some.stack.Frame\n'
        '2017-01-01 10:00:10 ERROR Something failed\n'
    )
    filename, common = check.get_error_pattern_counts(str(path))
    assert filename == str(path)
    assert list(common.keys()) == ['ERROR Something failed']
    assert common['ERROR Something failed']['count'] == 2
    assert common['ERROR Something failed']['first_seen'] == '2017-01-01 10:00:00'


def test_get_error_pattern_counts_no_errors(check, tmp_path):
    path = tmp_path / 'quiet.log'
    path.write_text('2017-01-01 10:00:00 INFO fine\n')
    filename, common = check.get_error_pattern_counts(str(path))
    assert filename == str(path)
    assert len(common) == 0


# ---------- run ----------

def test_run_without_logfiles_returns_none(check):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None
    assert check.run({'LogFiles': None}) is None


def test_run_formats_output(check, fake_pool, tmp_path):
    path = tmp_path / 'app.log'
    path.write_text(
        '2017-01-01 10:00:00 ERROR Something failed\n'
        '2017-01-01 10:00:05 INFO ok\n'
        '2017-01-01 10:00:10 ERROR Something failed\n'
    )
    result = check.run({'LogFiles': [str(path)]})
    assert result == (f"filename: {path}\n"
                      "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Something failed\n\n")


def test_run_line_without_date_has_no_first_seen_text(check, fake_pool, tmp_path):
    path = tmp_path / 'app.log'
    path.write_text('Exception in thread main\n')
    result = check.run({'LogFiles': [str(path)]})
    assert result == f"filename: {path}\nCount: 1, Exception in thread main\n\n"


def test_run_returns_none_when_no_errors(check, fake_pool, tmp_path):
    path = tmp_path / 'app.log'
    path.write_text('2017-01-01 10:00:00 INFO ok\n')
    assert check.run({'LogFiles': [str(path)]}) is None


def test_run_ignores_blacklisted_logs(check, fake_pool, tmp_path):
    good = tmp_path / 'app.log'
    bad = tmp_path / 'gc.log'
    content = '2017-01-01 10:00:00 ERROR Something failed\n'
    good.write_text(content)
    bad.write_text(content)
    result = check.run({'LogFiles': [str(good), str(bad)]})
    assert f"filename: {good}" in result
    assert str(bad) not in result


def test_run_multiple_files(check, fake_pool, tmp_path):
    a = tmp_path / 'a.log'
    b = tmp_path / 'b.log'
    a.write_text('2017-01-01 10:00:00 ERROR alpha\n')
    b.write_text('2017-01-02 10:00:00 ERROR zzzzzzzzzz\n')
    result = check.run({'LogFiles': [str(a), str(b)]})
    assert f"filename: {a}\n" in result
    assert f"filename: {b}\n" in result
    assert result.index(str(a)) < result.index(str(b))
