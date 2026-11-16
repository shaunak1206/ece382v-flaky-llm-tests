import logging

import pytest

import fossor.checks.similar_log_errors as sle_module
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
    monkeypatch.setattr(sle_module, 'Pool', FakePool)


# ---------- _process_line ----------

def test_process_line_skips_indented_stacktrace_line(check):
    assert check._process_line('    at com.foo.Bar(Bar.java:10)\n') == (None, None)


def test_process_line_skips_tab_indented_line(check):
    assert check._process_line('\tat com.foo.Bar(Bar.java:10)\n') == (None, None)


def test_process_line_skips_alpha_line_without_exception(check):
    assert check._process_line('Some random message\n') == (None, None)


def test_process_line_keeps_alpha_line_with_exception(check):
    line = 'java.lang.NullPointerException: oops\n'
    assert check._process_line(line) == (line, None)


def test_process_line_skips_caused_by(check):
    assert check._process_line('Caused by: java.lang.RuntimeException: x\n') == (None, None)


def test_process_line_error_line_extracts_date_and_strips_it(check):
    line, date = check._process_line('2017-01-01 10:00:00 ERROR Something bad happened\n')
    assert line == 'ERROR Something bad happened'
    assert date == '2017-01-01 10:00:00'


def test_process_line_non_error_level_is_skipped(check):
    assert check._process_line('2017-01-01 10:00:00 INFO All good\n') == (None, None)


def test_process_line_too_few_fields_is_skipped(check):
    assert check._process_line('2017-01-01 10:00\n') == (None, None)


def test_process_line_truncates_long_lines_to_200_chars(check):
    long_msg = 'x' * 500
    line, date = check._process_line(f'2017-01-01 10:00:00 ERROR {long_msg}\n')
    assert len(line) == 200
    assert line.startswith('ERROR xxx')
    assert date == '2017-01-01 10:00:00'


def test_process_line_other_leading_char_returned_unchanged(check):
    line = '[main] something\n'
    assert check._process_line(line) == (line, None)


# ---------- _group_similar_log_lines ----------

def test_group_new_line_creates_entry_with_count_and_first_seen(check):
    result = check._group_similar_log_lines(line='ERROR foo', date='2017-01-01 10:00:00', common_lines={})
    assert result['ERROR foo']['count'] == 1
    assert result['ERROR foo']['first_seen'] == '2017-01-01 10:00:00'


def test_group_without_date_has_no_first_seen(check):
    result = check._group_similar_log_lines(line='ERROR foo', date=None, common_lines={})
    assert result['ERROR foo']['count'] == 1
    assert 'first_seen' not in result['ERROR foo']


def test_group_similar_lines_increment_and_keep_first_seen(check):
    common = {}
    check._group_similar_log_lines('ERROR Connection failed to host 1', 'd1', common)
    check._group_similar_log_lines('ERROR Connection failed to host 2', 'd2', common)
    assert list(common.keys()) == ['ERROR Connection failed to host 1']
    assert common['ERROR Connection failed to host 1']['count'] == 2
    assert common['ERROR Connection failed to host 1']['first_seen'] == 'd1'


def test_group_dissimilar_lines_are_tracked_separately(check):
    common = {}
    check._group_similar_log_lines('aaaaaaaaaa', None, common)
    check._group_similar_log_lines('zzzzzzzzzz', None, common)
    assert len(common) == 2
    assert common['aaaaaaaaaa']['count'] == 1
    assert common['zzzzzzzzzz']['count'] == 1


def test_group_overflow_goes_to_other_error_lines(check):
    check.MAX_COMMON_LINES = 2
    common = {}
    check._group_similar_log_lines('zzzzzzzzzz', None, common)
    check._group_similar_log_lines('yyyyyyyyyy', None, common)
    check._group_similar_log_lines('xxxxxxxxxx', None, common)
    check._group_similar_log_lines('wwwwwwwwww', None, common)
    assert 'xxxxxxxxxx' not in common
    assert 'wwwwwwwwww' not in common
    assert common['Other Error Lines']['count'] == 2


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/x.out',
             '/var/log/service_configuration.log', '/var/log/other.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log', '/var/log/other.log']


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_reads_file(check, tmp_path):
    p = tmp_path / 'app.log'
    p.write_text(
        '2017-01-01 10:00:00 INFO started\n'
        '2017-01-01 10:00:01 ERROR Connection failed to host 1\n'
        '    at com.foo.Bar(Bar.java:10)\n'
        'Caused by: something\n'
        '2017-01-01 10:00:05 ERROR Connection failed to host 2\n'
    )
    filename, common = check.get_error_pattern_counts(str(p))
    assert filename == str(p)
    assert len(common) == 1
    values = common['ERROR Connection failed to host 1']
    assert values['count'] == 2
    assert values['first_seen'] == '2017-01-01 10:00:01'


# ---------- run ----------

def test_run_returns_none_without_log_files(check, fake_pool):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, fake_pool, tmp_path):
    p = tmp_path / 'app.log'
    p.write_text(
        '2017-01-01 10:00:01 ERROR Connection failed to host 1\n'
        '2017-01-01 10:00:05 ERROR Connection failed to host 2\n'
    )
    result = check.run({'LogFiles': [str(p)]})
    assert f'filename: {p}' in result
    assert 'First seen: 2017-01-01 10:00:01, ' in result
    assert 'Count: 2, ' in result
    assert 'ERROR Connection failed to host 1' in result


def test_run_returns_none_when_no_errors(check, fake_pool, tmp_path):
    p = tmp_path / 'app.log'
    p.write_text('2017-01-01 10:00:00 INFO fine\n')
    assert check.run({'LogFiles': [str(p)]}) is None


def test_run_ignores_blacklisted_logs(check, fake_pool, tmp_path):
    p = tmp_path / 'gc.log'
    p.write_text('2017-01-01 10:00:00 ERROR bad thing\n')
    assert check.run({'LogFiles': [str(p)]}) is None
