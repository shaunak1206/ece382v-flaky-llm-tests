import logging
from collections import OrderedDict

import pytest

import fossor.checks.similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Synchronous stand-in for multiprocessing.Pool."""

    def __init__(self, *args, **kwargs):
        pass

    def map(self, func, iterable):
        return [func(x) for x in iterable]

    def close(self):
        pass

    def join(self):
        pass


@pytest.fixture
def check():
    c = SimilarLogErrors()
    if not hasattr(c, 'log'):
        c.log = logging.getLogger('test_similar_log_errors')
    return c


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(sle_module, 'Pool', FakePool)


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs_filters_known_names(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/x.out',
             '/var/log/a_configuration.log', '/var/log/oor.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log']


def test_remove_blacklisted_logs_empty_list(check):
    assert check._remove_blacklisted_logs([]) == []


def test_remove_blacklisted_logs_keeps_all_when_none_match(check):
    files = ['/a/one.log', '/a/two.log']
    assert check._remove_blacklisted_logs(files) == files


# ---------- _process_line ----------

def test_process_line_error_with_date(check):
    line, date = check._process_line('2017-01-01 10:00:00 ERROR Something failed\n')
    assert line == 'ERROR Something failed'
    assert date == '2017-01-01 10:00:00'


def test_process_line_non_error_level_ignored(check):
    line, date = check._process_line('2017-01-01 10:00:00 INFO all good\n')
    assert line is None
    assert date is None


def test_process_line_too_short_line_ignored(check):
    line, date = check._process_line('2017-01-01\n')
    assert line is None
    assert date is None


def test_process_line_whitespace_start_ignored(check):
    line, date = check._process_line('    at com.foo.Bar(Bar.java:10)\n')
    assert line is None
    assert date is None


def test_process_line_alpha_without_exception_ignored(check):
    line, date = check._process_line('Some random text\n')
    assert line is None


def test_process_line_alpha_with_exception_kept(check):
    raw = 'java.lang.NullPointerException: boom\n'
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


def test_process_line_caused_by_ignored(check):
    line, date = check._process_line('Caused by: java.lang.Exception: x\n')
    assert line is None


def test_process_line_truncates_to_200_chars(check):
    raw = '2017-01-01 10:00:00 ERROR ' + 'x' * 500 + '\n'
    line, date = check._process_line(raw)
    assert len(line) == 200
    assert line.startswith('ERROR xxx')
    assert date == '2017-01-01 10:00:00'


# ---------- _group_similar_log_lines ----------

def test_group_adds_new_line_with_count_and_first_seen(check):
    result = check._group_similar_log_lines(
        line='ERROR first', date='2017-01-01 10:00:00', common_lines=OrderedDict())
    assert result['ERROR first']['count'] == 1
    assert result['ERROR first']['first_seen'] == '2017-01-01 10:00:00'


def test_group_increments_count_and_keeps_first_seen(check):
    common = OrderedDict()
    check._group_similar_log_lines('ERROR same thing', '2017-01-01 10:00:00', common)
    check._group_similar_log_lines('ERROR same thing', '2017-01-02 11:00:00', common)
    assert len(common) == 1
    assert common['ERROR same thing']['count'] == 2
    assert common['ERROR same thing']['first_seen'] == '2017-01-01 10:00:00'


def test_group_similar_lines_are_merged_into_first(check):
    common = OrderedDict()
    check._group_similar_log_lines('ERROR Connection to host 10.0.0.1 failed', None, common)
    check._group_similar_log_lines('ERROR Connection to host 10.0.0.2 failed', None, common)
    assert list(common.keys()) == ['ERROR Connection to host 10.0.0.1 failed']
    assert common['ERROR Connection to host 10.0.0.1 failed']['count'] == 2
    assert 'first_seen' not in common['ERROR Connection to host 10.0.0.1 failed']


def test_group_dissimilar_lines_kept_separate(check):
    common = OrderedDict()
    check._group_similar_log_lines('aaaaaaaaaa', None, common)
    check._group_similar_log_lines('zzzzzzzzzz', None, common)
    assert len(common) == 2
    assert common['aaaaaaaaaa']['count'] == 1
    assert common['zzzzzzzzzz']['count'] == 1


def test_group_overflow_goes_to_other_error_lines(check):
    common = OrderedDict()
    for i in range(1, SimilarLogErrors.MAX_COMMON_LINES + 1):
        common['a' * i] = {'count': 1}
    check._group_similar_log_lines('zzzz', None, common)
    check._group_similar_log_lines('zzzz', None, common)
    assert 'zzzz' not in common
    assert common['Other Error Lines']['count'] == 2


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_reads_file(check, tmp_path):
    p = tmp_path / 'app.log'
    p.write_text(
        '2017-01-01 10:00:00 ERROR Something failed\n'
        '2017-01-01 10:00:01 INFO fine\n'
        '    at com.foo.Bar(Bar.java:10)\n'
        'Caused by: something\n'
        '2017-01-01 10:00:02 ERROR Something failed\n'
    )
    filename, common = check.get_error_pattern_counts(str(p))
    assert filename == str(p)
    assert list(common.keys()) == ['ERROR Something failed']
    assert common['ERROR Something failed']['count'] == 2
    assert common['ERROR Something failed']['first_seen'] == '2017-01-01 10:00:00'


# ---------- run ----------

def test_run_without_logfiles_returns_none(check, fake_pool):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, fake_pool, tmp_path):
    p = tmp_path / 'service.log'
    p.write_text(
        '2017-01-01 10:00:00 ERROR Something failed\n'
        '2017-01-01 10:00:05 ERROR Something failed\n'
    )
    result = check.run({'LogFiles': [str(p)]})
    assert result == (
        f'filename: {p}\n'
        'First seen: 2017-01-01 10:00:00, Count: 2, ERROR Something failed\n\n'
    )


def test_run_returns_none_when_no_errors(check, fake_pool, tmp_path):
    p = tmp_path / 'quiet.log'
    p.write_text('2017-01-01 10:00:00 INFO all good\n')
    assert check.run({'LogFiles': [str(p)]}) is None


def test_run_skips_blacklisted_logs(check, fake_pool, tmp_path):
    good = tmp_path / 'service.log'
    bad = tmp_path / 'gc.log'
    good.write_text('2017-01-01 10:00:00 ERROR Real problem\n')
    bad.write_text('2017-01-01 10:00:00 ERROR Ignored problem\n')
    result = check.run({'LogFiles': [str(good), str(bad)]})
    assert 'Real problem' in result
    assert 'Ignored problem' not in result
    assert str(bad) not in result
