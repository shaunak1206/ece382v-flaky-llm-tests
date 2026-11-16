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
def check(monkeypatch):
    monkeypatch.setattr(similar_log_errors, 'Pool', FakePool)
    c = SimilarLogErrors()
    if not hasattr(c, 'log'):
        c.log = logging.getLogger('test_similar_log_errors')
    return c


def write_log(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs_filters_matching(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/stdout.out',
             '/var/log/x_configuration.log', '/var/log/engine_access_log.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log']


def test_remove_blacklisted_logs_empty_list(check):
    assert check._remove_blacklisted_logs([]) == []


def test_remove_blacklisted_logs_keeps_all_when_none_match(check):
    files = ['/a/one.log', '/a/two.log']
    assert check._remove_blacklisted_logs(files) == files


# ---------- _process_line ----------

def test_process_line_error_with_date(check):
    line, date = check._process_line('2017-01-01 10:00:00 ERROR something bad\n')
    assert line == 'ERROR something bad'
    assert date == '2017-01-01 10:00:00'


def test_process_line_non_error_level_ignored(check):
    line, date = check._process_line('2017-01-01 10:00:00 INFO all fine\n')
    assert line is None
    assert date is None


def test_process_line_too_few_fields_ignored(check):
    line, date = check._process_line('2017-01-01\n')
    assert line is None
    assert date is None


def test_process_line_whitespace_start_ignored(check):
    assert check._process_line('\tat com.foo.Bar(Bar.java:10)\n') == (None, None)
    assert check._process_line('   indented Exception\n') == (None, None)
    assert check._process_line('\n') == (None, None)


def test_process_line_alpha_without_exception_ignored(check):
    assert check._process_line('Some random text\n') == (None, None)


def test_process_line_alpha_with_exception_kept(check):
    text = 'java.lang.RuntimeException: boom\n'
    line, date = check._process_line(text)
    assert line == text
    assert date is None


def test_process_line_caused_by_ignored(check):
    assert check._process_line('Caused by: java.lang.Exception: x\n') == (None, None)


def test_process_line_truncates_to_200_chars(check):
    line, date = check._process_line('2017-01-01 10:00:00 ERROR ' + 'x' * 500 + '\n')
    assert len(line) == 200
    assert line.startswith('ERROR xxx')
    assert date == '2017-01-01 10:00:00'


# ---------- _group_similar_log_lines ----------

def test_group_new_line_creates_entry(check):
    result = check._group_similar_log_lines(line='ERROR foo', date='2017-01-01 10:00:00',
                                            common_lines=OrderedDict())
    assert result['ERROR foo']['count'] == 1
    assert result['ERROR foo']['first_seen'] == '2017-01-01 10:00:00'


def test_group_similar_lines_increment_and_keep_first_seen(check):
    common = OrderedDict()
    common = check._group_similar_log_lines('ERROR connection failed host 1', 'd1', common)
    common = check._group_similar_log_lines('ERROR connection failed host 2', 'd2', common)
    assert list(common.keys()) == ['ERROR connection failed host 1']
    assert common['ERROR connection failed host 1']['count'] == 2
    assert common['ERROR connection failed host 1']['first_seen'] == 'd1'


def test_group_dissimilar_lines_are_separate(check):
    common = OrderedDict()
    common = check._group_similar_log_lines('a' * 20, None, common)
    common = check._group_similar_log_lines('z' * 20, None, common)
    assert len(common) == 2
    assert all(v['count'] == 1 for v in common.values())
    assert 'first_seen' not in common['a' * 20]


def test_group_overflow_goes_to_other_error_lines(check):
    check.MAX_COMMON_LINES = 2
    common = OrderedDict()
    for ch in 'abc':
        common = check._group_similar_log_lines(ch * 20, None, common)
    common = check._group_similar_log_lines('d' * 20, None, common)
    assert 'Other Error Lines' in common
    assert common['Other Error Lines']['count'] == 2
    assert 'd' * 20 not in common


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_groups_errors(check, tmp_path):
    content = (
        '2017-01-01 10:00:00 ERROR Connection failed to host 1\n'
        '2017-01-01 10:00:05 INFO ignored line\n'
        '\tat com.foo.Bar(Bar.java:10)\n'
        '2017-01-01 10:01:00 ERROR Connection failed to host 2\n'
        '\n'
    )
    path = write_log(tmp_path, 'app.log', content)
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert len(common) == 1
    values = list(common.values())[0]
    assert values['count'] == 2
    assert values['first_seen'] == '2017-01-01 10:00:00'


def test_get_error_pattern_counts_no_errors(check, tmp_path):
    path = write_log(tmp_path, 'app.log', '2017-01-01 10:00:00 INFO ok\n')
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert len(common) == 0


# ---------- run ----------

def test_run_without_logfiles_returns_none(check):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, tmp_path):
    content = (
        '2017-01-01 10:00:00 ERROR Connection failed to host 1\n'
        '2017-01-01 10:01:00 ERROR Connection failed to host 2\n'
    )
    path = write_log(tmp_path, 'app.log', content)
    result = check.run({'LogFiles': [path]})
    expected = (
        f'filename: {path}\n'
        'First seen: 2017-01-01 10:00:00, Count: 2, ERROR Connection failed to host 1\n\n'
    )
    assert result == expected


def test_run_returns_none_when_no_errors(check, tmp_path):
    path = write_log(tmp_path, 'app.log', '2017-01-01 10:00:00 INFO fine\n')
    assert check.run({'LogFiles': [path]}) is None


def test_run_skips_blacklisted_logs(check, tmp_path):
    content = '2017-01-01 10:00:00 ERROR bad thing\n'
    gc = write_log(tmp_path, 'gc.log', content)
    app = write_log(tmp_path, 'app.log', content)
    result = check.run({'LogFiles': [gc, app]})
    assert f'filename: {app}' in result
    assert f'filename: {gc}' not in result


def test_run_multiple_files(check, tmp_path):
    a = write_log(tmp_path, 'a.log', '2017-01-01 10:00:00 ERROR alpha failure\n')
    b = write_log(tmp_path, 'b.log', '2017-01-02 11:00:00 ERROR zzzzzzzz qqqqqq\n')
    result = check.run({'LogFiles': [a, b]})
    assert f'filename: {a}' in result
    assert f'filename: {b}' in result
    assert result.count('Count: 1') == 2
