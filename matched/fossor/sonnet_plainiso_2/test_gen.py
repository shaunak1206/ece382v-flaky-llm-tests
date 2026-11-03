from collections import OrderedDict
from unittest import mock

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Serial stand-in for multiprocessing.Pool."""

    def __init__(self, *args, **kwargs):
        pass

    def map(self, func, iterable):
        return [func(x) for x in iterable]


@pytest.fixture
def check():
    return SimilarLogErrors()


def write_log(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text(''.join(lines))
    return str(path)


# ---------------------------------------------------------------- _process_line

def test_process_line_skips_indented_stacktrace(check):
    assert check._process_line('    at com.foo.Bar(Bar.java:10)\n') == (None, None)


def test_process_line_skips_blank_newline(check):
    assert check._process_line('\n') == (None, None)


def test_process_line_skips_plain_alpha_line_without_exception(check):
    assert check._process_line('Starting server on port 80\n') == (None, None)


def test_process_line_keeps_alpha_line_with_exception(check):
    line = 'java.lang.RuntimeException: boom\n'
    assert check._process_line(line) == (line, None)


def test_process_line_skips_caused_by(check):
    assert check._process_line('Caused by: java.io.IOException: nope\n') == (None, None)


def test_process_line_error_level_extracts_date_and_message(check):
    line, date = check._process_line('2017-01-01 10:00:00 ERROR failed to connect\n')
    assert line == 'ERROR failed to connect'
    assert date == '2017-01-01 10:00:00'


def test_process_line_non_error_level_is_skipped(check):
    assert check._process_line('2017-01-01 10:00:00 INFO all good\n') == (None, None)
    assert check._process_line('2017-01-01 10:00:00 WARN hmm\n') == (None, None)


def test_process_line_too_few_fields_is_skipped(check):
    assert check._process_line('2017-01-01 10:00:00\n') == (None, None)


def test_process_line_truncates_to_200_chars(check):
    long_line = '2017-01-01 10:00:00 ERROR ' + 'x' * 500 + '\n'
    line, date = check._process_line(long_line)
    assert len(line) == 200
    assert line.startswith('ERROR xxx')
    assert date == '2017-01-01 10:00:00'


# ------------------------------------------------------ _group_similar_log_lines

def test_group_first_line_creates_entry_with_first_seen(check):
    result = check._group_similar_log_lines(line='ERROR a', date='2017-01-01 10:00:00', common_lines=OrderedDict())
    assert result == OrderedDict([('ERROR a', {'count': 1, 'first_seen': '2017-01-01 10:00:00'})])


def test_group_identical_lines_increment_and_keep_first_seen(check):
    common = OrderedDict()
    check._group_similar_log_lines('ERROR same thing', 'd1', common)
    check._group_similar_log_lines('ERROR same thing', 'd2', common)
    assert list(common.keys()) == ['ERROR same thing']
    assert common['ERROR same thing']['count'] == 2
    assert common['ERROR same thing']['first_seen'] == 'd1'


def test_group_similar_lines_merge_under_first_line(check):
    common = OrderedDict()
    check._group_similar_log_lines('ERROR connection to host 10.0.0.1 failed', 'd1', common)
    check._group_similar_log_lines('ERROR connection to host 10.0.0.2 failed', 'd2', common)
    assert list(common.keys()) == ['ERROR connection to host 10.0.0.1 failed']
    assert common['ERROR connection to host 10.0.0.1 failed']['count'] == 2


def test_group_dissimilar_lines_stay_separate(check):
    common = OrderedDict()
    check._group_similar_log_lines('aaaaaaaaaa', None, common)
    check._group_similar_log_lines('zzzzzzzzzz', None, common)
    assert list(common.keys()) == ['aaaaaaaaaa', 'zzzzzzzzzz']
    assert all(v['count'] == 1 for v in common.values())


def test_group_without_date_has_no_first_seen(check):
    common = check._group_similar_log_lines('ERROR x', None, OrderedDict())
    assert common['ERROR x'] == {'count': 1}


def test_group_overflow_goes_to_other_error_lines(check):
    check.MAX_COMMON_LINES = 2
    check.SIMILARITY_RATIO = 2.0  # nothing can ever be considered similar
    common = OrderedDict()
    for line in ['line a', 'line b', 'line c', 'line d']:
        check._group_similar_log_lines(line, None, common)
    assert list(common.keys()) == ['line a', 'line b', 'Other Error Lines']
    assert common['Other Error Lines']['count'] == 2


# ----------------------------------------------------- _remove_blacklisted_logs

def test_remove_blacklisted_logs_filters_known_names(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/stdout.out',
             '/var/log/x_configuration.log', '/var/log/other.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log', '/var/log/other.log']


# ----------------------------------------------------- get_error_pattern_counts

def test_get_error_pattern_counts_counts_only_error_lines(check, tmp_path):
    path = write_log(tmp_path, 'app.log', [
        '2017-01-01 10:00:00 ERROR failed to connect\n',
        '2017-01-01 10:00:05 ERROR failed to connect\n',
        '2017-01-01 10:00:06 INFO ok\n',
        '    at some.stack.Frame\n',
    ])
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert list(common.keys()) == ['ERROR failed to connect']
    assert common['ERROR failed to connect']['count'] == 2
    assert common['ERROR failed to connect']['first_seen'] == '2017-01-01 10:00:00'


# -------------------------------------------------------------------------- run

def test_run_without_logfiles_returns_none(check):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, tmp_path):
    path = write_log(tmp_path, 'app.log', [
        '2017-01-01 10:00:00 ERROR failed to connect\n',
        '2017-01-01 10:00:05 ERROR failed to connect\n',
    ])
    with mock.patch('fossor.checks.similar_log_errors.Pool', FakePool):
        out = check.run({'LogFiles': [path]})
    assert out == (f"filename: {path}\n"
                   "First seen: 2017-01-01 10:00:00, Count: 2, ERROR failed to connect\n\n")


def test_run_returns_none_when_no_errors_or_all_blacklisted(check, tmp_path):
    clean = write_log(tmp_path, 'clean.log', ['2017-01-01 10:00:00 INFO fine\n'])
    gc = write_log(tmp_path, 'gc.log', ['2017-01-01 10:00:00 ERROR ignored\n'])
    with mock.patch('fossor.checks.similar_log_errors.Pool', FakePool):
        assert check.run({'LogFiles': [clean]}) is None
        assert check.run({'LogFiles': [gc]}) is None


def test_run_exception_line_has_no_first_seen_text(check, tmp_path):
    path = write_log(tmp_path, 'app.log', ['java.lang.RuntimeException: boom\n'])
    with mock.patch('fossor.checks.similar_log_errors.Pool', FakePool):
        out = check.run({'LogFiles': [path]})
    assert out == f"filename: {path}\nCount: 1, java.lang.RuntimeException: boom\n\n"


def test_run_multiple_files_and_run_helper(check, tmp_path):
    p1 = write_log(tmp_path, 'one.log', ['2017-01-01 10:00:00 ERROR alpha problem\n'])
    p2 = write_log(tmp_path, 'two.log', ['2017-01-02 11:00:00 ERROR zzz\n'])
    with mock.patch('fossor.checks.similar_log_errors.Pool', FakePool):
        out = check.run_helper({'LogFiles': [p1, p2]})
    assert f"filename: {p1}\n" in out
    assert f"filename: {p2}\n" in out
    assert out.index(p1) < out.index(p2)
    assert 'Count: 1, ERROR alpha problem' in out
    assert 'First seen: 2017-01-02 11:00:00, Count: 1, ERROR zzz' in out
