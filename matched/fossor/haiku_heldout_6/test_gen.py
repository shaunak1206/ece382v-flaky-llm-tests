import logging
from collections import OrderedDict

import pytest

import fossor.checks.similar_log_errors as sle
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool(object):
    """Runs work serially so run() can be tested without spawning processes."""

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
    monkeypatch.setattr(sle, 'Pool', FakePool)
    return FakePool


# ---------------------------------------------------------------------------
# _process_line
# ---------------------------------------------------------------------------

def test_process_line_skips_indented_stacktrace_line(check):
    assert check._process_line('    at com.example.Foo.bar(Foo.java:10)\n') == (None, None)


def test_process_line_skips_blank_line(check):
    assert check._process_line('\n') == (None, None)


def test_process_line_keeps_exception_line_without_date(check):
    line, date = check._process_line('java.lang.NullPointerException: boom\n')
    assert line == 'java.lang.NullPointerException: boom\n'
    assert date is None


def test_process_line_skips_alpha_line_without_exception(check):
    assert check._process_line('Starting server now\n') == (None, None)


def test_process_line_skips_caused_by_line(check):
    assert check._process_line('Caused by: something went wrong\n') == (None, None)


def test_process_line_extracts_error_and_date(check):
    line, date = check._process_line('2017-01-01 10:00:00 ERROR disk full\n')
    assert line == 'ERROR disk full'
    assert date == '2017-01-01 10:00:00'


def test_process_line_skips_non_error_level(check):
    assert check._process_line('2017-01-01 10:00:00 INFO all good\n') == (None, None)


def test_process_line_truncates_long_error_line(check):
    long_message = 'x' * 500
    line, date = check._process_line(f'2017-01-01 10:00:00 ERROR {long_message}\n')
    assert len(line) == 200
    assert line.startswith('ERROR xxx')
    assert date == '2017-01-01 10:00:00'


def test_process_line_skips_short_dated_line_without_level(check):
    assert check._process_line('2017 ERROR\n') == (None, None)


def test_process_line_passes_through_line_with_other_leading_character(check):
    line, date = check._process_line('- some dash-prefixed text\n')
    assert line == '- some dash-prefixed text\n'
    assert date is None


# ---------------------------------------------------------------------------
# _group_similar_log_lines
# ---------------------------------------------------------------------------

def test_group_adds_first_line_with_count_one(check):
    common = check._group_similar_log_lines(
        line='ERROR first', date='2017-01-01 10:00:00', common_lines=OrderedDict())
    assert list(common.keys()) == ['ERROR first']
    assert common['ERROR first']['count'] == 1


def test_group_increments_identical_line(check):
    common = OrderedDict()
    common = check._group_similar_log_lines(line='ERROR same', date=None, common_lines=common)
    common = check._group_similar_log_lines(line='ERROR same', date=None, common_lines=common)
    assert list(common.keys()) == ['ERROR same']
    assert common['ERROR same']['count'] == 2


def test_group_merges_similar_line_into_existing_key(check):
    common = OrderedDict()
    common = check._group_similar_log_lines(
        line='ERROR connection refused to host alpha', date=None, common_lines=common)
    common = check._group_similar_log_lines(
        line='ERROR connection refused to host beta', date=None, common_lines=common)
    assert list(common.keys()) == ['ERROR connection refused to host alpha']
    assert common['ERROR connection refused to host alpha']['count'] == 2


def test_group_keeps_dissimilar_lines_separate(check):
    common = OrderedDict()
    common = check._group_similar_log_lines(line='AAAAAAAAAA', date=None, common_lines=common)
    common = check._group_similar_log_lines(line='bbbbbbbbbb', date=None, common_lines=common)
    assert list(common.keys()) == ['AAAAAAAAAA', 'bbbbbbbbbb']
    assert common['AAAAAAAAAA']['count'] == 1
    assert common['bbbbbbbbbb']['count'] == 1


def test_group_sets_first_seen_only_once(check):
    common = OrderedDict()
    common = check._group_similar_log_lines(
        line='ERROR repeat', date='2017-01-01 10:00:00', common_lines=common)
    common = check._group_similar_log_lines(
        line='ERROR repeat', date='2017-01-02 11:00:00', common_lines=common)
    assert common['ERROR repeat']['first_seen'] == '2017-01-01 10:00:00'
    assert common['ERROR repeat']['count'] == 2


def test_group_without_date_has_no_first_seen(check):
    common = check._group_similar_log_lines(
        line='ERROR no date', date=None, common_lines=OrderedDict())
    assert 'first_seen' not in common['ERROR no date']


def test_group_similarity_threshold_respected(check):
    # With a threshold of 1.0 only identical lines can merge.
    check.SIMILARITY_RATIO = 1.0
    common = OrderedDict()
    common = check._group_similar_log_lines(line='ERROR abc', date=None, common_lines=common)
    common = check._group_similar_log_lines(line='ERROR abd', date=None, common_lines=common)
    assert list(common.keys()) == ['ERROR abc', 'ERROR abd']


def test_group_picks_most_similar_existing_line(check):
    common = OrderedDict()
    common = check._group_similar_log_lines(
        line='ERROR database timeout', date=None, common_lines=common)
    common = check._group_similar_log_lines(
        line='ERROR network failure', date=None, common_lines=common)
    common = check._group_similar_log_lines(
        line='ERROR database timeouts', date=None, common_lines=common)
    assert common['ERROR database timeout']['count'] == 2
    assert common['ERROR network failure']['count'] == 1


def test_group_overflow_goes_to_other_error_lines(check):
    check.MAX_COMMON_LINES = 2
    common = OrderedDict()
    common = check._group_similar_log_lines(line='AAAA', date=None, common_lines=common)
    common = check._group_similar_log_lines(line='bbbb', date=None, common_lines=common)
    common = check._group_similar_log_lines(line='cccc', date=None, common_lines=common)
    assert list(common.keys()) == ['AAAA', 'bbbb', 'Other Error Lines']
    assert common['Other Error Lines']['count'] == 1


def test_group_similar_match_preferred_over_overflow(check):
    check.MAX_COMMON_LINES = 2
    common = OrderedDict()
    common = check._group_similar_log_lines(
        line='ERROR disk full', date=None, common_lines=common)
    common = check._group_similar_log_lines(
        line='AAAAAAAAAAAAAAAA', date=None, common_lines=common)
    common = check._group_similar_log_lines(
        line='ERROR disk full now', date=None, common_lines=common)
    assert 'Other Error Lines' not in common
    assert common['ERROR disk full']['count'] == 2


# ---------------------------------------------------------------------------
# get_error_pattern_counts
# ---------------------------------------------------------------------------

def test_get_error_pattern_counts_aggregates_file(check, tmp_path):
    log_path = tmp_path / 'app.log'
    log_path.write_text(
        '2017-01-01 10:00:00 ERROR disk full\n'
        '2017-01-01 10:00:01 ERROR disk full\n'
        '    at some.stack.Frame(Frame.java:1)\n'
        '2017-01-01 10:00:02 INFO all good\n'
        'Caused by: nothing\n'
    )
    filename, common = check.get_error_pattern_counts(str(log_path))
    assert filename == str(log_path)
    assert list(common.keys()) == ['ERROR disk full']
    assert common['ERROR disk full']['count'] == 2
    assert common['ERROR disk full']['first_seen'] == '2017-01-01 10:00:00'


# ---------------------------------------------------------------------------
# _remove_blacklisted_logs
# ---------------------------------------------------------------------------

def test_remove_blacklisted_logs_filters_names(check):
    files = [
        '/logs/app.log',
        '/logs/x_configuration.log',
        '/logs/gc.log.0',
        '/logs/db_databus.log',
        '/logs/server.out',
        '/logs/web.log',
    ]
    assert check._remove_blacklisted_logs(files) == ['/logs/app.log', '/logs/web.log']


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_returns_none_without_log_files(check):
    assert check.run({}) is None


def test_run_returns_none_for_empty_log_file_list(check):
    assert check.run({'LogFiles': []}) is None


def test_run_formats_report(check, fake_pool, tmp_path):
    log_path = tmp_path / 'service.log'
    log_path.write_text(
        '2017-01-01 10:00:00 ERROR disk full\n'
        '2017-01-01 10:00:01 ERROR disk full\n'
    )
    result = check.run({'LogFiles': [str(log_path)]})
    expected = (
        f"filename: {log_path}\n"
        "First seen: 2017-01-01 10:00:00, Count: 2, ERROR disk full"
        "\n\n"
    )
    assert result == expected


def test_run_returns_none_when_no_error_lines_match(check, fake_pool, tmp_path):
    log_path = tmp_path / 'quiet.log'
    log_path.write_text('2017-01-01 10:00:00 INFO everything fine\n')
    assert check.run({'LogFiles': [str(log_path)]}) is None


def test_run_returns_none_when_all_logs_are_blacklisted(check, fake_pool, tmp_path):
    log_path = tmp_path / 'server.out'
    log_path.write_text('2017-01-01 10:00:00 ERROR disk full\n')
    assert check.run({'LogFiles': [str(log_path)]}) is None
