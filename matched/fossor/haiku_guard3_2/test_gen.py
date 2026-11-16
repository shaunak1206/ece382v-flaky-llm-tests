import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


def _make_check():
    return SimilarLogErrors()


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_returns_none_when_log_files_key_missing():
    check = _make_check()
    assert check.run({}) is None


def test_run_returns_none_when_log_files_empty():
    check = _make_check()
    assert check.run({'LogFiles': []}) is None


# ---------------------------------------------------------------------------
# _remove_blacklisted_logs()
# ---------------------------------------------------------------------------

def test_remove_blacklisted_logs_filters_known_suffixes():
    check = _make_check()
    files = [
        '/logs/app.log',
        '/logs/app_configuration.log',
        '/logs/gc.log',
        '/logs/server.out',
        '/logs/db.log',
        '/logs/app_databus.log',
    ]
    assert check._remove_blacklisted_logs(files) == ['/logs/app.log', '/logs/db.log']


def test_remove_blacklisted_logs_keeps_all_when_none_match():
    check = _make_check()
    files = ['/logs/a.log', '/logs/b.log']
    assert check._remove_blacklisted_logs(files) == ['/logs/a.log', '/logs/b.log']


def test_remove_blacklisted_logs_empty_input():
    check = _make_check()
    assert check._remove_blacklisted_logs([]) == []


# ---------------------------------------------------------------------------
# _process_line()
# ---------------------------------------------------------------------------

def test_process_line_skips_stacktrace_continuation():
    check = _make_check()
    assert check._process_line('    at com.example.Foo.bar(Foo.java:10)\n') == (None, None)


def test_process_line_skips_alpha_line_without_exception():
    check = _make_check()
    assert check._process_line('random text without a date\n') == (None, None)


def test_process_line_keeps_alpha_exception_line_without_date():
    check = _make_check()
    line = 'java.lang.NullPointerException happened\n'
    assert check._process_line(line) == (line, None)


def test_process_line_skips_caused_by_even_with_exception():
    check = _make_check()
    assert check._process_line('Caused by: java.lang.Exception: boom\n') == (None, None)


def test_process_line_parses_error_line_into_message_and_date():
    check = _make_check()
    line = '2017-01-01 10:00:00 ERROR Something failed badly\n'
    message, date = check._process_line(line)
    assert message == 'ERROR Something failed badly'
    assert date == '2017-01-01 10:00:00'


def test_process_line_ignores_non_error_level():
    check = _make_check()
    assert check._process_line('2017-01-01 10:00:00 INFO all good\n') == (None, None)


def test_process_line_handles_short_dated_line():
    check = _make_check()
    assert check._process_line('2017-01-01 x\n') == (None, None)


def test_process_line_truncates_long_message_to_200_chars():
    check = _make_check()
    line = '2017-01-01 10:00:00 ERROR ' + 'x' * 300 + '\n'
    message, date = check._process_line(line)
    assert len(message) == 200
    assert message.startswith('ERROR ' + 'x' * 194)
    assert date == '2017-01-01 10:00:00'


# ---------------------------------------------------------------------------
# _group_similar_log_lines()
# ---------------------------------------------------------------------------

def test_group_increments_count_for_similar_line():
    check = _make_check()
    common = {}
    common = check._group_similar_log_lines(
        line='ERROR connecting to database host one',
        date='2017-01-01 10:00:00',
        common_lines=common,
    )
    common = check._group_similar_log_lines(
        line='ERROR connecting to database host onf',
        date='2017-01-01 10:00:05',
        common_lines=common,
    )
    assert list(common.keys()) == ['ERROR connecting to database host one']
    assert common['ERROR connecting to database host one']['count'] == 2


def test_group_adds_dissimilar_line_as_new_key():
    check = _make_check()
    common = check._group_similar_log_lines(
        line='aaaaaaaaaa', date=None, common_lines={}
    )
    common = check._group_similar_log_lines(
        line='bbbbbbbbbb', date=None, common_lines=common
    )
    assert set(common.keys()) == {'aaaaaaaaaa', 'bbbbbbbbbb'}
    assert common['aaaaaaaaaa']['count'] == 1
    assert common['bbbbbbbbbb']['count'] == 1


def test_group_keeps_first_seen_date_from_first_occurrence():
    check = _make_check()
    common = check._group_similar_log_lines(
        line='ERROR disk full on volume one',
        date='2017-01-01 10:00:00',
        common_lines={},
    )
    common = check._group_similar_log_lines(
        line='ERROR disk full on volume one',
        date='2017-01-02 11:00:00',
        common_lines=common,
    )
    values = common['ERROR disk full on volume one']
    assert values['count'] == 2
    assert values['first_seen'] == '2017-01-01 10:00:00'


def test_group_without_date_sets_no_first_seen():
    check = _make_check()
    common = check._group_similar_log_lines(
        line='ERROR no date here', date=None, common_lines={}
    )
    assert 'first_seen' not in common['ERROR no date here']


def test_group_collapses_into_other_bucket_when_cap_reached(monkeypatch):
    monkeypatch.setattr(SimilarLogErrors, 'MAX_COMMON_LINES', 2)
    check = _make_check()
    common = {}
    for line in ['aaaaaaaaaa', 'bbbbbbbbbb']:
        common = check._group_similar_log_lines(line=line, date=None, common_lines=common)
    assert len(common) == 2

    common = check._group_similar_log_lines(line='cccccccccc', date=None, common_lines=common)
    common = check._group_similar_log_lines(line='dddddddddd', date=None, common_lines=common)

    assert len(common) == 3
    assert common['Other Error Lines']['count'] == 2
    assert 'cccccccccc' not in common
    assert 'dddddddddd' not in common


# ---------------------------------------------------------------------------
# get_error_pattern_counts()
# ---------------------------------------------------------------------------

def test_get_error_pattern_counts_end_to_end(tmp_path):
    log_file = tmp_path / 'service.log'
    log_file.write_text(
        '2017-01-01 10:00:00 ERROR Failed to connect to database alpha\n'
        '    at com.example.Client.connect(Client.java:42)\n'
        'Caused by: java.net.ConnectException\n'
        '2017-01-01 10:00:01 INFO informational message\n'
        '2017-01-01 10:00:02 ERROR Failed to connect to database beta\n'
        'random text with no date\n',
        encoding='ascii',
    )
    check = _make_check()
    filename, common = check.get_error_pattern_counts(str(log_file))

    assert filename == str(log_file)
    assert len(common) == 1
    values = common['ERROR Failed to connect to database alpha']
    assert values['count'] == 2
    assert values['first_seen'] == '2017-01-01 10:00:00'
