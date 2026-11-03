from collections import OrderedDict

import fossor.checks.similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class _SerialPool(object):
    '''Stands in for multiprocessing.Pool so run() can be exercised in-process.'''

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]


def _check():
    return SimilarLogErrors()


def test_remove_blacklisted_logs_drops_noise_files():
    files = [
        '/logs/a_configuration.log',
        '/logs/app.out',
        '/logs/gc.log',
        '/logs/oor.log',
        '/logs/x_databus.log',
        '/logs/x_public_access.log',
        '/logs/jetty_thread_pool_stats.log',
        '/logs/engine_access_log.log',
    ]
    assert _check()._remove_blacklisted_logs(files) == []


def test_remove_blacklisted_logs_keeps_order_of_ordinary_logs():
    files = ['/logs/web.log', '/logs/gc.log', '/logs/app.log', '/logs/db.log']
    assert _check()._remove_blacklisted_logs(files) == ['/logs/web.log', '/logs/app.log', '/logs/db.log']
    assert _check()._remove_blacklisted_logs([]) == []


def test_process_line_blank_and_indented_lines_are_ignored():
    check = _check()
    assert check._process_line('\n') == (None, None)
    assert check._process_line('    at com.example.Foo.bar(Foo.java:10)\n') == (None, None)
    assert check._process_line('\tERROR tabbed stack line\n') == (None, None)


def test_process_line_plain_text_without_exception_is_ignored():
    assert _check()._process_line('Starting service on port 8080\n') == (None, None)


def test_process_line_exception_text_is_returned_verbatim():
    line = 'IllegalStateException: pool exhausted\n'
    assert _check()._process_line(line) == (line, None)


def test_process_line_caused_by_lines_are_ignored():
    check = _check()
    assert check._process_line('Caused by: retry later\n') == (None, None)
    assert check._process_line('Caused by: java.io.IOException: disk full\n') == (None, None)


def test_process_line_error_line_splits_timestamp_and_message():
    line = '2020-01-01 10:00:00 ERROR Something bad happened\n'
    assert _check()._process_line(line) == ('ERROR Something bad happened', '2020-01-01 10:00:00')


def test_process_line_collapses_whitespace_and_strips_newline():
    line = '2020-01-01 10:00:00 ERROR   too   many   spaces  \n'
    assert _check()._process_line(line) == ('ERROR too many spaces', '2020-01-01 10:00:00')


def test_process_line_non_error_levels_are_ignored():
    check = _check()
    assert check._process_line('2020-01-01 10:00:00 WARN disk almost full\n') == (None, None)
    assert check._process_line('2020-01-01 10:00:00 INFO heartbeat ok\n') == (None, None)
    assert check._process_line('2020-01-01 10:00:00 error lowercase level\n') == (None, None)


def test_process_line_truncates_message_to_200_characters():
    message = 'x' * 300
    line = '2020-01-01 10:00:00 ERROR ' + message + '\n'
    processed, date = _check()._process_line(line)
    assert date == '2020-01-01 10:00:00'
    assert len(processed) == 200
    assert processed == ('ERROR ' + message)[:200]


def test_process_line_timestamp_without_level_is_ignored():
    check = _check()
    assert check._process_line('2020-01-01 10:00:00\n') == (None, None)
    assert check._process_line('2020-01-01 ERROR\n') == (None, None)


def test_process_line_other_leading_character_passes_through_unchanged():
    line = '[2020-01-01] ERROR bracketed prefix\n'
    assert _check()._process_line(line) == (line, None)


def test_group_first_line_creates_entry_with_count_and_first_seen():
    check = _check()
    common = OrderedDict()
    result = check._group_similar_log_lines(line='ERROR disk full', date='2020-01-01 10:00:00', common_lines=common)
    assert result is common
    assert list(result.keys()) == ['ERROR disk full']
    assert result['ERROR disk full'] == {'count': 1, 'first_seen': '2020-01-01 10:00:00'}


def test_group_similar_line_merges_and_keeps_first_seen():
    check = _check()
    common = OrderedDict()
    check._group_similar_log_lines(line='ERROR connection refused to host alpha',
                                   date='2020-01-01 10:00:00', common_lines=common)
    check._group_similar_log_lines(line='ERROR connection refused to host beta',
                                   date='2020-01-01 10:05:00', common_lines=common)
    assert list(common.keys()) == ['ERROR connection refused to host alpha']
    assert common['ERROR connection refused to host alpha'] == {'count': 2, 'first_seen': '2020-01-01 10:00:00'}


def test_group_dissimilar_line_creates_new_entry():
    check = _check()
    common = OrderedDict()
    check._group_similar_log_lines(line='ERROR disk full', date=None, common_lines=common)
    check._group_similar_log_lines(line='1234567890 !@#$%^&*()', date=None, common_lines=common)
    assert list(common.keys()) == ['ERROR disk full', '1234567890 !@#$%^&*()']


def test_group_ratio_exactly_at_threshold_is_not_merged():
    # quick_ratio = 2 * 7 / (10 + 10) = 0.7, and the comparison is strictly greater than SIMILARITY_RATIO.
    check = _check()
    common = OrderedDict([('abcdefghij', {'count': 1})])
    check._group_similar_log_lines(line='abcdefgXYZ', date=None, common_lines=common)
    assert list(common.keys()) == ['abcdefghij', 'abcdefgXYZ']


def test_group_ratio_above_threshold_is_merged():
    # quick_ratio = 2 * 8 / (10 + 10) = 0.8
    check = _check()
    common = OrderedDict([('abcdefghij', {'count': 1})])
    check._group_similar_log_lines(line='abcdefghXY', date=None, common_lines=common)
    assert list(common.keys()) == ['abcdefghij']
    assert common['abcdefghij']['count'] == 2


def test_group_without_date_records_no_first_seen():
    common = OrderedDict()
    _check()._group_similar_log_lines(line='ERROR no timestamp here', date=None, common_lines=common)
    assert common['ERROR no timestamp here'] == {'count': 1}


def test_group_overflow_collapses_into_other_error_lines():
    check = _check()
    common = OrderedDict((letter * 10, {'count': 1}) for letter in 'abcdefghijklmnopqrst')
    assert len(common) == check.MAX_COMMON_LINES

    check._group_similar_log_lines(line='zzzzzzzzzz', date=None, common_lines=common)
    assert len(common) == check.MAX_COMMON_LINES + 1
    assert common['Other Error Lines'] == {'count': 1}

    check._group_similar_log_lines(line='yyyyyyyyyy', date=None, common_lines=common)
    assert len(common) == check.MAX_COMMON_LINES + 1
    assert common['Other Error Lines']['count'] == 2


def test_get_error_pattern_counts_counts_similar_error_lines(tmp_path):
    log = tmp_path / 'app.log'
    log.write_text(
        '2020-01-01 10:00:00 INFO starting up\n'
        '2020-01-01 10:00:01 ERROR disk full on /dev/sda1\n'
        '2020-01-01 10:00:02 ERROR disk full on /dev/sdb1\n'
        '    at com.example.Foo.bar(Foo.java:10)\n'
        'Caused by: java.io.IOException: boom\n'
        '2020-01-01 10:00:03 WARN slow response\n'
    )
    filename, common = _check().get_error_pattern_counts(str(log))
    assert filename == str(log)
    assert list(common.keys()) == ['ERROR disk full on /dev/sda1']
    assert common['ERROR disk full on /dev/sda1'] == {'count': 2, 'first_seen': '2020-01-01 10:00:01'}


def test_get_error_pattern_counts_without_errors_is_empty(tmp_path):
    log = tmp_path / 'quiet.log'
    log.write_text('2020-01-01 10:00:00 INFO all good\n\n')
    filename, common = _check().get_error_pattern_counts(str(log))
    assert filename == str(log)
    assert len(common) == 0


def test_run_without_log_files_returns_none():
    check = _check()
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_report_per_file_and_skips_blacklisted(tmp_path, monkeypatch):
    monkeypatch.setattr(sle_module, 'Pool', _SerialPool)
    app = tmp_path / 'app.log'
    other = tmp_path / 'other.log'
    gc = tmp_path / 'gc.log'
    app.write_text(
        '2020-01-01 10:00:00 ERROR disk full on /dev/sda1\n'
        '2020-01-01 10:00:01 ERROR disk full on /dev/sdb1\n'
    )
    other.write_text('2020-01-02 11:00:00 ERROR network unreachable\n')
    gc.write_text('2020-01-01 10:00:00 ERROR gc pause\n')

    variables = {'LogFiles': [str(app), str(gc), str(other)]}
    result = _check().run(variables)

    expected = (
        'filename: {app}\n'
        'First seen: 2020-01-01 10:00:00, Count: 2, ERROR disk full on /dev/sda1\n\n'
        'filename: {other}\n'
        'First seen: 2020-01-02 11:00:00, Count: 1, ERROR network unreachable\n\n'
    ).format(app=str(app), other=str(other))
    assert result == expected


def test_run_returns_none_when_every_log_is_blacklisted(tmp_path, monkeypatch):
    monkeypatch.setattr(sle_module, 'Pool', _SerialPool)
    gc = tmp_path / 'gc.log'
    gc.write_text('2020-01-01 10:00:00 ERROR gc pause\n')
    assert _check().run({'LogFiles': [str(gc)]}) is None
