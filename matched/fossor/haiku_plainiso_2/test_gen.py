from collections import OrderedDict

import pytest

from fossor.checks import similar_log_errors as sle
from fossor.checks.similar_log_errors import SimilarLogErrors


OTHER_BUCKET = 'Other Error Lines'

APP_LOG = '\n'.join([
    '2024-01-01 10:00:00 ERROR Connection refused to db1',
    '2024-01-01 10:00:05 INFO heartbeat ok',
    '2024-01-01 10:00:09 ERROR Connection refused to db2',
    '    at com.example.Client.connect(Client.java:42)',
    'Caused by: java.lang.RuntimeException: socket closed',
]) + '\n'

WORKER_LOG = '2024-01-01 11:00:00 ERROR Worker crashed with code 3\n'

INFO_ONLY_LOG = '2024-01-01 10:00:00 INFO all good\n2024-01-01 10:00:01 DEBUG tick\n'


def _write(path, text):
    path.write_text(text)
    return str(path)


@pytest.fixture
def serial_pool(monkeypatch):
    sizes = []

    class FakePool(object):
        def __init__(self, processes=None):
            sizes.append(processes)

        def map(self, func, iterable):
            return list(map(func, iterable))

    monkeypatch.setattr(sle, 'Pool', FakePool)
    return sizes


def test_run_without_log_files_returns_none():
    check = SimilarLogErrors()
    assert check.run({}) is None
    assert check.run({'LogFiles': None}) is None
    assert check.run({'LogFiles': []}) is None


def test_blacklist_removes_known_log_names():
    check = SimilarLogErrors()
    files = ['/logs/app_configuration.log', '/logs/server.out', '/logs/gc.log', '/logs/keep.log']
    assert check._remove_blacklisted_logs(files) == ['/logs/keep.log']


def test_blacklist_keeps_unmatched_names():
    check = SimilarLogErrors()
    files = ['/logs/app.log', '/logs/web.log']
    assert check._remove_blacklisted_logs(files) == files


def test_blacklist_matches_substrings():
    check = SimilarLogErrors()
    files = ['/logs/door.log', '/logs/app.log']
    assert check._remove_blacklisted_logs(files) == ['/logs/app.log']


def test_process_line_skips_stacktrace_continuation():
    check = SimilarLogErrors()
    assert check._process_line('    at com.example.Client.connect(Client.java:42)\n') == (None, None)


def test_process_line_skips_blank_line():
    check = SimilarLogErrors()
    assert check._process_line('\n') == (None, None)


def test_process_line_skips_plain_text_without_exception():
    check = SimilarLogErrors()
    assert check._process_line('hello world\n') == (None, None)


def test_process_line_keeps_exception_text_without_date():
    check = SimilarLogErrors()
    line, date = check._process_line('NullPointerException thrown here\n')
    assert line == 'NullPointerException thrown here\n'
    assert date is None


def test_process_line_skips_caused_by_lines():
    check = SimilarLogErrors()
    assert check._process_line('Caused by: java.lang.RuntimeException: boom\n') == (None, None)


def test_process_line_parses_error_with_timestamp():
    check = SimilarLogErrors()
    assert check._process_line('2024-03-01 12:00:00 ERROR disk is full\n') == (
        'ERROR disk is full', '2024-03-01 12:00:00')


def test_process_line_skips_non_error_levels():
    check = SimilarLogErrors()
    assert check._process_line('2024-03-01 12:00:00 INFO all good\n') == (None, None)


def test_process_line_truncates_to_200_chars():
    check = SimilarLogErrors()
    raw = '2024-03-01 12:00:00 ERROR ' + 'x' * 300 + '\n'
    line, date = check._process_line(raw)
    assert len(line) == 200
    assert line.startswith('ERROR xxx')
    assert date == '2024-03-01 12:00:00'


def test_process_line_short_digit_line_returns_none():
    check = SimilarLogErrors()
    assert check._process_line('2024-03-01 12:00:00\n') == (None, None)


def test_group_first_line_creates_entry():
    check = SimilarLogErrors()
    common = check._group_similar_log_lines(
        line='ERROR first', date='2024-03-01 12:00:00', common_lines=OrderedDict())
    assert list(common.keys()) == ['ERROR first']
    assert common['ERROR first'] == {'count': 1, 'first_seen': '2024-03-01 12:00:00'}


def test_group_near_duplicate_increments_count():
    check = SimilarLogErrors()
    common = OrderedDict()
    common = check._group_similar_log_lines(
        line='ERROR connection refused to host1', date='2024-03-01 12:00:00', common_lines=common)
    common = check._group_similar_log_lines(
        line='ERROR connection refused to host2', date='2024-03-01 12:00:05', common_lines=common)
    assert list(common.keys()) == ['ERROR connection refused to host1']
    assert common['ERROR connection refused to host1']['count'] == 2


def test_group_dissimilar_lines_stay_separate():
    check = SimilarLogErrors()
    common = OrderedDict()
    common = check._group_similar_log_lines(line='A' * 20, date=None, common_lines=common)
    common = check._group_similar_log_lines(line='b' * 20, date=None, common_lines=common)
    assert list(common.keys()) == ['A' * 20, 'b' * 20]
    assert common['A' * 20]['count'] == 1
    assert common['b' * 20]['count'] == 1


def test_group_respects_instance_similarity_ratio():
    check = SimilarLogErrors()
    check.SIMILARITY_RATIO = 0.99
    common = OrderedDict()
    common = check._group_similar_log_lines(
        line='ERROR connection refused to host1', date=None, common_lines=common)
    common = check._group_similar_log_lines(
        line='ERROR connection refused to host2', date=None, common_lines=common)
    assert len(common) == 2


def test_group_caps_unique_lines_into_other_bucket():
    check = SimilarLogErrors()
    check.MAX_COMMON_LINES = 2
    common = OrderedDict()
    for line in ['A' * 10, 'b' * 10, '1234567890', '!' * 10]:
        common = check._group_similar_log_lines(line=line, date=None, common_lines=common)
    assert list(common.keys()) == ['A' * 10, 'b' * 10, OTHER_BUCKET]
    assert common[OTHER_BUCKET]['count'] == 2


def test_group_first_seen_keeps_earliest_date():
    check = SimilarLogErrors()
    common = check._group_similar_log_lines(
        line='ERROR x', date='2024-03-01 12:00:00', common_lines=OrderedDict())
    common = check._group_similar_log_lines(
        line='ERROR x', date='2024-03-02 09:00:00', common_lines=common)
    assert common['ERROR x'] == {'count': 2, 'first_seen': '2024-03-01 12:00:00'}
    undated = check._group_similar_log_lines(line='ERROR y', date=None, common_lines=OrderedDict())
    assert 'first_seen' not in undated['ERROR y']


def test_get_error_pattern_counts_reads_and_groups(tmp_path):
    path = _write(tmp_path / 'app.log', APP_LOG)
    check = SimilarLogErrors()
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert list(common.keys()) == ['ERROR Connection refused to db1']
    assert common['ERROR Connection refused to db1'] == {
        'count': 2, 'first_seen': '2024-01-01 10:00:00'}


def test_run_formats_report_per_file(tmp_path, serial_pool):
    app = _write(tmp_path / 'app.log', APP_LOG)
    worker = _write(tmp_path / 'worker.log', WORKER_LOG)
    report = SimilarLogErrors().run({'LogFiles': [app, worker]})
    lines = report.splitlines()
    assert f'filename: {app}' in lines
    assert f'filename: {worker}' in lines
    assert 'First seen: 2024-01-01 10:00:00, Count: 2, ERROR Connection refused to db1' in lines
    assert 'First seen: 2024-01-01 11:00:00, Count: 1, ERROR Worker crashed with code 3' in lines
    assert lines.index(f'filename: {app}') < lines.index(f'filename: {worker}')


def test_run_skips_blacklisted_files(tmp_path, serial_pool):
    app = _write(tmp_path / 'app.log', APP_LOG)
    gc = _write(tmp_path / 'gc.log', WORKER_LOG)
    report = SimilarLogErrors().run({'LogFiles': [gc, app]})
    assert f'filename: {gc}' not in report
    assert f'filename: {app}' in report


def test_run_returns_none_when_no_errors(tmp_path, serial_pool):
    info = _write(tmp_path / 'info.log', INFO_ONLY_LOG)
    assert SimilarLogErrors().run({'LogFiles': [info]}) is None


def test_run_pool_size_is_twice_cpu_count(tmp_path, serial_pool, monkeypatch):
    monkeypatch.setattr(sle, 'cpu_count', lambda: 3)
    app = _write(tmp_path / 'app.log', APP_LOG)
    SimilarLogErrors().run({'LogFiles': [app]})
    assert serial_pool == [6]
