# Tests for fossor/checks/similar_log_errors.py
#
# Deterministic by design:
# - The code under test uses no random numbers and no hash() of strings.
# - Output ordering comes from OrderedDict insertion order and list order.
# - Files are written only under pytest's tmp_path.
# - Module attributes are replaced only with monkeypatch.
# - Multiprocessing Pool is replaced by a serial stand-in so no processes start.

from collections import OrderedDict
from difflib import SequenceMatcher

import pytest

from fossor.checks import similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class _SerialPool(object):
    """Serial stand-in for multiprocessing.Pool, so tests spawn no processes."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def close(self):
        pass

    def join(self):
        pass


def _group_all(check, entries):
    """Feed (line, date) pairs through _group_similar_log_lines in order."""
    common_lines = OrderedDict()
    for line, date in entries:
        common_lines = check._group_similar_log_lines(line=line, date=date, common_lines=common_lines)
    return common_lines


# ---------------------------------------------------------------- blacklist

def test_remove_blacklisted_logs_drops_every_blacklisted_name():
    check = SimilarLogErrors()
    files = ['/logs/' + name for name in SimilarLogErrors.log_name_blacklist]
    files.append('/logs/keep_me.log')

    assert check._remove_blacklisted_logs(files) == ['/logs/keep_me.log']


def test_remove_blacklisted_logs_preserves_order_of_kept_files():
    check = SimilarLogErrors()
    files = ['/logs/b.log', '/logs/gc.log', '/logs/a.log', '/logs/c.log']

    assert check._remove_blacklisted_logs(files) == ['/logs/b.log', '/logs/a.log', '/logs/c.log']


def test_remove_blacklisted_logs_empty_input_returns_empty_list():
    check = SimilarLogErrors()

    assert check._remove_blacklisted_logs([]) == []


# -------------------------------------------------------------- _process_line

def test_process_line_leading_whitespace_is_skipped():
    check = SimilarLogErrors()

    assert check._process_line('    at com.example.Foo.bar(Foo.java:10)\n') == (None, None)


def test_process_line_alpha_start_without_exception_is_skipped():
    check = SimilarLogErrors()

    assert check._process_line('Application started\n') == (None, None)


def test_process_line_alpha_start_with_exception_is_kept_unchanged():
    check = SimilarLogErrors()
    line = 'NullPointerException happened\n'

    assert check._process_line(line) == (line, None)


def test_process_line_caused_by_is_skipped_even_with_exception_word():
    check = SimilarLogErrors()

    assert check._process_line('Caused by: java.lang.Exception: boom\n') == (None, None)


def test_process_line_error_with_date_returns_message_and_date():
    check = SimilarLogErrors()

    line, date = check._process_line('2017-01-01 12:00:00 ERROR something bad\n')

    assert line == 'ERROR something bad'
    assert date == '2017-01-01 12:00:00'


def test_process_line_info_level_is_skipped():
    check = SimilarLogErrors()

    assert check._process_line('2017-01-01 12:00:00 INFO all good\n') == (None, None)


def test_process_line_warn_level_is_skipped():
    check = SimilarLogErrors()

    assert check._process_line('2017-01-01 12:00:00 WARN disk almost full\n') == (None, None)


def test_process_line_digit_start_with_too_few_tokens_is_skipped():
    check = SimilarLogErrors()

    # Only two tokens, so reading index 2 raises IndexError inside the code.
    assert check._process_line('2017 ERROR\n') == (None, None)


def test_process_line_truncates_long_message_to_200_chars():
    check = SimilarLogErrors()
    line = '2017-01-01 10:00:00 ERROR ' + 'x' * 300 + '\n'

    message, date = check._process_line(line)

    assert date == '2017-01-01 10:00:00'
    assert len(message) == 200
    assert message.startswith('ERROR ')


# ------------------------------------------------- _group_similar_log_lines

def test_group_identical_line_increments_count():
    check = SimilarLogErrors()

    common_lines = _group_all(check, [
        ('ERROR disk full', None),
        ('ERROR disk full', None),
        ('ERROR disk full', None),
    ])

    assert list(common_lines.keys()) == ['ERROR disk full']
    assert common_lines['ERROR disk full']['count'] == 3


def test_group_similar_line_joins_existing_entry():
    check = SimilarLogErrors()
    first = 'ERROR connection refused to host alpha'
    second = 'ERROR connection refused to host beta'
    # Precondition: the two lines really are similar enough to be grouped.
    ratio = SequenceMatcher(a=first, b=second).quick_ratio()
    assert ratio > SimilarLogErrors.SIMILARITY_RATIO

    common_lines = _group_all(check, [(first, None), (second, None)])

    assert list(common_lines.keys()) == [first]
    assert common_lines[first]['count'] == 2


def test_group_dissimilar_lines_create_separate_entries():
    check = SimilarLogErrors()
    first = 'aaaa'
    second = 'bbbb'
    # Precondition: the two lines really are below the similarity threshold.
    ratio = SequenceMatcher(a=first, b=second).quick_ratio()
    assert ratio <= SimilarLogErrors.SIMILARITY_RATIO

    common_lines = _group_all(check, [(first, None), (second, None)])

    assert list(common_lines.keys()) == [first, second]
    assert common_lines[first]['count'] == 1
    assert common_lines[second]['count'] == 1


def test_group_first_seen_keeps_earliest_date():
    check = SimilarLogErrors()

    common_lines = _group_all(check, [
        ('ERROR disk full', '2017-01-01 10:00:00'),
        ('ERROR disk full', '2017-01-02 11:00:00'),
    ])

    assert common_lines['ERROR disk full']['first_seen'] == '2017-01-01 10:00:00'
    assert common_lines['ERROR disk full']['count'] == 2


def test_group_without_date_sets_no_first_seen():
    check = SimilarLogErrors()

    common_lines = _group_all(check, [('ERROR disk full', None)])

    assert 'first_seen' not in common_lines['ERROR disk full']


def test_group_overflow_goes_to_other_error_lines(monkeypatch):
    monkeypatch.setattr(SimilarLogErrors, 'MAX_COMMON_LINES', 2)
    check = SimilarLogErrors()

    common_lines = _group_all(check, [
        ('aaaa', None),
        ('bbbb', None),
        ('cccc', None),  # dissimilar to both, and the table is already full
    ])

    assert list(common_lines.keys()) == ['aaaa', 'bbbb', 'Other Error Lines']
    assert common_lines['Other Error Lines']['count'] == 1


# ------------------------------------------------------ get_error_pattern_counts

def test_get_error_pattern_counts_reads_and_groups_file(tmp_path):
    check = SimilarLogErrors()
    log = tmp_path / 'app.log'
    log.write_text(
        'Application started\n'
        '2017-01-01 10:00:00 ERROR Disk full on node1\n'
        '\tat com.example.Foo\n'
        '2017-01-01 10:05:00 INFO heartbeat ok\n'
        '2017-01-01 10:10:00 ERROR Disk full on node1\n'
    )

    filename, common_lines = check.get_error_pattern_counts(str(log))

    assert filename == str(log)
    assert dict(common_lines) == {
        'ERROR Disk full on node1': {'count': 2, 'first_seen': '2017-01-01 10:00:00'},
    }


# ------------------------------------------------------------------------- run

def test_run_without_log_files_returns_none():
    check = SimilarLogErrors()

    assert check.run({}) is None


def test_run_with_empty_log_file_list_returns_none():
    check = SimilarLogErrors()

    assert check.run({'LogFiles': []}) is None


def test_run_with_only_blacklisted_files_returns_none(monkeypatch):
    monkeypatch.setattr(sle_module, 'Pool', _SerialPool)
    monkeypatch.setattr(sle_module, 'cpu_count', lambda: 1)
    check = SimilarLogErrors()

    assert check.run({'LogFiles': ['/logs/server.out']}) is None


def test_run_with_no_error_lines_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(sle_module, 'Pool', _SerialPool)
    monkeypatch.setattr(sle_module, 'cpu_count', lambda: 1)
    check = SimilarLogErrors()
    log = tmp_path / 'app.log'
    log.write_text('2017-01-01 10:05:00 INFO heartbeat ok\n')

    assert check.run({'LogFiles': [str(log)]}) is None


def test_run_formats_output_and_skips_blacklisted_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sle_module, 'Pool', _SerialPool)
    monkeypatch.setattr(sle_module, 'cpu_count', lambda: 1)
    check = SimilarLogErrors()

    log = tmp_path / 'app.log'
    log.write_text(
        'Application started\n'
        '2017-01-01 10:00:00 ERROR Disk full on node1\n'
        '\tat com.example.Foo\n'
        '2017-01-01 10:05:00 INFO heartbeat ok\n'
        '2017-01-01 10:10:00 ERROR Disk full on node1\n'
    )
    ignored = tmp_path / 'gc.log'
    ignored.write_text('2017-01-01 10:00:00 ERROR should be ignored\n')

    result = check.run({'LogFiles': [str(log), str(ignored)]})

    expected = (
        f"filename: {log}\n"
        "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Disk full on node1\n\n"
    )
    assert result == expected
