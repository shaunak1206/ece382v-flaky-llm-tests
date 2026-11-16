import os
from collections import OrderedDict

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


@pytest.fixture
def check():
    return SimilarLogErrors()


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_returns_none_when_log_files_missing(check):
    assert check.run({}) is None


def test_run_returns_none_when_log_files_empty(check):
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output_for_file(check, tmp_path):
    log_path = tmp_path / 'app.log'
    log_path.write_text(
        "2017-01-01 10:00:00 ERROR disk failure on node\n"
        "2017-01-01 10:05:00 INFO started\n"
        "2017-01-01 10:06:00 ERROR disk failure on node\n"
        "    at some.stack.frame\n",
        encoding='utf-8',
    )

    result = check.run({'LogFiles': [str(log_path)]})

    assert result is not None
    assert f"filename: {log_path}\n" in result
    assert "First seen: 2017-01-01 10:00:00, Count: 2, ERROR disk failure on node" in result


# ---------------------------------------------------------------------------
# _remove_blacklisted_logs()
# ---------------------------------------------------------------------------

def test_remove_blacklisted_logs_filters_known_names(check):
    files = [
        '/var/log/app_configuration.log',
        '/var/log/server.out',
        '/var/log/gc.log',
        '/var/log/oor.log',
        '/var/log/app_databus.log',
        '/var/log/app_public_access.log',
        '/var/log/jetty_thread_pool_stats.log',
        '/var/log/engine_access_log.log',
    ]

    assert check._remove_blacklisted_logs(files) == []


def test_remove_blacklisted_logs_keeps_other_logs(check):
    files = [
        '/var/log/app.log',
        '/var/log/app_configuration.log',
        '/var/log/error.log',
    ]

    assert check._remove_blacklisted_logs(files) == [
        '/var/log/app.log',
        '/var/log/error.log',
    ]


# ---------------------------------------------------------------------------
# _process_line()
# ---------------------------------------------------------------------------

def test_process_line_error_line_returns_message_and_date(check):
    line, date = check._process_line("2017-01-01 10:00:00 ERROR something broke\n")

    assert line == 'ERROR something broke'
    assert date == '2017-01-01 10:00:00'


def test_process_line_truncates_long_error_to_200_chars(check):
    long_line = "2017-01-01 10:00:00 ERROR " + 'x' * 300 + "\n"

    line, date = check._process_line(long_line)

    assert len(line) == 200
    assert line.startswith('ERROR xxx')
    assert date == '2017-01-01 10:00:00'


def test_process_line_non_error_level_is_ignored(check):
    line, date = check._process_line("2017-01-01 10:00:00 INFO all good\n")

    assert line is None
    assert date is None


def test_process_line_too_few_tokens_is_ignored(check):
    line, date = check._process_line("2017 short\n")

    assert line is None
    assert date is None


def test_process_line_leading_whitespace_is_ignored(check):
    line, date = check._process_line("    at com.example.Foo.bar(Foo.java:12)\n")

    assert line is None
    assert date is None


def test_process_line_caused_by_is_ignored(check):
    line, date = check._process_line("Caused by: java.io.IOException: boom\n")

    assert line is None
    assert date is None


def test_process_line_alpha_start_without_exception_is_ignored(check):
    line, date = check._process_line("Something happened here\n")

    assert line is None
    assert date is None


def test_process_line_alpha_start_with_exception_is_kept_without_date(check):
    raw = "IllegalStateException thrown in worker\n"

    line, date = check._process_line(raw)

    assert line == raw
    assert date is None


# ---------------------------------------------------------------------------
# _group_similar_log_lines()
# ---------------------------------------------------------------------------

def test_group_identical_lines_increment_count(check):
    common_lines = OrderedDict()
    line = 'ERROR identical message'

    common_lines = check._group_similar_log_lines(line, '2017-01-01 10:00:00', common_lines)
    common_lines = check._group_similar_log_lines(line, '2017-01-01 10:01:00', common_lines)

    assert len(common_lines) == 1
    assert common_lines[line]['count'] == 2


def test_group_similar_line_merges_into_existing_key(check):
    common_lines = OrderedDict()
    first = 'ERROR connection refused to host a'
    second = 'ERROR connection refused to host b'

    common_lines = check._group_similar_log_lines(first, None, common_lines)
    common_lines = check._group_similar_log_lines(second, None, common_lines)

    assert len(common_lines) == 1
    assert list(common_lines.keys()) == [first]
    assert common_lines[first]['count'] == 2


def test_group_dissimilar_line_creates_new_key(check):
    common_lines = OrderedDict()

    common_lines = check._group_similar_log_lines('ABCDEFGHIJ', None, common_lines)
    common_lines = check._group_similar_log_lines('klmnopqrst', None, common_lines)

    assert len(common_lines) == 2
    assert common_lines['ABCDEFGHIJ']['count'] == 1
    assert common_lines['klmnopqrst']['count'] == 1


def test_group_first_seen_is_set_only_once(check):
    common_lines = OrderedDict()
    line = 'ERROR repeated event'

    common_lines = check._group_similar_log_lines(line, '2017-01-01 09:00:00', common_lines)
    common_lines = check._group_similar_log_lines(line, '2017-01-01 12:00:00', common_lines)

    assert common_lines[line]['first_seen'] == '2017-01-01 09:00:00'


def test_group_too_many_unique_lines_goes_to_other_bucket(check):
    check.MAX_COMMON_LINES = 2
    common_lines = OrderedDict()

    common_lines = check._group_similar_log_lines('AAAA', None, common_lines)
    common_lines = check._group_similar_log_lines('BBBB', None, common_lines)
    common_lines = check._group_similar_log_lines('CCCC', None, common_lines)

    assert 'CCCC' not in common_lines
    assert common_lines['Other Error Lines']['count'] == 1


def test_group_similar_line_still_merges_when_over_limit(check):
    check.MAX_COMMON_LINES = 2
    common_lines = OrderedDict()
    existing = 'ERROR timeout contacting service alpha'

    common_lines = check._group_similar_log_lines(existing, None, common_lines)
    common_lines = check._group_similar_log_lines('XXXX', None, common_lines)
    common_lines = check._group_similar_log_lines(
        'ERROR timeout contacting service beta', None, common_lines)

    assert common_lines[existing]['count'] == 2
    assert 'Other Error Lines' not in common_lines


# ---------------------------------------------------------------------------
# get_error_pattern_counts()
# ---------------------------------------------------------------------------

def test_get_error_pattern_counts_reads_and_groups_file(check, tmp_path):
    log_path = tmp_path / 'service.log'
    log_path.write_text(
        "2017-01-01 10:00:00 ERROR disk failure on node\n"
        "2017-01-01 10:05:00 INFO started\n"
        "2017-01-01 10:06:00 ERROR disk failure on node\n"
        "    at some.stack.frame\n"
        "Caused by: something\n",
        encoding='utf-8',
    )

    filename, common_lines = check.get_error_pattern_counts(str(log_path))

    assert filename == str(log_path)
    assert list(common_lines.keys()) == ['ERROR disk failure on node']
    entry = common_lines['ERROR disk failure on node']
    assert entry['count'] == 2
    assert entry['first_seen'] == '2017-01-01 10:00:00'
