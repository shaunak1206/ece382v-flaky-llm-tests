import logging
from collections import OrderedDict

import pytest

from fossor.checks import similar_log_errors
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Serial stand-in for multiprocessing.Pool to keep tests deterministic."""

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
    monkeypatch.setattr(similar_log_errors, 'Pool', FakePool)


def test_process_line_skips_indented_stacktrace_line(check):
    assert check._process_line("    at com.foo.Bar(Bar.java:10)\n") == (None, None)
    assert check._process_line("\tat com.foo.Bar(Bar.java:10)\n") == (None, None)


def test_process_line_skips_alpha_line_without_exception(check):
    assert check._process_line("Some random text\n") == (None, None)


def test_process_line_keeps_alpha_line_with_exception(check):
    line = "NullPointerException: something broke\n"
    result_line, date = check._process_line(line)
    assert result_line == line
    assert date is None


def test_process_line_skips_caused_by(check):
    assert check._process_line("Caused by: java.lang.Exception: boom\n") == (None, None)


def test_process_line_error_line_extracts_date_and_message(check):
    line, date = check._process_line("2017-01-01 12:00:00 ERROR Something bad happened\n")
    assert line == "ERROR Something bad happened"
    assert date == "2017-01-01 12:00:00"


def test_process_line_ignores_non_error_levels(check):
    assert check._process_line("2017-01-01 12:00:00 INFO All good\n") == (None, None)
    assert check._process_line("2017-01-01 12:00:00 WARN Hmm\n") == (None, None)


def test_process_line_short_digit_line_is_ignored(check):
    assert check._process_line("2017-01-01\n") == (None, None)


def test_process_line_truncates_long_lines(check):
    message = "x" * 500
    line, date = check._process_line(f"2017-01-01 12:00:00 ERROR {message}\n")
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 12:00:00"


def test_process_line_other_leading_character_passes_through(check):
    line = "[ERROR] unformatted line\n"
    result_line, date = check._process_line(line)
    assert result_line == line
    assert date is None


def test_group_new_line_creates_entry_with_first_seen(check):
    common = check._group_similar_log_lines(line="ERROR disk full", date="2017-01-01 10:00:00",
                                            common_lines=OrderedDict())
    assert list(common.keys()) == ["ERROR disk full"]
    assert common["ERROR disk full"]['count'] == 1
    assert common["ERROR disk full"]['first_seen'] == "2017-01-01 10:00:00"


def test_group_without_date_has_no_first_seen(check):
    common = check._group_similar_log_lines(line="SomeException", date=None, common_lines=OrderedDict())
    assert common["SomeException"]['count'] == 1
    assert 'first_seen' not in common["SomeException"]


def test_group_similar_lines_increment_count_and_keep_first_seen(check):
    common = OrderedDict()
    common = check._group_similar_log_lines("ERROR connection to host 10.0.0.1 failed", "d1", common)
    common = check._group_similar_log_lines("ERROR connection to host 10.0.0.2 failed", "d2", common)
    assert len(common) == 1
    values = common["ERROR connection to host 10.0.0.1 failed"]
    assert values['count'] == 2
    assert values['first_seen'] == "d1"


def test_group_dissimilar_lines_are_tracked_separately(check):
    common = OrderedDict()
    common = check._group_similar_log_lines("a" * 30, "d1", common)
    common = check._group_similar_log_lines("b" * 30, "d2", common)
    assert len(common) == 2
    assert common["a" * 30]['count'] == 1
    assert common["b" * 30]['count'] == 1


def test_group_overflow_goes_to_other_error_lines(check):
    check.MAX_COMMON_LINES = 2
    common = OrderedDict()
    common = check._group_similar_log_lines("a" * 30, "d1", common)
    common = check._group_similar_log_lines("b" * 30, "d2", common)
    common = check._group_similar_log_lines("z" * 30, "d3", common)
    common = check._group_similar_log_lines("y" * 30, "d4", common)
    assert len(common) == 3
    assert common['Other Error Lines']['count'] == 2
    assert common['Other Error Lines']['first_seen'] == "d3"


def test_remove_blacklisted_logs(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/app.out',
             '/var/log/x_configuration.log', '/var/log/service.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log', '/var/log/service.log']


def test_get_error_pattern_counts_reads_file(check, tmp_path):
    path = tmp_path / "app.log"
    path.write_text(
        "2017-01-01 10:00:00 ERROR Database connection failed\n"
        "    at com.foo.Bar(Bar.java:1)\n"
        "2017-01-01 10:01:00 INFO fine\n"
        "Caused by: something\n"
        "2017-01-01 10:05:00 ERROR Database connection failed\n"
    )
    filename, common = check.get_error_pattern_counts(str(path))
    assert filename == str(path)
    assert list(common.keys()) == ["ERROR Database connection failed"]
    assert common["ERROR Database connection failed"]['count'] == 2
    assert common["ERROR Database connection failed"]['first_seen'] == "2017-01-01 10:00:00"


def test_run_returns_none_without_log_files(check):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, tmp_path, fake_pool):
    path = tmp_path / "app.log"
    path.write_text(
        "2017-01-01 10:00:00 ERROR Database connection failed\n"
        "2017-01-01 10:05:00 ERROR Database connection failed\n"
        "2017-01-01 10:06:00 INFO fine\n"
    )
    result = check.run({'LogFiles': [str(path)]})
    expected = (f"filename: {path}\n"
                "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Database connection failed\n\n")
    assert result == expected


def test_run_returns_none_when_no_errors(check, tmp_path, fake_pool):
    path = tmp_path / "quiet.log"
    path.write_text("2017-01-01 10:00:00 INFO all good\n")
    assert check.run({'LogFiles': [str(path)]}) is None


def test_run_ignores_blacklisted_files(check, tmp_path, fake_pool):
    gc = tmp_path / "gc.log"
    gc.write_text("2017-01-01 10:00:00 ERROR should be ignored\n")
    assert check.run({'LogFiles': [str(gc)]}) is None


def test_run_multiple_files_only_reports_files_with_errors(check, tmp_path, fake_pool):
    good = tmp_path / "good.log"
    good.write_text("2017-01-01 10:00:00 INFO ok\n")
    bad = tmp_path / "bad.log"
    bad.write_text("2017-01-01 11:00:00 ERROR kaboom\n")
    result = check.run({'LogFiles': [str(good), str(bad)]})
    assert f"filename: {bad}" in result
    assert str(good) not in result
    assert "Count: 1, ERROR kaboom" in result
