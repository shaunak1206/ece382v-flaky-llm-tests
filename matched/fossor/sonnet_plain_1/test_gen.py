from collections import OrderedDict

import pytest

from fossor.checks import similar_log_errors
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Synchronous stand-in for multiprocessing.Pool."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return list(map(func, iterable))


@pytest.fixture
def check():
    return SimilarLogErrors()


@pytest.fixture
def fake_pool(monkeypatch):
    monkeypatch.setattr(similar_log_errors, 'Pool', FakePool)


def write_log(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


SAMPLE_LOG = (
    "2017-01-01 10:00:00 ERROR Connection failed to host 1\n"
    "    at foo.bar(Baz.java:10)\n"
    "Caused by: something else\n"
    "2017-01-01 10:00:05 INFO all good\n"
    "2017-01-01 10:00:09 ERROR Connection failed to host 2\n"
)


# ---------------------------------------------------------------- _process_line

def test_process_line_error_line_splits_date_and_message(check):
    line, date = check._process_line("2017-01-01 12:00:00 ERROR something bad happened\n")
    assert line == "ERROR something bad happened"
    assert date == "2017-01-01 12:00:00"


def test_process_line_non_error_level_is_ignored(check):
    assert check._process_line("2017-01-01 12:00:00 INFO all fine\n") == (None, None)
    assert check._process_line("2017-01-01 12:00:00 WARN careful\n") == (None, None)


def test_process_line_indented_stacktrace_line_is_ignored(check):
    assert check._process_line("    at com.foo.Bar.baz(Bar.java:42)\n") == (None, None)
    assert check._process_line("\tat com.foo.Bar.baz(Bar.java:42)\n") == (None, None)
    assert check._process_line("\n") == (None, None)


def test_process_line_alpha_line_without_exception_is_ignored(check):
    assert check._process_line("Starting up the server\n") == (None, None)


def test_process_line_alpha_line_with_exception_is_kept_unchanged(check):
    raw = "java.lang.NullPointerException: boom\n"
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


def test_process_line_caused_by_is_ignored_even_with_exception(check):
    assert check._process_line("Caused by: java.io.IOException: disk\n") == (None, None)


def test_process_line_short_digit_line_is_ignored(check):
    # Only two tokens, so the level lookup raises IndexError internally.
    assert check._process_line("2017-01-01 12:00:00\n") == (None, None)


def test_process_line_truncates_long_messages_to_200_chars(check):
    raw = "2017-01-01 12:00:00 ERROR " + "x" * 500 + "\n"
    line, date = check._process_line(raw)
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 12:00:00"


def test_process_line_other_first_character_passes_through(check):
    raw = "[main] something odd\n"
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


# ------------------------------------------------------- _group_similar_log_lines

def test_group_first_line_creates_entry_with_first_seen(check):
    result = check._group_similar_log_lines(line="ERROR boom", date="2017-01-01 00:00:00",
                                            common_lines=OrderedDict())
    assert list(result.keys()) == ["ERROR boom"]
    assert result["ERROR boom"] == {'count': 1, 'first_seen': "2017-01-01 00:00:00"}


def test_group_similar_lines_are_merged_and_keep_first_seen(check):
    common = OrderedDict()
    check._group_similar_log_lines("ERROR Connection failed to host 1", "2017-01-01 00:00:01", common)
    check._group_similar_log_lines("ERROR Connection failed to host 2", "2017-01-01 00:00:09", common)
    assert list(common.keys()) == ["ERROR Connection failed to host 1"]
    assert common["ERROR Connection failed to host 1"]['count'] == 2
    assert common["ERROR Connection failed to host 1"]['first_seen'] == "2017-01-01 00:00:01"


def test_group_dissimilar_lines_get_separate_entries(check):
    common = OrderedDict()
    check._group_similar_log_lines("ERROR aaaaaaaaaa", None, common)
    check._group_similar_log_lines("ERROR zzzzzzzzzz", None, common)
    assert len(common) == 2
    assert all(v['count'] == 1 for v in common.values())


def test_group_without_date_has_no_first_seen(check):
    common = check._group_similar_log_lines("ERROR boom", None, OrderedDict())
    assert common["ERROR boom"] == {'count': 1}


def test_group_ratio_exactly_at_threshold_is_not_merged(check):
    common = OrderedDict()
    check._group_similar_log_lines("a" * 10, None, common)
    # quick_ratio == 2 * 7 / 20 == 0.7, which is not strictly greater than the threshold.
    check._group_similar_log_lines("a" * 7 + "b" * 3, None, common)
    assert len(common) == 2


def test_group_ratio_above_threshold_is_merged(check):
    common = OrderedDict()
    check._group_similar_log_lines("a" * 10, None, common)
    # quick_ratio == 2 * 8 / 20 == 0.8
    check._group_similar_log_lines("a" * 8 + "b" * 2, None, common)
    assert list(common.keys()) == ["a" * 10]
    assert common["a" * 10]['count'] == 2


def test_group_overflow_goes_to_other_error_lines(check):
    common = OrderedDict()
    for i in range(SimilarLogErrors.MAX_COMMON_LINES):
        common[chr(ord('a') + i) * 10] = {'count': 1}
    assert len(common) == SimilarLogErrors.MAX_COMMON_LINES

    check._group_similar_log_lines("ZZZZZZZZZZ", "2017-01-01 00:00:00", common)
    check._group_similar_log_lines("YYYYYYYYYY", "2017-01-01 00:00:05", common)

    assert len(common) == SimilarLogErrors.MAX_COMMON_LINES + 1
    assert "ZZZZZZZZZZ" not in common
    assert common['Other Error Lines']['count'] == 2
    assert common['Other Error Lines']['first_seen'] == "2017-01-01 00:00:00"


# ------------------------------------------------------------ blacklist / counting

def test_remove_blacklisted_logs_filters_known_names(check):
    files = [
        '/var/log/app.log',
        '/var/log/gc.log',
        '/var/log/server.out',
        '/var/log/x_databus.log',
        '/var/log/y_configuration.log',
        '/var/log/engine_access_log.log',
        '/var/log/error.log',
    ]
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log', '/var/log/error.log']


def test_get_error_pattern_counts_reads_and_groups_file(check, tmp_path):
    path = write_log(tmp_path, 'app.log', SAMPLE_LOG)
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert list(common.keys()) == ["ERROR Connection failed to host 1"]
    assert common["ERROR Connection failed to host 1"] == {
        'count': 2,
        'first_seen': "2017-01-01 10:00:00",
    }


def test_get_error_pattern_counts_empty_file(check, tmp_path):
    path = write_log(tmp_path, 'empty.log', '')
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert len(common) == 0


# ------------------------------------------------------------------------- run

def test_run_without_log_files_returns_none(check):
    assert check.run({}) is None
    assert check.run({'LogFiles': None}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output_for_log_file(check, tmp_path, fake_pool):
    path = write_log(tmp_path, 'app.log', SAMPLE_LOG)
    result = check.run({'LogFiles': [path]})
    expected = (
        "filename: " + path + "\n"
        "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Connection failed to host 1\n\n"
    )
    assert result == expected


def test_run_returns_none_when_no_errors_or_only_blacklisted(check, tmp_path, fake_pool):
    clean = write_log(tmp_path, 'clean.log', "2017-01-01 10:00:00 INFO fine\n")
    blacklisted = write_log(tmp_path, 'gc.log', "2017-01-01 10:00:00 ERROR ignored\n")
    assert check.run({'LogFiles': [clean]}) is None
    assert check.run({'LogFiles': [blacklisted]}) is None


def test_run_reports_each_file_with_errors_in_order(check, tmp_path, fake_pool):
    first = write_log(tmp_path, 'a.log', "2017-01-01 10:00:00 ERROR first problem\n")
    quiet = write_log(tmp_path, 'b.log', "2017-01-01 10:00:00 INFO nothing\n")
    second = write_log(tmp_path, 'c.log', "2017-01-02 11:00:00 ERROR second problem\n")
    result = check.run({'LogFiles': [first, quiet, second]})
    assert "filename: " + first in result
    assert "filename: " + second in result
    assert "filename: " + quiet not in result
    assert result.index(first) < result.index(second)
    assert result.endswith("\n\n")


def test_run_omits_first_seen_and_strips_line_when_missing(check, fake_pool):
    check.get_error_pattern_counts = lambda f: (f, OrderedDict([("  ERROR padded  ", {'count': 3})]))
    result = check.run({'LogFiles': ['/some/app.log']})
    assert result == "filename: /some/app.log\nCount: 3, ERROR padded\n\n"


def test_run_helper_returns_output_string(check, tmp_path, fake_pool):
    path = write_log(tmp_path, 'app.log', SAMPLE_LOG)
    output = check.run_helper({'LogFiles': [path]})
    assert isinstance(output, str)
    assert "Count: 2" in output
