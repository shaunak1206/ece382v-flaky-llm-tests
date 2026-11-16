import collections
import logging
import os

import pytest

from fossor.checks import similar_log_errors as mod
from fossor.checks.similar_log_errors import SimilarLogErrors


@pytest.fixture
def check():
    c = SimilarLogErrors()
    if not hasattr(c, "log"):
        c.log = logging.getLogger("test_similar_log_errors")
    return c


class FakePool:
    """Stands in for multiprocessing.Pool so tests run in-process."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]


def _write(path, lines):
    with open(path, "wt") as f:
        f.write("".join(lines))
    return str(path)


def test_constants_have_expected_values():
    assert SimilarLogErrors.SIMILARITY_RATIO == 0.7
    assert SimilarLogErrors.MAX_COMMON_LINES == 20


def test_blacklist_contains_expected_names():
    blacklist = SimilarLogErrors.log_name_blacklist
    for name in ["_configuration.log", ".out", "gc.log", "oor.log",
                 "_databus.log", "_public_access.log",
                 "jetty_thread_pool_stats.log", "engine_access_log.log"]:
        assert name in blacklist


def test_process_line_error_line_extracts_date_and_message(check):
    line, date = check._process_line("2017-01-01 10:00:00 ERROR disk full\n")
    assert line == "ERROR disk full"
    assert date == "2017-01-01 10:00:00"


def test_process_line_truncates_long_error_to_200_chars(check):
    raw = "2017-01-01 10:00:00 ERROR " + "x" * 300 + "\n"
    line, date = check._process_line(raw)
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 10:00:00"


def test_process_line_non_error_level_is_ignored(check):
    line, date = check._process_line("2017-01-01 10:00:00 WARN something minor\n")
    assert line is None
    assert date is None


def test_process_line_leading_whitespace_is_ignored(check):
    line, date = check._process_line("    at com.example.Foo.bar(Foo.java:10)\n")
    assert line is None
    assert date is None


def test_process_line_alpha_without_exception_is_ignored(check):
    line, date = check._process_line("random text without a date\n")
    assert line is None
    assert date is None


def test_process_line_alpha_with_exception_is_kept(check):
    raw = "java.lang.NullPointerException: boom\n"
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


def test_process_line_caused_by_is_ignored(check):
    line, date = check._process_line("Caused by: java.lang.IllegalStateException\n")
    assert line is None
    assert date is None


def test_process_line_too_few_tokens_is_ignored(check):
    line, date = check._process_line("2017-01-01 10:00:00\n")
    assert line is None
    assert date is None


def test_group_new_line_creates_entry_without_first_seen(check):
    common = check._group_similar_log_lines(line="ERROR first", date=None,
                                            common_lines=collections.OrderedDict())
    assert list(common.keys()) == ["ERROR first"]
    assert common["ERROR first"]["count"] == 1
    assert "first_seen" not in common["ERROR first"]


def test_group_new_line_records_first_seen_date(check):
    common = check._group_similar_log_lines(line="ERROR first", date="2017-01-01 10:00:00",
                                            common_lines=collections.OrderedDict())
    assert common["ERROR first"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_identical_line_increments_count(check):
    common = collections.OrderedDict()
    common = check._group_similar_log_lines("ERROR same", None, common)
    common = check._group_similar_log_lines("ERROR same", None, common)
    assert len(common) == 1
    assert common["ERROR same"]["count"] == 2


def test_group_similar_line_merges_into_existing_key(check):
    common = collections.OrderedDict()
    common = check._group_similar_log_lines(
        "ERROR database connection refused to host alpha", None, common)
    common = check._group_similar_log_lines(
        "ERROR database connection refused to host beta", None, common)
    assert len(common) == 1
    assert list(common.keys()) == ["ERROR database connection refused to host alpha"]
    assert common["ERROR database connection refused to host alpha"]["count"] == 2


def test_group_dissimilar_line_creates_separate_entry(check):
    common = collections.OrderedDict()
    common = check._group_similar_log_lines("aaaaaaaaaa", None, common)
    common = check._group_similar_log_lines("bbbbbbbbbb", None, common)
    assert list(common.keys()) == ["aaaaaaaaaa", "bbbbbbbbbb"]


def test_group_first_seen_keeps_earliest_date(check):
    common = collections.OrderedDict()
    common = check._group_similar_log_lines("ERROR x", "2017-01-01 10:00:00", common)
    common = check._group_similar_log_lines("ERROR x", "2017-01-02 11:00:00", common)
    assert common["ERROR x"]["first_seen"] == "2017-01-01 10:00:00"
    assert common["ERROR x"]["count"] == 2


def test_group_overflow_goes_to_other_bucket(check):
    # A ratio above 1.0 can never match, so only the overflow rule applies.
    check.SIMILARITY_RATIO = 1.01
    common = collections.OrderedDict()
    for i in range(check.MAX_COMMON_LINES):
        common["existing line %d" % i] = {"count": 1}
    common = check._group_similar_log_lines("brand new line", None, common)
    assert "Other Error Lines" in common
    assert common["Other Error Lines"]["count"] == 1
    assert "brand new line" not in common
    assert len(common) == check.MAX_COMMON_LINES + 1


def test_group_below_max_adds_new_line_even_when_dissimilar(check):
    check.SIMILARITY_RATIO = 1.01
    common = collections.OrderedDict()
    for i in range(check.MAX_COMMON_LINES - 1):
        common["existing line %d" % i] = {"count": 1}
    common = check._group_similar_log_lines("brand new line", None, common)
    assert "brand new line" in common
    assert "Other Error Lines" not in common


def test_remove_blacklisted_logs_filters_names(check):
    files = [
        "/logs/app.log",
        "/logs/app_configuration.log",
        "/logs/server.out",
        "/logs/gc.log",
        "/logs/service.log",
        "/logs/engine_access_log.log",
    ]
    assert check._remove_blacklisted_logs(files) == ["/logs/app.log", "/logs/service.log"]


def test_get_error_pattern_counts_reads_file(check, tmp_path):
    path = _write(tmp_path / "app.log", [
        "2017-01-01 10:00:00 ERROR disk full\n",
        "\tat com.example.Foo(Foo.java:1)\n",
        "2017-01-01 10:00:05 WARN minor\n",
        "2017-01-01 10:01:00 ERROR disk full\n",
    ])
    filename, common = check.get_error_pattern_counts(path)
    assert filename == path
    assert list(common.keys()) == ["ERROR disk full"]
    assert common["ERROR disk full"]["count"] == 2
    assert common["ERROR disk full"]["first_seen"] == "2017-01-01 10:00:00"


def test_run_returns_none_without_log_files(check):
    assert check.run({}) is None
    assert check.run({"LogFiles": []}) is None


def test_run_returns_none_when_all_logs_blacklisted(check, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "Pool", FakePool)
    path = _write(tmp_path / "x_configuration.log", [
        "2017-01-01 10:00:00 ERROR disk full\n",
    ])
    assert check.run({"LogFiles": [path]}) is None


def test_run_formats_output_and_skips_blacklisted(check, monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "Pool", FakePool)
    good = _write(tmp_path / "app.log", [
        "2017-01-01 10:00:00 ERROR disk full\n",
        "2017-01-01 10:01:00 ERROR disk full\n",
    ])
    skipped = _write(tmp_path / "gc.log", [
        "2017-01-01 10:00:00 ERROR should not appear\n",
    ])
    result = check.run({"LogFiles": [good, skipped]})
    expected = (
        f"filename: {good}\n"
        "First seen: 2017-01-01 10:00:00, Count: 2, ERROR disk full\n\n"
    )
    assert result == expected
    assert "should not appear" not in result
