from unittest.mock import MagicMock, patch

import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


@pytest.fixture
def check():
    return SimilarLogErrors()


def _fake_pool():
    pool = MagicMock()
    pool.map.side_effect = lambda func, items: [func(i) for i in items]
    return pool


# ---------- _process_line ----------

def test_process_line_whitespace_start_is_skipped(check):
    line, date = check._process_line("    at com.example.Foo.bar(Foo.java:10)\n")
    assert line is None
    assert date is None


def test_process_line_newline_only_is_skipped(check):
    line, date = check._process_line("\n")
    assert line is None
    assert date is None


def test_process_line_alpha_without_exception_is_skipped(check):
    line, date = check._process_line("Something happened here\n")
    assert line is None
    assert date is None


def test_process_line_alpha_with_exception_is_kept(check):
    raw = "java.lang.NullPointerException: boom\n"
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


def test_process_line_caused_by_is_skipped(check):
    line, date = check._process_line("Caused by: java.lang.RuntimeException: bad\n")
    assert line is None
    assert date is None


def test_process_line_error_with_date_is_parsed(check):
    line, date = check._process_line("2017-01-01 10:00:00 ERROR something bad\n")
    assert line == "ERROR something bad"
    assert date == "2017-01-01 10:00:00"


def test_process_line_non_error_level_is_skipped(check):
    line, date = check._process_line("2017-01-01 10:00:00 INFO all good\n")
    assert line is None
    assert date is None


def test_process_line_too_few_fields_is_skipped(check):
    line, date = check._process_line("2017-01-01 10:00:00\n")
    assert line is None
    assert date is None


def test_process_line_truncates_to_200_chars(check):
    raw = "2017-01-01 10:00:00 ERROR " + "x" * 500 + "\n"
    line, date = check._process_line(raw)
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 10:00:00"


def test_process_line_other_leading_character_returned_unchanged(check):
    raw = "[main] something\n"
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


# ---------- _group_similar_log_lines ----------

def test_group_new_line_creates_entry_with_first_seen(check):
    result = check._group_similar_log_lines(line="ERROR boom", date="2017-01-01 10:00:00", common_lines={})
    assert result["ERROR boom"]["count"] == 1
    assert result["ERROR boom"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_without_date_has_no_first_seen(check):
    result = check._group_similar_log_lines(line="SomeException", date=None, common_lines={})
    assert result["SomeException"] == {"count": 1}


def test_group_similar_lines_increment_and_keep_first_seen(check):
    common = {}
    check._group_similar_log_lines("ERROR Connection failed to host 1", "2017-01-01 10:00:00", common)
    check._group_similar_log_lines("ERROR Connection failed to host 2", "2017-01-02 11:00:00", common)
    assert list(common.keys()) == ["ERROR Connection failed to host 1"]
    values = common["ERROR Connection failed to host 1"]
    assert values["count"] == 2
    assert values["first_seen"] == "2017-01-01 10:00:00"


def test_group_dissimilar_lines_are_separate(check):
    common = {}
    check._group_similar_log_lines("aaaaaaaaaa", None, common)
    check._group_similar_log_lines("bbbbbbbbbb", None, common)
    assert set(common.keys()) == {"aaaaaaaaaa", "bbbbbbbbbb"}


def test_group_overflow_goes_to_other_error_lines(check):
    check.MAX_COMMON_LINES = 2
    common = {}
    check._group_similar_log_lines("aaaaaaaaaa", None, common)
    check._group_similar_log_lines("bbbbbbbbbb", None, common)
    check._group_similar_log_lines("cccccccccc", None, common)
    check._group_similar_log_lines("dddddddddd", None, common)
    assert "cccccccccc" not in common
    assert common["Other Error Lines"]["count"] == 2


# ---------- _remove_blacklisted_logs ----------

def test_remove_blacklisted_logs(check):
    files = [
        "/var/log/app.log",
        "/var/log/gc.log",
        "/var/log/service.out",
        "/var/log/x_configuration.log",
        "/var/log/other.log",
    ]
    assert check._remove_blacklisted_logs(files) == ["/var/log/app.log", "/var/log/other.log"]


# ---------- get_error_pattern_counts ----------

def test_get_error_pattern_counts_reads_file(check, tmp_path):
    p = tmp_path / "app.log"
    p.write_text(
        "2017-01-01 10:00:00 INFO started\n"
        "2017-01-01 10:00:01 ERROR Connection failed to host 1\n"
        "    at some.Stack.frame(Stack.java:1)\n"
        "2017-01-01 10:00:02 ERROR Connection failed to host 2\n"
        "Caused by: java.lang.Exception\n"
        "plain text line\n"
    )
    filename, common = check.get_error_pattern_counts(str(p))
    assert filename == str(p)
    assert len(common) == 1
    values = list(common.values())[0]
    assert values["count"] == 2
    assert values["first_seen"] == "2017-01-01 10:00:01"


# ---------- run ----------

def test_run_without_log_files_returns_none(check):
    assert check.run({}) is None
    assert check.run({"LogFiles": []}) is None


def test_run_formats_output(check, tmp_path):
    p = tmp_path / "app.log"
    p.write_text(
        "2017-01-01 10:00:01 ERROR Connection failed to host 1\n"
        "2017-01-01 10:00:02 ERROR Connection failed to host 2\n"
    )
    with patch("fossor.checks.similar_log_errors.Pool", return_value=_fake_pool()):
        result = check.run({"LogFiles": [str(p)]})
    assert f"filename: {p}\n" in result
    assert "First seen: 2017-01-01 10:00:01, Count: 2, ERROR Connection failed to host 1" in result
    assert result.endswith("\n\n")


def test_run_no_errors_returns_none(check, tmp_path):
    p = tmp_path / "app.log"
    p.write_text("2017-01-01 10:00:01 INFO fine\n")
    with patch("fossor.checks.similar_log_errors.Pool", return_value=_fake_pool()):
        assert check.run({"LogFiles": [str(p)]}) is None


def test_run_skips_blacklisted_files(check, tmp_path):
    good = tmp_path / "app.log"
    bad = tmp_path / "gc.log"
    content = "2017-01-01 10:00:01 ERROR something broke\n"
    good.write_text(content)
    bad.write_text(content)
    with patch("fossor.checks.similar_log_errors.Pool", return_value=_fake_pool()):
        result = check.run({"LogFiles": [str(good), str(bad)]})
    assert f"filename: {good}" in result
    assert f"filename: {bad}" not in result


def test_run_exception_line_without_date(check, tmp_path):
    p = tmp_path / "app.log"
    p.write_text("java.lang.IllegalStateException: oops\n")
    with patch("fossor.checks.similar_log_errors.Pool", return_value=_fake_pool()):
        result = check.run({"LogFiles": [str(p)]})
    assert "Count: 1, java.lang.IllegalStateException: oops" in result
    assert "First seen" not in result
