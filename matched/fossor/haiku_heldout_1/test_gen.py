import pytest

from fossor.checks.similar_log_errors import SimilarLogErrors


@pytest.fixture
def check():
    return SimilarLogErrors()


def write_log(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return str(path)


class FakePool:
    """Serial stand-in for multiprocessing.Pool so tests do not spawn processes."""

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        return [func(item) for item in iterable]


# ---------------------------------------------------------------------------
# _process_line
# ---------------------------------------------------------------------------

def test_process_line_skips_indented_stacktrace_line(check):
    assert check._process_line("    at com.example.Foo.bar(Foo.java:10)\n") == (None, None)


def test_process_line_skips_alpha_line_without_exception(check):
    assert check._process_line("Something happened here\n") == (None, None)


def test_process_line_keeps_alpha_line_containing_exception(check):
    line = "NullPointerException: boom"
    assert check._process_line(line) == (line, None)


def test_process_line_skips_caused_by_even_with_exception_text(check):
    line = "Caused by: java.lang.NullPointerException"
    assert check._process_line(line) == (None, None)


def test_process_line_skips_dated_line_that_is_not_error(check):
    line = "2017-01-01 10:00:00 INFO all good\n"
    assert check._process_line(line) == (None, None)


def test_process_line_extracts_error_and_date(check):
    line = "2017-01-01 10:00:00 ERROR something bad\n"
    assert check._process_line(line) == ("ERROR something bad", "2017-01-01 10:00:00")


def test_process_line_truncates_long_error_to_200_chars(check):
    line = "2017-01-01 10:00:00 ERROR " + "x" * 300
    result, date = check._process_line(line)
    assert len(result) == 200
    assert result.startswith("ERROR ")
    assert date == "2017-01-01 10:00:00"


def test_process_line_short_dated_line_returns_none(check):
    # Only two tokens, so indexing the level raises IndexError internally.
    assert check._process_line("2017-01-01 10:00\n") == (None, None)


# ---------------------------------------------------------------------------
# _group_similar_log_lines
# ---------------------------------------------------------------------------

def test_group_adds_new_line_with_count_one(check):
    common = check._group_similar_log_lines(
        line="ERROR alpha", date="2017-01-01 10:00:00", common_lines={}
    )
    assert common["ERROR alpha"]["count"] == 1
    assert common["ERROR alpha"]["first_seen"] == "2017-01-01 10:00:00"


def test_group_increments_identical_line(check):
    common = {}
    common = check._group_similar_log_lines("ERROR alpha", "d1", common)
    common = check._group_similar_log_lines("ERROR alpha", "d2", common)
    assert len(common) == 1
    assert common["ERROR alpha"]["count"] == 2


def test_group_merges_highly_similar_line_into_existing_key(check):
    common = {}
    common = check._group_similar_log_lines("ERROR failed task A", "d1", common)
    common = check._group_similar_log_lines("ERROR failed task B", "d2", common)
    assert list(common.keys()) == ["ERROR failed task A"]
    assert common["ERROR failed task A"]["count"] == 2


def test_group_keeps_dissimilar_line_separate(check):
    common = {}
    common = check._group_similar_log_lines("ERROR alpha", "d1", common)
    common = check._group_similar_log_lines("1234567890", "d2", common)
    assert set(common.keys()) == {"ERROR alpha", "1234567890"}
    assert common["1234567890"]["count"] == 1


def test_group_first_seen_is_not_overwritten(check):
    common = {}
    common = check._group_similar_log_lines("ERROR alpha", "first", common)
    common = check._group_similar_log_lines("ERROR alpha", "second", common)
    assert common["ERROR alpha"]["first_seen"] == "first"


def test_group_without_date_sets_no_first_seen(check):
    common = check._group_similar_log_lines("ERROR alpha", None, {})
    assert "first_seen" not in common["ERROR alpha"]
    assert common["ERROR alpha"]["count"] == 1


def test_group_routes_to_other_bucket_when_max_reached(check, monkeypatch):
    monkeypatch.setattr(SimilarLogErrors, "MAX_COMMON_LINES", 2)
    common = {}
    common = check._group_similar_log_lines("aaaa", "d", common)
    common = check._group_similar_log_lines("bbbb", "d", common)
    common = check._group_similar_log_lines("cccc", "d", common)
    assert "Other Error Lines" in common
    assert common["Other Error Lines"]["count"] == 1
    assert "cccc" not in common


# ---------------------------------------------------------------------------
# _remove_blacklisted_logs
# ---------------------------------------------------------------------------

def test_remove_blacklisted_logs_filters_known_names(check):
    files = [
        "/logs/app.log",
        "/logs/server.out",
        "/logs/gc.log",
        "/logs/service_configuration.log",
        "/logs/other.log",
    ]
    assert check._remove_blacklisted_logs(files) == ["/logs/app.log", "/logs/other.log"]


def test_remove_blacklisted_logs_empty_input(check):
    assert check._remove_blacklisted_logs([]) == []


# ---------------------------------------------------------------------------
# get_error_pattern_counts
# ---------------------------------------------------------------------------

def test_get_error_pattern_counts_counts_and_skips_noise(check, tmp_path):
    content = (
        "2017-01-01 10:00:00 ERROR Something failed badly\n"
        "2017-01-01 10:00:01 ERROR Something failed badly\n"
        "    at some.stack.frame\n"
        "2017-01-01 10:00:02 INFO ok\n"
    )
    path = write_log(tmp_path, "app.log", content)

    filename, common = check.get_error_pattern_counts(path)

    assert filename == path
    assert list(common.keys()) == ["ERROR Something failed badly"]
    entry = common["ERROR Something failed badly"]
    assert entry["count"] == 2
    assert entry["first_seen"] == "2017-01-01 10:00:00"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def test_run_returns_none_without_log_files(check):
    assert check.run({}) is None


def test_run_returns_none_when_no_error_lines(check, tmp_path, monkeypatch):
    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", FakePool)
    path = write_log(tmp_path, "app.log", "2017-01-01 10:00:00 INFO fine\n")
    assert check.run({"LogFiles": [path]}) is None


def test_run_reports_grouped_errors(check, tmp_path, monkeypatch):
    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", FakePool)
    path = write_log(
        tmp_path,
        "app.log",
        "2017-01-01 10:00:00 ERROR Something failed badly\n"
        "2017-01-01 10:00:01 ERROR Something failed badly\n",
    )

    result = check.run({"LogFiles": [path]})

    assert result is not None
    assert f"filename: {path}\n" in result
    assert "First seen: 2017-01-01 10:00:00, Count: 2, ERROR Something failed badly" in result


def test_run_excludes_blacklisted_logs(check, tmp_path, monkeypatch):
    monkeypatch.setattr("fossor.checks.similar_log_errors.Pool", FakePool)
    path = write_log(
        tmp_path,
        "service_configuration.log",
        "2017-01-01 10:00:00 ERROR Should be ignored\n",
    )
    assert check.run({"LogFiles": [path]}) is None
