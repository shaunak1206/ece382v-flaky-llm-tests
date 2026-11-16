import logging

import pytest

import fossor.checks.similar_log_errors as sle_module
from fossor.checks.similar_log_errors import SimilarLogErrors


class FakePool:
    """Sequential stand-in for multiprocessing.Pool."""
    calls = []

    def __init__(self, processes=None):
        self.processes = processes

    def map(self, func, iterable):
        items = list(iterable)
        FakePool.calls.append(items)
        return [func(x) for x in items]


@pytest.fixture
def check():
    return SimilarLogErrors()


@pytest.fixture
def fake_pool(monkeypatch):
    FakePool.calls = []
    monkeypatch.setattr(sle_module, 'Pool', FakePool)
    monkeypatch.setattr(sle_module, 'cpu_count', lambda: 1)
    return FakePool


def test_process_line_skips_indented_stacktrace_lines(check):
    line, date = check._process_line("   at com.foo.Bar(Bar.java:10)\n")
    assert line is None
    assert date is None


def test_process_line_skips_alpha_line_without_exception(check):
    line, date = check._process_line("Some random warning text\n")
    assert line is None
    assert date is None


def test_process_line_keeps_alpha_line_with_exception(check):
    raw = "java.lang.RuntimeException: boom\n"
    line, date = check._process_line(raw)
    assert line == raw
    assert date is None


def test_process_line_skips_caused_by(check):
    line, date = check._process_line("Caused by: java.lang.NullPointerException\n")
    assert line is None
    assert date is None


def test_process_line_error_with_date(check):
    line, date = check._process_line("2017-01-01 12:00:00 ERROR something bad happened\n")
    assert line == "ERROR something bad happened"
    assert date == "2017-01-01 12:00:00"


def test_process_line_non_error_level_skipped(check):
    line, date = check._process_line("2017-01-01 12:00:00 INFO all is fine\n")
    assert line is None
    assert date is None


def test_process_line_too_few_fields_skipped(check):
    line, date = check._process_line("2017-01-01 12:00\n")
    assert line is None
    assert date is None


def test_process_line_truncates_to_200_chars(check):
    raw = "2017-01-01 00:00:00 ERROR " + "x" * 300 + "\n"
    line, date = check._process_line(raw)
    assert len(line) == 200
    assert line.startswith("ERROR xxx")
    assert date == "2017-01-01 00:00:00"


def test_group_similar_lines_are_merged_and_keep_first_seen(check):
    common = {}
    from collections import OrderedDict
    common = OrderedDict()
    check._group_similar_log_lines("ERROR connection failed to host1", "2017-01-01 10:00:00", common)
    check._group_similar_log_lines("ERROR connection failed to host2", "2017-01-01 11:00:00", common)
    assert list(common.keys()) == ["ERROR connection failed to host1"]
    values = common["ERROR connection failed to host1"]
    assert values['count'] == 2
    assert values['first_seen'] == "2017-01-01 10:00:00"


def test_group_dissimilar_lines_are_separate(check):
    from collections import OrderedDict
    common = OrderedDict()
    check._group_similar_log_lines("ERROR aaaaaaaaaa", "2017-01-01 10:00:00", common)
    check._group_similar_log_lines("ERROR zzzzzzzzzz", "2017-01-01 10:00:01", common)
    assert list(common.keys()) == ["ERROR aaaaaaaaaa", "ERROR zzzzzzzzzz"]
    assert common["ERROR aaaaaaaaaa"]['count'] == 1
    assert common["ERROR zzzzzzzzzz"]['count'] == 1


def test_group_without_date_has_no_first_seen(check):
    from collections import OrderedDict
    common = OrderedDict()
    check._group_similar_log_lines("java.lang.RuntimeException: boom", None, common)
    assert common["java.lang.RuntimeException: boom"] == {'count': 1}


def test_group_overflow_goes_to_other_error_lines(check, monkeypatch):
    from collections import OrderedDict
    monkeypatch.setattr(SimilarLogErrors, 'MAX_COMMON_LINES', 2)
    common = OrderedDict()
    for text in ("ERROR aaaaaaaaaa", "ERROR bbbbbbbbbb", "ERROR cccccccccc", "ERROR dddddddddd"):
        check._group_similar_log_lines(text, "2017-01-01 10:00:00", common)
    assert "ERROR aaaaaaaaaa" in common
    assert "ERROR bbbbbbbbbb" in common
    assert "ERROR cccccccccc" not in common
    assert "ERROR dddddddddd" not in common
    assert common["Other Error Lines"]['count'] == 2


def test_group_existing_line_still_merges_when_full(check, monkeypatch):
    from collections import OrderedDict
    monkeypatch.setattr(SimilarLogErrors, 'MAX_COMMON_LINES', 2)
    common = OrderedDict()
    for text in ("ERROR aaaaaaaaaa", "ERROR bbbbbbbbbb", "ERROR aaaaaaaaaa"):
        check._group_similar_log_lines(text, None, common)
    assert common["ERROR aaaaaaaaaa"]['count'] == 2
    assert "Other Error Lines" not in common


def test_remove_blacklisted_logs(check):
    files = ['/var/log/app.log', '/var/log/gc.log', '/var/log/app.out',
             '/var/log/x_configuration.log', '/var/log/other.log']
    assert check._remove_blacklisted_logs(files) == ['/var/log/app.log', '/var/log/other.log']


def test_get_error_pattern_counts_reads_file(check, tmp_path):
    path = tmp_path / "service.log"
    path.write_text(
        "2017-01-01 10:00:00 INFO started\n"
        "2017-01-01 10:00:01 ERROR connection failed to host1\n"
        "    at com.foo.Bar(Bar.java:10)\n"
        "Caused by: something\n"
        "2017-01-01 10:00:02 ERROR connection failed to host2\n"
        "2017-01-01 10:00:03 ERROR aaaaaaaaaaaaaaaaaaaa\n"
    )
    filename, common = check.get_error_pattern_counts(str(path))
    assert filename == str(path)
    assert len(common) == 2
    items = list(common.items())
    assert items[0][0] == "ERROR connection failed to host1"
    assert items[0][1] == {'count': 2, 'first_seen': "2017-01-01 10:00:01"}
    assert items[1][1]['count'] == 1


def test_run_returns_none_without_log_files(check):
    assert check.run({}) is None
    assert check.run({'LogFiles': []}) is None


def test_run_formats_output(check, fake_pool, tmp_path):
    path = tmp_path / "service.log"
    path.write_text(
        "2017-01-01 10:00:01 ERROR connection failed to host1\n"
        "2017-01-01 10:00:02 ERROR connection failed to host2\n"
    )
    result = check.run({'LogFiles': [str(path)]})
    expected = (
        f"filename: {path}\n"
        "First seen: 2017-01-01 10:00:01, Count: 2, ERROR connection failed to host1\n\n"
    )
    assert result == expected


def test_run_without_date_omits_first_seen(check, fake_pool, tmp_path):
    path = tmp_path / "service.log"
    path.write_text("java.lang.RuntimeException: boom\n")
    result = check.run({'LogFiles': [str(path)]})
    assert result == f"filename: {path}\nCount: 1, java.lang.RuntimeException: boom\n\n"


def test_run_returns_none_when_no_errors_found(check, fake_pool, tmp_path):
    path = tmp_path / "quiet.log"
    path.write_text("2017-01-01 10:00:00 INFO fine\n   indented\n")
    assert check.run({'LogFiles': [str(path)]}) is None


def test_run_filters_blacklisted_logs_before_processing(check, fake_pool, tmp_path):
    good = tmp_path / "service.log"
    good.write_text("2017-01-01 10:00:00 ERROR bad thing\n")
    bad = tmp_path / "gc.log"
    bad.write_text("2017-01-01 10:00:00 ERROR gc thing\n")
    result = check.run({'LogFiles': [str(good), str(bad)]})
    assert fake_pool.calls == [[str(good)]]
    assert str(good) in result
    assert str(bad) not in result


def test_run_multiple_files_in_input_order(check, fake_pool, tmp_path):
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    first.write_text("2017-01-01 10:00:00 ERROR aaaaaaaaaa\n")
    second.write_text("2017-01-02 10:00:00 ERROR zzzzzzzzzz\n")
    result = check.run({'LogFiles': [str(first), str(second)]})
    assert result.index(f"filename: {first}") < result.index(f"filename: {second}")
    assert "First seen: 2017-01-01 10:00:00, Count: 1, ERROR aaaaaaaaaa" in result
    assert "First seen: 2017-01-02 10:00:00, Count: 1, ERROR zzzzzzzzzz" in result
