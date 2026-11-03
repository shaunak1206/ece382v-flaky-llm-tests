import os
import tempfile
import pytest
from collections import OrderedDict
from fossor.checks.similar_log_errors import SimilarLogErrors


class DummyLog:
    def __init__(self):
        self.debug_messages = []

    def debug(self, msg):
        self.debug_messages.append(msg)


@pytest.fixture
def sl_errors():
    obj = SimilarLogErrors()
    # replace logger with a dummy logger capturing debug calls
    obj.log = DummyLog()
    return obj


def test_remove_blacklisted_logs_removes_blacklisted(sl_errors):
    logs = [
        'app_configuration.log',
        'test.out',
        'normal.log',
        'gc.log',
        'something_oor.log',
        'random.log'
    ]
    result = sl_errors._remove_blacklisted_logs(logs)
    assert 'normal.log' in result
    assert 'random.log' in result
    # blacklisted partial matches should remove
    assert 'app_configuration.log' not in result
    assert 'test.out' not in result
    assert 'gc.log' not in result
    assert 'something_oor.log' not in result


def test_remove_blacklisted_logs_empty_list(sl_errors):
    assert sl_errors._remove_blacklisted_logs([]) == []


def test_process_line_none_for_stacktrace_line(sl_errors):
    line = '    at some_function()'
    result_line, date = sl_errors._process_line(line)
    assert result_line is None


def test_process_line_none_for_alpha_start_no_exception(sl_errors):
    line = 'Warning: this is a warning'
    result_line, date = sl_errors._process_line(line)
    assert result_line is None


def test_process_line_none_for_caused_by_line(sl_errors):
    line = 'Caused by some error'
    result_line, date = sl_errors._process_line(line)
    assert result_line is None


def test_process_line_none_for_non_error_digit_line(sl_errors):
    line = '2023-05-01 12:12:12 INFO Some info message'
    result_line, date = sl_errors._process_line(line)
    assert result_line is None


def test_process_line_returns_line_and_date_for_error_line(sl_errors):
    line = '2023-05-01 15:13:14 ERROR This is an error message'
    processed_line, date = sl_errors._process_line(line)
    assert processed_line.startswith('ERROR')
    assert processed_line.endswith('message')
    assert date == '2023-05-01 15:13:14'


def test_process_line_truncates_line(sl_errors):
    long_msg = 'x' * 500
    line = f"2023-05-01 15:13:14 ERROR {long_msg}"
    processed_line, _ = sl_errors._process_line(line)
    # The processed line is truncated to 200 chars after 'ERROR '
    assert len(processed_line) <= 204  # 5 chars for 'ERROR'
    assert processed_line.startswith('ERROR')


def test_process_line_handles_short_line_and_index_error(sl_errors):
    # Should not throw IndexError and return None gracefully
    line = '2023-05-01'
    processed_line, date = sl_errors._process_line(line)
    assert processed_line is None
    # Also test a line without enough parts
    line2 = '2023-05-01 15:13:14'
    processed_line2, date2 = sl_errors._process_line(line2)
    assert processed_line2 is None


def make_common_lines_with_variants(sl_errors):
    cl = OrderedDict()
    cl['ERROR message one'] = {'count': 3, 'first_seen': '2023-01-01 00:00:00'}
    cl['ERROR message two'] = {'count': 1, 'first_seen': '2023-01-02 00:00:00'}
    return cl


def test_group_similar_log_lines_increments_existing(sl_errors):
    common_lines = make_common_lines_with_variants(sl_errors)
    line_similar = 'ERROR message one with extra text'  # close to 'ERROR message one'
    updated = sl_errors._group_similar_log_lines(line_similar, '2023-01-05 00:00:00', common_lines)
    # Should increment count for 'ERROR message one' not add new key
    assert 'ERROR message one' in updated
    assert updated['ERROR message one']['count'] == 4
    # The date should remain original (not overwritten)
    assert updated['ERROR message one']['first_seen'] == '2023-01-01 00:00:00'


def test_group_similar_log_lines_adds_new_unique_line(sl_errors):
    common_lines = make_common_lines_with_variants(sl_errors)
    new_line = 'ERROR completely different message'
    updated = sl_errors._group_similar_log_lines(new_line, '2023-01-06 00:00:00', common_lines)
    assert new_line in updated
    assert updated[new_line]['count'] == 1
    # First seen should be set
    assert updated[new_line]['first_seen'] == '2023-01-06 00:00:00'


def test_group_similar_log_lines_adds_to_other_error_lines_when_too_many(sl_errors):
    common_lines = OrderedDict()
    # Populate common_lines with MAX_COMMON_LINES unique keys
    for i in range(sl_errors.MAX_COMMON_LINES):
        common_lines[f"ERROR line {i}"] = {'count': 1}
    new_line = 'ERROR a new unique error'
    updated = sl_errors._group_similar_log_lines(new_line, None, common_lines)
    # Should add under 'Other Error Lines' key
    assert 'Other Error Lines' in updated
    assert updated['Other Error Lines']['count'] == 1
    # No new unique key other than 'Other Error Lines'
    assert new_line not in updated


def test_group_similar_log_lines_similar_but_below_threshold_counts_new(sl_errors):
    common_lines = OrderedDict()
    common_lines['ERROR abcdef'] = {'count': 1}
    # Construct line slightly less similar than threshold
    line = 'ERROR zzzzzz'  # very different
    # Force threshold to higher than quick_ratio for test
    orig_threshold = sl_errors.SIMILARITY_RATIO
    sl_errors.SIMILARITY_RATIO = 0.95
    updated = sl_errors._group_similar_log_lines(line, None, common_lines)
    sl_errors.SIMILARITY_RATIO = orig_threshold
    # Should add new unique line because similarity below threshold
    assert line in updated
    assert updated[line]['count'] == 1


def test_get_error_pattern_counts_creates_ordered_dict_and_counts(sl_errors):
    with tempfile.NamedTemporaryFile('w+t', delete=False) as tf:
        tf.write(
            "2023-05-01 15:13:14 ERROR error message one\n"
            "2023-05-01 15:13:14 ERROR error message one\n"
            "2023-05-01 15:13:14 ERROR different error\n"
            "Some random irrelevant line\n"
        )
        tf.flush()
        filename = tf.name

    try:
        fname, common_lines = sl_errors.get_error_pattern_counts(filename)
        assert fname == filename
        # It should have counted 2 occurrences of similar lines and 1 unique different line
        counts = {k: v['count'] for k, v in common_lines.items()}
        # There should be at least two keys
        assert len(counts) >= 1
        total_counts = sum(counts.values())
        assert total_counts == 3  # 3 valid lines, the irrelevant line ignored
    finally:
        os.unlink(filename)


def test_run_returns_none_if_no_logfiles_in_variables(sl_errors):
    assert sl_errors.run({}) is None
    assert sl_errors.run({'LogFiles': []}) is None


def test_run_filters_blacklisted_logfiles_and_returns_formatted_output(sl_errors):
    # Prepare two temp log files, one blacklisted
    blacklist_name = '_configuration.log'
    normal_name = 'normal.log'
    temp_files = []
    try:
        # blacklisted file
        with tempfile.NamedTemporaryFile('w+t', delete=False, prefix='foo'+blacklist_name) as tmp1:
            tmp1.write("2023-01-01 00:00:00 ERROR error in blacklisted\n")
            tmp1.flush()
            bl_file = tmp1.name
            temp_files.append(bl_file)

        # normal file
        with tempfile.NamedTemporaryFile('w+t', delete=False, prefix=normal_name) as tmp2:
            tmp2.write("2023-01-01 00:00:00 ERROR error in normal\n")
            tmp2.flush()
            norm_file = tmp2.name
            temp_files.append(norm_file)

        result = sl_errors.run({'LogFiles': temp_files})
        # Only normal file included, blacklisted removed
        assert norm_file in result
        assert bl_file not in result
        assert "Count: 1" in result
        assert "error in normal" in result
        # It includes 'filename: ' line
        assert f"filename: {norm_file}" in result
    finally:
        for f in temp_files:
            if os.path.exists(f):
                os.unlink(f)


def test_run_returns_none_if_all_logs_filtered_blacklist(sl_errors):
    logs = ['my_configuration.log', 'test.gc.log', '_public_access.log']
    result = sl_errors.run({'LogFiles': logs})
    assert result is None


def test_run_multiple_errors_logged(sl_errors):
    with tempfile.NamedTemporaryFile('w+t', delete=False) as tf:
        tf.write(
            "2023-01-01 00:00:00 ERROR error one\n"
            "2023-01-01 00:00:01 ERROR error two\n"
            "2023-01-01 00:00:02 ERROR error one\n"
        )
        tf.flush()
        filename = tf.name
    try:
        output = sl_errors.run({'LogFiles': [filename]})
        assert filename in output
        assert "error one" in output
        assert "Count: 2" in output
        assert "error two" in output
        assert "Count: 1" in output
    finally:
        os.unlink(filename)
