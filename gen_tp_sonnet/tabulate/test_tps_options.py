import tabulate

T = tabulate.tabulate


def test_missingval_replaces_none():
    out = T([["a", None], ["b", 2]], missingval="?", tablefmt="plain")
    assert out == "a  ?\nb  2"


def test_missingval_default_is_empty_string():
    out = T([["a", None], ["b", 2]], tablefmt="plain")
    lines = out.splitlines()
    assert lines[0] == "a"
    assert lines[1] == "b  2"


def test_maxcolwidths_wraps_at_whitespace():
    out = T([["hello world"]], maxcolwidths=5, tablefmt="plain")
    assert out == "hello\nworld"


def test_maxcolwidths_per_column_list():
    out = T([["x", "hello world"]], maxcolwidths=[None, 5], tablefmt="plain")
    lines = out.splitlines()
    assert lines[0].split() == ["x", "hello"]
    assert lines[1].split() == ["world"]


def test_maxcolwidths_breaks_long_words_by_default():
    out = T([["abcdefghij"]], maxcolwidths=4, tablefmt="plain")
    assert out == "abcd\nefgh\nij"


def test_break_long_words_false_keeps_word_intact():
    out = T([["abcdefghij"]], maxcolwidths=4, break_long_words=False, tablefmt="plain")
    assert out == "abcdefghij"


def test_maxheadercolwidths_wraps_header():
    out = T([["a"]], headers=["hello world"], maxheadercolwidths=5, tablefmt="plain")
    lines = out.splitlines()
    assert lines[0] == "hello"
    assert lines[1] == "world"


def test_leading_whitespace_stripped_by_default():
    out = T([["  a"], ["b"]], tablefmt="plain")
    assert out == "a\nb"


def test_preserve_whitespace_keeps_leading_spaces():
    out = T([["  a"], ["b"]], preserve_whitespace=True, tablefmt="plain")
    assert out == "  a\nb"


def test_ansi_color_codes_do_not_count_toward_width():
    red = "\x1b[31mred\x1b[0m"
    out = T([[red, "x"], ["abcde", "y"]], tablefmt="plain")
    lines = out.splitlines()
    assert lines[0] == red + "    x"
    assert lines[1] == "abcde  y"


def test_all_lines_same_width_in_grid():
    data = [["alpha", 1.5], ["be", 22], ["gamma delta", 3]]
    lines = T(data, headers=["name", "val"], tablefmt="grid").splitlines()
    assert len({len(line) for line in lines}) == 1


