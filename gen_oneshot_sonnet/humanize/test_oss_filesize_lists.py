"""Tests for humanize.filesize.naturalsize and humanize.lists.natural_list."""

import pytest

import humanize


@pytest.fixture(autouse=True)
def _reset_locale():
    humanize.deactivate()
    yield
    humanize.deactivate()


# ---------------------------------------------------------------- naturalsize


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0 Bytes"),
        (1, "1 Byte"),
        (2, "2 Bytes"),
        (300, "300 Bytes"),
        (999, "999 Bytes"),
        (1000, "1.0 kB"),
        (1500, "1.5 kB"),
        (3_000_000, "3.0 MB"),
        (2_500_000_000, "2.5 GB"),
        (4_200_000_000_000, "4.2 TB"),
        (10**28, "10.0 RB"),
        (10**34 * 3, "30000.0 QB"),
        ("3000000", "3.0 MB"),
        (-1, "-1 Byte"),
        (-1500, "-1.5 kB"),
        (999_999, "1.0 MB"),
    ],
)
def test_naturalsize_decimal(value, expected):
    assert humanize.naturalsize(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, "1 Byte"),
        (1023, "1023 Bytes"),
        (1024, "1.0 KiB"),
        (3000, "2.9 KiB"),
        (5 * 1024**2, "5.0 MiB"),
        (3 * 1024**3, "3.0 GiB"),
        (-4096, "-4.0 KiB"),
    ],
)
def test_naturalsize_binary(value, expected):
    assert humanize.naturalsize(value, binary=True) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (1, "1B"),
        (300, "300B"),
        (1024, "1.0K"),
        (3000, "2.9K"),
        (5 * 1024**2, "5.0M"),
    ],
)
def test_naturalsize_gnu(value, expected):
    assert humanize.naturalsize(value, gnu=True) == expected


def test_naturalsize_positional_flags_and_gnu_precedence():
    assert humanize.naturalsize(300, False, True) == "300B"
    assert humanize.naturalsize(3000, False, True) == "2.9K"
    assert humanize.naturalsize(3000, True) == "2.9 KiB"
    assert humanize.naturalsize(3000, binary=True, gnu=True) == "2.9K"


def test_naturalsize_custom_format():
    assert humanize.naturalsize(3000, False, True, "%.3f") == "2.930K"
    assert humanize.naturalsize(1500, format="%.2f") == "1.50 kB"
    assert humanize.naturalsize(1500, format="%d") == "1 kB"


def test_naturalsize_invalid_string_raises():
    with pytest.raises(ValueError):
        humanize.naturalsize("not a size")


# ---------------------------------------------------------------- natural_list


@pytest.mark.parametrize(
    "items, expected",
    [
        ([], ""),
        (["one"], "one"),
        (["one", "two"], "one and two"),
        (["one", "two", "three"], "one, two and three"),
        (list("abcde"), "a, b, c, d and e"),
        ([1, 2, 3], "1, 2 and 3"),
        ([None, 3.5], "None and 3.5"),
        ([42], "42"),
    ],
)
def test_natural_list(items, expected):
    assert humanize.natural_list(items) == expected


def test_natural_list_does_not_mutate_input():
    items = ["x", "y", "z"]
    humanize.natural_list(items)
    assert items == ["x", "y", "z"]


def test_natural_list_accepts_tuple():
    assert humanize.natural_list(("a", "b", "c")) == "a, b and c"
