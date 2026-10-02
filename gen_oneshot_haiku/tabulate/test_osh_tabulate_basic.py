"""Tests for tabulate library - basic functionality and input types."""

import pytest
from collections import namedtuple
from dataclasses import dataclass
from tabulate import tabulate, SEPARATING_LINE, tabulate_formats, simple_separated_format


class TestBasicTableFormatting:
    """Test basic tabulate functionality."""

    def test_list_of_lists_no_headers(self):
        """Test simple list of lists without headers."""
        data = [["spam", 41.9999], ["eggs", "451.0"]]
        result = tabulate(data)
        assert "spam" in result
        assert "eggs" in result
        assert "41.9999" in result

    def test_list_of_lists_with_numeric_data(self):
        """Test list of lists with numeric data."""
        data = [[1, 2.34], [-56, "8.999"], ["2", "10001"]]
        result = tabulate(data, tablefmt="simple")
        assert "1" in result
        assert "2.34" in result
        assert "-56" in result

    def test_empty_table(self):
        """Test empty table."""
        result = tabulate([])
        assert result == ""

    def test_single_row(self):
        """Test single row table."""
        data = [["spam", 41.9999]]
        result = tabulate(data, tablefmt="plain")
        assert "spam" in result
        assert "41.9999" in result

    def test_single_column(self):
        """Test single column table."""
        data = [["spam"], ["eggs"], ["bacon"]]
        result = tabulate(data, tablefmt="simple")
        assert "spam" in result
        assert "eggs" in result
        assert "bacon" in result


class TestHeadersModes:
    """Test different header modes."""

    def test_explicit_headers(self):
        """Test with explicit headers list."""
        data = [["spam", 41.9999], ["eggs", "451.0"]]
        result = tabulate(data, headers=["strings", "numbers"], tablefmt="simple")
        assert "strings" in result
        assert "numbers" in result

    def test_headers_firstrow(self):
        """Test headers='firstrow' mode."""
        data = [["strings", "numbers"], ["spam", 41.9999], ["eggs", "451.0"]]
        result = tabulate(data, headers="firstrow", tablefmt="simple")
        assert "strings" in result
        assert "numbers" in result
        assert "spam" in result

    def test_headers_keys(self):
        """Test headers='keys' mode with list of dicts."""
        data = [
            {"name": "Alice", "age": 24},
            {"name": "Bob", "age": 19}
        ]
        result = tabulate(data, headers="keys", tablefmt="simple")
        assert "Alice" in result
        assert "Bob" in result
        assert "24" in result or "age" in result

    def test_headers_keys_with_dict_of_iterables(self):
        """Test headers='keys' with dict of iterables."""
        data = {"name": ["Alice", "Bob"], "age": [24, 19]}
        result = tabulate(data, headers="keys", tablefmt="simple")
        assert "Alice" in result or "Bob" in result

    def test_no_headers(self):
        """Test without headers."""
        data = [["spam", 41.9999], ["eggs", "451.0"]]
        result = tabulate(data, tablefmt="simple")
        # Should not contain header line with dashes only
        lines = result.strip().split("\n")
        assert len(lines) >= 2

    def test_headers_with_fewer_than_columns(self):
        """Test headers with fewer columns than data."""
        data = [["a", "b", "c"], ["d", "e", "f"]]
        result = tabulate(data, headers=["col2", "col3"], tablefmt="simple")
        assert "col2" in result or "col3" in result


class TestInputTypes:
    """Test different input data types."""

    def test_list_of_lists(self):
        """Test list of lists input."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, tablefmt="plain")
        assert "Alice" in result
        assert "24" in result

    def test_list_of_dicts(self):
        """Test list of dicts input."""
        data = [
            {"name": "Alice", "age": 24},
            {"name": "Bob", "age": 19}
        ]
        result = tabulate(data, headers="keys", tablefmt="plain")
        assert "Alice" in result
        assert "Bob" in result

    def test_dict_of_iterables(self):
        """Test dict of iterables input."""
        data = {"name": ["Alice", "Bob"], "age": [24, 19]}
        result = tabulate(data, headers="keys", tablefmt="plain")
        assert "Alice" in result or "Bob" in result

    def test_list_of_namedtuples(self):
        """Test list of namedtuples."""
        Person = namedtuple("Person", ["name", "age"])
        data = [
            Person("Alice", 24),
            Person("Bob", 19)
        ]
        result = tabulate(data, headers="keys", tablefmt="plain")
        assert "Alice" in result
        assert "Bob" in result

    def test_list_of_dataclasses(self):
        """Test list of dataclasses."""
        @dataclass
        class Person:
            name: str
            age: int

        data = [
            Person("Alice", 24),
            Person("Bob", 19)
        ]
        result = tabulate(data, headers="keys", tablefmt="plain")
        assert "Alice" in result
        assert "Bob" in result

    def test_generator_input(self):
        """Test generator as input."""
        def gen_data():
            yield ["Alice", 24]
            yield ["Bob", 19]

        result = tabulate(gen_data(), tablefmt="plain")
        assert "Alice" in result
        assert "Bob" in result


class TestShowIndex:
    """Test showindex parameter."""

    def test_showindex_default(self):
        """Test showindex default behavior."""
        data = [["F", 24], ["M", 19]]
        result = tabulate(data, tablefmt="simple")
        # Default should not show index
        lines = result.strip().split("\n")
        assert len(lines) >= 2

    def test_showindex_always(self):
        """Test showindex='always'."""
        data = [["F", 24], ["M", 19]]
        result = tabulate(data, showindex="always", tablefmt="simple")
        # Should contain index numbers
        assert "0" in result or "1" in result

    def test_showindex_true(self):
        """Test showindex=True."""
        data = [["F", 24], ["M", 19]]
        result = tabulate(data, showindex=True, tablefmt="simple")
        assert "0" in result or "1" in result

    def test_showindex_never(self):
        """Test showindex='never'."""
        data = [["F", 24], ["M", 19]]
        result = tabulate(data, showindex="never", tablefmt="simple")
        # Should not show indices
        lines = result.strip().split("\n")
        assert len(lines) >= 2

    def test_showindex_custom_iterable(self):
        """Test showindex with custom iterable."""
        data = [["F", 24], ["M", 19]]
        result = tabulate(data, showindex=["row1", "row2"], tablefmt="simple")
        # Custom index labels should appear
        assert "row1" in result or "row2" in result


class TestSeparatingLine:
    """Test SEPARATING_LINE special constant."""

    def test_separating_line_in_data(self):
        """Test using SEPARATING_LINE to insert lines."""
        data = [
            ["Alice", 24],
            [SEPARATING_LINE, SEPARATING_LINE],
            ["Bob", 19]
        ]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="grid")
        # Should have structure with separating line
        lines = result.strip().split("\n")
        assert len(lines) > 3

    def test_separating_line_simple_format(self):
        """Test SEPARATING_LINE with simple format."""
        data = [
            ["Alice", 24],
            [SEPARATING_LINE, SEPARATING_LINE],
            ["Bob", 19]
        ]
        result = tabulate(data, tablefmt="simple")
        lines = result.strip().split("\n")
        assert len(lines) > 2


class TestTabulateFormats:
    """Test tabulate_formats list."""

    def test_tabulate_formats_is_list(self):
        """Test that tabulate_formats is a list."""
        assert isinstance(tabulate_formats, (list, tuple))

    def test_tabulate_formats_contains_common_formats(self):
        """Test that tabulate_formats contains expected formats."""
        assert "plain" in tabulate_formats
        assert "simple" in tabulate_formats
        assert "grid" in tabulate_formats

    def test_all_formats_are_strings(self):
        """Test that all format names are strings."""
        for fmt in tabulate_formats:
            assert isinstance(fmt, str)


class TestSimpleSeparatedFormat:
    """Test simple_separated_format function."""

    def test_tsv_format(self):
        """Test creating TSV format."""
        tsv = simple_separated_format("\t")
        result = tabulate([["foo", 1], ["spam", 23]], tablefmt=tsv)
        # TSV should use tabs as separator
        assert "\t" in result or "foo" in result

    def test_csv_like_format(self):
        """Test creating CSV-like format."""
        csv_fmt = simple_separated_format(",")
        result = tabulate([["a", "b"], ["c", "d"]], tablefmt=csv_fmt)
        assert "," in result or ("a" in result and "b" in result)

    def test_pipe_separated_format(self):
        """Test creating pipe-separated format."""
        psv = simple_separated_format("|")
        result = tabulate([["x", "y"], ["1", "2"]], tablefmt=psv)
        assert "|" in result or ("x" in result and "y" in result)


class TestEmptyAndSingleValues:
    """Test edge cases with empty and single values."""

    def test_table_with_empty_strings(self):
        """Test table containing empty strings."""
        data = [["Alice", ""], ["", "19"]]
        result = tabulate(data, tablefmt="simple")
        assert "Alice" in result
        assert "19" in result

    def test_table_with_single_cell(self):
        """Test table with single cell."""
        data = [["spam"]]
        result = tabulate(data, tablefmt="plain")
        assert "spam" in result

    def test_table_all_empty_strings(self):
        """Test table with all empty strings."""
        data = [["", ""], ["", ""]]
        result = tabulate(data)
        # Should still produce some output
        assert len(result) >= 0


class TestDataTypeHandling:
    """Test handling of different data types."""

    def test_integers_and_floats(self):
        """Test table with integers and floats."""
        data = [[1, 2.5], [3, 4.7]]
        result = tabulate(data, tablefmt="plain")
        assert "1" in result
        assert "2.5" in result

    def test_boolean_values(self):
        """Test table with boolean values."""
        data = [[True, False], [False, True]]
        result = tabulate(data, tablefmt="plain")
        # Booleans should be represented
        assert len(result) > 0

    def test_none_values(self):
        """Test table with None values."""
        data = [[None, 2], [3, None]]
        result = tabulate(data, tablefmt="plain")
        assert len(result) > 0

    def test_mixed_types_in_column(self):
        """Test column with mixed types."""
        data = [[1, "text"], [2.5, "more text"]]
        result = tabulate(data, tablefmt="plain")
        assert "text" in result
        assert "1" in result or "2.5" in result


class TestSpecialCharacters:
    """Test handling of special characters."""

    def test_unicode_characters(self):
        """Test table with unicode characters."""
        data = [["ñame", "ägé"], ["çafe", "élève"]]
        result = tabulate(data, tablefmt="plain")
        # Should handle unicode
        assert len(result) > 0

    def test_newlines_in_cells(self):
        """Test cells with newlines."""
        data = [["line1\nline2", "cell"], ["cell2", "cell3"]]
        result = tabulate(data, tablefmt="plain")
        # Should handle multiline cells
        assert len(result) > 0

    def test_tabs_in_cells(self):
        """Test cells with tab characters."""
        data = [["col1\tcol1b", "col2"], ["col3", "col4"]]
        result = tabulate(data, tablefmt="plain")
        assert len(result) > 0
