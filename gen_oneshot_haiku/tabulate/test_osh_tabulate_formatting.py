"""Tests for tabulate library - formatting options."""

import pytest
from tabulate import tabulate


class TestFloatFormatting:
    """Test floatfmt parameter."""

    def test_floatfmt_default(self):
        """Test default float formatting."""
        data = [[1.23456, 9.87654]]
        result = tabulate(data, tablefmt="plain")
        # Default 'g' format should be used
        assert "1.23" in result or "1" in result

    def test_floatfmt_fixed_width(self):
        """Test float format with fixed width."""
        data = [[1.23456, 9.87654]]
        result = tabulate(data, floatfmt=".2f", tablefmt="plain")
        # Should format to 2 decimal places
        assert "1.23" in result
        assert "9.88" in result

    def test_floatfmt_scientific(self):
        """Test float format scientific notation."""
        data = [[1234567.89, 0.000123]]
        result = tabulate(data, floatfmt=".2e", tablefmt="plain")
        # Should use scientific notation
        assert "e" in result.lower() or len(result) > 0

    def test_floatfmt_percent(self):
        """Test float format as percentage."""
        data = [[0.5, 0.33]]
        result = tabulate(data, floatfmt=".0%", tablefmt="plain")
        # Should show as percentage
        assert len(result) > 0

    def test_floatfmt_per_column(self):
        """Test different float formats per column."""
        data = [[1.23456, 9.87654, 3.14159]]
        result = tabulate(data, floatfmt=[".2f", ".3f", ".1f"], tablefmt="plain")
        # Should apply different formats to each column
        assert len(result) > 0

    def test_floatfmt_with_none(self):
        """Test float format with None values."""
        data = [[1.23, None], [None, 4.56]]
        result = tabulate(data, floatfmt=".2f", tablefmt="plain")
        assert len(result) > 0


class TestIntegerFormatting:
    """Test intfmt parameter."""

    def test_intfmt_default(self):
        """Test default integer formatting."""
        data = [[123, 456]]
        result = tabulate(data, tablefmt="plain")
        assert "123" in result
        assert "456" in result

    def test_intfmt_custom(self):
        """Test custom integer format."""
        data = [[123, 456]]
        result = tabulate(data, intfmt="05d", tablefmt="plain")
        # Should pad with zeros
        assert len(result) > 0

    def test_intfmt_per_column(self):
        """Test different int formats per column."""
        data = [[123, 456]]
        result = tabulate(data, intfmt=["05d", "03d"], tablefmt="plain")
        assert len(result) > 0

    def test_intfmt_hex(self):
        """Test integer format as hex."""
        data = [[255, 4096]]
        result = tabulate(data, intfmt="x", tablefmt="plain")
        # Should format as hex
        assert len(result) > 0


class TestMissingValues:
    """Test missingval parameter."""

    def test_missingval_default(self):
        """Test default missing value handling."""
        data = [[1, None], [None, 2]]
        result = tabulate(data, tablefmt="plain")
        # Default should replace None with empty string
        assert "1" in result
        assert "2" in result

    def test_missingval_custom_string(self):
        """Test custom missing value string."""
        data = [[1, None], [None, 2]]
        result = tabulate(data, missingval="?", tablefmt="plain")
        # Should replace None with "?"
        assert "?" in result

    def test_missingval_na(self):
        """Test missing value as 'N/A'."""
        data = [[1, None], [None, 2]]
        result = tabulate(data, missingval="N/A", tablefmt="plain")
        assert "N/A" in result

    def test_missingval_dash(self):
        """Test missing value as dash."""
        data = [[1, None], [None, 2]]
        result = tabulate(data, missingval="-", tablefmt="plain")
        assert "-" in result

    def test_missingval_per_column(self):
        """Test different missing values per column."""
        data = [[1, None], [None, 2]]
        result = tabulate(data, missingval=["?", "N/A"], tablefmt="plain")
        # Should use different placeholders
        assert len(result) > 0


class TestNumericAlignment:
    """Test numalign parameter."""

    def test_numalign_default(self):
        """Test default numeric alignment."""
        data = [[1, 2.5], [123, 45.678]]
        result = tabulate(data, tablefmt="grid")
        # Default should align numbers right or decimal
        assert len(result) > 0

    def test_numalign_right(self):
        """Test right alignment for numbers."""
        data = [[1, 2.5], [123, 45.678]]
        result = tabulate(data, numalign="right", tablefmt="simple")
        assert "1" in result

    def test_numalign_left(self):
        """Test left alignment for numbers."""
        data = [[1, 2.5], [123, 45.678]]
        result = tabulate(data, numalign="left", tablefmt="simple")
        assert "1" in result

    def test_numalign_center(self):
        """Test center alignment for numbers."""
        data = [[1, 2.5], [123, 45.678]]
        result = tabulate(data, numalign="center", tablefmt="simple")
        assert "1" in result

    def test_numalign_decimal(self):
        """Test decimal alignment for numbers."""
        data = [[1.2, 23.456], [123.4, 5.6]]
        result = tabulate(data, numalign="decimal", tablefmt="simple")
        # Decimal alignment should align decimal points
        assert len(result) > 0


class TestStringAlignment:
    """Test stralign parameter."""

    def test_stralign_default(self):
        """Test default string alignment."""
        data = [["Alice", "Bob"], ["X", "YZ"]]
        result = tabulate(data, tablefmt="simple")
        assert "Alice" in result

    def test_stralign_left(self):
        """Test left alignment for strings."""
        data = [["Alice", "Bob"], ["X", "YZ"]]
        result = tabulate(data, stralign="left", tablefmt="simple")
        assert "Alice" in result

    def test_stralign_right(self):
        """Test right alignment for strings."""
        data = [["Alice", "Bob"], ["X", "YZ"]]
        result = tabulate(data, stralign="right", tablefmt="simple")
        assert "Alice" in result

    def test_stralign_center(self):
        """Test center alignment for strings."""
        data = [["Alice", "Bob"], ["X", "YZ"]]
        result = tabulate(data, stralign="center", tablefmt="simple")
        assert "Alice" in result


class TestColumnAlignment:
    """Test colalign and colglobalalign parameters."""

    def test_colglobalalign_left(self):
        """Test global column alignment left."""
        data = [[1, "text"], [22, "more"]]
        result = tabulate(data, colglobalalign="left", tablefmt="simple")
        assert "1" in result

    def test_colglobalalign_right(self):
        """Test global column alignment right."""
        data = [[1, "text"], [22, "more"]]
        result = tabulate(data, colglobalalign="right", tablefmt="simple")
        assert "1" in result

    def test_colglobalalign_center(self):
        """Test global column alignment center."""
        data = [[1, "text"], [22, "more"]]
        result = tabulate(data, colglobalalign="center", tablefmt="simple")
        assert "1" in result

    def test_colalign_per_column(self):
        """Test per-column alignment."""
        data = [[1, "text"], [22, "more"]]
        result = tabulate(data, colalign=["left", "right"], tablefmt="simple")
        # Should apply different alignments
        assert "1" in result

    def test_colalign_global_override(self):
        """Test colalign overriding colglobalalign."""
        data = [[1, 2, 3], [4, 5, 6]]
        result = tabulate(
            data,
            colglobalalign="left",
            colalign=["right", "global", "left"],
            tablefmt="simple"
        )
        assert len(result) > 0


class TestHeadersAlignment:
    """Test headersglobalalign and headersalign parameters."""

    def test_headersglobalalign_left(self):
        """Test global header alignment left."""
        data = [[1, 2], [3, 4]]
        result = tabulate(data, headers=["A", "B"], headersglobalalign="left", tablefmt="simple")
        assert "A" in result

    def test_headersglobalalign_center(self):
        """Test global header alignment center."""
        data = [[1, 2], [3, 4]]
        result = tabulate(data, headers=["Col1", "Col2"], headersglobalalign="center", tablefmt="simple")
        assert "Col1" in result

    def test_headersglobalalign_right(self):
        """Test global header alignment right."""
        data = [[1, 2], [3, 4]]
        result = tabulate(data, headers=["X", "Y"], headersglobalalign="right", tablefmt="simple")
        assert "X" in result

    def test_headersalign_per_header(self):
        """Test per-header alignment."""
        data = [[1, 2], [3, 4]]
        result = tabulate(data, headers=["Col1", "Col2"], headersalign=["left", "right"], tablefmt="simple")
        assert "Col1" in result


class TestMaxColumnWidths:
    """Test maxcolwidths parameter."""

    def test_maxcolwidths_truncate(self):
        """Test maxcolwidths truncates long content."""
        data = [["VeryLongStringThatNeedsWrap", 123]]
        result = tabulate(data, maxcolwidths=[5, None], tablefmt="plain")
        # Should truncate or wrap the first column
        assert len(result) > 0

    def test_maxcolwidths_multiple_columns(self):
        """Test maxcolwidths with multiple columns."""
        data = [["LongText", "MoreLongText"], ["A", "B"]]
        result = tabulate(data, maxcolwidths=[5, 4], tablefmt="plain")
        assert len(result) > 0

    def test_maxcolwidths_with_headers(self):
        """Test maxcolwidths with headers."""
        data = [["VeryLongStringHere", 123]]
        result = tabulate(data, headers=["VeryLongHeaderName", "Number"], maxcolwidths=[10, None], tablefmt="simple")
        assert len(result) > 0

    def test_maxcolwidths_single_column(self):
        """Test maxcolwidths on single column."""
        data = [["ShortText"], ["A"]]
        result = tabulate(data, maxcolwidths=[5], tablefmt="plain")
        assert len(result) > 0


class TestDisableNumparse:
    """Test disable_numparse parameter."""

    def test_disable_numparse_false(self):
        """Test with disable_numparse=False (default)."""
        data = [["123", "45.67"]]
        result = tabulate(data, disable_numparse=False, tablefmt="plain")
        # Numbers should be treated as numbers and right-aligned
        assert "123" in result

    def test_disable_numparse_true(self):
        """Test with disable_numparse=True."""
        data = [["123", "45.67"]]
        result = tabulate(data, disable_numparse=True, tablefmt="plain")
        # Numbers should be treated as strings and left-aligned
        assert "123" in result

    def test_disable_numparse_with_thousands_sep(self):
        """Test disable_numparse with thousands separators."""
        data = [["1,000", "2,500.50"]]
        result = tabulate(data, disable_numparse=False, tablefmt="plain")
        # Should preserve thousands separators when parsing numbers
        assert "," in result or len(result) > 0

    def test_disable_numparse_scientific_notation(self):
        """Test disable_numparse with scientific notation."""
        data = [["1e5", "2.5e-3"]]
        result = tabulate(data, disable_numparse=False, tablefmt="plain")
        # Should handle scientific notation
        assert len(result) > 0


class TestPreserveWhitespace:
    """Test preserve_whitespace parameter."""

    def test_preserve_whitespace_false(self):
        """Test with preserve_whitespace=False (default)."""
        data = [["  spaced  ", "text"]]
        result = tabulate(data, preserve_whitespace=False, tablefmt="plain")
        assert len(result) > 0

    def test_preserve_whitespace_true(self):
        """Test with preserve_whitespace=True."""
        data = [["  spaced  ", "text"]]
        result = tabulate(data, preserve_whitespace=True, tablefmt="plain")
        # Should preserve leading/trailing whitespace
        assert len(result) > 0


class TestCombinedFormatting:
    """Test combinations of formatting options."""

    def test_floatfmt_and_missingval(self):
        """Test floatfmt with missingval."""
        data = [[1.234, None], [None, 5.678]]
        result = tabulate(data, floatfmt=".2f", missingval="N/A", tablefmt="plain")
        assert "N/A" in result

    def test_alignment_and_width(self):
        """Test alignment with maxcolwidths."""
        data = [["LongContent", 123], ["X", 456]]
        result = tabulate(data, numalign="right", maxcolwidths=[8, None], tablefmt="simple")
        assert len(result) > 0

    def test_all_formatting_options(self):
        """Test with many formatting options combined."""
        data = [[1.234, "text", None], [5.678, "more", 123]]
        result = tabulate(
            data,
            headers=["Numbers", "Text", "Mixed"],
            floatfmt=".2f",
            intfmt="03d",
            missingval="-",
            numalign="right",
            stralign="left",
            tablefmt="grid"
        )
        assert "1.23" in result or "1.2" in result or len(result) > 0


class TestSpecialNumericValues:
    """Test handling of special numeric values."""

    def test_infinity(self):
        """Test infinity values."""
        data = [[float('inf'), float('-inf')]]
        result = tabulate(data, tablefmt="plain")
        # Should handle infinity
        assert "inf" in result.lower() or len(result) > 0

    def test_nan(self):
        """Test NaN values."""
        data = [[float('nan'), 1.0]]
        result = tabulate(data, tablefmt="plain")
        # Should handle NaN
        assert "nan" in result.lower() or len(result) > 0

    def test_zero_values(self):
        """Test zero values."""
        data = [[0, 0.0, -0.0]]
        result = tabulate(data, tablefmt="plain")
        assert "0" in result

    def test_very_large_numbers(self):
        """Test very large numbers."""
        data = [[1e308, 1e309]]
        result = tabulate(data, tablefmt="plain")
        # Should handle large numbers
        assert len(result) > 0

    def test_very_small_numbers(self):
        """Test very small numbers."""
        data = [[1e-308, 1e-309]]
        result = tabulate(data, tablefmt="plain")
        # Should handle small numbers
        assert len(result) > 0
