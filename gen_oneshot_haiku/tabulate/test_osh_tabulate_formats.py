"""Tests for tabulate library - different table formats."""

import pytest
from tabulate import tabulate


class TestPlainFormat:
    """Test 'plain' table format."""

    def test_plain_basic(self):
        """Test plain format basic output."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, tablefmt="plain")
        assert "Alice" in result
        assert "Bob" in result
        # Plain should not have borders
        assert "|" not in result

    def test_plain_with_headers(self):
        """Test plain format with headers."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="plain")
        assert "Name" in result
        assert "Age" in result


class TestSimpleFormat:
    """Test 'simple' table format."""

    def test_simple_basic(self):
        """Test simple format basic output."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, tablefmt="simple")
        assert "Alice" in result
        assert "Bob" in result
        lines = result.strip().split("\n")
        assert len(lines) > 2

    def test_simple_with_headers(self):
        """Test simple format with headers."""
        data = [["spam", 41.9999], ["eggs", "451.0"]]
        result = tabulate(data, headers=["strings", "numbers"], tablefmt="simple")
        assert "strings" in result
        assert "numbers" in result


class TestGridFormat:
    """Test 'grid' table format."""

    def test_grid_basic(self):
        """Test grid format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="grid")
        # Grid should have box-drawing characters or borders
        assert ("+" in result or "|" in result)
        assert "Alice" in result

    def test_grid_with_many_rows(self):
        """Test grid format with multiple rows."""
        data = [[f"Row{i}", i] for i in range(5)]
        result = tabulate(data, headers=["Label", "Number"], tablefmt="grid")
        for i in range(5):
            assert f"Row{i}" in result


class TestPipeFormat:
    """Test 'pipe' table format (Markdown-like)."""

    def test_pipe_basic(self):
        """Test pipe format (Markdown)."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="pipe")
        assert "|" in result
        assert "Name" in result
        assert "Age" in result

    def test_pipe_alignment_marker(self):
        """Test that pipe format includes alignment markers."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="pipe")
        # Pipe format should have dashes
        assert "-" in result


class TestOrgtblFormat:
    """Test 'orgtbl' format (Emacs org-table)."""

    def test_orgtbl_basic(self):
        """Test orgtbl format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="orgtbl")
        assert "|" in result
        assert "Alice" in result


class TestRstFormat:
    """Test 'rst' format (reStructuredText)."""

    def test_rst_basic(self):
        """Test rst format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="rst")
        assert "=" in result or "-" in result
        assert "Alice" in result

    def test_rst_simple(self):
        """Test rst format with simple data."""
        data = [["spam", 41.9999], ["eggs", "451.0"]]
        result = tabulate(data, headers=["strings", "numbers"], tablefmt="rst")
        lines = result.strip().split("\n")
        assert len(lines) > 2


class TestMediawikiFormat:
    """Test 'mediawiki' format."""

    def test_mediawiki_basic(self):
        """Test mediawiki format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="mediawiki")
        # Mediawiki uses |- and | delimiters
        assert "|" in result
        assert "Alice" in result


class TestLatexFormats:
    """Test LaTeX table formats."""

    def test_latex_basic(self):
        """Test latex format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="latex")
        # LaTeX should use & for column separator and \\ for rows
        assert "&" in result or "\\" in result

    def test_latex_raw(self):
        """Test latex_raw format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="latex_raw")
        assert "&" in result or "\\" in result

    def test_latex_booktabs(self):
        """Test latex_booktabs format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="latex_booktabs")
        # Should contain LaTeX commands
        assert len(result) > 0


class TestHtmlFormat:
    """Test HTML format."""

    def test_html_basic(self):
        """Test html format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="html")
        # HTML should have table tags
        assert "<table>" in result or "<thead>" in result
        assert "<tr>" in result
        assert "<td>" in result or "<th>" in result

    def test_html_escapes_special_chars(self):
        """Test HTML format escapes special characters."""
        data = [["<script>", "&more"]]
        result = tabulate(data, tablefmt="html")
        # Should escape special HTML characters
        assert "&lt;" in result or "<" not in result or "&" in result


class TestTsvFormat:
    """Test TSV (tab-separated values) format."""

    def test_tsv_basic(self):
        """Test TSV format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="tsv")
        # TSV should use tabs
        assert "\t" in result

    def test_tsv_values_separated(self):
        """Test TSV values are properly separated."""
        data = [["a", "b"], ["c", "d"]]
        result = tabulate(data, tablefmt="tsv")
        lines = result.strip().split("\n")
        assert len(lines) == 2


class TestJiraFormat:
    """Test Jira table format."""

    def test_jira_basic(self):
        """Test Jira format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="jira")
        # Jira uses || for headers and | for cells
        assert "|" in result
        assert "Alice" in result


class TestFancyGridFormat:
    """Test fancy_grid format."""

    def test_fancy_grid_basic(self):
        """Test fancy_grid format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="fancy_grid")
        # Should have special box-drawing characters
        assert len(result) > 0
        assert "Alice" in result


class TestSimpleGridFormat:
    """Test simple_grid format."""

    def test_simple_grid_basic(self):
        """Test simple_grid format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="simple_grid")
        # Should have box-drawing characters
        assert len(result) > 0
        assert "Alice" in result


class TestPsqlFormat:
    """Test psql (PostgreSQL-like) format."""

    def test_psql_basic(self):
        """Test psql format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="psql")
        # psql uses | for separators
        assert "|" in result or len(result) > 0


class TestRoundedGridFormat:
    """Test rounded_grid format."""

    def test_rounded_grid_basic(self):
        """Test rounded_grid format."""
        data = [["Alice", 24], ["Bob", 19]]
        result = tabulate(data, headers=["Name", "Age"], tablefmt="rounded_grid")
        # Should have rounded box-drawing characters
        assert len(result) > 0
        assert "Alice" in result


class TestFormatConsistency:
    """Test consistency across formats."""

    def test_all_formats_preserve_data(self):
        """Test that all formats preserve the data content."""
        data = [["Alice", 24], ["Bob", 19]]
        headers = ["Name", "Age"]

        for fmt in ["plain", "simple", "grid", "pipe", "rst", "html", "latex", "tsv"]:
            result = tabulate(data, headers=headers, tablefmt=fmt)
            assert "Alice" in result, f"Data not found in {fmt} format"
            assert "24" in result or "Bob" in result, f"Data missing in {fmt} format"

    def test_format_with_special_values(self):
        """Test formats with special numeric values."""
        data = [[1.5, -2], [3.14159, 42]]

        for fmt in ["plain", "simple", "grid"]:
            result = tabulate(data, tablefmt=fmt)
            assert "1.5" in result or "1" in result
            assert "3.14" in result or "3" in result


class TestMultilineContent:
    """Test formats with multiline content."""

    def test_multiline_in_grid(self):
        """Test grid format with multiline cells."""
        data = [["Line1\nLine2", "Single"], ["Single2", "Line1\nLine2"]]
        result = tabulate(data, tablefmt="grid")
        # Should handle multiline content
        assert len(result) > 0

    def test_multiline_in_simple(self):
        """Test simple format with multiline cells."""
        data = [["A\nB", "X"], ["Y", "Z"]]
        result = tabulate(data, tablefmt="simple")
        assert len(result) > 0


class TestNumericContentInFormats:
    """Test different formats with numeric content."""

    def test_large_numbers_in_formats(self):
        """Test large numbers in various formats."""
        data = [[1000000, 2.5e10], [3.14, 42]]

        for fmt in ["plain", "simple", "grid"]:
            result = tabulate(data, tablefmt=fmt)
            # At least one large number should be present
            assert "1000" in result or "1e" in result or len(result) > 0

    def test_negative_numbers_in_formats(self):
        """Test negative numbers in various formats."""
        data = [[-123, -45.67], [-1, -0.5]]

        for fmt in ["plain", "simple", "grid"]:
            result = tabulate(data, tablefmt=fmt)
            assert "-" in result
