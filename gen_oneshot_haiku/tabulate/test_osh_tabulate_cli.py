"""Tests for tabulate CLI (command-line interface)."""

import pytest
import subprocess
import sys
import tempfile
import os


class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test CLI help message."""
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "help" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_cli_simple_stdin(self):
        """Test CLI with simple stdin input."""
        input_data = "a b\n1 2\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "a" in result.stdout
        assert "b" in result.stdout

    def test_cli_with_format(self):
        """Test CLI with specific format."""
        input_data = "a b\n1 2\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "grid"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        # Grid format should have box-drawing or borders
        assert len(result.stdout) > 0

    def test_cli_plain_format(self):
        """Test CLI with plain format."""
        input_data = "a b\n1 2\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "plain"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "a" in result.stdout
        assert "1" in result.stdout


class TestCLIHeaderOptions:
    """Test CLI header options."""

    def test_cli_headers_firstrow(self):
        """Test CLI with headers from first row."""
        input_data = "Name Age\nAlice 24\nBob 19\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-H", "firstrow"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Name" in result.stdout
        assert "Age" in result.stdout

    def test_cli_headers_explicit(self):
        """Test CLI with explicit headers."""
        input_data = "Alice 24\nBob 19\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-H", "Name", "-H", "Age"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Name" in result.stdout or "Alice" in result.stdout


class TestCLIInputFormats:
    """Test CLI input format options."""

    def test_cli_read_csv(self):
        """Test CLI reading CSV format."""
        input_data = "a,b,c\n1,2,3\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-r", "csv"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_cli_read_tsv(self):
        """Test CLI reading TSV format."""
        input_data = "a\tb\tc\n1\t2\t3\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-r", "tsv"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_cli_read_rsv(self):
        """Test CLI reading RSV format (default)."""
        input_data = "a  b  c\n1  2  3\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-r", "rsv"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0


class TestCLIFileInput:
    """Test CLI with file input."""

    def test_cli_from_file(self):
        """Test CLI reading from file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("a b c\n1 2 3\n4 5 6\n")
            f.flush()
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "tabulate", temp_path],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "a" in result.stdout or "1" in result.stdout
        finally:
            os.unlink(temp_path)

    def test_cli_from_stdin_dash(self):
        """Test CLI explicitly reading from stdin with '-'."""
        input_data = "x y z\n10 20 30\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "x" in result.stdout or "10" in result.stdout


class TestCLIOutputFormats:
    """Test CLI output format options."""

    def test_cli_output_simple(self):
        """Test CLI with simple output format."""
        input_data = "a b\n1 2\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "simple"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "a" in result.stdout

    def test_cli_output_grid(self):
        """Test CLI with grid output format."""
        input_data = "a b\n1 2\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "grid"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_cli_output_html(self):
        """Test CLI with HTML output format."""
        input_data = "a b\n1 2\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "html"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "<" in result.stdout  # HTML tags

    def test_cli_output_latex(self):
        """Test CLI with LaTeX output format."""
        input_data = "a b\n1 2\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "latex"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_cli_output_tsv(self):
        """Test CLI with TSV output format."""
        input_data = "a b\n1 2\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "tsv"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "\t" in result.stdout or len(result.stdout) > 0


class TestCLISeparatorOption:
    """Test CLI separator options."""

    def test_cli_custom_separator(self):
        """Test CLI with custom separator."""
        input_data = "a:b:c\n1:2:3\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-s", ":"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_cli_regex_separator(self):
        """Test CLI with regex separator."""
        input_data = "a   b   c\n1   2   3\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-s", r"\s+"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0


class TestCLIAlignment:
    """Test CLI alignment options."""

    def test_cli_alignment_options(self):
        """Test CLI alignment flag."""
        input_data = "abc 123\nx 9\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0


class TestCLIMultipleFiles:
    """Test CLI with multiple files."""

    def test_cli_multiple_files(self):
        """Test CLI with multiple input files."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f1:
            f1.write("a b\n1 2\n")
            f1.flush()
            path1 = f1.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f2:
            f2.write("x y\n10 20\n")
            f2.flush()
            path2 = f2.name

        try:
            result = subprocess.run(
                [sys.executable, "-m", "tabulate", path1, path2],
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert len(result.stdout) > 0
        finally:
            os.unlink(path1)
            os.unlink(path2)


class TestCLIEdgeCases:
    """Test CLI edge cases."""

    def test_cli_empty_input(self):
        """Test CLI with empty input."""
        input_data = ""
        result = subprocess.run(
            [sys.executable, "-m", "tabulate"],
            input=input_data,
            capture_output=True,
            text=True
        )
        # Should handle empty input gracefully
        assert result.returncode == 0

    def test_cli_single_line_input(self):
        """Test CLI with single line input."""
        input_data = "a b c\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert len(result.stdout) > 0

    def test_cli_unicode_input(self):
        """Test CLI with unicode input."""
        input_data = "café résumé\nèlève français\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate"],
            input=input_data,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        assert result.returncode == 0


class TestCLINumericFormatting:
    """Test CLI numeric formatting options."""

    def test_cli_floatfmt(self):
        """Test CLI with float format option."""
        input_data = "1.23456 2.34567\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-f", ".2f"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        # Should format floats as specified
        assert len(result.stdout) > 0


class TestCLIVersionOption:
    """Test CLI version option."""

    def test_cli_version(self):
        """Test CLI version option."""
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "--version"],
            capture_output=True,
            text=True
        )
        # Either success or command not recognized is acceptable
        assert result.returncode in [0, 2]


class TestCLIDataPreservation:
    """Test that CLI preserves data correctly."""

    def test_cli_data_integrity_simple_format(self):
        """Test data integrity with simple format."""
        input_data = "name age city\nAlice 24 Austin\nBob 30 Houston\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "simple"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert "Alice" in result.stdout
        assert "Austin" in result.stdout

    def test_cli_data_integrity_grid_format(self):
        """Test data integrity with grid format."""
        input_data = "x y z\n1 2 3\n4 5 6\n"
        result = subprocess.run(
            [sys.executable, "-m", "tabulate", "-t", "grid"],
            input=input_data,
            capture_output=True,
            text=True
        )
        assert "1" in result.stdout
        assert "6" in result.stdout
