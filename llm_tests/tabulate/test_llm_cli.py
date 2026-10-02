"""Unit tests for the `tabulate` command-line interface (`python -m tabulate`),
exercised via subprocess with input piped through stdin, matching how a
user would actually invoke the CLI tool.
"""

import subprocess
import sys

import pytest


def run_cli(args, stdin_text=""):
    return subprocess.run(
        [sys.executable, "-m", "tabulate"] + args,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_default_rsv_input_simple_format():
    result = run_cli([], "a b\n1 2\n3 4\n")
    assert result.returncode == 0
    expected = "-  -\na  b\n1  2\n3  4\n-  -\n"
    assert result.stdout == expected


def test_cli_with_firstrow_header_flag():
    result = run_cli(["-1"], "name age\nAlice 30\nBob 25\n")
    assert result.returncode == 0
    expected = "name      age\n------  -----\nAlice      30\nBob        25\n"
    assert result.stdout == expected


def test_cli_output_format_grid():
    result = run_cli(["-1", "-f", "grid"], "a b\n1 2\n")
    expected = "+-----+-----+\n|   a |   b |\n+=====+=====+\n|   1 |   2 |\n+-----+-----+\n"
    assert result.returncode == 0
    assert result.stdout == expected


def test_cli_output_format_pipe():
    result = run_cli(["-1", "-f", "pipe"], "name age\nAlice 30\nBob 25\n")
    expected = (
        "| name   |   age |\n"
        "|:-------|------:|\n"
        "| Alice  |    30 |\n"
        "| Bob    |    25 |\n"
    )
    assert result.stdout == expected


def test_cli_csv_input():
    result = run_cli(["-1", "-r", "csv", "-f", "grid"], "a,b\n1,2\n3,4\n")
    expected = (
        "+-----+-----+\n"
        "|   a |   b |\n"
        "+=====+=====+\n"
        "|   1 |   2 |\n"
        "+-----+-----+\n"
        "|   3 |   4 |\n"
        "+-----+-----+\n"
    )
    assert result.stdout == expected


def test_cli_jsonl_input_headers_keys():
    stdin = '{"a": 1, "b": 2}\n{"a": 3, "b": 4}\n'
    result = run_cli(["-r", "jsonl", "-f", "github"], stdin)
    expected = "|   a |   b |\n|----:|----:|\n|   1 |   2 |\n|   3 |   4 |\n"
    assert result.stdout == expected


def test_cli_float_format_option():
    result = run_cli(["-1", "-F", ".2f"], "a b\n1.23456 2\n")
    assert result.returncode == 0
    assert "1.23" in result.stdout
    assert "1.23456" not in result.stdout


def test_cli_help_exits_zero_and_prints_usage():
    result = run_cli(["-h"])
    assert result.returncode == 0
    assert "Usage: tabulate" in result.stdout


def test_cli_invalid_format_exits_nonzero():
    result = run_cli(["-f", "not_a_real_format"], "a b\n1 2\n")
    assert result.returncode == 3
    assert "is not a supported output format" in result.stderr


def test_cli_output_to_file(tmp_path):
    outfile = tmp_path / "out.txt"
    result = run_cli(["-1", "-o", str(outfile)], "a b\n1 2\n")
    assert result.returncode == 0
    assert result.stdout == ""
    content = outfile.read_text()
    assert "a" in content and "b" in content


def test_cli_reads_from_named_file(tmp_path):
    infile = tmp_path / "in.txt"
    infile.write_text("a b\n1 2\n3 4\n")
    result = run_cli(["-1", str(infile)])
    assert result.returncode == 0
    expected = "  a    b\n---  ---\n  1    2\n  3    4\n"
    assert result.stdout == expected


@pytest.mark.parametrize("fmt", ["simple", "grid", "github", "pipe", "plain", "rst"])
def test_cli_accepts_all_common_formats_without_error(fmt):
    result = run_cli(["-1", "-f", fmt], "a b\n1 2\n")
    assert result.returncode == 0
    assert result.stdout.strip() != ""
