"""Tests for the ``python -m tabulate`` command line interface."""

import os
import subprocess
import sys
import tempfile

import tabulate as tabulate_pkg

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(tabulate_pkg.__file__)))


def run_cli(*args, stdin=""):
    env = dict(os.environ)
    paths = [PKG_ROOT]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable, "-m", "tabulate", *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        cwd=tempfile.gettempdir(),
        timeout=30,
    )


def test_cli_reads_whitespace_separated_stdin_with_simple_format():
    res = run_cli(stdin="a 1\nb 22\n")
    assert res.returncode == 0
    assert res.stdout == "-  --\na   1\nb  22\n-  --\n"


def test_cli_dash_one_uses_first_row_as_header():
    res = run_cli("-1", stdin="name age\nal 3\nbo 12\n")
    assert res.returncode == 0
    expected = "\n".join(
        [
            "name" + " " * 6 + "age",
            "------  -----",
            "al" + " " * 10 + "3",
            "bo" + " " * 9 + "12",
        ]
    )
    assert res.stdout == expected + "\n"


def test_cli_headers_option_and_pipe_format():
    res = run_cli("--headers", "k,v", "-f", "pipe", stdin="a 1\nb 2\n")
    assert res.returncode == 0
    expected = "\n".join(
        [
            "| k   |   v |",
            "|:----|----:|",
            "| a   |   1 |",
            "| b   |   2 |",
        ]
    )
    assert res.stdout == expected + "\n"


def test_cli_csv_input_with_first_row_header():
    res = run_cli("-r", "csv", "-1", stdin="a,b\n1,2\n3,4\n")
    assert res.returncode == 0
    assert res.stdout == "  a    b\n---  ---\n  1    2\n  3    4\n"


def test_cli_jsonl_input_uses_keys_as_headers():
    stdin = '{"x": 1, "y": 2}\n{"x": 3, "y": 4}\n'
    res = run_cli("-r", "jsonl", stdin=stdin)
    assert res.returncode == 0
    assert res.stdout == "  x    y\n---  ---\n  1    2\n  3    4\n"


def test_cli_custom_separator_skips_blank_lines():
    res = run_cli("-s", ":", stdin="a:1\n\nb:2\n")
    assert res.returncode == 0
    assert res.stdout == "-  -\na  1\nb  2\n-  -\n"


def test_cli_float_format_option():
    res = run_cli("-F", ".2f", stdin="a 1.234\n")
    assert res.returncode == 0
    assert res.stdout == "-  ----\na  1.23\n-  ----\n"


def test_cli_output_file_option(tmp_path):
    out_file = tmp_path / "table.txt"
    res = run_cli("-o", str(out_file), stdin="a 1\n")
    assert res.returncode == 0
    assert res.stdout == ""
    assert out_file.read_text() == "-  -\na  1\n-  -\n"


def test_cli_reads_one_or_more_files(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a 1\n")
    second.write_text("b 2\n")
    single = run_cli(str(first))
    assert single.returncode == 0
    assert single.stdout == "-  -\na  1\n-  -\n"
    both = run_cli(str(first), str(second))
    assert both.returncode == 0
    assert both.stdout == "-  -\na  1\n-  -\n-  -\nb  2\n-  -\n"


def test_cli_help_and_error_exit_codes():
    helped = run_cli("-h")
    assert helped.returncode == 0
    assert "Usage: tabulate" in helped.stdout

    bad_format = run_cli("-f", "nonsense")
    assert bad_format.returncode == 3
    assert "not a supported output format" in bad_format.stderr

    bad_option = run_cli("--bogus")
    assert bad_option.returncode == 2

    bad_reader = run_cli("-r", "xml", stdin="a 1\n")
    assert bad_reader.returncode == 3
