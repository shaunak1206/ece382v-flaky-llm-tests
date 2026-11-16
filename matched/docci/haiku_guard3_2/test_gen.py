"""Unit tests for xlsx_to_bytes and xlsx_to_file in docci/xlsx.py."""

from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_module
from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


ZIP_MAGIC = b"PK\x03\x04"


def _load(data):
    """Load xlsx bytes back into a Workbook."""
    return load_workbook(BytesIO(data))


# ---------------------------------------------------------------------------
# xlsx_to_bytes
# ---------------------------------------------------------------------------

def test_xlsx_to_bytes_returns_bytes_instance():
    result = xlsx_to_bytes(Workbook())

    assert isinstance(result, bytes)


def test_xlsx_to_bytes_output_is_zip_archive():
    # .xlsx files are OOXML packages, i.e. zip archives.
    result = xlsx_to_bytes(Workbook())

    assert result.startswith(ZIP_MAGIC)


def test_xlsx_to_bytes_empty_workbook_round_trip():
    result = xlsx_to_bytes(Workbook())
    loaded = _load(result)

    assert loaded.sheetnames == ["Sheet"]
    assert loaded.active.max_row == 1
    assert loaded.active.max_column == 1
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_preserves_cell_values_of_each_type():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = "text"
    ws["B1"] = 42
    ws["C1"] = 3.5
    ws["D1"] = True
    ws["E1"] = False

    loaded = _load(xlsx_to_bytes(wb)).active

    assert loaded["A1"].value == "text"
    assert loaded["B1"].value == 42
    assert loaded["C1"].value == pytest.approx(3.5)
    assert loaded["D1"].value is True
    assert loaded["E1"].value is False


def test_xlsx_to_bytes_float_value_round_trip_within_tolerance():
    wb = Workbook()
    wb.active["A1"] = 0.1 + 0.2

    loaded = _load(xlsx_to_bytes(wb)).active

    assert loaded["A1"].value == pytest.approx(0.1 + 0.2, rel=1e-12)


def test_xlsx_to_bytes_preserves_unicode_strings():
    wb = Workbook()
    wb.active["A1"] = "héllo wörld 日本語 ✓"

    loaded = _load(xlsx_to_bytes(wb)).active

    assert loaded["A1"].value == "héllo wörld 日本語 ✓"


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    wb.active["A1"] = 1
    wb.active["A2"] = "=A1+1"

    loaded = _load(xlsx_to_bytes(wb)).active

    assert loaded["A2"].value == "=A1+1"


def test_xlsx_to_bytes_preserves_sheet_names_and_order():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_preserves_content_of_every_sheet():
    wb = Workbook()
    wb.active.title = "Alpha"
    wb.active["A1"] = "a"
    other = wb.create_sheet("Beta")
    other["A1"] = "b"
    other["B2"] = 7

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded["Alpha"]["A1"].value == "a"
    assert loaded["Beta"]["A1"].value == "b"
    assert loaded["Beta"]["B2"].value == 7


def test_xlsx_to_bytes_preserves_active_sheet_selection():
    wb = Workbook()
    wb.active.title = "Main"
    wb.create_sheet("Selected")
    wb.active = 1

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active.title == "Selected"


def test_xlsx_to_bytes_preserves_row_and_column_dimensions():
    wb = Workbook()
    ws = wb.active
    ws["C3"] = "corner"

    loaded = _load(xlsx_to_bytes(wb)).active

    assert loaded.max_row == 3
    assert loaded.max_column == 3
    assert loaded["C3"].value == "corner"


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = Workbook()
    wb.active.title = "Keep"
    wb.active["A1"] = "unchanged"
    wb.create_sheet("Extra")

    xlsx_to_bytes(wb)

    assert wb.sheetnames == ["Keep", "Extra"]
    assert wb["Keep"]["A1"].value == "unchanged"


def test_xlsx_to_bytes_repeated_calls_yield_equivalent_content():
    wb = Workbook()
    wb.active["A1"] = "repeat"

    first = _load(xlsx_to_bytes(wb))
    second = _load(xlsx_to_bytes(wb))

    assert first.sheetnames == second.sheetnames
    assert first.active["A1"].value == second.active["A1"].value == "repeat"


def test_xlsx_to_bytes_different_workbooks_yield_different_content():
    wb_a = Workbook()
    wb_a.active["A1"] = "alpha"
    wb_b = Workbook()
    wb_b.active["A1"] = "beta"

    loaded_a = _load(xlsx_to_bytes(wb_a)).active
    loaded_b = _load(xlsx_to_bytes(wb_b)).active

    assert loaded_a["A1"].value == "alpha"
    assert loaded_b["A1"].value == "beta"


# ---------------------------------------------------------------------------
# xlsx_to_file
# ---------------------------------------------------------------------------

def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(Workbook(), "report.xlsx")

    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(Workbook(), "report.xlsx")

    assert result.name == "report.xlsx"


def test_xlsx_to_file_name_passed_through_unchanged():
    name = "données 2024 (final).xlsx"

    result = xlsx_to_file(Workbook(), name)

    assert result.name == name


def test_xlsx_to_file_content_is_xlsx_bytes():
    wb = Workbook()
    wb.active["A1"] = "in file"

    result = xlsx_to_file(wb, "data.xlsx")

    assert isinstance(result.content, bytes)
    assert result.content.startswith(ZIP_MAGIC)
    assert _load(result.content).active["A1"].value == "in file"


def test_xlsx_to_file_delegates_to_xlsx_to_bytes(monkeypatch):
    wb = Workbook()
    calls = []

    def fake_xlsx_to_bytes(xlsx):
        calls.append(xlsx)
        return b"sentinel-bytes"

    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", fake_xlsx_to_bytes)

    result = xlsx_to_file(wb, "any.xlsx")

    assert calls == [wb]
    assert result.content == b"sentinel-bytes"


def test_xlsx_to_file_does_not_modify_workbook():
    wb = Workbook()
    wb.active.title = "Stable"
    wb.active["B2"] = "value"

    xlsx_to_file(wb, "stable.xlsx")

    assert wb.sheetnames == ["Stable"]
    assert wb["Stable"]["B2"].value == "value"


def test_xlsx_to_file_content_round_trips_all_sheets():
    wb = Workbook()
    wb.active.title = "One"
    wb.active["A1"] = 1
    two = wb.create_sheet("Two")
    two["A1"] = "two"

    result = xlsx_to_file(wb, "multi.xlsx")
    loaded = _load(result.content)

    assert loaded.sheetnames == ["One", "Two"]
    assert loaded["One"]["A1"].value == 1
    assert loaded["Two"]["A1"].value == "two"
