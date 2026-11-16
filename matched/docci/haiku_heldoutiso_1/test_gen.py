"""Unit tests for docci.xlsx: xlsx_to_bytes and xlsx_to_file."""

from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _make_workbook():
    """Create a workbook with a single sheet containing a few values."""
    wb = Workbook()
    sheet = wb.active
    sheet["A1"] = "name"
    sheet["B1"] = "value"
    sheet["A2"] = "alpha"
    sheet["B2"] = 1
    sheet["A3"] = "beta"
    sheet["B3"] = 2.5
    return wb


def _load(data):
    """Load xlsx bytes back into a workbook."""
    return load_workbook(BytesIO(data))


def _sheet_values(wb):
    """Return {sheet_title: [[cell values row by row]]} for all sheets."""
    result = {}
    for ws in wb.worksheets:
        result[ws.title] = [list(row) for row in ws.iter_rows(values_only=True)]
    return result


# ---------------------------------------------------------------------------
# xlsx_to_bytes
# ---------------------------------------------------------------------------

def test_xlsx_to_bytes_returns_bytes_instance():
    result = xlsx_to_bytes(_make_workbook())
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_returns_non_empty_output():
    result = xlsx_to_bytes(_make_workbook())
    assert len(result) > 0


def test_xlsx_to_bytes_output_is_zip_container():
    # XLSX files are OOXML packages, which are ZIP archives starting with "PK".
    result = xlsx_to_bytes(_make_workbook())
    assert result[:2] == b"PK"


def test_xlsx_to_bytes_round_trip_preserves_cell_values():
    wb = _make_workbook()
    loaded = _load(xlsx_to_bytes(wb))
    assert _sheet_values(loaded) == _sheet_values(wb)


def test_xlsx_to_bytes_preserves_numeric_types():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 42
    ws["A2"] = 3.75
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A1"].value == 42
    assert isinstance(loaded["A1"].value, int)
    assert loaded["A2"].value == pytest.approx(3.75)


def test_xlsx_to_bytes_preserves_unicode_strings():
    wb = Workbook()
    wb.active["A1"] = "héllo wörld ✓ 日本語"
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A1"].value == "héllo wörld ✓ 日本語"


def test_xlsx_to_bytes_preserves_formulas():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=A1+A2"
    loaded = _load(xlsx_to_bytes(wb)).active
    assert loaded["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_preserves_multiple_sheets_and_titles():
    wb = Workbook()
    first = wb.active
    first.title = "Summary"
    first["A1"] = "one"
    second = wb.create_sheet("Details")
    second["A1"] = "two"

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["Summary", "Details"]
    assert loaded["Summary"]["A1"].value == "one"
    assert loaded["Details"]["A1"].value == "two"


def test_xlsx_to_bytes_empty_workbook_round_trips():
    wb = Workbook()
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == wb.sheetnames


def test_xlsx_to_bytes_does_not_modify_workbook_values():
    wb = _make_workbook()
    before = _sheet_values(wb)
    xlsx_to_bytes(wb)
    assert _sheet_values(wb) == before


def test_xlsx_to_bytes_can_be_called_repeatedly_on_same_workbook():
    wb = _make_workbook()
    first = _sheet_values(_load(xlsx_to_bytes(wb)))
    second = _sheet_values(_load(xlsx_to_bytes(wb)))
    assert first == second == _sheet_values(wb)


def test_xlsx_to_bytes_accepts_workbook_loaded_from_bytes():
    original = xlsx_to_bytes(_make_workbook())
    reloaded = _load(original)
    again = _load(xlsx_to_bytes(reloaded))
    assert _sheet_values(again) == _sheet_values(_load(original))


def test_xlsx_to_bytes_raises_for_non_workbook_input():
    with pytest.raises(AttributeError):
        xlsx_to_bytes(object())


# ---------------------------------------------------------------------------
# xlsx_to_file
# ---------------------------------------------------------------------------

def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_name_is_stored_verbatim():
    name = "My Report (final) 2024.xlsx"
    result = xlsx_to_file(_make_workbook(), name)
    assert result.name == name


def test_xlsx_to_file_content_matches_xlsx_to_bytes_round_trip():
    wb = _make_workbook()
    result = xlsx_to_file(wb, "data.xlsx")
    assert _sheet_values(_load(result.content)) == _sheet_values(wb)


def test_xlsx_to_file_content_is_zip_container():
    result = xlsx_to_file(_make_workbook(), "data.xlsx")
    assert isinstance(result.content, bytes)
    assert result.content[:2] == b"PK"


def test_xlsx_to_file_raises_for_non_workbook_input():
    with pytest.raises(AttributeError):
        xlsx_to_file(object(), "bad.xlsx")
