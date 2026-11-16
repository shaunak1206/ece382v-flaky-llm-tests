"""
Unit tests for docci.xlsx.xlsx_to_bytes and docci.xlsx.xlsx_to_file.
"""

import datetime
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


ZIP_MAGIC = b"PK\x03\x04"


@pytest.fixture
def simple_workbook():
    wb = Workbook()
    sheet = wb.active
    sheet["A1"] = "name"
    sheet["B1"] = "value"
    sheet["A2"] = "alpha"
    sheet["B2"] = 1
    sheet["A3"] = "beta"
    sheet["B3"] = 2
    return wb


def _load(data: bytes) -> Workbook:
    return load_workbook(BytesIO(data))


# ---------------------------------------------------------------------------
# xlsx_to_bytes
# ---------------------------------------------------------------------------

def test_xlsx_to_bytes_returns_bytes(simple_workbook):
    result = xlsx_to_bytes(simple_workbook)
    assert isinstance(result, bytes)


def test_xlsx_to_bytes_returns_non_empty(simple_workbook):
    assert len(xlsx_to_bytes(simple_workbook)) > 0


def test_xlsx_to_bytes_has_zip_signature(simple_workbook):
    # XLSX files are ZIP archives
    assert xlsx_to_bytes(simple_workbook).startswith(ZIP_MAGIC)


def test_xlsx_to_bytes_output_is_loadable(simple_workbook):
    wb = _load(xlsx_to_bytes(simple_workbook))
    assert isinstance(wb, Workbook)


def test_xlsx_to_bytes_preserves_cell_values(simple_workbook):
    wb = _load(xlsx_to_bytes(simple_workbook))
    sheet = wb.active
    assert sheet["A1"].value == "name"
    assert sheet["B1"].value == "value"
    assert sheet["A2"].value == "alpha"
    assert sheet["B2"].value == 1
    assert sheet["A3"].value == "beta"
    assert sheet["B3"].value == 2


def test_xlsx_to_bytes_preserves_sheet_names():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_empty_workbook_round_trips():
    wb = Workbook()
    data = xlsx_to_bytes(wb)

    assert data.startswith(ZIP_MAGIC)
    loaded = _load(data)
    assert loaded.sheetnames == wb.sheetnames


def test_xlsx_to_bytes_does_not_modify_workbook(simple_workbook):
    before = [[c.value for c in row] for row in simple_workbook.active.iter_rows()]
    xlsx_to_bytes(simple_workbook)
    after = [[c.value for c in row] for row in simple_workbook.active.iter_rows()]
    assert before == after


def test_xlsx_to_bytes_returns_independent_bytes(simple_workbook):
    first = xlsx_to_bytes(simple_workbook)
    simple_workbook.active["A2"] = "changed"
    second = xlsx_to_bytes(simple_workbook)

    assert _load(first).active["A2"].value == "alpha"
    assert _load(second).active["A2"].value == "changed"


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    wb.active["A1"] = 2
    wb.active["A2"] = "=A1*3"

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A2"].value == "=A1*3"


def test_xlsx_to_bytes_preserves_datetime_values():
    wb = Workbook()
    when = datetime.datetime(2023, 5, 17, 12, 30, 0)
    wb.active["A1"] = when

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == when


def test_xlsx_to_bytes_preserves_unicode_text():
    wb = Workbook()
    wb.active["A1"] = "héllo wörld ✓ 日本語"

    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == "héllo wörld ✓ 日本語"


def test_xlsx_to_bytes_rejects_non_workbook():
    with pytest.raises(AttributeError):
        xlsx_to_bytes("not a workbook")


# ---------------------------------------------------------------------------
# xlsx_to_file
# ---------------------------------------------------------------------------

def test_xlsx_to_file_returns_file_attachment(simple_workbook):
    result = xlsx_to_file(simple_workbook, "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_stores_name(simple_workbook):
    result = xlsx_to_file(simple_workbook, "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_xlsx_bytes(simple_workbook):
    result = xlsx_to_file(simple_workbook, "report.xlsx")
    assert isinstance(result.content, bytes)
    assert result.content.startswith(ZIP_MAGIC)


def test_xlsx_to_file_content_matches_xlsx_to_bytes_values(simple_workbook):
    attachment = xlsx_to_file(simple_workbook, "report.xlsx")
    loaded = _load(attachment.content)

    assert loaded.active["A1"].value == "name"
    assert loaded.active["B2"].value == 1
    assert loaded.active["A3"].value == "beta"


def test_xlsx_to_file_round_trip_preserves_sheet_names():
    wb = Workbook()
    wb.active.title = "Data"
    wb.create_sheet("Summary")

    attachment = xlsx_to_file(wb, "book.xlsx")
    assert _load(attachment.content).sheetnames == ["Data", "Summary"]


def test_xlsx_to_file_rejects_non_workbook():
    with pytest.raises(AttributeError):
        xlsx_to_file("not a workbook", "bad.xlsx")
