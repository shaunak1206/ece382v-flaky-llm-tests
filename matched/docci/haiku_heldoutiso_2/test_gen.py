"""
Unit tests for docci.xlsx: xlsx_to_bytes and xlsx_to_file.
"""

import datetime
import zipfile
from io import BytesIO

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _load(data: bytes) -> Workbook:
    return load_workbook(BytesIO(data))


def _make_workbook() -> Workbook:
    wb = Workbook()
    sheet = wb.active
    sheet.append(["name", "value"])
    sheet.append(["alpha", 1])
    sheet.append(["beta", 2.5])
    return wb


def test_xlsx_to_bytes_returns_bytes_instance():
    result = xlsx_to_bytes(Workbook())

    assert isinstance(result, bytes)


def test_xlsx_to_bytes_returns_non_empty_output():
    result = xlsx_to_bytes(_make_workbook())

    assert len(result) > 0


def test_xlsx_to_bytes_output_is_zip_archive():
    result = xlsx_to_bytes(_make_workbook())

    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_output_contains_workbook_part():
    result = xlsx_to_bytes(_make_workbook())

    with zipfile.ZipFile(BytesIO(result)) as archive:
        assert "xl/workbook.xml" in archive.namelist()


def test_xlsx_to_bytes_round_trip_preserves_cell_values():
    result = xlsx_to_bytes(_make_workbook())

    sheet = _load(result).active
    assert sheet["A1"].value == "name"
    assert sheet["B1"].value == "value"
    assert sheet["A2"].value == "alpha"
    assert sheet["B2"].value == 1
    assert sheet["A3"].value == "beta"
    assert sheet["B3"].value == 2.5


def test_xlsx_to_bytes_preserves_multiple_sheets_and_titles():
    wb = Workbook()
    wb.active.title = "First"
    wb.active["A1"] = "one"
    second = wb.create_sheet("Second")
    second["A1"] = "two"

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.sheetnames == ["First", "Second"]
    assert loaded["First"]["A1"].value == "one"
    assert loaded["Second"]["A1"].value == "two"


def test_xlsx_to_bytes_preserves_active_sheet():
    wb = Workbook()
    wb.create_sheet("Other")
    wb.active = 1

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active.title == "Other"


def test_xlsx_to_bytes_empty_workbook_round_trip():
    loaded = _load(xlsx_to_bytes(Workbook()))

    assert len(loaded.sheetnames) == 1


def test_xlsx_to_bytes_preserves_datetime_values():
    wb = Workbook()
    when = datetime.datetime(2023, 5, 17, 12, 30, 0)
    wb.active["A1"] = when

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active["A1"].value == when


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    wb.active["A1"] = 2
    wb.active["A2"] = 3
    wb.active["A3"] = "=A1+A2"

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active["A3"].value == "=A1+A2"


def test_xlsx_to_bytes_preserves_unicode_text():
    wb = Workbook()
    wb.active["A1"] = "héllo wörld 日本語"

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active["A1"].value == "héllo wörld 日本語"


def test_xlsx_to_bytes_preserves_empty_cells():
    wb = Workbook()
    wb.active["A1"] = "filled"
    wb.active["C3"] = "far"

    loaded = _load(xlsx_to_bytes(wb))

    assert loaded.active["A1"].value == "filled"
    assert loaded.active["B1"].value is None
    assert loaded.active["C3"].value == "far"


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = _make_workbook()
    before = [list(row) for row in wb.active.iter_rows(values_only=True)]

    xlsx_to_bytes(wb)

    after = [list(row) for row in wb.active.iter_rows(values_only=True)]
    assert after == before


def test_xlsx_to_bytes_snapshot_unaffected_by_later_changes():
    wb = _make_workbook()
    data = xlsx_to_bytes(wb)

    wb.active["A2"] = "changed"

    assert _load(data).active["A2"].value == "alpha"


def test_xlsx_to_bytes_repeated_calls_yield_same_content():
    wb = _make_workbook()

    first = _load(xlsx_to_bytes(wb))
    second = _load(xlsx_to_bytes(wb))

    assert [list(r) for r in first.active.iter_rows(values_only=True)] == [
        list(r) for r in second.active.iter_rows(values_only=True)
    ]


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")

    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(Workbook(), "report.xlsx")

    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_matches_xlsx_to_bytes_content():
    wb = _make_workbook()

    result = xlsx_to_file(wb, "data.xlsx")

    loaded = _load(result.content)
    assert loaded.active["A2"].value == "alpha"
    assert loaded.active["B3"].value == 2.5


def test_xlsx_to_file_content_is_bytes():
    result = xlsx_to_file(Workbook(), "empty.xlsx")

    assert isinstance(result.content, bytes)
    assert zipfile.is_zipfile(BytesIO(result.content))
