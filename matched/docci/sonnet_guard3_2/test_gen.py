from datetime import datetime
from io import BytesIO
from zipfile import ZipFile, is_zipfile

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_module
from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _simple_workbook():
    wb = Workbook()
    ws = wb.active
    ws.append(["a", "b", "c"])
    ws.append([1, 2, 3])
    ws.append(["x", "y", "z"])
    return wb


def _rows(wb):
    return [tuple(row) for row in wb.active.iter_rows(values_only=True)]


def test_xlsx_to_bytes_returns_bytes():
    result = xlsx_to_bytes(_simple_workbook())
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_is_valid_zip_archive():
    result = xlsx_to_bytes(_simple_workbook())
    assert result[:2] == b"PK"
    assert is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_contains_workbook_part():
    result = xlsx_to_bytes(_simple_workbook())
    with ZipFile(BytesIO(result)) as zf:
        names = zf.namelist()
    assert "xl/workbook.xml" in names
    assert "[Content_Types].xml" in names


def test_xlsx_to_bytes_roundtrip_values():
    result = xlsx_to_bytes(_simple_workbook())
    loaded = load_workbook(BytesIO(result))
    assert _rows(loaded) == [("a", "b", "c"), (1, 2, 3), ("x", "y", "z")]


def test_xlsx_to_bytes_empty_workbook():
    result = xlsx_to_bytes(Workbook())
    loaded = load_workbook(BytesIO(result))
    assert len(loaded.sheetnames) == 1
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_preserves_sheet_titles_and_order():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_multiple_sheets_content():
    wb = Workbook()
    wb.active.title = "One"
    wb["One"]["A1"] = "hello"
    other = wb.create_sheet("Two")
    other["B2"] = 42
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded["One"]["A1"].value == "hello"
    assert loaded["Two"]["B2"].value == 42


def test_xlsx_to_bytes_unicode_and_types():
    wb = Workbook()
    ws = wb.active
    moment = datetime(2020, 1, 2, 3, 4, 5)
    ws.append(["привет", "日本語", 3.5, True, moment])
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    row = _rows(loaded)[0]
    assert row[0] == "привет"
    assert row[1] == "日本語"
    assert row[2] == pytest.approx(3.5)
    assert row[3] is True
    assert row[4] == moment


def test_xlsx_to_bytes_preserves_formula_text():
    wb = Workbook()
    ws = wb.active
    ws["A1"] = 1
    ws["A2"] = 2
    ws["A3"] = "=SUM(A1:A2)"
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded.active["A3"].value == "=SUM(A1:A2)"


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = _simple_workbook()
    before = _rows(wb)
    xlsx_to_bytes(wb)
    assert _rows(wb) == before
    assert wb.sheetnames == ["Sheet"]


def test_xlsx_to_bytes_can_be_called_twice_with_same_content_type():
    wb = _simple_workbook()
    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)
    assert _rows(load_workbook(BytesIO(first))) == _rows(load_workbook(BytesIO(second)))


def test_xlsx_to_bytes_uses_workbook_save_with_stream():
    class FakeWorkbook:
        def __init__(self):
            self.calls = 0

        def save(self, stream):
            self.calls += 1
            stream.write(b"fake-")
            stream.write(b"content")

    fake = FakeWorkbook()
    assert xlsx_to_bytes(fake) == b"fake-content"
    assert fake.calls == 1


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_loadable_workbook():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert isinstance(result.content, bytes)
    loaded = load_workbook(BytesIO(result.content))
    assert _rows(loaded) == [("a", "b", "c"), (1, 2, 3), ("x", "y", "z")]


def test_xlsx_to_file_uses_xlsx_to_bytes(monkeypatch):
    calls = []

    def fake_to_bytes(wb):
        calls.append(wb)
        return b"sentinel-bytes"

    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", fake_to_bytes)
    wb = Workbook()
    result = xlsx_module.xlsx_to_file(wb, "x.xlsx")
    assert calls == [wb]
    assert result.content == b"sentinel-bytes"
    assert result.name == "x.xlsx"
