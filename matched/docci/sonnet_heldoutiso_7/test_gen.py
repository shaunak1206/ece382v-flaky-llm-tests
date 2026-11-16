from io import BytesIO
import zipfile

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import (
    xlsx_to_bytes,
    xlsx_to_file,
    dicts_to_xlsx,
    xlsx_from_bytes,
    xlsx_from_file,
)


def _simple_workbook():
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    ws.append(["a", "b"])
    ws.append([1, 2])
    ws.append(["x", "y"])
    return wb


class FakeWorkbook:
    """Minimal stand-in that only implements save()."""

    def __init__(self, payload=b"fake-content"):
        self.payload = payload
        self.saved_to = None

    def save(self, stream):
        self.saved_to = stream
        stream.write(self.payload)


def test_xlsx_to_bytes_returns_bytes():
    result = xlsx_to_bytes(_simple_workbook())
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_is_valid_zip_with_xlsx_structure():
    result = xlsx_to_bytes(_simple_workbook())
    assert result[:2] == b"PK"
    with zipfile.ZipFile(BytesIO(result)) as zf:
        names = zf.namelist()
    assert "[Content_Types].xml" in names
    assert "xl/workbook.xml" in names


def test_xlsx_to_bytes_roundtrip_preserves_values():
    result = xlsx_to_bytes(_simple_workbook())
    loaded = load_workbook(BytesIO(result))
    ws = loaded.active
    assert ws.title == "Data"
    rows = [tuple(c.value for c in row) for row in ws.iter_rows()]
    assert rows == [("a", "b"), (1, 2), ("x", "y")]


def test_xlsx_to_bytes_empty_workbook():
    result = xlsx_to_bytes(Workbook())
    loaded = load_workbook(BytesIO(result))
    assert len(loaded.sheetnames) == 1
    assert loaded.active.max_row == 1
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_multiple_sheets():
    wb = Workbook()
    wb.active.title = "One"
    wb.create_sheet("Two")["A1"] = "hello"
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded.sheetnames == ["One", "Two"]
    assert loaded["Two"]["A1"].value == "hello"


def test_xlsx_to_bytes_uses_workbook_save_and_returns_written_data():
    fake = FakeWorkbook(b"abc123")
    result = xlsx_to_bytes(fake)
    assert result == b"abc123"
    assert isinstance(fake.saved_to, BytesIO)


def test_xlsx_to_bytes_unicode_values():
    wb = Workbook()
    wb.active["A1"] = "Привет, мир ✓"
    loaded = load_workbook(BytesIO(xlsx_to_bytes(wb)))
    assert loaded.active["A1"].value == "Привет, мир ✓"


def test_xlsx_to_bytes_does_not_modify_original_workbook():
    wb = _simple_workbook()
    xlsx_to_bytes(wb)
    assert wb.active["A1"].value == "a"
    assert wb.active.max_row == 3


def test_xlsx_to_bytes_can_be_called_twice():
    wb = _simple_workbook()
    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)
    for data in (first, second):
        loaded = load_workbook(BytesIO(data))
        assert loaded.active["B2"].value == 2


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_workbook_bytes():
    result = xlsx_to_file(_simple_workbook(), "report.xlsx")
    assert isinstance(result.content, bytes)
    loaded = load_workbook(BytesIO(result.content))
    assert loaded.active["A1"].value == "a"
    assert loaded.active["B2"].value == 2


def test_xlsx_to_file_with_fake_workbook_content():
    fake = FakeWorkbook(b"payload")
    result = xlsx_to_file(fake, "f.xlsx")
    assert result.content == b"payload"
    assert result.name == "f.xlsx"


def test_xlsx_to_file_roundtrip_with_xlsx_from_file():
    wb = dicts_to_xlsx([{"id": 1, "name": "A"}, {"id": 2, "name": "B"}])
    file = xlsx_to_file(wb, "data.xlsx")
    loaded = xlsx_from_file(file)
    rows = [tuple(c.value for c in row) for row in loaded.active.iter_rows()]
    assert rows == [("id", "name"), (1, "A"), (2, "B")]


def test_xlsx_to_bytes_roundtrip_with_xlsx_from_bytes():
    data = xlsx_to_bytes(_simple_workbook())
    loaded = xlsx_from_bytes(data)
    assert loaded.active["A3"].value == "x"


def test_xlsx_to_bytes_raises_on_object_without_save():
    with pytest.raises(AttributeError):
        xlsx_to_bytes(object())
