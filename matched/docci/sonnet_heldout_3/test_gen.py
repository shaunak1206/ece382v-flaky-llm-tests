import zipfile
from io import BytesIO
from unittest import mock

import pytest
from openpyxl import Workbook, load_workbook

from docci.file import FileAttachment
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _make_workbook():
    wb = Workbook()
    ws = wb.active
    ws.append(["name", "age"])
    ws.append(["Alice", 30])
    ws.append(["Bob", 25])
    return wb


def _load(data):
    return load_workbook(BytesIO(data))


def test_xlsx_to_bytes_returns_bytes():
    result = xlsx_to_bytes(_make_workbook())
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_is_zip_archive():
    result = xlsx_to_bytes(_make_workbook())
    assert result.startswith(b"PK")
    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_contains_content_types():
    result = xlsx_to_bytes(_make_workbook())
    with zipfile.ZipFile(BytesIO(result)) as archive:
        assert "[Content_Types].xml" in archive.namelist()


def test_xlsx_to_bytes_roundtrip_values():
    result = xlsx_to_bytes(_make_workbook())
    rows = [tuple(r) for r in _load(result).active.iter_rows(values_only=True)]
    assert rows == [("name", "age"), ("Alice", 30), ("Bob", 25)]


def test_xlsx_to_bytes_empty_workbook():
    result = xlsx_to_bytes(Workbook())
    wb = _load(result)
    assert wb.active.max_row == 1
    assert wb.active["A1"].value is None


def test_xlsx_to_bytes_preserves_sheet_title():
    wb = Workbook()
    wb.active.title = "Report"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["Report"]


def test_xlsx_to_bytes_multiple_sheets():
    wb = Workbook()
    wb.active.title = "First"
    wb.active["A1"] = "one"
    second = wb.create_sheet("Second")
    second["B2"] = "two"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second"]
    assert loaded["First"]["A1"].value == "one"
    assert loaded["Second"]["B2"].value == "two"


def test_xlsx_to_bytes_preserves_value_types():
    wb = Workbook()
    ws = wb.active
    ws.append([1, 2.5, "text", True])
    loaded = _load(xlsx_to_bytes(wb)).active
    values = [c.value for c in loaded[1]]
    assert values == [1, 2.5, "text", True]
    assert isinstance(values[0], int)
    assert isinstance(values[1], float)


def test_xlsx_to_bytes_unicode_values():
    wb = Workbook()
    wb.active["A1"] = "Привет, мир ✓"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["A1"].value == "Привет, мир ✓"


def test_xlsx_to_bytes_can_be_called_repeatedly():
    wb = _make_workbook()
    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)
    rows1 = list(_load(first).active.iter_rows(values_only=True))
    rows2 = list(_load(second).active.iter_rows(values_only=True))
    assert rows1 == rows2


def test_xlsx_to_bytes_does_not_modify_workbook():
    wb = _make_workbook()
    xlsx_to_bytes(wb)
    assert wb.active["A2"].value == "Alice"
    assert wb.active.max_row == 3


def test_xlsx_to_bytes_calls_save_with_stream():
    wb = mock.MagicMock()
    wb.save.side_effect = lambda stream: stream.write(b"data")
    result = xlsx_to_bytes(wb)
    assert result == b"data"
    wb.save.assert_called_once()
    assert isinstance(wb.save.call_args[0][0], BytesIO)


def test_xlsx_to_bytes_invalid_object_raises():
    with pytest.raises(AttributeError):
        xlsx_to_bytes("not a workbook")


def test_xlsx_to_file_returns_file_attachment():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert isinstance(result, FileAttachment)


def test_xlsx_to_file_sets_name():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert result.name == "report.xlsx"


def test_xlsx_to_file_content_is_valid_xlsx():
    result = xlsx_to_file(_make_workbook(), "report.xlsx")
    assert isinstance(result.content, bytes)
    rows = [tuple(r) for r in _load(result.content).active.iter_rows(values_only=True)]
    assert rows == [("name", "age"), ("Alice", 30), ("Bob", 25)]


def test_xlsx_to_file_uses_xlsx_to_bytes():
    with mock.patch("docci.xlsx.xlsx_to_bytes", return_value=b"fake") as patched:
        wb = Workbook()
        result = xlsx_to_file(wb, "x.xlsx")
    patched.assert_called_once_with(wb)
    assert result.content == b"fake"
    assert result.name == "x.xlsx"


def test_xlsx_to_file_different_names():
    wb = _make_workbook()
    a = xlsx_to_file(wb, "a.xlsx")
    b = xlsx_to_file(wb, "b.xlsx")
    assert a.name == "a.xlsx"
    assert b.name == "b.xlsx"
    assert _load(a.content).active["A1"].value == _load(b.content).active["A1"].value
