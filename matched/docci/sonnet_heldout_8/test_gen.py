import datetime
import zipfile
from io import BytesIO
from unittest import mock

import pytest
from openpyxl import Workbook, load_workbook

import docci.xlsx as xlsx_module
from docci.xlsx import xlsx_to_bytes, xlsx_to_file


def _make_workbook(rows=None):
    wb = Workbook()
    ws = wb.active
    for row in rows or []:
        ws.append(row)
    return wb


def _load(data):
    return load_workbook(BytesIO(data))


def test_xlsx_to_bytes_returns_bytes():
    result = xlsx_to_bytes(_make_workbook([["a", "b"]]))
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_xlsx_to_bytes_is_valid_zip_with_zip_signature():
    result = xlsx_to_bytes(_make_workbook([[1, 2]]))
    assert result[:2] == b"PK"
    assert zipfile.is_zipfile(BytesIO(result))


def test_xlsx_to_bytes_contains_workbook_part():
    result = xlsx_to_bytes(_make_workbook([[1]]))
    with zipfile.ZipFile(BytesIO(result)) as zf:
        names = zf.namelist()
    assert "xl/workbook.xml" in names
    assert "[Content_Types].xml" in names


def test_xlsx_to_bytes_roundtrip_values():
    rows = [["name", "age"], ["Alice", 30], ["Bob", 25]]
    loaded = _load(xlsx_to_bytes(_make_workbook(rows)))
    values = [list(r) for r in loaded.active.iter_rows(values_only=True)]
    assert values == rows


def test_xlsx_to_bytes_empty_workbook():
    loaded = _load(xlsx_to_bytes(Workbook()))
    assert len(loaded.sheetnames) == 1
    assert loaded.active.max_row == 1
    assert loaded.active["A1"].value is None


def test_xlsx_to_bytes_preserves_sheet_titles_and_order():
    wb = Workbook()
    wb.active.title = "First"
    wb.create_sheet("Second")
    wb.create_sheet("Third")
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.sheetnames == ["First", "Second", "Third"]


def test_xlsx_to_bytes_multiple_sheets_data():
    wb = Workbook()
    wb.active.title = "One"
    wb["One"]["A1"] = "x"
    ws2 = wb.create_sheet("Two")
    ws2["B2"] = 42
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded["One"]["A1"].value == "x"
    assert loaded["Two"]["B2"].value == 42


def test_xlsx_to_bytes_unicode_values():
    text = "Привет, мир! ñ 你好"
    loaded = _load(xlsx_to_bytes(_make_workbook([[text]])))
    assert loaded.active["A1"].value == text


def test_xlsx_to_bytes_various_types():
    dt = datetime.datetime(2020, 1, 2, 3, 4, 5)
    rows = [[1, 2.5, True, "s", dt]]
    loaded = _load(xlsx_to_bytes(_make_workbook(rows)))
    values = list(next(loaded.active.iter_rows(values_only=True)))
    assert values[0] == 1
    assert values[1] == 2.5
    assert values[2] is True
    assert values[3] == "s"
    assert values[4] == dt


def test_xlsx_to_bytes_formula_preserved():
    wb = _make_workbook([[1, 2]])
    wb.active["C1"] = "=A1+B1"
    loaded = _load(xlsx_to_bytes(wb))
    assert loaded.active["C1"].value == "=A1+B1"


def test_xlsx_to_bytes_many_rows():
    rows = [[i, str(i)] for i in range(500)]
    loaded = _load(xlsx_to_bytes(_make_workbook(rows)))
    assert loaded.active.max_row == 500
    assert loaded.active["A500"].value == 499


def test_xlsx_to_bytes_does_not_modify_workbook_and_can_repeat():
    wb = _make_workbook([["a", 1]])
    first = xlsx_to_bytes(wb)
    second = xlsx_to_bytes(wb)
    assert wb.active["A1"].value == "a"
    assert _load(first).active["B1"].value == 1
    assert _load(second).active["B1"].value == 1


def test_xlsx_to_bytes_uses_workbook_save_with_bytesio():
    wb = mock.MagicMock()
    wb.save.side_effect = lambda stream: stream.write(b"fake-content")
    result = xlsx_to_bytes(wb)
    assert result == b"fake-content"
    wb.save.assert_called_once()
    assert isinstance(wb.save.call_args[0][0], BytesIO)


def test_xlsx_to_file_content_roundtrip():
    wb = _make_workbook([["k", "v"], ["a", 1]])
    file = xlsx_to_file(wb, "report.xlsx")
    assert isinstance(file.content, bytes)
    loaded = _load(file.content)
    assert [list(r) for r in loaded.active.iter_rows(values_only=True)] == [
        ["k", "v"],
        ["a", 1],
    ]


def test_xlsx_to_file_has_name():
    file = xlsx_to_file(_make_workbook([[1]]), "my_file.xlsx")
    assert file.name == "my_file.xlsx"


def test_xlsx_to_file_uses_xlsx_to_bytes(monkeypatch):
    monkeypatch.setattr(xlsx_module, "xlsx_to_bytes", lambda wb: b"patched-bytes")
    file = xlsx_module.xlsx_to_file(Workbook(), "x.xlsx")
    assert file.content == b"patched-bytes"
    assert file.name == "x.xlsx"


def test_xlsx_to_file_content_equals_xlsx_to_bytes_structure():
    wb = _make_workbook([[1, 2, 3]])
    file = xlsx_to_file(wb, "a.xlsx")
    direct = xlsx_to_bytes(wb)
    with zipfile.ZipFile(BytesIO(file.content)) as z1, zipfile.ZipFile(
        BytesIO(direct)
    ) as z2:
        assert sorted(z1.namelist()) == sorted(z2.namelist())


def test_xlsx_to_bytes_invalid_input_raises():
    with pytest.raises(AttributeError):
        xlsx_to_bytes(None)
